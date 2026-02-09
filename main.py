from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import csv
import io
import os
import logging

from sqlalchemy import (
    create_engine,
    Column,
    Float,
    String,
    Boolean,
    DateTime,
    Integer
)
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("air-quality-server")

# =========================================================
# DATABASE CONFIG
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Fix postgres URL for SQLAlchemy
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://", "postgresql+psycopg://"
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://"
    )

MAX_RECORDS_PER_DEVICE = 1000

# =========================================================
# SQLALCHEMY SETUP
# =========================================================
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()

# =========================================================
# DATABASE MODEL
# =========================================================
class AirQuality(Base):
    __tablename__ = "air_quality"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True)
    device_id = Column(String, index=True)

    temperature = Column(Float)
    humidity = Column(Float)
    co_ppm = Column(Float)
    h2_ppm = Column(Float)
    butane_ppm = Column(Float)

    alert = Column(Boolean)
    co_alert = Column(Boolean)
    butane_alert = Column(Boolean)
    temperature_alert = Column(Boolean)
    humidity_alert = Column(Boolean)

# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(
    title="Air Quality IoT Server",
    description="ESP32 → FastAPI → PostgreSQL → CSV / ML",
    version="1.0.0"
)

# =========================================================
# STARTUP EVENT
# =========================================================
@app.on_event("startup")
def on_startup():
    logger.info("Creating database tables (if not exist)")
    Base.metadata.create_all(bind=engine)

# =========================================================
# DEPENDENCY
# =========================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================================
# ALERT THRESHOLDS
# =========================================================
CO_THRESHOLD = 50.0
BUTANE_THRESHOLD = 10.0
TEMP_MIN = 15.0
TEMP_MAX = 30.0
HUMIDITY_MIN = 20.0
HUMIDITY_MAX = 70.0

# =========================================================
# Pydantic MODEL
# =========================================================
class ESP32Data(BaseModel):
    device_id: str
    temperature: float
    humidity: float
    co_ppm: float
    h2_ppm: float
    butane_ppm: float

# =========================================================
# HELPERS
# =========================================================
def compute_alerts(data: ESP32Data):
    co_alert = data.co_ppm > CO_THRESHOLD
    butane_alert = data.butane_ppm > BUTANE_THRESHOLD
    temperature_alert = not (TEMP_MIN <= data.temperature <= TEMP_MAX)
    humidity_alert = not (HUMIDITY_MIN <= data.humidity <= HUMIDITY_MAX)

    alert = any([
        co_alert,
        butane_alert,
        temperature_alert,
        humidity_alert
    ])

    return alert, co_alert, butane_alert, temperature_alert, humidity_alert


def cleanup_old_records(db: Session, device_id: str):
    count = (
        db.query(AirQuality)
        .filter(AirQuality.device_id == device_id)
        .count()
    )

    if count > MAX_RECORDS_PER_DEVICE:
        to_delete = (
            db.query(AirQuality)
            .filter(AirQuality.device_id == device_id)
            .order_by(AirQuality.timestamp.asc())
            .limit(count - MAX_RECORDS_PER_DEVICE)
            .all()
        )

        for row in to_delete:
            db.delete(row)

        db.commit()
        logger.info(
            f"Cleaned {len(to_delete)} old records for {device_id}"
        )

# =========================================================
# ROUTES
# =========================================================
@app.post("/api/data")
def receive_data(
    data: ESP32Data,
    db: Session = Depends(get_db)
):
    timestamp = datetime.utcnow()

    alert, co_alert, butane_alert, temp_alert, hum_alert = compute_alerts(data)

    record = AirQuality(
        timestamp=timestamp,
        device_id=data.device_id,
        temperature=data.temperature,
        humidity=data.humidity,
        co_ppm=data.co_ppm,
        h2_ppm=data.h2_ppm,
        butane_ppm=data.butane_ppm,
        alert=alert,
        co_alert=co_alert,
        butane_alert=butane_alert,
        temperature_alert=temp_alert,
        humidity_alert=hum_alert
    )

    db.add(record)
    db.commit()

    cleanup_old_records(db, data.device_id)

    return {"status": "ok"}


@app.get("/latest")
def latest(db: Session = Depends(get_db)):
    row = (
        db.query(AirQuality)
        .order_by(AirQuality.timestamp.desc())
        .first()
    )

    if not row:
        return {"message": "No data yet"}

    return {
        "timestamp": row.timestamp.isoformat(),
        "device_id": row.device_id,
        "temperature": row.temperature,
        "humidity": row.humidity,
        "co_ppm": row.co_ppm,
        "h2_ppm": row.h2_ppm,
        "butane_ppm": row.butane_ppm,
        "alert": row.alert
    }


@app.get("/download/csv")
def download_csv(db: Session = Depends(get_db)):
    rows = db.query(AirQuality).all()

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "timestamp",
        "device_id",
        "temperature",
        "humidity",
        "co_ppm",
        "h2_ppm",
        "butane_ppm",
        "alert",
        "co_alert",
        "butane_alert",
        "temperature_alert",
        "humidity_alert"
    ])

    for r in rows:
        writer.writerow([
            r.timestamp.isoformat(),
            r.device_id,
            r.temperature,
            r.humidity,
            r.co_ppm,
            r.h2_ppm,
            r.butane_ppm,
            r.alert,
            r.co_alert,
            r.butane_alert,
            r.temperature_alert,
            r.humidity_alert
        ])

    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=air_quality_data.csv"
        }
    )


@app.get("/health")
def health():
    return {"status": "running"}
