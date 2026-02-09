from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime
import csv
import io
import os

from sqlalchemy import (
    create_engine, Column, Float, String,
    Boolean, DateTime, Integer
)
from sqlalchemy.orm import sessionmaker, declarative_base

# ================= CONFIG =================
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://")

MAX_RECORDS_PER_DEVICE = 1000

# ================= SQLALCHEMY =================
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class AirQuality(Base):
    __tablename__ = "air_quality"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime)
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

Base.metadata.create_all(bind=engine)

# ================= FASTAPI =================
app = FastAPI(title="Air Quality IoT Server")

# ================= THRESHOLDS =================
CO_THRESHOLD = 50.0
BUTANE_THRESHOLD = 10.0
TEMP_MIN = 15.0
TEMP_MAX = 30.0
HUMIDITY_MIN = 20.0
HUMIDITY_MAX = 70.0

# ================= MODEL =================
class ESP32Data(BaseModel):
    device_id: str
    temperature: float
    humidity: float
    co_ppm: float
    h2_ppm: float
    butane_ppm: float

# ================= HELPERS =================
def compute_alerts(data: ESP32Data):
    co_alert = data.co_ppm > CO_THRESHOLD
    butane_alert = data.butane_ppm > BUTANE_THRESHOLD
    temperature_alert = not (TEMP_MIN <= data.temperature <= TEMP_MAX)
    humidity_alert = not (HUMIDITY_MIN <= data.humidity <= HUMIDITY_MAX)
    alert = any([co_alert, butane_alert, temperature_alert, humidity_alert])
    return alert, co_alert, butane_alert, temperature_alert, humidity_alert

def cleanup_old_records(db, device_id: str):
    count = db.query(AirQuality).filter(
        AirQuality.device_id == device_id
    ).count()

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

# ================= ROUTES =================
@app.post("/api/data")
async def receive_data(data: ESP32Data):
    db = SessionLocal()

    alert, co_alert, butane_alert, temp_alert, hum_alert = compute_alerts(data)

    record = AirQuality(
        timestamp=datetime.utcnow(),
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
    db.close()

    return {"status": "ok"}

@app.get("/latest")
async def latest():
    db = SessionLocal()
    row = db.query(AirQuality).order_by(
        AirQuality.timestamp.desc()
    ).first()
    db.close()

    if not row:
        return {"message": "No data yet"}

    return {
        "timestamp": row.timestamp,
        "device_id": row.device_id,
        "temperature": row.temperature,
        "humidity": row.humidity,
        "co_ppm": row.co_ppm,
        "h2_ppm": row.h2_ppm,
        "butane_ppm": row.butane_ppm,
        "alert": row.alert
    }

@app.get("/download/csv")
async def download_csv():
    db = SessionLocal()
    rows = db.query(AirQuality).all()
    db.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "timestamp","device_id","temperature","humidity",
        "co_ppm","h2_ppm","butane_ppm",
        "alert","co_alert","butane_alert",
        "temperature_alert","humidity_alert"
    ])

    for r in rows:
        writer.writerow([
            r.timestamp, r.device_id, r.temperature, r.humidity,
            r.co_ppm, r.h2_ppm, r.butane_ppm,
            r.alert, r.co_alert, r.butane_alert,
            r.temperature_alert, r.humidity_alert
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=air_quality_data.csv"}
    )

@app.get("/health")
async def health():
    return {"status": "running"}
