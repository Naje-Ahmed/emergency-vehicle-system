"""
SQLite Event Logging and Incident Telemetry for AegisFlow.
Logs all detected emergency events, priority overrides, and clearance statistics.
"""

import sqlite3
import datetime
import os
import pandas as pd
from typing import List, Dict, Any, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events.db")

def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes the database schema if it doesn't already exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            vehicle_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            lane_id TEXT NOT NULL,
            urgency_score REAL NOT NULL,
            speed_px_s REAL NOT NULL,
            siren_detected INTEGER NOT NULL,
            override_status TEXT NOT NULL,
            clearance_time_sec REAL,
            snapshot_path TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            log_level TEXT NOT NULL,
            message TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def log_incident(
    vehicle_type: str,
    confidence: float,
    lane_id: str,
    urgency_score: float,
    speed_px_s: float,
    siren_detected: bool,
    override_status: str = "TRIGGERED",
    clearance_time_sec: Optional[float] = None,
    snapshot_path: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """Logs an emergency vehicle detection and priority preemption event."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO incidents (
            timestamp, vehicle_type, confidence, lane_id,
            urgency_score, speed_px_s, siren_detected,
            override_status, clearance_time_sec, snapshot_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        vehicle_type,
        round(float(confidence), 3),
        str(lane_id),
        round(float(urgency_score), 2),
        round(float(speed_px_s), 1),
        1 if siren_detected else 0,
        override_status,
        round(float(clearance_time_sec), 2) if clearance_time_sec is not None else 0.0,
        snapshot_path or ""
    ))
    
    incident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return incident_id

def update_clearance_time(incident_id: int, clearance_time_sec: float, db_path: str = DEFAULT_DB_PATH):
    """Updates the clearance duration for a specific incident."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE incidents
        SET clearance_time_sec = ?, override_status = 'CLEARED'
        WHERE id = ?
    """, (round(float(clearance_time_sec), 2), incident_id))
    conn.commit()
    conn.close()

def get_recent_incidents(limit: int = 50, db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Fetches recent emergency incidents as a pandas DataFrame."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM incidents ORDER BY id DESC LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_incident_summary(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculates high-level telemetry summary statistics."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM incidents")
    total_incidents = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT AVG(urgency_score) FROM incidents")
    avg_urgency = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT AVG(clearance_time_sec) FROM incidents WHERE clearance_time_sec > 0")
    avg_clearance = cursor.fetchone()[0] or 0.0
    
    cursor.execute("SELECT COUNT(*) FROM incidents WHERE siren_detected = 1")
    siren_confirmed = cursor.fetchone()[0] or 0
    
    cursor.execute("SELECT vehicle_type, COUNT(*) FROM incidents GROUP BY vehicle_type")
    by_type = dict(cursor.fetchall())
    
    conn.close()
    return {
        "total_incidents": total_incidents,
        "avg_urgency": round(avg_urgency, 2),
        "avg_clearance_sec": round(avg_clearance, 2),
        "siren_confirmed_count": siren_confirmed,
        "incidents_by_type": by_type
    }

def clear_all_records(db_path: str = DEFAULT_DB_PATH):
    """Clears all records for clean resets."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents")
    cursor.execute("DELETE FROM system_logs")
    conn.commit()
    conn.close()
