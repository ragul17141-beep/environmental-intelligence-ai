import os
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


# =========================================================
# LOAD LOCAL .ENV
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

load_dotenv(
    os.path.join(BASE_DIR, ".env")
)


# =========================================================
# READ CONFIG
# Supports local .env + Streamlit Cloud Secrets
# =========================================================

def get_secret(name, default=""):

    # Local .env
    value = os.getenv(name)

    if value is not None and str(value).strip():
        return str(value).strip()

    # Streamlit Cloud
    try:
        value = st.secrets[name]

        if value is not None:
            return str(value).strip()

    except Exception:
        pass

    return default


INFLUX_URL = get_secret("INFLUX_URL")
INFLUX_TOKEN = get_secret("INFLUX_TOKEN")
INFLUX_ORG = get_secret("INFLUX_ORG")
INFLUX_BUCKET = get_secret(
    "INFLUX_BUCKET",
    "environmental_intelligence"
)


# =========================================================
# CREATE INFLUX CLIENT
# =========================================================

def create_influx_client():

    url = str(INFLUX_URL).strip()

    if not url:
        raise ValueError(
            "INFLUX_URL is missing."
        )

    if not (
        url.startswith("https://")
        or url.startswith("http://")
    ):
        raise ValueError(
            "INFLUX_URL must start with http:// or https://"
        )

    if not INFLUX_TOKEN:
        raise ValueError(
            "INFLUX_TOKEN is missing."
        )

    if not INFLUX_ORG:
        raise ValueError(
            "INFLUX_ORG is missing."
        )

    if not INFLUX_BUCKET:
        raise ValueError(
            "INFLUX_BUCKET is missing."
        )

    return InfluxDBClient(
        url=url,
        token=str(INFLUX_TOKEN),
        org=str(INFLUX_ORG),
        timeout=30000
    )


# =========================================================
# SAVE OPENAQ DATA
# =========================================================

def save_openaq_data(
    station_name,
    location_id,
    latitude,
    longitude,
    pm25=None,
    pm10=None,
    no2=None,
    o3=None
):

    client = create_influx_client()

    try:

        write_api = client.write_api(
            write_options=SYNCHRONOUS
        )

        point = (
            Point("openaq_air_quality")
            .tag(
                "station",
                str(station_name)
            )
            .tag(
                "location_id",
                str(location_id)
            )
            .field(
                "latitude",
                float(latitude)
            )
            .field(
                "longitude",
                float(longitude)
            )
        )

        if pm25 is not None:
            point = point.field(
                "pm25",
                float(pm25)
            )

        if pm10 is not None:
            point = point.field(
                "pm10",
                float(pm10)
            )

        if no2 is not None:
            point = point.field(
                "no2",
                float(no2)
            )

        if o3 is not None:
            point = point.field(
                "o3",
                float(o3)
            )

        point = point.time(
            datetime.now(timezone.utc)
        )

        write_api.write(
            bucket=str(INFLUX_BUCKET),
            org=str(INFLUX_ORG),
            record=point
        )

        write_api.close()

        return True

    finally:
        client.close()


# =========================================================
# GET HISTORICAL DATA
# =========================================================

def get_historical_data(hours=24):

    client = create_influx_client()

    try:

        query_api = client.query_api()

        query = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: -{int(hours)}h)
  |> filter(
      fn: (r) =>
      r["_measurement"] == "openaq_air_quality"
  )
  |> pivot(
      rowKey: ["_time"],
      columnKey: ["_field"],
      valueColumn: "_value"
  )
  |> sort(columns: ["_time"])
'''

        tables = query_api.query(
            query=query,
            org=str(INFLUX_ORG)
        )

        rows = []

        for table in tables:

            for record in table.records:

                rows.append({
                    "time": record.get_time(),
                    "station": record.values.get("station"),
                    "location_id": record.values.get("location_id"),
                    "latitude": record.values.get("latitude"),
                    "longitude": record.values.get("longitude"),
                    "pm25": record.values.get("pm25"),
                    "pm10": record.values.get("pm10"),
                    "no2": record.values.get("no2"),
                    "o3": record.values.get("o3")
                })

        return rows

    finally:

        client.close()
