import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")


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

    if not all([
        INFLUX_URL,
        INFLUX_TOKEN,
        INFLUX_ORG,
        INFLUX_BUCKET
    ]):
        raise ValueError(
            "InfluxDB configuration is missing from .env"
        )

    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    write_api = client.write_api(
        write_options=SYNCHRONOUS
    )

    point = (
        Point("openaq_air_quality")
        .tag("station", str(station_name))
        .tag("location_id", str(location_id))
        .field("latitude", float(latitude))
        .field("longitude", float(longitude))
    )

    if pm25 is not None:
        point = point.field("pm25", float(pm25))

    if pm10 is not None:
        point = point.field("pm10", float(pm10))

    if no2 is not None:
        point = point.field("no2", float(no2))

    if o3 is not None:
        point = point.field("o3", float(o3))

    point = point.time(datetime.now(timezone.utc))

    write_api.write(
        bucket=INFLUX_BUCKET,
        org=INFLUX_ORG,
        record=point
    )

    client.close()

    return True
def get_historical_data(hours=24):
    """
    Read historical OpenAQ air-quality data from InfluxDB.
    """

    client = InfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG
    )

    query_api = client.query_api()

    query = f'''
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -{hours}h)
      |> filter(fn: (r) => r["_measurement"] == "openaq_air_quality")
      |> pivot(
          rowKey: ["_time"],
          columnKey: ["_field"],
          valueColumn: "_value"
      )
      |> sort(columns: ["_time"])
    '''

    try:
        tables = query_api.query(
            query=query,
            org=INFLUX_ORG
        )

        rows = []

        for table in tables:
            for record in table.records:

                row = {
                    "time": record.get_time(),
                    "station": record.values.get("station"),
                    "location_id": record.values.get("location_id"),
                    "latitude": record.values.get("latitude"),
                    "longitude": record.values.get("longitude"),
                    "pm25": record.values.get("pm25"),
                    "pm10": record.values.get("pm10"),
                    "no2": record.values.get("no2"),
                    "o3": record.values.get("o3"),
                }

                rows.append(row)

        return rows

    finally:
        client.close()