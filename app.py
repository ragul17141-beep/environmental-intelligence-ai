import streamlit as st
import pandas as pd
import joblib
import requests
import os
import json
import html
import certifi
from datetime import datetime

import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

import streamlit.components.v1 as components
from dotenv import load_dotenv
from streamlit_geolocation import streamlit_geolocation
from influxdb_manager import get_historical_data

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Environmental Intelligence",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

GOOGLE_MAPS_API_KEY = os.getenv(
    "GOOGLE_MAPS_API_KEY",
    ""
).strip()

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "").strip()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "").strip()

INFLUX_URL = os.getenv("INFLUX_URL", "").strip()
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "").strip()
INFLUX_ORG = os.getenv("INFLUX_ORG", "").strip()
INFLUX_BUCKET = os.getenv(
    "INFLUX_BUCKET",
    "environmental_intelligence"
).strip()


# =========================================================
# INFLUXDB CONNECTION
# =========================================================

@st.cache_resource
def get_influx_client():
    """Create a reusable InfluxDB Cloud client when configured."""

    if not all([
        INFLUX_URL,
        INFLUX_TOKEN,
        INFLUX_ORG,
        INFLUX_BUCKET
    ]):
        return None

    try:
        return influxdb_client.InfluxDBClient(
            url=INFLUX_URL,
            token=INFLUX_TOKEN,
            org=INFLUX_ORG,
            verify_ssl=True,
            ssl_ca_cert=certifi.where(),
            timeout=30000
        )
    except Exception:
        return None


influx_client = get_influx_client()


def save_environment_to_influx(
    latitude,
    longitude,
    live_env,
    health_profile,
    selected_station=None
):
    """Save the current environmental reading and GPS location."""

    if influx_client is None:
        return False, "InfluxDB is not configured."

    try:
        write_api = influx_client.write_api(
            write_options=SYNCHRONOUS
        )

        point = (
            Point("environmental_readings")
            .tag("health_profile", str(health_profile))
            .tag("aqi_source", str(live_env.get("aqi_source") or "Unknown"))
        )

        if selected_station:
            point = point.tag(
                "monitoring_station",
                str(selected_station)
            )

        point = (
            point
            .field("latitude", float(latitude))
            .field("longitude", float(longitude))
        )

        field_map = {
            "aqi": "aqi",
            "pm25": "pm25",
            "pm10": "pm10",
            "no2": "no2",
            "o3": "o3",
            "co": "co",
            "so2": "so2"
        }

        for source_key, field_name in field_map.items():
            value = live_env.get(source_key)
            if value is not None:
                point = point.field(
                    field_name,
                    float(value)
                )

        weather = live_env.get("weather", {}) or {}

        weather_map = {
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "apparent_temperature": "apparent_temperature",
            "wind_speed_10m": "wind_speed",
            "wind_direction_10m": "wind_direction",
            "precipitation": "precipitation",
            "weather_code": "weather_code"
        }

        for source_key, field_name in weather_map.items():
            value = weather.get(source_key)
            if value is not None:
                point = point.field(
                    field_name,
                    float(value)
                )

        # Preserve the AI prediction when one exists.
        prediction = st.session_state.get("prediction")
        if prediction is not None:
            point = point.field(
                "predicted_aqi",
                float(prediction)
            )

        write_api.write(
            bucket=INFLUX_BUCKET,
            org=INFLUX_ORG,
            record=point
        )
        write_api.close()

        return True, "Environmental reading saved to InfluxDB."

    except Exception as exc:
        return False, str(exc)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_files():

    model_path = os.path.join(
        BASE_DIR,
        "models",
        "air_quality_model.pkl"
    )

    imputer_path = os.path.join(
        BASE_DIR,
        "models",
        "air_quality_imputer.pkl"
    )

    features_path = os.path.join(
        BASE_DIR,
        "models",
        "model_features.pkl"
    )

    model = joblib.load(
        model_path
    )

    imputer = joblib.load(
        imputer_path
    )

    features = joblib.load(
        features_path
    )

    return model, imputer, features


model, imputer, features = load_model_files()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_data():

    data_path = os.path.join(
        BASE_DIR,
        "delhi_station_hour_with_coords.csv"
    )

    return pd.read_csv(
        data_path
    )


df = load_data()


# =========================================================
# LIVE AQI + WEATHER API
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def get_live_environment_data(latitude, longitude):
    """
    Fetch live air-quality and weather data for the supplied GPS coordinates.

    AQI:
      1. WAQI is used when WAQI_TOKEN is configured.
      2. Otherwise Open-Meteo Air Quality API provides a no-key fallback
         using US AQI and pollutant concentrations.

    Weather:
      OpenWeatherMap Current Weather API is used when
      OPENWEATHER_API_KEY is configured.
    """
    latitude = float(latitude)
    longitude = float(longitude)

    result = {
        "aqi": None,
        "aqi_source": None,
        "pm25": None,
        "pm10": None,
        "no2": None,
        "o3": None,
        "co": None,
        "so2": None,
        "weather": {},
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "errors": []
    }

    # ---------------------------------------------------------
    # WAQI live AQI (optional)
    # ---------------------------------------------------------
    if WAQI_TOKEN:
        try:
            waqi_url = (
                "https://api.waqi.info/feed/geo:"
                f"{latitude:.6f};{longitude:.6f}/"
            )

            waqi_response = requests.get(
                waqi_url,
                params={"token": WAQI_TOKEN},
                timeout=10
            )

            if waqi_response.ok:
                waqi_data = waqi_response.json()

                if waqi_data.get("status") == "ok":
                    waqi_data_block = waqi_data.get("data", {})
                    waqi_aqi = waqi_data_block.get("aqi")

                    if waqi_aqi is not None:
                        result["aqi"] = float(waqi_aqi)
                        result["aqi_source"] = "WAQI"

                    iaqi = waqi_data_block.get("iaqi", {})

                    def waqi_value(name):
                        value = iaqi.get(name, {})
                        if isinstance(value, dict):
                            return value.get("v")
                        return value

                    result["pm25"] = waqi_value("pm25")
                    result["pm10"] = waqi_value("pm10")
                    result["no2"] = waqi_value("no2")
                    result["o3"] = waqi_value("o3")
                    result["co"] = waqi_value("co")
                    result["so2"] = waqi_value("so2")
                else:
                    result["errors"].append(
                        "WAQI did not return live data for this location."
                    )
            else:
                result["errors"].append(
                    f"WAQI request failed ({waqi_response.status_code})."
                )

        except requests.exceptions.RequestException as exc:
            result["errors"].append(f"WAQI connection error: {exc}")
        except Exception as exc:
            result["errors"].append(f"WAQI data error: {exc}")

    # ---------------------------------------------------------
    # Open-Meteo Air Quality fallback / pollutant completion
    # ---------------------------------------------------------
    try:
        air_url = "https://air-quality-api.open-meteo.com/v1/air-quality"

        air_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "us_aqi,pm2_5,pm10,nitrogen_dioxide,"
                "ozone,carbon_monoxide,sulphur_dioxide"
            ),
            "timezone": "auto"
        }

        air_response = requests.get(
            air_url,
            params=air_params,
            timeout=30
        )
        air_response.raise_for_status()
        air_data = air_response.json()
        current_air = air_data.get("current", {})

        if result["aqi"] is None and current_air.get("us_aqi") is not None:
            result["aqi"] = float(current_air["us_aqi"])
            result["aqi_source"] = "Open-Meteo US AQI"

        pollutant_map = {
            "pm25": "pm2_5",
            "pm10": "pm10",
            "no2": "nitrogen_dioxide",
            "o3": "ozone",
            "co": "carbon_monoxide",
            "so2": "sulphur_dioxide"
        }

        for result_key, api_key in pollutant_map.items():
            if result[result_key] is None:
                value = current_air.get(api_key)
                if value is not None:
                    result[result_key] = float(value)

        air_time = current_air.get("time")
        if air_time:
            result["updated_at"] = str(air_time)

    except requests.exceptions.RequestException as exc:
        result["errors"].append(
            f"Open-Meteo air-quality connection error: {exc}"
        )
    except Exception as exc:
        result["errors"].append(
            f"Open-Meteo air-quality data error: {exc}"
        )

    # ---------------------------------------------------------
    # OpenWeatherMap current weather
    # ---------------------------------------------------------
    if not OPENWEATHER_API_KEY:
        result["errors"].append(
            "OpenWeatherMap API key is missing. Add OPENWEATHER_API_KEY to .env."
        )
    else:
        try:
            weather_url = (
                "https://api.openweathermap.org/data/2.5/weather"
            )

            weather_params = {
                "lat": latitude,
                "lon": longitude,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            }

            weather_response = requests.get(
                weather_url,
                params=weather_params,
                timeout=15
            )
            weather_response.raise_for_status()
            weather_data = weather_response.json()

            main = weather_data.get("main", {}) or {}
            wind = weather_data.get("wind", {}) or {}
            weather_items = weather_data.get("weather", []) or []

            description = "Unknown"
            if weather_items:
                description = weather_items[0].get(
                    "description",
                    "Unknown"
                )

            # OpenWeatherMap gives wind speed in m/s for metric units.
            wind_speed_ms = wind.get("speed")
            wind_speed_kmh = (
                float(wind_speed_ms) * 3.6
                if wind_speed_ms is not None
                else None
            )

            # Use rainfall from the last 1 hour when available.
            rain = weather_data.get("rain", {}) or {}
            precipitation = rain.get("1h")
            if precipitation is None:
                precipitation = 0.0

            weather = {
                "temperature_2m": main.get("temp"),
                "relative_humidity_2m": main.get("humidity"),
                "apparent_temperature": main.get("feels_like"),
                "precipitation": precipitation,
                "wind_speed_10m": wind_speed_kmh,
                "wind_direction_10m": wind.get("deg"),
                "description": description,
                "city_name": weather_data.get("name"),
                "country": (
                    weather_data.get("sys", {}) or {}
                ).get("country"),
                "pressure": main.get("pressure"),
                "visibility": weather_data.get("visibility"),
                "openweathermap_timestamp": weather_data.get("dt")
            }

            result["weather"] = weather

            weather_timestamp = weather_data.get("dt")
            if weather_timestamp:
                try:
                    result["weather_updated_at"] = datetime.fromtimestamp(
                        float(weather_timestamp)
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    result["weather_updated_at"] = str(weather_timestamp)

        except requests.exceptions.HTTPError as exc:
            status = getattr(
                weather_response,
                "status_code",
                "unknown"
            )
            result["errors"].append(
                f"OpenWeatherMap request failed ({status}): {exc}"
            )
        except requests.exceptions.RequestException as exc:
            result["errors"].append(
                f"OpenWeatherMap connection error: {exc}"
            )
        except Exception as exc:
            result["errors"].append(
                f"OpenWeatherMap data error: {exc}"
            )

    return result


def weather_description(weather_code):
    """Keep compatibility with older Open-Meteo-style weather data."""
    if isinstance(weather_code, str) and weather_code:
        return weather_code

    try:
        code = int(weather_code)
    except (TypeError, ValueError):
        return "Unknown"

    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    return descriptions.get(code, "Unknown")
def weather_description(weather_code):
    """Convert Open-Meteo WMO weather code to readable text."""
    try:
        code = int(weather_code)
    except (TypeError, ValueError):
        return "Unknown"

    descriptions = {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail"
    }

    return descriptions.get(code, "Unknown")


def live_aqi_category(aqi):
    """Category for the live US-AQI fallback / WAQI numeric value."""
    if aqi is None:
        return "Unavailable"
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


# =========================================================
# GOOGLE ROUTES API
# =========================================================

def get_route(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    url = (
        "https://routes.googleapis.com/"
        "directions/v2:computeRoutes"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": (
            "routes.distanceMeters,"
            "routes.duration,"
            "routes.staticDuration,"
            "routes.polyline.encodedPolyline,"
            "routes.description"
        )
    }

    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": float(start_lat),
                    "longitude": float(start_lon)
                }
            }
        },

        "destination": {
            "location": {
                "latLng": {
                    "latitude": float(end_lat),
                    "longitude": float(end_lon)
                }
            }
        },

        "travelMode": "DRIVE",

        "routingPreference": (
            "TRAFFIC_AWARE_OPTIMAL"
        ),

        "computeAlternativeRoutes": True,

        "units": "METRIC"
    }

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=30
    )

    if response.status_code != 200:

        try:

            error_data = response.json()

            error_message = (
                error_data
                .get("error", {})
                .get(
                    "message",
                    response.text
                )
            )

        except Exception:

            error_message = response.text

        raise Exception(
            f"Google Routes API error "
            f"{response.status_code}: "
            f"{error_message}"
        )

    data = response.json()

    routes = data.get(
        "routes",
        []
    )

    if not routes:

        raise Exception(
            "Google Routes API returned no routes."
        )

    return data


# =========================================================
# DECODE GOOGLE POLYLINE
# =========================================================

def decode_polyline(encoded):

    points = []

    index = 0
    lat = 0
    lng = 0

    while index < len(encoded):

        result = 0
        shift = 0

        while True:

            if index >= len(encoded):
                break

            b = (
                ord(encoded[index])
                - 63
            )

            index += 1

            result |= (
                (b & 0x1F)
                << shift
            )

            shift += 5

            if b < 0x20:
                break

        if result & 1:

            dlat = ~(result >> 1)

        else:

            dlat = result >> 1

        lat += dlat

        result = 0
        shift = 0

        while True:

            if index >= len(encoded):
                break

            b = (
                ord(encoded[index])
                - 63
            )

            index += 1

            result |= (
                (b & 0x1F)
                << shift
            )

            shift += 5

            if b < 0x20:
                break

        if result & 1:

            dlng = ~(result >> 1)

        else:

            dlng = result >> 1

        lng += dlng

        points.append(
            [
                lat / 100000.0,
                lng / 100000.0
            ]
        )

    return points


# =========================================================
# GET ROUTE POINTS
# =========================================================

def get_route_points(route):

    try:

        polyline_data = route.get(
            "polyline",
            {}
        )

        encoded = polyline_data.get(
            "encodedPolyline",
            ""
        )

        if not encoded:

            return []

        return decode_polyline(
            encoded
        )

    except Exception:

        return []


# =========================================================
# GOOGLE DURATION TO MINUTES
# =========================================================

def duration_to_minutes(
    duration_string
):

    if not duration_string:

        return 0.0

    try:

        seconds = float(
            str(duration_string)
            .replace("s", "")
        )

        return seconds / 60.0

    except Exception:

        return 0.0


# =========================================================
# SMART MOBILITY RECOMMENDATION
# =========================================================

def mobility_recommendation(
    aqi,
    traffic_delay
):

    if (
        aqi <= 50
        and
        traffic_delay <= 5
    ):

        return (
            "Normal travel is recommended. "
            "Walking, cycling, public transport or "
            "private vehicles can be considered."
        )

    elif (
        aqi <= 100
        and
        traffic_delay > 15
    ):

        return (
            "Air quality is acceptable, but traffic is high. "
            "Consider public transport or an alternative route."
        )

    elif aqi <= 200:

        return (
            "Moderate pollution detected. "
            "Prefer public transport and reduce prolonged "
            "outdoor exposure."
        )

    elif aqi <= 300:

        return (
            "Poor air quality detected. "
            "Avoid unnecessary travel and consider public transport."
        )

    else:

        return (
            "Very poor air quality detected. "
            "Avoid unnecessary outdoor travel."
        )


# =========================================================
# AQI CATEGORY
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:

        return "Good"

    elif aqi <= 100:

        return "Satisfactory"

    elif aqi <= 200:

        return "Moderate"

    elif aqi <= 300:

        return "Poor"

    elif aqi <= 400:

        return "Very Poor"

    return "Severe"


# =========================================================
# TRAFFIC LEVEL
# =========================================================

def get_traffic_level(delay):

    if delay <= 5:

        return "Low"

    elif delay <= 15:

        return "Moderate"

    elif delay <= 30:

        return "High"

    return "Severe"


# =========================================================
# GET STATION COORDINATES
# =========================================================

def get_station_coordinates(
    data,
    station_name
):

    if (
        "StationName" not in data.columns
        or
        "Latitude" not in data.columns
        or
        "Longitude" not in data.columns
    ):

        return None, None

    rows = data[
        data["StationName"] == station_name
    ].copy()

    if rows.empty:

        return None, None

    rows["Latitude"] = pd.to_numeric(
        rows["Latitude"],
        errors="coerce"
    )

    rows["Longitude"] = pd.to_numeric(
        rows["Longitude"],
        errors="coerce"
    )

    rows = rows.dropna(
        subset=[
            "Latitude",
            "Longitude"
        ]
    )

    if rows.empty:

        return None, None

    return (
        float(rows["Latitude"].iloc[0]),
        float(rows["Longitude"].iloc[0])
    )


# =========================================================
# STATION AQI TABLE
# =========================================================

@st.cache_data
def get_station_aqi_table(data):

    needed = [
        "StationName",
        "Latitude",
        "Longitude",
        "AQI"
    ]

    if not all(
        column in data.columns
        for column in needed
    ):

        return pd.DataFrame(
            columns=needed
        )

    temp = data[
        needed
    ].copy()

    temp["AQI"] = pd.to_numeric(
        temp["AQI"],
        errors="coerce"
    )

    temp["Latitude"] = pd.to_numeric(
        temp["Latitude"],
        errors="coerce"
    )

    temp["Longitude"] = pd.to_numeric(
        temp["Longitude"],
        errors="coerce"
    )

    temp = temp.dropna(
        subset=needed
    )

    if "Datetime" in data.columns:

        temp["_datetime"] = pd.to_datetime(
            data.loc[
                temp.index,
                "Datetime"
            ],
            errors="coerce"
        ).values

        temp = temp.sort_values(
            "_datetime"
        )

    temp = temp.drop_duplicates(
        "StationName",
        keep="last"
    )

    temp = temp.drop(
        columns=["_datetime"],
        errors="ignore"
    )

    return temp.reset_index(
        drop=True
    )


station_table = get_station_aqi_table(
    df
)


# =========================================================
# INTERPOLATE ROUTE AQI USING IDW
# =========================================================

def interpolate_route_aqi(
    route_points,
    station_table
):

    if (
        not route_points
        or
        station_table.empty
    ):

        return 100.0

    station_values = (
        station_table[
            [
                "Latitude",
                "Longitude",
                "AQI"
            ]
        ]
        .to_numpy(dtype=float)
    )

    estimates = []

    step = max(
        1,
        len(route_points) // 60
    )

    for lat, lon in route_points[::step]:

        distances = (
            (
                station_values[:, 0]
                - lat
            ) ** 2
            +
            (
                station_values[:, 1]
                - lon
            ) ** 2
        ) ** 0.5

        nearest = distances.argsort()[:5]

        d = distances[nearest]

        aqi_values = station_values[
            nearest,
            2
        ]

        if d[0] < 1e-8:

            estimates.append(
                float(aqi_values[0])
            )

        else:

            weights = 1 / (
                d ** 2
            )

            estimate = (
                weights * aqi_values
            ).sum() / weights.sum()

            estimates.append(
                float(estimate)
            )

    if not estimates:

        return 100.0

    return float(
        sum(estimates)
        /
        len(estimates)
    )


# =========================================================
# EXPOSURE SCORE
# =========================================================

def calculate_exposure_score(
    avg_aqi,
    travel_time
):

    return max(
        0.0,
        float(avg_aqi)
        *
        float(travel_time)
    )


# =========================================================
# ROUTE SAFETY SCORE
# =========================================================

def calculate_route_safety_scores(
    route_results
):

    if not route_results:

        return route_results

    doses = [
        item["dose"]
        for item in route_results
    ]

    delays = [
        item["delay"]
        for item in route_results
    ]

    aqis = [
        item["avg_aqi"]
        for item in route_results
    ]

    def lower_is_better_score(
        value,
        values
    ):

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:

            return 100.0

        score = (
            100.0
            -
            (
                (
                    value
                    -
                    minimum
                )
                /
                (
                    maximum
                    -
                    minimum
                )
                *
                100.0
            )
        )

        return max(
            0.0,
            min(
                100.0,
                score
            )
        )

    for item in route_results:

        dose_score = lower_is_better_score(
            item["dose"],
            doses
        )

        traffic_score = lower_is_better_score(
            item["delay"],
            delays
        )

        aqi_score = lower_is_better_score(
            item["avg_aqi"],
            aqis
        )

        item["aqi_score"] = aqi_score
        item["exposure_score"] = dose_score
        item["traffic_score"] = traffic_score

        item["safety_score"] = (
            0.45 * dose_score
            +
            0.35 * aqi_score
            +
            0.20 * traffic_score
        )

    return route_results


# =========================================================
# ROUTE ANALYSIS MESSAGE
# =========================================================

def route_analysis_message(
    route_info,
    recommended
):

    if recommended:

        return (
            "🏆 Recommended route: this route has the best "
            "combined pollution-exposure and traffic score."
        )

    messages = []

    if route_info["aqi_difference"] > 1:

        messages.append(
            f"AQI is "
            f"{route_info['aqi_difference']:.0f}% higher "
            f"than the recommended route"
        )

    elif route_info["aqi_difference"] < -1:

        messages.append(
            f"AQI is "
            f"{abs(route_info['aqi_difference']):.0f}% lower "
            f"than the recommended route"
        )

    if route_info["dose_difference"] > 1:

        messages.append(
            f"inhaled dose is "
            f"{route_info['dose_difference']:.0f}% higher"
        )

    elif route_info["dose_difference"] < -1:

        messages.append(
            f"inhaled dose is "
            f"{abs(route_info['dose_difference']):.0f}% lower"
        )

    if route_info["time_difference"] > 1:

        messages.append(
            f"{route_info['time_difference']:.0f} min longer"
        )

    elif route_info["time_difference"] < -1:

        messages.append(
            f"{abs(route_info['time_difference']):.0f} min faster"
        )

    if not messages:

        return (
            "This route has similar conditions "
            "to the recommended route."
        )

    return (
        "• "
        +
        "\n• ".join(messages)
        +
        "."
    )


# =========================================================
# INITIALIZE ROUTE SESSION STATE
# =========================================================

if "exposure_routes" not in st.session_state:

    st.session_state.exposure_routes = None

if "exposure_start" not in st.session_state:

    st.session_state.exposure_start = None

if "exposure_end" not in st.session_state:

    st.session_state.exposure_end = None

if "route_from_name" not in st.session_state:

    st.session_state.route_from_name = None

if "route_to_name" not in st.session_state:

    st.session_state.route_to_name = None

if "route_result_start_mode" not in st.session_state:

    st.session_state.route_result_start_mode = None

if "last_influx_save_key" not in st.session_state:

    st.session_state.last_influx_save_key = None


# =========================================================
# PAGE TITLE
# =========================================================

st.title(
    "🌍 AI-Powered Environmental Intelligence System"
)

st.subheader(
    "Air Quality Prediction and Smart Mobility Recommendations"
)

st.write(
    "An AI-powered system for predicting Air Quality Index (AQI) "
    "and providing environment-aware mobility recommendations."
)

st.markdown("---")


# =========================================================
# MONITORING STATIONS
# =========================================================

st.sidebar.header(
    "📍 Monitoring Station"
)

if "StationName" not in df.columns:

    st.error(
        "StationName column is missing from the dataset."
    )

    st.stop()


stations = sorted(
    df["StationName"]
    .dropna()
    .unique()
    .tolist()
)

if not stations:

    st.error(
        "No monitoring stations were found."
    )

    st.stop()


selected_station = st.sidebar.selectbox(
    "Select Station",
    stations
)


# =========================================================
# SELECTED STATION
# =========================================================

station_data = df[
    df["StationName"] == selected_station
].copy()


if not station_data.empty:

    latitude = pd.to_numeric(
        station_data["Latitude"].iloc[0],
        errors="coerce"
    )

    longitude = pd.to_numeric(
        station_data["Longitude"].iloc[0],
        errors="coerce"
    )

else:

    latitude = 28.7041
    longitude = 77.1025


if pd.isna(latitude):

    latitude = 28.7041


if pd.isna(longitude):

    longitude = 77.1025


# Coordinates used by the trained AQI model
start_lat = float(latitude)
start_lon = float(longitude)


# =========================================================
# CURRENT GPS LOCATION
# =========================================================

st.subheader(
    "📍 Current Location"
)

st.caption(
    "Allow browser location access to use your current "
    "GPS position for route analysis."
)

location = streamlit_geolocation()


if (
    location
    and
    location.get("latitude") is not None
    and
    location.get("longitude") is not None
):

    user_lat = float(
        location["latitude"]
    )

    user_lon = float(
        location["longitude"]
    )

    st.success(
        "✅ Current location detected"
    )

    gps_col1, gps_col2 = st.columns(2)

    with gps_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📍 GPS LATITUDE</div>
                <div class="metric-value">{user_lat:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with gps_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">📍 GPS LONGITUDE</div>
                <div class="metric-value">{user_lon:.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

else:

    user_lat = None
    user_lon = None

    st.warning(
        "📍 Please allow location access in your browser."
    )


# =========================================================
# LIVE AQI + WEATHER DASHBOARD
# =========================================================

st.markdown("---")
st.header("🌫️ Live AQI & 🌦️ Weather Dashboard")

st.caption(
    "Real-time environmental conditions are fetched using the "
    "current GPS coordinates. AQI uses WAQI when a WAQI_TOKEN is "
    "configured; otherwise Open-Meteo US AQI is used as a no-key fallback. "
    "Weather is fetched from OpenWeatherMap."
)

if user_lat is not None and user_lon is not None:

    refresh_col1, refresh_col2 = st.columns([4, 1])

    with refresh_col1:
        st.write(
            f"📍 Monitoring live conditions near "
            f"**{user_lat:.6f}, {user_lon:.6f}**"
        )

    with refresh_col2:
        refresh_live_data = st.button(
            "🔄 Refresh",
            key="refresh_live_environment"
        )

    if refresh_live_data:
        get_live_environment_data.clear()

    try:
        with st.spinner("Fetching live AQI and weather..."):
            live_env = get_live_environment_data(
                user_lat,
                user_lon
            )

            # Keep the latest live environment data available
            # for the personalized health-alert section below.
            st.session_state.live_environment_data = live_env

        live_aqi = live_env.get("aqi")
        aqi_source = live_env.get("aqi_source") or "No AQI source"
        live_weather = live_env.get("weather", {})

        if live_aqi is not None:

            live_category = live_aqi_category(live_aqi)

            st.subheader("🌫️ Live Air Quality")

            aqi_col1, aqi_col2, aqi_col3, aqi_col4 = st.columns(4)

            with aqi_col1:
                st.metric(
                    "Live AQI",
                    f"{live_aqi:.0f}"
                )

            with aqi_col2:
                st.metric(
                    "AQI Category",
                    live_category
                )

            with aqi_col3:
                pm25_live = live_env.get("pm25")
                st.metric(
                    "PM2.5",
                    (
                        f"{float(pm25_live):.1f} µg/m³"
                        if pm25_live is not None
                        else "N/A"
                    )
                )

            with aqi_col4:
                pm10_live = live_env.get("pm10")
                st.metric(
                    "PM10",
                    (
                        f"{float(pm10_live):.1f} µg/m³"
                        if pm10_live is not None
                        else "N/A"
                    )
                )

            pollutant_col1, pollutant_col2, pollutant_col3, pollutant_col4 = (
                st.columns(4)
            )

            with pollutant_col1:
                no2_live = live_env.get("no2")
                st.metric(
                    "NO₂",
                    (
                        f"{float(no2_live):.1f} µg/m³"
                        if no2_live is not None
                        else "N/A"
                    )
                )

            with pollutant_col2:
                o3_live = live_env.get("o3")
                st.metric(
                    "O₃",
                    (
                        f"{float(o3_live):.1f} µg/m³"
                        if o3_live is not None
                        else "N/A"
                    )
                )

            with pollutant_col3:
                co_live = live_env.get("co")
                st.metric(
                    "CO",
                    (
                        f"{float(co_live):.1f} µg/m³"
                        if co_live is not None
                        else "N/A"
                    )
                )

            with pollutant_col4:
                so2_live = live_env.get("so2")
                st.metric(
                    "SO₂",
                    (
                        f"{float(so2_live):.1f} µg/m³"
                        if so2_live is not None
                        else "N/A"
                    )
                )

            st.caption(
                f"AQI source: **{aqi_source}** | "
                f"Last data timestamp: **{live_env.get('updated_at', 'N/A')}**"
            )

            if live_aqi <= 50:
                st.success(
                    "🟢 Live air quality is currently good."
                )
            elif live_aqi <= 100:
                st.info(
                    "🟡 Live air quality is moderate."
                )
            elif live_aqi <= 150:
                st.warning(
                    "🟠 Air quality may affect sensitive individuals."
                )
            elif live_aqi <= 200:
                st.warning(
                    "🔴 Live air quality is unhealthy."
                )
            elif live_aqi <= 300:
                st.error(
                    "🟣 Live air quality is very unhealthy."
                )
            else:
                st.error(
                    "⚠️ Live air quality is hazardous."
                )

        else:
            st.warning(
                "Live AQI is currently unavailable. "
                "Check your AQI data source configuration or internet connection."
            )

        st.subheader("🌦️ Current Weather")

        if live_weather:

            temperature = live_weather.get("temperature_2m")
            humidity = live_weather.get("relative_humidity_2m")
            apparent_temperature = live_weather.get("apparent_temperature")
            wind_speed = live_weather.get("wind_speed_10m")
            wind_direction = live_weather.get("wind_direction_10m")
            precipitation = live_weather.get("precipitation")
            weather_condition = live_weather.get("description", "Unknown")

            weather_col1, weather_col2, weather_col3, weather_col4 = (
                st.columns(4)
            )

            with weather_col1:
                st.metric(
                    "🌡️ Temperature",
                    (
                        f"{float(temperature):.1f} °C"
                        if temperature is not None
                        else "N/A"
                    )
                )

            with weather_col2:
                st.metric(
                    "💧 Humidity",
                    (
                        f"{float(humidity):.0f}%"
                        if humidity is not None
                        else "N/A"
                    )
                )

            with weather_col3:
                st.metric(
                    "💨 Wind Speed",
                    (
                        f"{float(wind_speed):.1f} km/h"
                        if wind_speed is not None
                        else "N/A"
                    )
                )

            with weather_col4:
                st.metric(
                    "🌧️ Precipitation",
                    (
                        f"{float(precipitation):.1f} mm"
                        if precipitation is not None
                        else "N/A"
                    )
                )

            weather_col5, weather_col6, weather_col7 = st.columns(3)

            with weather_col5:
                st.metric(
                    "Feels Like",
                    (
                        f"{float(apparent_temperature):.1f} °C"
                        if apparent_temperature is not None
                        else "N/A"
                    )
                )

            with weather_col6:
                st.metric(
                    "Wind Direction",
                    (
                        f"{float(wind_direction):.0f}°"
                        if wind_direction is not None
                        else "N/A"
                    )
                )

            with weather_col7:
                st.metric(
                    "Condition",
                    weather_description(weather_condition)
                )

            weather_time = live_weather.get("time")

            st.caption(
                f"Weather source: **OpenWeatherMap** | "
                f"Last updated: **{live_env.get('weather_updated_at') or weather_time or 'N/A'}**"
            )

        else:
            st.warning(
                "Current weather data is unavailable."
            )

        # Show API errors only when there is useful diagnostic information.
        api_errors = live_env.get("errors", [])

        if api_errors:
            with st.expander("ℹ️ Live data diagnostics"):
                for error_message in api_errors:
                    st.write(f"• {error_message}")

    except Exception as e:
        st.error(
            f"❌ Live AQI/Weather error: {e}"
        )

else:

    st.info(
        "📍 Allow browser location access above to load "
        "live AQI and weather for your current coordinates."
    )


# =========================================================
# HEALTH SENSITIVITY PROFILE
# =========================================================

st.markdown("---")

st.header(
    "❤️ Health Sensitivity Profile"
)

st.caption(
    "Select a sensitivity profile to personalize pollution alerts. "
    "The thresholds below are project-defined alert settings for demonstration "
    "and are not medical advice."
)

health_profile = st.selectbox(
    "Select Health Profile",
    [
        "General",
        "Sensitive Individual",
        "Asthma / Respiratory Sensitive",
        "Child",
        "Elderly"
    ],
    key="health_profile"
)

# Project-defined alert thresholds.
# They are intended only for the application's personalized-alert feature.
health_thresholds = {
    "General": {
        "warning": 100,
        "critical": 200
    },
    "Sensitive Individual": {
        "warning": 75,
        "critical": 150
    },
    "Asthma / Respiratory Sensitive": {
        "warning": 50,
        "critical": 100
    },
    "Child": {
        "warning": 50,
        "critical": 100
    },
    "Elderly": {
        "warning": 50,
        "critical": 100
    }
}

selected_thresholds = health_thresholds[
    health_profile
]

warning_limit = selected_thresholds[
    "warning"
]

critical_limit = selected_thresholds[
    "critical"
]

profile_col1, profile_col2, profile_col3 = st.columns(3)

with profile_col1:

    st.metric(
        "Selected Profile",
        health_profile
    )

with profile_col2:

    st.metric(
        "Warning Threshold",
        f"AQI {warning_limit}"
    )

with profile_col3:

    st.metric(
        "Critical Threshold",
        f"AQI {critical_limit}"
    )


# ---------------------------------------------------------
# GET CURRENT AQI FOR PERSONALIZED ALERT
# ---------------------------------------------------------

personalized_aqi = None
personalized_aqi_source = None

live_environment_data = st.session_state.get(
    "live_environment_data"
)

if (
    live_environment_data
    and
    live_environment_data.get("aqi") is not None
):

    personalized_aqi = float(
        live_environment_data["aqi"]
    )

    personalized_aqi_source = (
        live_environment_data.get(
            "aqi_source"
        )
        or
        "Live environmental data"
    )

elif st.session_state.get("prediction") is not None:

    personalized_aqi = float(
        st.session_state.prediction
    )

    personalized_aqi_source = (
        "AI model prediction"
    )


# ---------------------------------------------------------
# PERSONALIZED ALERT
# ---------------------------------------------------------

if personalized_aqi is None:

    st.info(
        "Run the Live AQI dashboard or the AI AQI prediction "
        "to generate a personalized alert."
    )

else:

    st.caption(
        f"Alert is based on AQI **{personalized_aqi:.0f}** "
        f"from **{personalized_aqi_source}**."
    )

    if personalized_aqi < warning_limit:

        st.success(
            f"🟢 AQI {personalized_aqi:.0f}: "
            f"Current air quality is below the configured "
            f"warning threshold for the **{health_profile}** profile."
        )

    elif personalized_aqi < critical_limit:

        st.warning(
            f"🟠 AQI {personalized_aqi:.0f}: "
            f"The configured warning threshold for the "
            f"**{health_profile}** profile has been reached. "
            "Consider reducing prolonged outdoor exposure."
        )

    else:

        st.error(
            f"🔴 AQI {personalized_aqi:.0f}: "
            f"The configured critical threshold for the "
            f"**{health_profile}** profile has been reached. "
            "Consider avoiding unnecessary outdoor exposure and "
            "follow appropriate local health guidance."
        )




# =========================================================
# INFLUXDB TIME-SERIES STORAGE
# =========================================================

st.markdown("---")
st.header("🗄️ InfluxDB Environmental Data Storage")

st.caption(
    "Live AQI, pollutants, weather, GPS coordinates, health profile "
    "and model prediction are stored as time-series data when InfluxDB is configured."
)

if influx_client is None:

    st.warning(
        "InfluxDB is not configured. Add INFLUX_URL, INFLUX_TOKEN, "
        "INFLUX_ORG and INFLUX_BUCKET to your .env file."
    )

else:

    influx_col1, influx_col2 = st.columns(2)

    with influx_col1:
        st.success("🟢 InfluxDB connection configured")
        st.write(
            f"**Bucket:** `{INFLUX_BUCKET}`"
        )

    with influx_col2:
        manual_save = st.button(
            "💾 Save Current Reading",
            use_container_width=True,
            key="save_current_environment"
        )

    # Automatically save one reading per cached live-data refresh/location.
    if (
        user_lat is not None
        and user_lon is not None
        and live_environment_data
    ):

        live_timestamp = live_environment_data.get(
            "updated_at",
            ""
        )

        auto_save_key = (
            round(float(user_lat), 5),
            round(float(user_lon), 5),
            str(live_timestamp),
            str(health_profile)
        )

        should_auto_save = (
            st.session_state.last_influx_save_key
            != auto_save_key
        )

        if should_auto_save or manual_save:

            saved, save_message = save_environment_to_influx(
                user_lat,
                user_lon,
                live_environment_data,
                health_profile,
                selected_station
            )

            if saved:
                st.session_state.last_influx_save_key = auto_save_key
                if manual_save:
                    st.success("✅ Current environmental reading saved to InfluxDB.")
                else:
                    st.caption("🗄️ Live environmental reading stored in InfluxDB.")
            else:
                st.error(
                    f"❌ InfluxDB storage error: {save_message}"
                )

    elif manual_save:

        st.warning(
            "Live GPS and environmental data are not available yet. "
            "Allow GPS and load Live AQI first."
        )


# =========================================================
# STATION INFORMATION
# =========================================================

st.header(
    "📍 Station Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Station",
        selected_station
    )

with col2:

    st.metric(
        "Latitude",
        f"{start_lat:.4f}"
    )

with col3:

    st.metric(
        "Longitude",
        f"{start_lon:.4f}"
    )


# =========================================================
# ENVIRONMENTAL CONDITIONS
# =========================================================

st.header(
    "🌫️ Environmental Conditions"
)

col1, col2, col3 = st.columns(3)


with col1:

    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=100.0
    )

    no = st.number_input(
        "NO",
        min_value=0.0,
        value=20.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=40.0
    )


with col2:

    nox = st.number_input(
        "NOx",
        min_value=0.0,
        value=50.0
    )

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=20.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.0
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=20.0
    )


with col3:

    o3 = st.number_input(
        "O3",
        min_value=0.0,
        value=30.0
    )

    benzene = st.number_input(
        "Benzene",
        min_value=0.0,
        value=5.0
    )

    toluene = st.number_input(
        "Toluene",
        min_value=0.0,
        value=10.0
    )

    xylene = st.number_input(
        "Xylene",
        min_value=0.0,
        value=5.0
    )


# =========================================================
# TIME INFORMATION
# =========================================================

st.header(
    "🕐 Time Information"
)

col1, col2, col3, col4 = st.columns(4)


with col1:

    hour = st.slider(
        "Hour",
        0,
        23,
        12
    )


with col2:

    day = st.slider(
        "Day",
        1,
        31,
        15
    )


with col3:

    month = st.slider(
        "Month",
        1,
        12,
        6
    )


with col4:

    day_of_week = st.slider(
        "Day of Week",
        0,
        6,
        2
    )


# =========================================================
# AQI PREDICTION
# =========================================================

if st.button(
    "🔮 Predict AQI",
    use_container_width=True
):

    try:

        input_values = [[
            pm25,
            pm10,
            no,
            no2,
            nox,
            nh3,
            co,
            so2,
            o3,
            benzene,
            toluene,
            xylene,
            hour,
            day,
            month,
            day_of_week,
            start_lat,
            start_lon
        ]]

        if len(features) != len(
            input_values[0]
        ):

            st.error(
                f"Model expects {len(features)} "
                f"features, but the app supplied "
                f"{len(input_values[0])} values."
            )

        else:

            input_data = pd.DataFrame(
                input_values,
                columns=features
            )

            input_data = imputer.transform(
                input_data
            )

            prediction = float(
                model.predict(
                    input_data
                )[0]
            )

            prediction = max(
                0.0,
                prediction
            )

            st.session_state.prediction = prediction

            category = get_aqi_category(
                prediction
            )

            if prediction <= 100:

                risk = "Low"

            elif prediction <= 200:

                risk = "Medium"

            elif prediction <= 300:

                risk = "High"

            else:

                risk = "Very High"


            st.markdown("---")

            st.header(
                "📊 AI Prediction Result"
            )

            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Predicted AQI",
                    f"{prediction:.2f}"
                )


            with col2:

                st.metric(
                    "Air Quality",
                    category
                )


            with col3:

                st.metric(
                    "Risk Level",
                    risk
                )


            st.subheader(
                "💡 Environmental Recommendation"
            )


            if prediction <= 50:

                st.success(
                    "Air quality is good. "
                    "Normal outdoor activities are suitable."
                )

            elif prediction <= 100:

                st.info(
                    "Air quality is satisfactory. "
                    "Sensitive individuals should remain cautious."
                )

            elif prediction <= 200:

                st.warning(
                    "Moderate pollution detected. "
                    "Consider reducing prolonged outdoor exposure."
                )

            elif prediction <= 300:

                st.warning(
                    "Poor air quality detected. "
                    "Avoid prolonged outdoor activities."
                )

            elif prediction <= 400:

                st.error(
                    "Very poor air quality. "
                    "Reduce outdoor exposure."
                )

            else:

                st.error(
                    "Severe air pollution. "
                    "Avoid unnecessary outdoor exposure."
                )


            st.subheader(
                "🚗 Smart Mobility Recommendation"
            )

            st.info(
                mobility_recommendation(
                    prediction,
                    0
                )
            )

    except Exception as e:

        st.error(
            f"AQI prediction error: {e}"
        )


# =========================================================
# SMART ROUTE EXPOSURE ANALYZER
# =========================================================

st.markdown("---")

st.header(
    "🛣️ Smart Route Exposure Analyzer"
)

st.caption(
    "Find a route that minimizes cumulative pollution exposure "
    "while considering Google live traffic."
)


# =========================================================
# ROUTE STARTING LOCATION
# =========================================================

st.subheader(
    "🚦 Route Starting Location"
)

route_start_mode = st.radio(
    "Choose starting location",
    [
        "📍 My Current Location",
        "🏭 Monitoring Station"
    ],
    horizontal=True,
    key="route_start_mode"
)


# =========================================================
# DETERMINE ROUTE START
# =========================================================

if route_start_mode == "📍 My Current Location":

    if (
        user_lat is not None
        and
        user_lon is not None
    ):

        from_lat = float(
            user_lat
        )

        from_lon = float(
            user_lon
        )

        route_from_name = (
            "📍 My Current Location"
        )

        st.success(
            f"GPS location will be used as the "
            f"route starting point: "
            f"{from_lat:.6f}, {from_lon:.6f}"
        )

    else:

        from_lat = None
        from_lon = None

        route_from_name = (
            "📍 My Current Location"
        )

        st.warning(
            "📍 Current GPS location is not available. "
            "Please allow location access in your browser."
        )


else:

    route_from = st.selectbox(
        "FROM — Starting Monitoring Station",
        stations,
        index=stations.index(
            selected_station
        ),
        key="route_from_station"
    )

    from_lat, from_lon = get_station_coordinates(
        df,
        route_from
    )

    route_from_name = route_from

    st.info(
        f"🏭 Starting station: {route_from}"
    )


# =========================================================
# DESTINATION OPTIONS
# =========================================================

if route_start_mode == "🏭 Monitoring Station":

    destination_options = [
        station
        for station in stations
        if station != route_from
    ]

else:

    destination_options = stations


if not destination_options:

    st.warning(
        "No destination monitoring stations are available."
    )

    st.stop()


# =========================================================
# DESTINATION
# =========================================================

route_to = st.selectbox(
    "TO — Destination Monitoring Station",
    destination_options,
    key="route_to_station"
)

route_to_name = route_to


# =========================================================
# DESTINATION COORDINATES
# =========================================================

to_lat, to_lon = get_station_coordinates(
    df,
    route_to
)


# =========================================================
# ROUTE LOCATION SUMMARY
# =========================================================

if (
    from_lat is not None
    and
    from_lon is not None
    and
    to_lat is not None
    and
    to_lon is not None
):

    st.write(
        f"**{route_from_name}** "
        f"({from_lat:.6f}, {from_lon:.6f}) "
        f"→ "
        f"**{route_to_name}** "
        f"({to_lat:.6f}, {to_lon:.6f})"
    )


# =========================================================
# ROUTE AQI INFORMATION
# =========================================================

if (
    not station_table.empty
    and
    route_to
):

    to_aqi_rows = station_table[
        station_table["StationName"]
        ==
        route_to
    ]


    # -----------------------------------------------------
    # STATION TO STATION
    # -----------------------------------------------------

    if route_start_mode == "🏭 Monitoring Station":

        from_aqi_rows = station_table[
            station_table["StationName"]
            ==
            route_from
        ]

        if (
            not from_aqi_rows.empty
            and
            not to_aqi_rows.empty
        ):

            from_aqi = float(
                from_aqi_rows[
                    "AQI"
                ].iloc[0]
            )

            to_aqi = float(
                to_aqi_rows[
                    "AQI"
                ].iloc[0]
            )

            aqi_col1, aqi_col2 = st.columns(2)

            with aqi_col1:

                st.metric(
                    f"AQI — {route_from}",
                    f"{from_aqi:.0f}"
                )

            with aqi_col2:

                st.metric(
                    f"AQI — {route_to}",
                    f"{to_aqi:.0f}"
                )


    # -----------------------------------------------------
    # GPS TO STATION
    # -----------------------------------------------------

    else:

        if not to_aqi_rows.empty:

            to_aqi = float(
                to_aqi_rows[
                    "AQI"
                ].iloc[0]
            )

            st.metric(
                f"Destination AQI — {route_to}",
                f"{to_aqi:.0f}"
            )

            st.caption(
                "Starting AQI is not directly available from "
                "the selected GPS point. Route AQI is estimated "
                "using available monitoring-station data."
            )


# =========================================================
# FIND SAFEST ROUTE
# =========================================================

find_safest_route = st.button(
    "🔍 Find Safest Route",
    type="primary",
    use_container_width=True,
    key="find_safest_route"
)


if find_safest_route:

    # -----------------------------------------------------
    # CHECK GOOGLE API KEY
    # -----------------------------------------------------

    if not GOOGLE_MAPS_API_KEY:

        st.error(
            "❌ Google Maps API key is missing. "
            "Check your .env file."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK START AND END COORDINATES
    # -----------------------------------------------------

    if (
        from_lat is None
        or
        from_lon is None
        or
        to_lat is None
        or
        to_lon is None
    ):

        st.error(
            "❌ Valid coordinates are required "
            "for the starting location and destination."
        )

        st.stop()


    # -----------------------------------------------------
    # CHECK SAME LOCATION
    # -----------------------------------------------------

    if (
        abs(from_lat - to_lat) < 0.000001
        and
        abs(from_lon - to_lon) < 0.000001
    ):

        st.warning(
            "⚠️ Starting location and destination "
            "have the same coordinates."
        )

        st.stop()


    try:

        with st.spinner(
            "Calculating routes, traffic and pollution exposure..."
        ):

            # -------------------------------------------------
            # CALL GOOGLE ROUTES API
            # -------------------------------------------------

            route_data = get_route(
                from_lat,
                from_lon,
                to_lat,
                to_lon
            )

            raw_routes = route_data.get(
                "routes",
                []
            )

            if not raw_routes:

                st.warning(
                    "Google did not return any routes."
                )

                st.stop()


            route_results = []


            # -------------------------------------------------
            # ANALYZE EACH ROUTE
            # -------------------------------------------------

            for route_index, route in enumerate(
                raw_routes
            ):

                # Distance
                distance_meters = float(
                    route.get(
                        "distanceMeters",
                        0
                    )
                )

                distance_km = (
                    distance_meters
                    /
                    1000.0
                )


                # Traffic-aware duration
                traffic_time_min = (
                    duration_to_minutes(
                        route.get(
                            "duration",
                            ""
                        )
                    )
                )


                # Normal/static duration
                normal_time_min = (
                    duration_to_minutes(
                        route.get(
                            "staticDuration",
                            ""
                        )
                    )
                )


                # Traffic delay
                delay_min = max(
                    0.0,
                    traffic_time_min
                    -
                    normal_time_min
                )


                # Route polyline points
                points = get_route_points(
                    route
                )


                # Estimate route AQI
                avg_aqi = interpolate_route_aqi(
                    points,
                    station_table
                )


                # Pollution exposure
                dose = calculate_exposure_score(
                    avg_aqi,
                    traffic_time_min
                )


                # Traffic category
                traffic_level = get_traffic_level(
                    delay_min
                )


                route_results.append({

                    "option":
                        route_index + 1,

                    "distance":
                        distance_km,

                    "time":
                        traffic_time_min,

                    "normal_time":
                        normal_time_min,

                    "delay":
                        delay_min,

                    "traffic":
                        traffic_level,

                    "avg_aqi":
                        avg_aqi,

                    "dose":
                        dose,

                    "points":
                        points

                })


            # -------------------------------------------------
            # CHECK ROUTE RESULTS
            # -------------------------------------------------

            if not route_results:

                st.warning(
                    "Google returned routes, "
                    "but route details were unavailable."
                )

                st.stop()


            # -------------------------------------------------
            # CALCULATE SAFETY SCORES
            # -------------------------------------------------

            route_results = calculate_route_safety_scores(
                route_results
            )


            # -------------------------------------------------
            # FIND RECOMMENDED ROUTE
            # -------------------------------------------------

            recommended = max(
                route_results,
                key=lambda item:
                    item["safety_score"]
            )


            # -------------------------------------------------
            # CALCULATE DIFFERENCES
            # -------------------------------------------------

            for item in route_results:

                item["aqi_difference"] = (
                    (
                        item["avg_aqi"]
                        -
                        recommended["avg_aqi"]
                    )
                    /
                    max(
                        recommended["avg_aqi"],
                        1.0
                    )
                ) * 100.0


                item["dose_difference"] = (
                    (
                        item["dose"]
                        -
                        recommended["dose"]
                    )
                    /
                    max(
                        recommended["dose"],
                        1.0
                    )
                ) * 100.0


                item["time_difference"] = (
                    item["time"]
                    -
                    recommended["time"]
                )


            # -------------------------------------------------
            # SAVE ROUTE RESULTS
            # -------------------------------------------------

            st.session_state.exposure_routes = (
                route_results
            )

            st.session_state.exposure_start = (
                float(from_lat),
                float(from_lon)
            )

            st.session_state.exposure_end = (
                float(to_lat),
                float(to_lon)
            )

            st.session_state.route_from_name = (
                route_from_name
            )

            st.session_state.route_to_name = (
                route_to_name
            )


            # IMPORTANT:
            #
            # Do NOT write:
            #
            # st.session_state.route_start_mode = ...
            #
            # because the radio widget already owns
            # that session-state key.
            #
            # We save the result mode separately.

            st.session_state.route_result_start_mode = (
                route_start_mode
            )


    except requests.exceptions.Timeout:

        st.error(
            "⏱️ Google Routes API request timed out. "
            "Please try again."
        )


    except requests.exceptions.RequestException as e:

        st.error(
            f"❌ Google Maps connection error: {e}"
        )


    except Exception as e:

        st.error(
            f"❌ Route exposure analysis error: {e}"
        )


# =========================================================
# ROUTE RESULTS
# =========================================================

route_results = st.session_state.get(
    "exposure_routes"
)


if route_results:

    st.markdown("---")

    route_from_result = (
        st.session_state.get(
            "route_from_name",
            "Starting Location"
        )
    )

    route_to_result = (
        st.session_state.get(
            "route_to_name",
            "Destination"
        )
    )

    st.subheader(
        f"🛣️ Route Analysis: "
        f"{route_from_result} → {route_to_result}"
    )

    st.caption(
        "Google Routes API + Traffic-Aware Routing + "
        "IDW Pollution Exposure Analysis"
    )


    # =====================================================
    # RECOMMENDED ROUTE
    # =====================================================

    recommended = max(
        route_results,
        key=lambda item:
            item["safety_score"]
    )


    st.success(
        f"🏆 Recommended Route: "
        f"Option #{recommended['option']} "
        f"with safety score "
        f"{recommended['safety_score']:.0f}/100"
    )


    # =====================================================
    # ROUTE CARDS
    # =====================================================

    for item in route_results:

        is_recommended = (
            item["option"]
            ==
            recommended["option"]
        )


        if is_recommended:

            st.success(
                f"🏆 OPTION #{item['option']} — "
                f"RECOMMENDED SAFEST ROUTE"
            )

        else:

            st.markdown(
                f"### 🔵 Option #{item['option']}"
            )


        with st.container(
            border=True
        ):

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Travel Time",
                    f"{item['time']:.0f} min"
                )


            with col2:

                st.metric(
                    "Distance",
                    f"{item['distance']:.2f} km"
                )


            with col3:

                st.metric(
                    "Average AQI",
                    f"{item['avg_aqi']:.0f}"
                )


            with col4:

                st.metric(
                    "Inhaled Dose",
                    f"{item['dose']:,.0f} AQI-min"
                )


            st.write(
                f"**Traffic:** "
                f"{item['traffic']} "
                f"({item['delay']:.0f} min delay)"
            )


            score_col1, score_col2, score_col3, score_col4 = (
                st.columns(4)
            )


            with score_col1:

                st.metric(
                    "AQI Score",
                    f"{item['aqi_score']:.0f}/100"
                )


            with score_col2:

                st.metric(
                    "Exposure Score",
                    f"{item['exposure_score']:.0f}/100"
                )


            with score_col3:

                st.metric(
                    "Traffic Score",
                    f"{item['traffic_score']:.0f}/100"
                )


            with score_col4:

                st.metric(
                    "Safety Score",
                    f"{item['safety_score']:.0f}/100"
                )


            st.info(
                "🤖 **Route Analysis:** "
                +
                route_analysis_message(
                    item,
                    is_recommended
                )
            )


# =========================================================
# GOOGLE LIVE TRAFFIC & SMART ROUTE MAP
# =========================================================

if route_results:

    st.markdown("---")

    st.subheader(
        "🗺️ Google Live Traffic & Smart Route Map"
    )

    st.caption(
        "Google Maps showing current traffic conditions, "
        "recommended route and alternative routes."
    )


    # =====================================================
    # MAP COORDINATES
    # =====================================================

    exposure_start = (
        st.session_state.get(
            "exposure_start"
        )
    )

    exposure_end = (
        st.session_state.get(
            "exposure_end"
        )
    )


    if (
        exposure_start is not None
        and
        exposure_end is not None
    ):

        start_lat_map = float(
            exposure_start[0]
        )

        start_lon_map = float(
            exposure_start[1]
        )

        end_lat_map = float(
            exposure_end[0]
        )

        end_lon_map = float(
            exposure_end[1]
        )


        # =================================================
        # START MODE
        # =================================================

        saved_start_mode = (
            st.session_state.get(
                "route_result_start_mode",
                "🏭 Monitoring Station"
            )
        )


        if (
            saved_start_mode
            ==
            "📍 My Current Location"
        ):

            start_marker_title = (
                "Current GPS Location"
            )

            start_marker_label = "U"

        else:

            start_marker_title = (
                "Starting Monitoring Station"
            )

            start_marker_label = "S"


        # =================================================
        # PREPARE GOOGLE MAP ROUTES
        # =================================================

        google_routes = []


        for item in route_results:

            points = item.get(
                "points",
                []
            )

            if not points:

                continue


            is_recommended = (
                item["option"]
                ==
                recommended["option"]
            )


            google_routes.append({

                "option":
                    item["option"],

                "points":
                    points,

                "recommended":
                    is_recommended,

                "distance":
                    float(item["distance"]),

                "time":
                    float(item["time"]),

                "normal_time":
                    float(item["normal_time"]),

                "delay":
                    float(item["delay"]),

                "traffic":
                    item["traffic"],

                "avg_aqi":
                    float(item["avg_aqi"]),

                "dose":
                    float(item["dose"]),

                "safety_score":
                    float(item["safety_score"])

            })


        routes_json = json.dumps(
            google_routes
        )


        # =================================================
        # MAP CENTER
        # =================================================

        map_center_lat = (
            start_lat_map
            +
            end_lat_map
        ) / 2.0


        map_center_lon = (
            start_lon_map
            +
            end_lon_map
        ) / 2.0


        # =================================================
        # GOOGLE MAP HTML
        # =================================================

        google_map_html = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<style>

html,
body,
#map {{
    height: 100%;
    width: 100%;
    margin: 0;
    padding: 0;
}}

#map {{
    border-radius: 12px;
}}

#traffic-status {{
    position: absolute;
    top: 10px;
    left: 10px;
    background: white;
    padding: 10px 14px;
    border-radius: 8px;
    font-family: Arial, sans-serif;
    font-size: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    z-index: 5;
}}


.metric-card {{ background: #ffffff; padding: 20px; border-radius: 16px; border: 1px solid #e5e7eb; box-shadow: 0 4px 12px rgba(0,0,0,.06); margin-bottom: 15px; }}
.metric-title {{ font-size: 12px; font-weight: 700; color: #64748b; letter-spacing: 1px; }}
.metric-value {{ font-size: 24px; font-weight: 800; color: #0f172a; margin-top: 8px; }}
</style>

</head>


<body>


<div id="traffic-status">

    🚦 <b>Live Traffic</b>

    <br>

    <span style="color:green;">
        🟢 Low
    </span>

    &nbsp;

    <span style="color:#d6a500;">
        🟡 Moderate
    </span>

    &nbsp;

    <span style="color:red;">
        🔴 Heavy
    </span>

</div>


<div id="map"></div>


<script>

let map;

let trafficLayer;


const start = {{

    lat: {start_lat_map},

    lng: {start_lon_map}

}};


const destination = {{

    lat: {end_lat_map},

    lng: {end_lon_map}

}};


const routes = {routes_json};


async function initMap() {{

    const {{ Map }} =
        await google.maps.importLibrary("maps");


    // =================================================
    // CREATE MAP
    // =================================================

    map = new Map(
        document.getElementById("map"),
        {{

            center: {{

                lat: {map_center_lat},

                lng: {map_center_lon}

            }},

            zoom: 11,

            mapTypeId: "roadmap",

            fullscreenControl: true,

            streetViewControl: false,

            mapTypeControl: true,

            zoomControl: true

        }}
    );


    // =================================================
    // LIVE TRAFFIC
    // =================================================

    trafficLayer =
        new google.maps.TrafficLayer({{

            autoRefresh: true

        }});


    trafficLayer.setMap(
        map
    );


    // =================================================
    // START MARKER
    // =================================================

    new google.maps.Marker({{

        position: start,

        map: map,

        title: "{start_marker_title}",

        label: {{

            text: "{start_marker_label}",

            color: "white"

        }}

    }});


    // =================================================
    // DESTINATION MARKER
    // =================================================

    new google.maps.Marker({{

        position: destination,

        map: map,

        title: "Destination",

        label: {{

            text: "D",

            color: "white"

        }}

    }});


    // =================================================
    // MAP BOUNDS
    // =================================================

    const bounds =
        new google.maps.LatLngBounds();


    bounds.extend(start);

    bounds.extend(destination);


    // =================================================
    // DRAW ROUTES
    // =================================================

    routes.forEach(
        function(route) {{

            let path = route.points.map(
                function(point) {{

                    return {{

                        lat: point[0],

                        lng: point[1]

                    }};

                }}
            );


            if (!path.length) {{

                return;

            }}


            let routeColor;

            let routeWeight;


            if (route.recommended) {{

                routeColor = "#00a000";

                routeWeight = 7;

            }}

            else {{

                routeColor = "#1557ff";

                routeWeight = 5;

            }}


            // =================================================
            // DRAW ROUTE POLYLINE
            // =================================================

            const polyline =
                new google.maps.Polyline({{

                    path: path,

                    geodesic: true,

                    strokeColor:
                        routeColor,

                    strokeOpacity:
                        0.85,

                    strokeWeight:
                        routeWeight,

                    map: map

                }});


            // =================================================
            // ROUTE INFO WINDOW
            // =================================================

            const infoWindow =
                new google.maps.InfoWindow();


            polyline.addListener(
                "click",
                function(event) {{

                    const title =
                        route.recommended
                        ?
                        "🏆 Recommended Safest Route"
                        :
                        "🔵 Alternative Route";


                    const content = `

                        <div style="
                            font-family:Arial;
                            min-width:230px;
                        ">

                            <h3>
                                ${{title}}
                            </h3>

                            <b>
                                Route #${{route.option}}
                            </b>

                            <br><br>

                            <b>
                                Distance:
                            </b>

                            ${{route.distance.toFixed(2)}}
                            km

                            <br><br>

                            <b>
                                Travel Time:
                            </b>

                            ${{route.time.toFixed(0)}}
                            min

                            <br><br>

                            <b>
                                Normal Time:
                            </b>

                            ${{route.normal_time.toFixed(0)}}
                            min

                            <br><br>

                            <b>
                                Traffic Delay:
                            </b>

                            ${{route.delay.toFixed(0)}}
                            min

                            <br><br>

                            <b>
                                Traffic:
                            </b>

                            ${{route.traffic}}

                            <br><br>

                            <b>
                                Average AQI:
                            </b>

                            ${{route.avg_aqi.toFixed(0)}}

                            <br><br>

                            <b>
                                Exposure:
                            </b>

                            ${{route.dose.toFixed(0)}}
                            AQI-min

                            <br><br>

                            <b>
                                Safety Score:
                            </b>

                            ${{route.safety_score.toFixed(0)}}
                            /100

                        </div>

                    `;


                    infoWindow.setContent(
                        content
                    );


                    infoWindow.setPosition(
                        event.latLng
                    );


                    infoWindow.open(
                        map
                    );

                }}
            );


            // =================================================
            // ADD ROUTE POINTS TO BOUNDS
            // =================================================

            path.forEach(
                function(point) {{

                    bounds.extend(point);

                }}
            );

        }}
    );


    // =================================================
    // FIT MAP TO ROUTES
    // =================================================

    map.fitBounds(
        bounds
    );


    console.log(
        "Google Live Traffic Layer enabled"
    );

}}


// =====================================================
// GOOGLE MAPS SCRIPT LOADER
// =====================================================

function loadGoogleMaps() {{

    const script =
        document.createElement("script");


    script.src =
        "https://maps.googleapis.com/maps/api/js"
        +
        "?key={GOOGLE_MAPS_API_KEY}"
        +
        "&callback=initMap";


    script.async = true;

    script.defer = true;


    script.onerror = function() {{

        document.getElementById("map")
            .innerHTML =
            "<div style='padding:30px;font-family:Arial;'>"
            +
            "<h3>Google Maps could not load</h3>"
            +
            "<p>"
            +
            "Check your Google Maps API key and "
            +
            "make sure Maps JavaScript API is enabled."
            +
            "</p>"
            +
            "</div>";

    }};


    document.head.appendChild(
        script
    );

}}


loadGoogleMaps();

</script>

</body>

</html>
"""


        # =================================================
        # DISPLAY MAP
        # =================================================

        components.html(
            google_map_html,
            height=700,
            scrolling=False
        )


# =========================================================
# STATION POLLUTION DATA
# =========================================================

st.markdown("---")

st.header(
    "📈 Station Pollution Data"
)


display_columns = [
    "Datetime",
    "PM2.5",
    "PM10",
    "NO2",
    "CO",
    "SO2",
    "O3",
    "AQI"
]


available_columns = [
    column
    for column in display_columns
    if column in station_data.columns
]


if available_columns:

    st.dataframe(
        station_data[
            available_columns
        ].tail(20),
        use_container_width=True
    )

else:

    st.info(
        "No pollution columns are available "
        "for display."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Powered Environmental Intelligence System | "
    "Air Quality Prediction & Smart Mobility Recommendations"
)
# ============================================================
# HISTORICAL AIR QUALITY ANALYTICS
# ============================================================

st.markdown("---")
st.header("📊 Historical Air Quality Analytics")

st.write(
    "Historical air-quality measurements collected from OpenAQ "
    "and stored in InfluxDB Cloud."
)

history_hours = st.selectbox(
    "Select historical period",
    [24, 48, 72, 168, 720],
    format_func=lambda x: (
        "Last 24 Hours" if x == 24 else
        "Last 48 Hours" if x == 48 else
        "Last 72 Hours" if x == 72 else
        "Last 7 Days" if x == 168 else
        "Last 30 Days"
    )
)

try:

    historical_data = get_historical_data(
        hours=history_hours
    )

    if historical_data:

        history_df = pd.DataFrame(historical_data)

        history_df["time"] = pd.to_datetime(
            history_df["time"]
        )

        history_df = history_df.sort_values("time")

        st.subheader("📈 Pollution Trends")

        # PM2.5
        if "pm25" in history_df.columns:

            pm25_df = history_df[
                ["time", "pm25"]
            ].dropna()

            if not pm25_df.empty:

                st.markdown("### PM2.5")

                st.line_chart(
                    pm25_df.set_index("time")["pm25"]
                )

        # PM10
        if "pm10" in history_df.columns:

            pm10_df = history_df[
                ["time", "pm10"]
            ].dropna()

            if not pm10_df.empty:

                st.markdown("### PM10")

                st.line_chart(
                    pm10_df.set_index("time")["pm10"]
                )

        # NO2
        if "no2" in history_df.columns:

            no2_df = history_df[
                ["time", "no2"]
            ].dropna()

            if not no2_df.empty:

                st.markdown("### NO₂")

                st.line_chart(
                    no2_df.set_index("time")["no2"]
                )

        # O3
        if "o3" in history_df.columns:

            o3_df = history_df[
                ["time", "o3"]
            ].dropna()

            if not o3_df.empty:

                st.markdown("### O₃")

                st.line_chart(
                    o3_df.set_index("time")["o3"]
                )

        st.subheader("📋 Historical Records")

        st.dataframe(
            history_df,
            use_container_width=True
        )

    else:

        st.info(
            "No historical air-quality data is available "
            "for the selected period."
        )

except Exception as e:

    st.error(
        f"Unable to load historical air-quality data: {e}"
    )