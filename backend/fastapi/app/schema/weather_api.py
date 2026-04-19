from pydantic import BaseModel, Field
from typing import List, Optional
from app.schema.distance import Coordinates


class CurrentWeatherRequest(BaseModel):
    """
    Request schema for current weather using geographic coordinates.
    """

    coordinates: Coordinates = Field(
        ...,
        title="Location Coordinates",
        description="Latitude and longitude for fetching current weather",
        examples=[{"lat": 12.9716, "lon": 77.5946}]
    )

# LAT/LON MODEL
class LatLon(BaseModel):
    latitude: float = Field(
        ...,
        description="Latitude in decimal degrees (-90 to 90)",
        examples=[12.9716],
        ge=-90,
        le=90
    )
    longitude: float = Field(
        ...,
        description="Longitude in decimal degrees (-180 to 180)",
        examples=[77.5946],
        ge=-180,
        le=180
    )


# AQI MODEL
class AQI(BaseModel):
    pm2_5: Optional[float] = Field(
        None,
        description="PM2.5 concentration (fine particulate matter ≤2.5µm) in µg/m³",
        examples=[35.4]
    )
    pm10: Optional[float] = Field(
        None,
        description="PM10 concentration (particles ≤10µm) in µg/m³",
        examples=[70.2]
    )
    co: Optional[float] = Field(
        None,
        description="Carbon Monoxide (CO) concentration in µg/m³",
        examples=[200.5]
    )
    no2: Optional[float] = Field(
        None,
        description="Nitrogen Dioxide (NO2) concentration in µg/m³",
        examples=[15.3]
    )
    o3: Optional[float] = Field(
        None,
        description="Ozone (O3) concentration in µg/m³",
        examples=[120.7]
    )
    so2: Optional[float] = Field(
        None,
        description="Sulfur Dioxide (SO2) concentration in µg/m³",
        examples=[5.2]
    )


# CURRENT WEATHER MODEL
class CurrentWeather(BaseModel):
    location: str = Field(
        ...,
        description="Name of the city or station",
        examples=["Bangalore"]
    )
    region: str = Field(
        ...,
        description="Administrative region/state of the location",
        examples=["Karnataka"]
    )
    country: str = Field(
        ...,
        description="Country of the location",
        examples=["India"]
    )
    lat: float = Field(
        ...,
        description="Latitude of the location",
        examples=[12.9716]
    )
    lon: float = Field(
        ...,
        description="Longitude of the location",
        examples=[77.5946]
    )

    temperature_c: float = Field(
        ...,
        description="Current temperature in Celsius",
        examples=[28.5]
    )
    feels_like_c: float = Field(
        ...,
        description="Perceived temperature considering humidity and wind",
        examples=[30.1]
    )

    humidity: int = Field(
        ...,
        description="Relative humidity percentage",
        ge=0,
        le=100,
        examples=[65]
    )
    pressure_mb: float = Field(
        ...,
        description="Atmospheric pressure in millibars",
        examples=[1012]
    )
    visibility_km: float = Field(
        ...,
        description="Visibility distance in kilometers",
        examples=[10.0]
    )

    wind_kph: float = Field(
        ...,
        description="Wind speed in kilometers per hour",
        examples=[15.2]
    )
    wind_degree: int = Field(
        ...,
        description="Wind direction in degrees (0–360)",
        ge=0,
        le=360,
        examples=[180]
    )
    wind_direction: str = Field(
        ...,
        description="Compass direction of wind (e.g., N, NE, SW)",
        examples=["SW"]
    )

    condition: str = Field(
        ...,
        description="Weather condition description",
        examples=["Partly cloudy"]
    )
    cloud: int = Field(
        ...,
        description="Cloud cover percentage",
        ge=0,
        le=100,
        examples=[75]
    )
    uv: float = Field(
        ...,
        description="UV index indicating sun exposure risk",
        examples=[6.5]
    )

    precip_mm: float = Field(
        ...,
        description="Precipitation in millimeters",
        examples=[2.3]
    )
    is_raining: bool = Field(
        ...,
        description="Indicates whether it is currently raining",
        examples=[True]
    )

    aqi: AQI = Field(
        ...,
        description="Air Quality Index metrics including pollutants"
    )


# FORECAST DAY MODEL
class WeatherForecastDay(BaseModel):
    date: str = Field(
        ...,
        description="Forecast date in YYYY-MM-DD format",
        examples=["2026-03-30"]
    )

    avg_temp_c: float = Field(
        ...,
        description="Average temperature for the day in Celsius",
        examples=[27.5]
    )
    max_temp_c: float = Field(
        ...,
        description="Maximum temperature for the day in Celsius",
        examples=[32.0]
    )
    min_temp_c: float = Field(
        ...,
        description="Minimum temperature for the day in Celsius",
        examples=[22.1]
    )

    total_precip_mm: float = Field(
        ...,
        description="Total precipitation expected in millimeters",
        examples=[5.6]
    )
    rain_probability: int = Field(
        ...,
        description="Probability of rain in percentage",
        ge=0,
        le=100,
        examples=[80]
    )
    will_rain: bool = Field(
        ...,
        description="Indicates if rain is likely based on probability threshold",
        examples=[True]
    )

    max_wind_kph: float = Field(
        ...,
        description="Maximum wind speed for the day in km/h",
        examples=[20.5]
    )
    condition: str = Field(
        ...,
        description="General weather condition for the day",
        examples=["Light rain"]
    )


class WeatherForecastResponse(BaseModel):
    """
    Weather forecast response using geographic coordinates.
    """

    location: Coordinates = Field(
        ...,
        title="Location Coordinates",
        description="Latitude and longitude for which the forecast is generated"
    )

    resolved_name: str = Field(
        ...,
        title="Resolved Location Name",
        description="Human-readable name returned by the weather API (e.g., city name)",
        examples=["Delhi"]
    )

    forecast: List[WeatherForecastDay] = Field(
        ...,
        title="Forecast Data",
        description="List of daily weather forecasts"
    )