from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import date
from app.schema.distance import Coordinates


class ZonewiseRainfallForecastItem(BaseModel):
    """
    Forecast output for a single day.
    """

    date_of_record: date = Field(
        ...,
        title="Forecast Date",
        description="The date for which rainfall is predicted",
        example="2024-07-01"
    )

    predicted_rainfall: float = Field(
        ...,
        ge=0,
        title="Predicted Rainfall (mm)",
        description="Estimated rainfall for the given date in millimeters",
        example=12.5
    )
    
    

class ZonewiseRainfallForecastRequest(BaseModel):
    """
    Request schema for zone-wise rainfall forecasting.
    """

    location: Coordinates = Field(
        ...,
        description="Latitude and longitude for prediction"
    )

    start_date: date = Field(
        ...,
        description="Forecast start date"
    )

    num_days: int = Field(
        ...,
        gt=0,
        description="Number of days to forecast"
    )

    sensitivity: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of nearest stations to consider"
    )


class ZonewiseRainfallForecastResponse(BaseModel):
    """
    Final response for zone-wise rainfall forecasting.
    """

    location: Coordinates = Field(
        ...,
        title="Input Location",
        description="Geographic coordinates for which forecast is generated"
    )

    dominant_zone: str = Field(
        ...,
        title="Dominant Monsoon Zone",
        description="Most influential monsoon zone determined from nearest weather stations",
        example="LOW_MONSOON"
    )

    start_date: date = Field(
        ...,
        title="Forecast Start Date",
        description="Start date of the rainfall prediction period",
        example="2024-07-01"
    )

    num_days: int = Field(
        ...,
        gt=0,
        title="Number of Days",
        description="Total number of days for which forecast is generated",
        example=7
    )

    sensitivity: int = Field(
        ...,
        ge=1,
        le=50,
        title="Station Sensitivity",
        description="Number of nearest weather stations considered for prediction",
        example=5
    )

    predictions: List[ZonewiseRainfallForecastItem] = Field(
        ...,
        title="Forecast Results",
        description="List of daily rainfall predictions"
    )

    @field_validator("num_days")
    @classmethod
    def validate_days(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_days must be > 0")
        return v

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, v: int) -> int:
        if not (1 <= v <= 50):
            raise ValueError("sensitivity must be between 1 and 50")
        return v