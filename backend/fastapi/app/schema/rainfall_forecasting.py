from typing import List
from datetime import date
from pydantic import BaseModel, Field


class ForecastItem(BaseModel):
    '''Represents a single day's rainfall forecast.'''
    date_of_record: date = Field(
        ...,
        description="Date for which rainfall is predicted"
    )
    predicted_rainfall: float = Field(
        ...,
        description="Predicted rainfall in mm"
    )


class ForecastResponse(BaseModel):
    '''Response model for rainfall forecast API.'''
    station_name: str = Field(
        ...,
        description="Name of the weather station"
    )
    start_date: date = Field(
        ...,
        description="Forecast start date"
    )
    num_days: int = Field(
        ...,
        description="Number of days forecasted",
        gt=0
    )
    predictions: List[ForecastItem] = Field(
        ...,
        description="List of daily rainfall predictions"
    )
    

class ForecastRequest(BaseModel):
    '''Request model for rainfall forecast API.'''
    station_name: str = Field(..., description="Weather station name")
    start_date: date = Field(..., description="Forecast start date")
    num_days: int = Field(..., gt=0, description="Number of days to forecast")