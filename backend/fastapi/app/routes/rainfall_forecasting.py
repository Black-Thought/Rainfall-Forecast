from datetime import date
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from app.schema.rainfall_forecasting import (
    ForecastResponse,
    ForecastRequest
)
from app.src.forecast_rainfall_pipeline import forecast_rainfall_pipeline


router = APIRouter(prefix="/forecast/rainfall", tags=["Forecast"])


@router.post("/", response_model=ForecastResponse)
def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
    """
    Forecast rainfall for a given station.
    """
    try:
        result = forecast_rainfall_pipeline(
            station_name=request.station_name,
            start_date=request.start_date,
            num_days=request.num_days
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))