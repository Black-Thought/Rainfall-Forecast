from fastapi import APIRouter, HTTPException

from app.schema.rainfall_forecasting_zonewise import (
    ZonewiseRainfallForecastRequest,
    ZonewiseRainfallForecastResponse
)

from app.src.forecast_rainfall_zonewise_pipeline import (
    forecast_rainfall_zonewise_pipeline
)

router = APIRouter(prefix="/forecast/rainfall/zonewise", tags=["Forecast"])


@router.post("/", response_model=ZonewiseRainfallForecastResponse)
def forecast_zonewise_endpoint(
    request: ZonewiseRainfallForecastRequest
) -> ZonewiseRainfallForecastResponse:
    """
    Forecast rainfall using zone-aware model based on nearest stations.
    """

    try:
        result = forecast_rainfall_zonewise_pipeline(
            coordinates=request.location,
            start_date=request.start_date,
            num_days=request.num_days,
            sensitivity=request.sensitivity,
        )
        return result

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))