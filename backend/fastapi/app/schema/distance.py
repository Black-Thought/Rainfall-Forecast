from typing import Union
import numpy as np
from pydantic import BaseModel, Field, field_validator


class Coordinates(BaseModel):
    """
    Schema for geographic coordinates with validation.
    """

    lat: float = Field(..., description="Latitude in degrees (-90 to 90)")
    lon: float = Field(..., description="Longitude in degrees (-180 to 180)")

    @field_validator("lat")
    @classmethod
    def validate_lat(cls, v: float) -> float:
        if not (-90 <= v <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        return v

    @field_validator("lon")
    @classmethod
    def validate_lon(cls, v: float) -> float:
        if not (-180 <= v <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")
        return v