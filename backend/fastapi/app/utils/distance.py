from typing import Union
import numpy as np
from pydantic import BaseModel, Field, field_validator
from app.schema.distance import Coordinates



# HAVERSINE FUNCTION
def haversine_distance(
    coord1: Coordinates,
    coord2: Coordinates
) -> float:
    """
    Compute the great-circle distance between two points on Earth
    using the Haversine formula.

    Args:
        coord1 (Coordinates): First geographic point (latitude, longitude).
        coord2 (Coordinates): Second geographic point (latitude, longitude).

    Returns:
        float: Distance between the two points in kilometers.

    Formula:
        Uses spherical trigonometry to calculate shortest distance over Earth's surface.
    """

    R: float = 6371.0  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [coord1.lat, coord1.lon, coord2.lat, coord2.lon]
    )

    dlat: float = lat2 - lat1
    dlon: float = lon2 - lon1

    a: float = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c: float = 2 * np.arcsin(np.sqrt(a))

    return float(R * c)