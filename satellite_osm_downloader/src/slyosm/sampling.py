from __future__ import annotations

from typing import Optional, Sequence, Set, Tuple

import numpy as np
import osmnx as ox
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


COORD_KEY_DECIMALS = 6
MAX_SAMPLE_ATTEMPTS_PER_IMAGE = 300


def build_area_polygon_wgs84(area_query: str):
    """Resolve a geocoder query to a valid WGS84 polygon.

    :param area_query: Geocoder query, for example ``Germany``.
    :type area_query: str
    :return: Resolved polygon geometry.
    :rtype: Any
    :raises RuntimeError: If the geocoder response does not contain valid geometry.
    """

    geodataframe = ox.geocode_to_gdf(area_query)
    if geodataframe.empty:
        raise RuntimeError("Area query returned no geometry: {query}".format(query=area_query))

    polygon = unary_union([geometry for geometry in geodataframe.geometry if geometry is not None])
    if polygon.is_empty:
        raise RuntimeError("Area geometry is empty: {query}".format(query=area_query))

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise RuntimeError("Area geometry is invalid after cleanup: {query}".format(query=area_query))

    return polygon


def coordinate_key(lat: float, lon: float) -> str:
    """Build a rounded coordinate key for de-duplication.

    :param lat: Latitude.
    :type lat: float
    :param lon: Longitude.
    :type lon: float
    :return: Stable coordinate key.
    :rtype: str
    """

    return "{lat:.{decimals}f},{lon:.{decimals}f}".format(
        lat=lat,
        lon=lon,
        decimals=COORD_KEY_DECIMALS,
    )


def generate_random_coordinates(
    area_polygon,
    count: int,
    rng: np.random.Generator,
    used_keys: Optional[Set[str]] = None,
) -> Sequence[Tuple[float, float]]:
    """Sample random unique coordinates inside a polygon.

    :param area_polygon: Sampling polygon in WGS84.
    :type area_polygon: Any
    :param count: Number of coordinates to sample.
    :type count: int
    :param rng: NumPy random generator.
    :type rng: np.random.Generator
    :param used_keys: Existing rounded coordinate keys to avoid.
    :type used_keys: Optional[Set[str]]
    :return: Sampled latitude-longitude pairs.
    :rtype: Sequence[Tuple[float, float]]
    :raises RuntimeError: If enough unique points can not be sampled.
    """

    if count <= 0:
        return []

    prepared = prep(area_polygon)
    min_lon, min_lat, max_lon, max_lat = area_polygon.bounds
    known_keys = set(used_keys or set())
    planned_keys = set()
    coordinates = []

    max_attempts = count * MAX_SAMPLE_ATTEMPTS_PER_IMAGE
    attempts = 0
    while len(coordinates) < count and attempts < max_attempts:
        attempts += 1

        lon = float(rng.uniform(min_lon, max_lon))
        lat = float(rng.uniform(min_lat, max_lat))
        if not prepared.contains(Point(lon, lat)):
            continue

        key = coordinate_key(lat, lon)
        if key in known_keys or key in planned_keys:
            continue

        planned_keys.add(key)
        coordinates.append((lat, lon))

    if len(coordinates) < count:
        raise RuntimeError(
            "Failed to sample enough unique points: requested={requested}, sampled={sampled}, attempts={attempts}".format(
                requested=count,
                sampled=len(coordinates),
                attempts=attempts,
            )
        )

    return coordinates
