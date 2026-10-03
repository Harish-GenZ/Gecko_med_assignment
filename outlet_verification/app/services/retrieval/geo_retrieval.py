import logging
from typing import Any
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger("outlet_verification.retrieval.geo")


def find_nearby_outlets(
    db: Session,
    latitude: float,
    longitude: float,
    radius_meters: float,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieves candidate outlets located within a specified geographic radius (in meters)
    from query coordinates using PostgreSQL earthdistance / GiST index.
    Does NOT evaluate whether candidate is a duplicate; only measures proximity.
    """
    if not (-90.0 <= latitude <= 90.0):
        raise ValueError(f"Latitude {latitude} must be between -90 and 90 degrees.")
    if not (-180.0 <= longitude <= 180.0):
        raise ValueError(f"Longitude {longitude} must be between -180 and 180 degrees.")
    if radius_meters <= 0:
        raise ValueError(f"Radius in meters must be greater than zero, got {radius_meters}.")
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than zero, got {top_k}.")

    query = text("""
        SELECT 
            id,
            name,
            latitude,
            longitude,
            image_url,
            earth_distance(
                ll_to_earth(latitude, longitude), 
                ll_to_earth(:query_lat, :query_lng)
            ) AS distance_meters
        FROM outlets
        WHERE earth_distance(
            ll_to_earth(latitude, longitude), 
            ll_to_earth(:query_lat, :query_lng)
        ) <= :radius_meters
        ORDER BY distance_meters ASC
        LIMIT :top_k;
    """)

    rows = db.execute(
        query,
        {
            "query_lat": float(latitude),
            "query_lng": float(longitude),
            "radius_meters": float(radius_meters),
            "top_k": int(top_k),
        },
    ).fetchall()

    candidates = []
    for row in rows:
        candidates.append({
            "outlet_id": str(row[0]),
            "name": str(row[1]),
            "latitude": float(row[2]),
            "longitude": float(row[3]),
            "image_url": str(row[4]),
            "distance_meters": round(float(row[5]), 2),
        })

    logger.debug(
        "Geographic retrieval at (%f, %f) with radius %.1fm returned %d candidates.",
        latitude,
        longitude,
        radius_meters,
        len(candidates),
    )
    return candidates
