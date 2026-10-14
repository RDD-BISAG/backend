"""
backend package — GIS and Geolocation Layer for Road Damage Detection.

Provides:
- GPXProcessor: GPX parsing, validation, and temporal GPS interpolation.
- GPSMatcher: Video FPS probing and defect frame-to-GPS synchronization.
- GeoJSONGenerator: RFC 7946 GeoJSON FeatureCollection generation with [lon, lat] ordering.
- GeoService: High-level pipeline enrichment orchestrator.
- enrich_inspection_with_gis: Convenience functional API.
"""

from backend.gpx_processor import GPXProcessor, GPXPoint, parse_iso_timestamp, haversine_distance
from backend.gps_matcher import GPSMatcher, SyncConfig, probe_video
from backend.geojson_generator import GeoJSONGenerator
from backend.geo_service import GeoService, enrich_inspection_with_gis, generate_leaflet_map_html

__all__ = [
    "GPXProcessor",
    "GPXPoint",
    "parse_iso_timestamp",
    "haversine_distance",
    "GPSMatcher",
    "SyncConfig",
    "probe_video",
    "GeoJSONGenerator",
    "GeoService",
    "enrich_inspection_with_gis",
    "generate_leaflet_map_html",
]
