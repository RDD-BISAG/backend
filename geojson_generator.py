"""
backend/geojson_generator.py — GeoJSON FeatureCollection Generation and Validation.

Constructs RFC 7946 compliant GeoJSON FeatureCollections for finalized road defects
and the survey route.

CRITICAL REQUIREMENT:
GeoJSON coordinate order is STRICTLY [longitude, latitude], NEVER [latitude, longitude].
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class GeoJSONGenerator:
    """
    Builds and validates GeoJSON FeatureCollection documents containing
    defect Point features and survey route LineString features.
    """

    @staticmethod
    def create_defect_feature(
        matched_defect: Dict[str, Any],
        image_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a GeoJSON Point feature for a single matched defect.
        
        Args:
            matched_defect: Dictionary of defect properties from GPSMatcher.
            image_uri: Resolved crop image path or base64 data URI.
            
        Returns:
            GeoJSON Feature dict.
        """
        lat = float(matched_defect["latitude"])
        lon = float(matched_defect["longitude"])

        conf = matched_defect.get("confidence", 0.0)
        conf_pct = f"{conf * 100:.1f}%" if conf <= 1.0 else f"{conf:.1f}%"

        props = {
            "defect_id": matched_defect.get("defect_id", ""),
            "defect_type": matched_defect.get("defect_type", "Unknown"),
            "confidence": conf,
            "confidence_pct": conf_pct,
            "avg_confidence": matched_defect.get("avg_confidence", conf),
            "representative_frame": matched_defect.get("representative_frame", 0),
            "video_time_seconds": matched_defect.get("video_time_seconds", 0.0),
            "video_time_formatted": matched_defect.get("video_time_formatted", ""),
            "latitude": lat,
            "longitude": lon,
            "image": image_uri or matched_defect.get("crop_path", ""),
            "feature_type": "defect",
        }

        # Optional metadata fields if available
        for opt_key in (
            "first_frame", "last_frame", "frames_observed",
            "source_track_ids", "pixel_area", "mask_path", "elevation",
            "real_area_m2"
        ):
            if matched_defect.get(opt_key) is not None:
                props[opt_key] = matched_defect[opt_key]

        return {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                # STRICT GEOJSON ORDER: [longitude, latitude]
                "coordinates": [lon, lat],
            },
            "properties": props,
        }

    @staticmethod
    def create_route_feature(
        route_coordinates: List[List[float]],
        route_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a GeoJSON LineString feature representing the survey trajectory.
        
        Args:
            route_coordinates: List of [longitude, latitude] coordinate pairs.
            route_metadata: Optional metadata (distance, start_time, etc.).
            
        Returns:
            GeoJSON Feature dict with LineString geometry.
        """
        props = {
            "name": "Survey Route",
            "feature_type": "survey_route",
            "total_points": len(route_coordinates),
        }
        if route_metadata:
            props.update(route_metadata)

        return {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": route_coordinates,
            },
            "properties": props,
        }

    @classmethod
    def generate_feature_collection(
        cls,
        matched_defects: List[Dict[str, Any]],
        route_coordinates: Optional[List[List[float]]] = None,
        route_metadata: Optional[Dict[str, Any]] = None,
        image_uri_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Assemble a complete GeoJSON FeatureCollection containing all defects
        and the optional survey route.
        
        Args:
            matched_defects: List of matched defect dictionaries.
            route_coordinates: Optional list of [longitude, latitude] pairs.
            route_metadata: Optional metadata about the route.
            image_uri_map: Map of defect_id -> resolved image URI.
            
        Returns:
            GeoJSON FeatureCollection dictionary.
        """
        features = []
        image_map = image_uri_map or {}

        # 1. Add defect Point features (one per unique finalized defect)
        for defect in matched_defects:
            did = defect.get("defect_id", "")
            img_uri = image_map.get(did, defect.get("crop_path"))
            feature = cls.create_defect_feature(defect, image_uri=img_uri)
            features.append(feature)

        # 2. Add survey route LineString feature if provided
        if route_coordinates and len(route_coordinates) >= 2:
            route_feat = cls.create_route_feature(route_coordinates, route_metadata)
            features.append(route_feat)

        return {
            "type": "FeatureCollection",
            "generator": "RDD-GIS-Backend",
            "features": features,
        }

    @staticmethod
    def validate_geojson(geojson_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate that a GeoJSON object conforms to standard specifications.
        
        Returns:
            Tuple of (is_valid: bool, issues: List[str])
        """
        issues = []
        if not isinstance(geojson_dict, dict):
            return False, ["Root object must be a dictionary"]

        if geojson_dict.get("type") != "FeatureCollection":
            issues.append(f"Expected type 'FeatureCollection', got '{geojson_dict.get('type')}'")

        features = geojson_dict.get("features")
        if not isinstance(features, list):
            issues.append("FeatureCollection must contain a 'features' array")
            return False, issues

        for idx, feat in enumerate(features):
            if not isinstance(feat, dict) or feat.get("type") != "Feature":
                issues.append(f"Feature at index {idx} has invalid format")
                continue

            geom = feat.get("geometry")
            if not isinstance(geom, dict):
                issues.append(f"Feature {idx} missing geometry")
                continue

            gtype = geom.get("type")
            coords = geom.get("coordinates")

            if gtype == "Point":
                if not isinstance(coords, (list, tuple)) or len(coords) < 2:
                    issues.append(f"Feature {idx} Point geometry must have [lon, lat] coordinates")
                else:
                    lon, lat = coords[0], coords[1]
                    if not (-180.0 <= lon <= 180.0):
                        issues.append(f"Feature {idx} longitude {lon} out of range [-180, 180]")
                    if not (-90.0 <= lat <= 90.0):
                        issues.append(f"Feature {idx} latitude {lat} out of range [-90, 90]")
            elif gtype == "LineString":
                if not isinstance(coords, list) or len(coords) < 2:
                    issues.append(f"Feature {idx} LineString must have at least 2 coordinate pairs")

        return len(issues) == 0, issues

    @classmethod
    def save_to_file(
        cls,
        geojson_dict: Dict[str, Any],
        output_path: str,
        indent: int = 2
    ) -> str:
        """
        Validate and write GeoJSON FeatureCollection to disk.
        """
        is_valid, issues = cls.validate_geojson(geojson_dict)
        if not is_valid:
            logger.warning("Saving GeoJSON with validation warnings: %s", issues)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, indent=indent, default=str)

        logger.info("GeoJSON saved to: %s", output_path)
        return output_path
