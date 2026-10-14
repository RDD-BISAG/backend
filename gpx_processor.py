"""
backend/gpx_processor.py — GPX Parsing, Validation, and Temporal GPS Interpolation.

Provides parsing of standard GPX files (1.0/1.1), chronological sorting,
timestamp validation, route extraction, and linear GPS interpolation between
timestamped trackpoints.
"""

import os
import math
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse an ISO 8601 timestamp string into a timezone-aware UTC datetime.
    Supports formats like:
      - 2026-07-26T06:18:25.526660Z
      - 2026-07-26T06:18:25Z
      - 2026-07-26T06:18:25+00:00
      - 2026-07-26T11:48:25+05:30
    """
    ts = timestamp_str.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        # Default naive datetime to UTC
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        # Convert any non-UTC timezone to UTC
        dt = dt.astimezone(timezone.utc)
    return dt


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth in meters.
    """
    radius_m = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_m * c


@dataclass
class GPXPoint:
    """A single timestamped geographic track point."""
    latitude: float
    longitude: float
    timestamp: datetime
    elevation: Optional[float] = None
    speed: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert point to serializable dictionary."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp.isoformat(),
            "elevation": self.elevation,
            "speed": self.speed,
        }

    @property
    def geojson_coordinates(self) -> List[float]:
        """GeoJSON standard ordering: [longitude, latitude]."""
        return [self.longitude, self.latitude]


class GPXProcessor:
    """
    Processor for reading, validating, and interpolating GPX track files.
    """

    def __init__(self, gpx_path: Optional[str] = None):
        self.gpx_path = gpx_path
        self.points: List[GPXPoint] = []
        if gpx_path:
            self.parse_file(gpx_path)

    def parse_file(self, gpx_path: str) -> List[GPXPoint]:
        """
        Parse a GPX file and extract timestamped track points.
        
        Args:
            gpx_path: Path to the .gpx file.
            
        Returns:
            List of chronologically sorted GPXPoint objects.
            
        Raises:
            FileNotFoundError: If GPX file does not exist.
            ValueError: If file is empty, invalid XML, has no points or no timestamps.
        """
        if not os.path.exists(gpx_path):
            raise FileNotFoundError(f"GPX file not found: {gpx_path}")

        try:
            tree = ET.parse(gpx_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise ValueError(f"Invalid GPX XML format in {gpx_path}: {e}") from e

        self.points = self._extract_points(root)
        self.gpx_path = gpx_path
        return self.points

    def parse_string(self, gpx_content: str) -> List[GPXPoint]:
        """Parse GPX content directly from an XML string."""
        if not gpx_content or not gpx_content.strip():
            raise ValueError("GPX content is empty")
        try:
            root = ET.fromstring(gpx_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid GPX XML content: {e}") from e

        self.points = self._extract_points(root)
        return self.points

    def _extract_points(self, root: ET.Element) -> List[GPXPoint]:
        """Extract and validate trackpoints from XML root element."""
        # Find all trkpt, rtept, and wpt elements regardless of namespace
        points: List[GPXPoint] = []

        # Iterate over all elements searching for trackpoints or route points
        for elem in root.iter():
            tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag_name in ("trkpt", "rtept", "wpt"):
                lat_str = elem.attrib.get("lat")
                lon_str = elem.attrib.get("lon")
                if lat_str is None or lon_str is None:
                    continue

                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                except ValueError:
                    continue

                # Coordinate bounds validation
                if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                    logger.warning("Skipping out-of-bounds coordinates: lat=%s, lon=%s", lat, lon)
                    continue

                # Extract time
                time_val = None
                ele_val = None
                speed_val = None

                for child in elem:
                    child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                    if child_tag == "time" and child.text:
                        try:
                            time_val = parse_iso_timestamp(child.text)
                        except Exception as e:
                            logger.warning("Could not parse timestamp '%s': %e", child.text, e)
                    elif child_tag == "ele" and child.text:
                        try:
                            ele_val = float(child.text)
                        except ValueError:
                            pass
                    elif child_tag == "speed" and child.text:
                        try:
                            speed_val = float(child.text)
                        except ValueError:
                            pass

                if time_val is None:
                    # Point has no timestamp
                    continue

                points.append(
                    GPXPoint(
                        latitude=lat,
                        longitude=lon,
                        timestamp=time_val,
                        elevation=ele_val,
                        speed=speed_val,
                    )
                )

        if not points:
            # Check if any points were present at all without time
            raw_pt_count = sum(
                1 for el in root.iter()
                if (el.tag.split("}")[-1] if "}" in el.tag else el.tag) in ("trkpt", "rtept", "wpt")
            )
            if raw_pt_count == 0:
                raise ValueError("GPX contains no track points (<trkpt>).")
            raise ValueError("GPX track points contain no usable timestamp (<time>) information.")

        # Sort chronologically by timestamp
        points.sort(key=lambda p: p.timestamp)
        return points

    @property
    def total_points(self) -> int:
        return len(self.points)

    @property
    def start_time(self) -> Optional[datetime]:
        return self.points[0].timestamp if self.points else None

    @property
    def end_time(self) -> Optional[datetime]:
        return self.points[-1].timestamp if self.points else None

    @property
    def duration_seconds(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].timestamp - self.points[0].timestamp).total_seconds()

    @property
    def bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """Return (min_lat, min_lon, max_lat, max_lon)."""
        if not self.points:
            return None
        lats = [p.latitude for p in self.points]
        lons = [p.longitude for p in self.points]
        return min(lats), min(lons), max(lats), max(lons)

    @property
    def total_distance_meters(self) -> float:
        """Calculate total trajectory distance in meters."""
        if len(self.points) < 2:
            return 0.0
        dist = 0.0
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            dist += haversine_distance(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
        return dist

    def get_route_coordinates(self) -> List[List[float]]:
        """
        Return the route trajectory as a list of [longitude, latitude] coordinates
        ready for GeoJSON LineString geometry.
        """
        return [[p.longitude, p.latitude] for p in self.points]

    def interpolate(
        self,
        target_time: datetime,
        tolerance_seconds: float = 3.0
    ) -> Dict[str, Any]:
        """
        Interpolate geographic coordinates at a given target UTC timestamp.
        
        Args:
            target_time: Target timezone-aware UTC datetime.
            tolerance_seconds: Grace window (in seconds) allowing clamping to
                               start/end points if target time falls slightly
                               outside GPX boundaries.
                               
        Returns:
            Dictionary containing:
                - 'latitude': float
                - 'longitude': float
                - 'timestamp': target_time
                - 'elevation': Optional[float]
                - 'speed': Optional[float]
                - 'status': 'exact' | 'interpolated' | 'clamped_start' | 'clamped_end'
                - 'segment_duration_sec': float
                - 'surrounding_points': tuple of surrounding GPXPoints (p1, p2)
        """
        if not self.points:
            raise ValueError("No GPX points available for interpolation.")

        # Ensure target_time is timezone-aware UTC
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)
        else:
            target_time = target_time.astimezone(timezone.utc)

        t_start = self.points[0].timestamp
        t_end = self.points[-1].timestamp

        # Case 1: Before start of GPX
        if target_time < t_start:
            gap = (t_start - target_time).total_seconds()
            if gap <= tolerance_seconds:
                p0 = self.points[0]
                return {
                    "latitude": p0.latitude,
                    "longitude": p0.longitude,
                    "timestamp": target_time,
                    "elevation": p0.elevation,
                    "speed": p0.speed,
                    "status": "clamped_start",
                    "segment_duration_sec": 0.0,
                    "surrounding_points": (p0, p0),
                    "warning": f"Time {target_time.isoformat()} is {gap:.2f}s before GPX start; clamped to first point.",
                }
            raise ValueError(
                f"Target time {target_time.isoformat()} is {gap:.2f}s before GPX track start "
                f"({t_start.isoformat()}), exceeding tolerance of {tolerance_seconds}s."
            )

        # Case 2: After end of GPX
        if target_time > t_end:
            gap = (target_time - t_end).total_seconds()
            if gap <= tolerance_seconds:
                pend = self.points[-1]
                return {
                    "latitude": pend.latitude,
                    "longitude": pend.longitude,
                    "timestamp": target_time,
                    "elevation": pend.elevation,
                    "speed": pend.speed,
                    "status": "clamped_end",
                    "segment_duration_sec": 0.0,
                    "surrounding_points": (pend, pend),
                    "warning": f"Time {target_time.isoformat()} is {gap:.2f}s after GPX end; clamped to last point.",
                }
            raise ValueError(
                f"Target time {target_time.isoformat()} is {gap:.2f}s after GPX track end "
                f"({t_end.isoformat()}), exceeding tolerance of {tolerance_seconds}s."
            )

        # Case 3: Binary search for surrounding points
        low = 0
        high = len(self.points) - 1

        while low <= high:
            mid = (low + high) // 2
            mid_t = self.points[mid].timestamp
            if mid_t == target_time:
                p = self.points[mid]
                return {
                    "latitude": p.latitude,
                    "longitude": p.longitude,
                    "timestamp": target_time,
                    "elevation": p.elevation,
                    "speed": p.speed,
                    "status": "exact",
                    "segment_duration_sec": 0.0,
                    "surrounding_points": (p, p),
                }
            elif mid_t < target_time:
                low = mid + 1
            else:
                high = mid - 1

        # target_time lies between points[high] and points[low]
        # (where high = low - 1)
        p1 = self.points[high]
        p2 = self.points[low]

        t1 = p1.timestamp
        t2 = p2.timestamp
        seg_duration = (t2 - t1).total_seconds()

        if seg_duration <= 0:
            alpha = 0.0
        else:
            alpha = (target_time - t1).total_seconds() / seg_duration

        # Linear interpolation of latitude and longitude
        interp_lat = p1.latitude + alpha * (p2.latitude - p1.latitude)
        interp_lon = p1.longitude + alpha * (p2.longitude - p1.longitude)

        # Elevation interpolation if present in both points
        interp_ele = None
        if p1.elevation is not None and p2.elevation is not None:
            interp_ele = p1.elevation + alpha * (p2.elevation - p1.elevation)

        # Speed interpolation if present in both points
        interp_speed = None
        if p1.speed is not None and p2.speed is not None:
            interp_speed = p1.speed + alpha * (p2.speed - p1.speed)

        return {
            "latitude": interp_lat,
            "longitude": interp_lon,
            "timestamp": target_time,
            "elevation": interp_ele,
            "speed": interp_speed,
            "status": "interpolated",
            "segment_duration_sec": seg_duration,
            "surrounding_points": (p1, p2),
            "alpha": alpha,
        }
