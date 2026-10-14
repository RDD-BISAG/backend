"""
backend/gps_matcher.py — Video Time Synchronization and Defect GPS Matching.

Responsible for:
1. Probing actual video FPS and frame duration via OpenCV.
2. Synchronizing video-relative time with absolute GPX timestamps.
3. Translating representative defect frames (max_confidence_frame) into
   interpolated geographic coordinates.
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2

from backend.gpx_processor import GPXProcessor, GPXPoint

logger = logging.getLogger(__name__)


def probe_video(video_path: str) -> Dict[str, Any]:
    """
    Extract accurate FPS, frame count, and duration from a video file.
    
    Args:
        video_path: Path to video file.
        
    Returns:
        Dict with 'fps', 'total_frames', 'duration_seconds', 'width', 'height'.
        
    Raises:
        FileNotFoundError: If video file does not exist.
        ValueError: If video cannot be opened or has invalid FPS / frame count.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file via OpenCV: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0 or math_is_invalid(fps):
        raise ValueError(f"Unable to determine valid FPS for video {video_path} (got {fps}).")

    duration_seconds = total_frames / fps if total_frames > 0 else 0.0

    return {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": duration_seconds,
        "width": width,
        "height": height,
    }


def math_is_invalid(val: float) -> bool:
    """Check if value is NaN or infinite."""
    import math
    return math.isnan(val) or math.isinf(val)


@dataclass
class SyncConfig:
    """
    Configuration establishing the synchronization relationship between
    video frame 0 and absolute GPX timestamps.
    """
    video_start_time: Optional[datetime] = None
    offset_seconds: float = 0.0
    sync_mode: str = "auto_align_gpx_start"  # "manual" | "offset" | "auto_align_gpx_start"
    tolerance_seconds: float = 4.0

    def get_effective_start_time(self, gpx_start: datetime) -> Tuple[datetime, str, List[str]]:
        """
        Compute effective video start datetime and return explanation and warnings.
        """
        warnings = []
        if self.video_start_time is not None:
            vt = self.video_start_time
            if vt.tzinfo is None:
                vt = vt.replace(tzinfo=timezone.utc)
            else:
                vt = vt.astimezone(timezone.utc)
            desc = f"Explicitly configured video start time: {vt.isoformat()}"
            return vt, desc, warnings

        # Mode: auto align with optional offset
        effective_start = gpx_start + timedelta(seconds=self.offset_seconds)
        if abs(self.offset_seconds) > 0.001:
            desc = (
                f"Video synchronized to GPX start ({gpx_start.isoformat()}) "
                f"with offset of {self.offset_seconds:+.2f}s -> {effective_start.isoformat()}"
            )
        else:
            desc = (
                f"Video synchronized assuming video frame 0 coincides with first GPX point "
                f"({gpx_start.isoformat()})"
            )
            warnings.append(
                "Notice: Video start time was not explicitly provided in metadata; "
                "synchronized by aligning video frame 0 to GPX start timestamp."
            )

        return effective_start, desc, warnings


class GPSMatcher:
    """
    Matches defect detection frames to GPS coordinates using video FPS
    and temporal GPX interpolation.
    """

    def __init__(
        self,
        gpx_processor: GPXProcessor,
        video_path: Optional[str] = None,
        fps: Optional[float] = None,
        sync_config: Optional[SyncConfig] = None,
    ):
        self.gpx_processor = gpx_processor
        self.sync_config = sync_config or SyncConfig()
        self.video_info: Optional[Dict[str, Any]] = None

        if video_path and os.path.exists(video_path):
            self.video_info = probe_video(video_path)
            self.fps = self.video_info["fps"]
        elif fps is not None and fps > 0:
            self.fps = float(fps)
        else:
            raise ValueError("Either valid video_path or explicit positive fps must be provided to GPSMatcher.")

        if not self.gpx_processor.points:
            raise ValueError("GPXProcessor contains no track points.")

        # Resolve video start timestamp
        self.video_start_time, self.sync_description, self.sync_warnings = (
            self.sync_config.get_effective_start_time(self.gpx_processor.start_time)
        )
        logger.info("GPSMatcher initialized: %s", self.sync_description)

    def match_defect(self, defect: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine geographic position for a finalized defect.
        
        Uses max_confidence_frame as the representative frame as required.
        
        Args:
            defect: Defect dictionary from summary.json or DefectDatabase.
            
        Returns:
            Dict containing matched geographic properties and metadata.
        """
        # 1. Identify representative frame: max_confidence_frame
        rep_frame = defect.get("max_confidence_frame")
        if rep_frame is None:
            # Fallback to first_frame or 0 if missing
            rep_frame = defect.get("first_frame", 0)

        # 2. Compute video-relative time in seconds
        video_time_seconds = float(rep_frame) / self.fps

        # Format video relative time for human reading (e.g. 00:03:03.88)
        mins = int(video_time_seconds // 60)
        secs = video_time_seconds % 60
        hrs = int(mins // 60)
        mins = mins % 60
        formatted_time = f"{hrs:02d}:{mins:02d}:{secs:05.2f}"

        # 3. Calculate absolute GPX timestamp
        defect_abs_time = self.video_start_time + timedelta(seconds=video_time_seconds)

        # 4. Interpolate GPS coordinates from GPX
        try:
            interp_result = self.gpx_processor.interpolate(
                defect_abs_time,
                tolerance_seconds=self.sync_config.tolerance_seconds
            )
            lat = interp_result["latitude"]
            lon = interp_result["longitude"]
            interp_status = interp_result.get("status", "interpolated")
            interp_warning = interp_result.get("warning")
        except ValueError as e:
            logger.error("Failed to interpolate GPS for defect %s: %s", defect.get("defect_id"), e)
            raise ValueError(
                f"Defect '{defect.get('defect_id', 'unknown')}' at frame {rep_frame} "
                f"({video_time_seconds:.2f}s) could not be matched to GPS: {e}"
            ) from e

        # Extract confidence
        conf = defect.get("max_confidence")
        if conf is None:
            conf = defect.get("avg_confidence", 0.0)
        try:
            conf = float(conf)
        except (ValueError, TypeError):
            conf = 0.0

        return {
            "defect_id": str(defect.get("defect_id", "")),
            "defect_type": defect.get("defect_type") or defect.get("class_name") or "Unknown",
            "confidence": round(conf, 4),
            "avg_confidence": round(float(defect.get("avg_confidence", conf)), 4),
            "representative_frame": int(rep_frame),
            "video_time_seconds": round(video_time_seconds, 2),
            "video_time_formatted": formatted_time,
            "latitude": lat,
            "longitude": lon,
            "elevation": interp_result.get("elevation"),
            "absolute_time": defect_abs_time.isoformat(),
            "crop_path": defect.get("crop_path") or defect.get("representative_crop_path") or "",
            "mask_path": defect.get("mask_path") or defect.get("segmentation_mask") or "",
            "pixel_area": defect.get("pixel_area"),
            "real_area_m2": defect.get("real_area_m2"),
            "first_frame": defect.get("first_frame"),
            "last_frame": defect.get("last_frame"),
            "frames_observed": defect.get("frames_observed"),
            "source_track_ids": defect.get("source_track_ids", []),
            "interpolation_status": interp_status,
            "interpolation_warning": interp_warning,
        }

    def match_all_defects(
        self,
        defects: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Match a collection of finalized defects to GPS locations.
        
        Returns:
            Tuple of (matched_defects_list, warnings_list)
        """
        matched = []
        all_warnings = list(self.sync_warnings)

        for d in defects:
            try:
                m = self.match_defect(d)
                if m.get("interpolation_warning"):
                    all_warnings.append(m["interpolation_warning"])
                matched.append(m)
            except Exception as e:
                all_warnings.append(f"Skipped defect {d.get('defect_id')}: {e}")

        return matched, all_warnings
