"""
backend/geo_service.py — High-Level GIS Enrichment Service Orchestrator.

Integrates GPX parsing, video FPS probing, time synchronization, defect location
matching, GeoJSON generation, and interactive Leaflet map rendering.
"""

import os
import json
import base64
import logging
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

from backend.gpx_processor import GPXProcessor
from backend.gps_matcher import GPSMatcher, SyncConfig, probe_video
from backend.geojson_generator import GeoJSONGenerator

logger = logging.getLogger(__name__)

# Standard color palette for defect visualization on map (supports lowercase, uppercase, and prefixes)
DEFECT_MAP_COLORS = {
    "Pothole": "#ef4444",              # Red
    "POTHOLE": "#ef4444",
    "D40: Pothole": "#ef4444",
    "D40": "#ef4444",
    "Crack_Long": "#f97316",           # Orange
    "CRACK_LONG": "#f97316",
    "Longitudinal Crack": "#f97316",
    "D00: Longitudinal Crack": "#f97316",
    "D00": "#f97316",
    "Crack_Trans": "#eab308",          # Yellow
    "CRACK_TRANS": "#eab308",
    "Transverse Crack": "#eab308",
    "D10: Transverse Crack": "#eab308",
    "D10": "#eab308",
    "Crack_Alligator": "#d946ef",      # Magenta
    "CRACK_ALLIGATOR": "#d946ef",
    "Alligator Crack": "#d946ef",
    "D20: Alligator Crack": "#d946ef",
    "D20": "#d946ef",
    "Patch": "#22c55e",                # Green
    "PATCH": "#22c55e",
    "Repair / Patch": "#22c55e",
    "Repair": "#22c55e",
    "LONG_REPAIR": "#10b981",          # Emerald
    "Manhole": "#8b5cf6",              # Purple
    "MANHOLE": "#8b5cf6",
    "Raveling": "#ec4899",             # Pink
    "RAVELING": "#ec4899",
    "Road_Edge_Damage": "#14b8a6",     # Teal
    "ROAD_EDGE_DAMAGE": "#14b8a6",
    "FADED_MARKING": "#94a3b8",        # Slate
    "CORRUGATIONS": "#f59e0b",         # Amber
    "RUTTING": "#d97706",              # Ochre
}
DEFAULT_DEFECT_COLOR = "#06b6d4"       # Cyan fallback


def resolve_video_path(video_path: Optional[str], search_dirs: List[str]) -> Optional[str]:
    """
    Find local video file for FPS and frame count probing.
    Handles remote/Kaggle paths by checking local search directories
    or falling back to output_video_*.mp4 in the run directory.
    """
    if video_path and os.path.exists(video_path):
        return video_path

    # Try matching filename in search dirs
    if video_path:
        base_name = os.path.basename(video_path)
        for sdir in search_dirs:
            if not sdir or not os.path.exists(sdir):
                continue
            cand = os.path.join(sdir, base_name)
            if os.path.exists(cand):
                return cand

    # Fallback to output_video_*.mp4 in target run directory
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        try:
            for f in os.listdir(sdir):
                if f.startswith("output_video_") and f.endswith(".mp4"):
                    return os.path.join(sdir, f)
            for f in os.listdir(sdir):
                if f.endswith(".mp4"):
                    return os.path.join(sdir, f)
        except OSError:
            pass

    return None


def resolve_crop_image_path(raw_crop_path: str, search_dirs: List[str]) -> Optional[str]:
    """
    Find local crop image file, handling remote/Kaggle absolute paths gracefully.
    """
    if not raw_crop_path:
        return None

    # Check if raw path exists directly
    if os.path.exists(raw_crop_path):
        return raw_crop_path

    # Extract filename
    fname = os.path.basename(raw_crop_path)
    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        # Direct check in dir
        cand1 = os.path.join(sdir, fname)
        if os.path.exists(cand1):
            return cand1
        # Check inside 'crops' subfolder
        cand2 = os.path.join(sdir, "crops", fname)
        if os.path.exists(cand2):
            return cand2

    return None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image to a base64 data URI string."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        ext = os.path.splitext(image_path)[1].lower().replace(".", "")
        if ext in ("jpg", "jpeg"):
            mime = "image/jpeg"
        elif ext == "png":
            mime = "image/png"
        elif ext == "webp":
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        logger.warning("Could not encode image %s to base64: %s", image_path, e)
        return None


def extract_defects_from_source(summary_source: Union[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract finalized unique defects from a summary dictionary, summary.json path,
    or directory containing summary.json / defects.db.
    
    Returns:
        Tuple of (defects_list, detected_output_directory)
    """
    detected_output_dir = "."

    if isinstance(summary_source, dict):
        defects = summary_source.get("finalized_defects", [])
        output_dir = summary_source.get("output_dir", ".")
        return defects, output_dir

    if not isinstance(summary_source, str) or not os.path.exists(summary_source):
        raise FileNotFoundError(f"Summary source path does not exist: {summary_source}")

    # Case A: Path is a JSON file
    if os.path.isfile(summary_source) and summary_source.endswith(".json"):
        with open(summary_source, "r", encoding="utf-8") as f:
            data = json.load(f)
        detected_output_dir = os.path.dirname(summary_source) or "."
        defects = data.get("finalized_defects", [])
        if defects:
            return defects, detected_output_dir

    # Case B: Path is a directory
    if os.path.isdir(summary_source):
        detected_output_dir = summary_source
        summary_json = os.path.join(summary_source, "summary.json")
        if os.path.exists(summary_json):
            with open(summary_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            defects = data.get("finalized_defects", [])
            if defects:
                return defects, detected_output_dir

        # Fallback to SQLite defects.db if summary.json has no finalized_defects
        db_path = os.path.join(summary_source, "defects.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM defects")
            rows = cursor.fetchall()
            defects = [dict(row) for row in rows]
            conn.close()
            if defects:
                return defects, detected_output_dir

    raise ValueError(f"No finalized unique defects found in summary source: {summary_source}")


def generate_leaflet_map_html(
    geojson_data: Dict[str, Any],
    map_height: str = "560px"
) -> str:
    """
    Generate a self-contained Leaflet.js interactive map wrapped in an iframe.
    The iframe isolation ensures that <script> tags execute reliably when updated
    dynamically within Gradio gr.HTML components.
    """
    import html

    geojson_json = json.dumps(geojson_data)
    color_map_json = json.dumps(DEFECT_MAP_COLORS)

    doc_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #0f172a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      overflow: hidden;
    }}
    #rdd-map-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: #1e293b;
    }}
    .defect-popup-card {{
      max-width: 290px;
      padding: 4px;
      color: #0f172a;
      font-size: 13px;
    }}
    .defect-popup-card h4 {{
      margin: 0 0 6px 0;
      font-size: 15px;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .defect-badge {{
      background: #e2e8f0;
      color: #0f172a;
      font-size: 11px;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
    }}
    .defect-img-wrap {{
      width: 100%;
      height: 145px;
      background: #020617;
      border-radius: 6px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 8px;
      border: 1px solid #cbd5e1;
    }}
    .defect-img-wrap img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }}
    .defect-meta-table {{
      width: 100%;
      font-size: 12px;
      border-collapse: collapse;
    }}
    .defect-meta-table td {{
      padding: 2px 0;
    }}
    .defect-meta-table td.label {{
      color: #64748b;
      font-weight: 500;
      width: 36%;
    }}
    .defect-meta-table td.val {{
      font-weight: 600;
      color: #0f172a;
    }}
    .map-legend {{
      background: rgba(15, 23, 42, 0.92);
      color: #f8fafc;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.5;
      backdrop-filter: blur(6px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      border: 1px solid rgba(255,255,255,0.1);
      max-height: 220px;
      overflow-y: auto;
    }}
    .map-legend-item {{
      display: flex;
      align-items: center;
      margin-bottom: 3px;
    }}
    .legend-color-dot {{
      width: 11px;
      height: 11px;
      border-radius: 50%;
      margin-right: 7px;
      display: inline-block;
      border: 1px solid #ffffff;
      flex-shrink: 0;
    }}
  </style>
</head>
<body>
  <div id="rdd-map-container"></div>
  <script>
    (function() {{
      const geoData = {geojson_json};
      const colorMap = {color_map_json};

      function resolveColor(dt) {{
        if (!dt) return '{DEFAULT_DEFECT_COLOR}';
        if (colorMap[dt]) return colorMap[dt];
        const upper = dt.toUpperCase();
        for (const [k, v] of Object.entries(colorMap)) {{
          if (k.toUpperCase() === upper) return v;
        }}
        if (upper.includes('POTHOLE')) return '#ef4444';
        if (upper.includes('LONG')) return '#f97316';
        if (upper.includes('TRANS')) return '#eab308';
        if (upper.includes('ALLIGATOR')) return '#d946ef';
        if (upper.includes('PATCH') || upper.includes('REPAIR')) return '#22c55e';
        if (upper.includes('MANHOLE')) return '#8b5cf6';
        if (upper.includes('RAVEL')) return '#ec4899';
        return '{DEFAULT_DEFECT_COLOR}';
      }}

      // Initialize Leaflet map
      const map = L.map('rdd-map-container', {{
        zoomControl: true,
        attributionControl: false
      }}).setView([22.025, 81.182], 14);

      // Base tile layers
      const googleHybrid = L.tileLayer('https://mt1.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
        maxZoom: 20,
        attribution: 'Google Satellite'
      }});

      const cartoVoyager = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        maxZoom: 20,
        subdomains: 'abcd',
        attribution: 'CartoDB Voyager'
      }});

      const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        maxZoom: 19,
        attribution: 'Esri World Imagery'
      }});

      const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: 'OpenStreetMap'
      }});

      // Default active base layer: Google Satellite Hybrid (highest road detail)
      googleHybrid.addTo(map);

      // Layer switcher control (top-right)
      const baseMaps = {{
        "🛰️ Satellite (Hybrid)": googleHybrid,
        "🗺️ Road Map (Carto)": cartoVoyager,
        "🛰️ Esri Satellite": esriSatellite,
        "🌐 OpenStreetMap": osmLayer
      }};
      L.control.layers(baseMaps, null, {{ position: 'topright', collapsed: false }}).addTo(map);

      let defectCount = 0;
      let routeCount = 0;
      const seenTypes = new Set();

      // Render GeoJSON Features
      const geoLayer = L.geoJSON(geoData, {{
        style: function(feature) {{
          if (feature.geometry && feature.geometry.type === 'LineString') {{
            return {{
              color: '#38bdf8',
              weight: 5,
              opacity: 0.9,
              lineCap: 'round',
              lineJoin: 'round'
            }};
          }}
        }},
        pointToLayer: function(feature, latlng) {{
          const props = feature.properties || {{}};
          const dtype = props.defect_type || 'Unknown';
          seenTypes.add(dtype);
          const color = resolveColor(dtype);
          defectCount++;

          const marker = L.circleMarker(latlng, {{
            radius: 10,
            fillColor: color,
            color: '#ffffff',
            weight: 3,
            opacity: 1,
            fillOpacity: 0.95,
            pane: 'markerPane'
          }});

          // Safe coordinate formatting
          const latVal = (typeof props.latitude === 'number') ? props.latitude.toFixed(7) : (latlng ? latlng.lat.toFixed(7) : 'N/A');
          const lonVal = (typeof props.longitude === 'number') ? props.longitude.toFixed(7) : (latlng ? latlng.lng.toFixed(7) : 'N/A');

          // Popup Card with image
          let imgHtml = '';
          if (props.image && props.image.length > 10) {{
            imgHtml = `<div class="defect-img-wrap"><img src="${{props.image}}" alt="${{dtype}}"/></div>`;
          }} else {{
            imgHtml = `<div class="defect-img-wrap" style="color:#64748b;font-size:12px;">No crop preview available</div>`;
          }}

          const confDisplay = props.confidence_pct || (props.confidence ? (Math.round(props.confidence * 100) + '%') : 'N/A');
          const timeDisplay = props.video_time_formatted || (props.video_time_seconds ? (props.video_time_seconds + 's') : '0s');

          const popupContent = `
            <div class="defect-popup-card">
              <h4>
                <span><span style="color:${{color}}">●</span> ${{dtype}}</span>
                <span class="defect-badge">${{confDisplay}}</span>
              </h4>
              ${{imgHtml}}
              <table class="defect-meta-table">
                <tr><td class="label">Latitude</td><td class="val">${{latVal}}</td></tr>
                <tr><td class="label">Longitude</td><td class="val">${{lonVal}}</td></tr>
                <tr><td class="label">Frame</td><td class="val">#${{props.representative_frame || 0}}</td></tr>
                <tr><td class="label">Video Time</td><td class="val">${{timeDisplay}}</td></tr>
                ${{props.real_area_m2 ? `<tr><td class="label">Ground Area</td><td class="val" style="color:#22c55e;font-weight:700;">${{props.real_area_m2}} m²</td></tr>` : ''}}
                ${{props.pixel_area ? `<tr><td class="label">Pixel Area</td><td class="val">${{props.pixel_area}} px</td></tr>` : ''}}
              </table>
            </div>
          `;

          marker.bindPopup(popupContent, {{ maxWidth: 300 }});
          marker.bindTooltip(`<b>${{dtype}}</b> (${{confDisplay}})`, {{
            direction: 'top',
            offset: [0, -10],
            permanent: false
          }});

          marker.on('mouseover', function() {{ this.openTooltip(); }});
          marker.on('click', function() {{ this.openPopup(); }});

          return marker;
        }},
        onEachFeature: function(feature, layer) {{
          if (feature.geometry && feature.geometry.type === 'LineString') {{
            routeCount++;
            const p = feature.properties || {{}};
            const pts = p.total_points || (feature.geometry.coordinates ? feature.geometry.coordinates.length : '');
            layer.bindTooltip(`<b>${{p.name || 'Survey Route'}}</b><br>${{pts}} GPS points`, {{
              sticky: true
            }});
          }}
        }}
      }}).addTo(map);

      // Auto-fit bounds reliably
      function fitToData() {{
        try {{
          map.invalidateSize();
          const layerBounds = geoLayer.getBounds();
          if (layerBounds && layerBounds.isValid()) {{
            if (layerBounds.getNorthEast().equals(layerBounds.getSouthWest())) {{
              map.setView(layerBounds.getCenter(), 16);
            }} else {{
              map.fitBounds(layerBounds, {{ padding: [45, 45], maxZoom: 18 }});
            }}
          }}
        }} catch(e) {{
          console.warn("Could not fit bounds:", e);
        }}
      }}

      fitToData();

      // Legend Control (placed at bottom-right to avoid layer switcher collision)
      const legend = L.control({{ position: 'bottomright' }});
      legend.onAdd = function() {{
        const div = L.DomUtil.create('div', 'map-legend');
        div.innerHTML = `<b style="display:block;margin-bottom:6px;border-bottom:1px solid rgba(255,255,255,0.2);padding-bottom:3px;">Survey Map Legend</b>`;
        if (routeCount > 0) {{
          div.innerHTML += `<div class="map-legend-item"><span class="legend-color-dot" style="background:#38bdf8;"></span> Survey Route</div>`;
        }}
        seenTypes.forEach(dtype => {{
          const c = resolveColor(dtype);
          div.innerHTML += `<div class="map-legend-item"><span class="legend-color-dot" style="background:${{c}};"></span> ${{dtype}}</div>`;
        }});
        return div;
      }};
      legend.addTo(map);

      // Periodic size invalidation and bounds re-fit to handle iframe loading
      window.addEventListener('load', fitToData);
      setTimeout(fitToData, 200);
      setTimeout(fitToData, 600);
      setTimeout(fitToData, 1500);
    }})();
  </script>
</body>
</html>
"""

    escaped_doc = html.escape(doc_html, quote=True)
    b64_doc = base64.b64encode(doc_html.encode("utf-8")).decode("utf-8")

    iframe_html = f'''<div style="width:100%; border-radius:10px; overflow:hidden; border:1px solid #334155; box-shadow:0 4px 14px rgba(0,0,0,0.3);">
  <iframe
    srcdoc="{escaped_doc}"
    src="data:text/html;charset=utf-8;base64,{b64_doc}"
    style="width:100%; height:{map_height}; border:none; display:block; background:#0f172a;"
    loading="lazy"
  ></iframe>
</div>'''

    return iframe_html


class GeoService:
    """
    Core service coordinating RDD defect extraction, video probing, GPX processing,
    and GeoJSON generation.
    """

    def __init__(self):
        pass

    def enrich_inspection_with_gis(
        self,
        video_path: str,
        gpx_path: str,
        summary_source: Union[str, Dict[str, Any]],
        output_dir: Optional[str] = None,
        video_start_time: Optional[datetime] = None,
        time_offset_seconds: float = 0.0,
        embed_images_as_base64: bool = True,
    ) -> Dict[str, Any]:
        """
        Main entrypoint for GIS enrichment.
        
        Args:
            video_path: Path to road footage video.
            gpx_path: Path to GPX survey file.
            summary_source: summary.json path, output dir path, or summary dict.
            output_dir: Destination folder for defects.geojson (defaults to summary's folder).
            video_start_time: Explicit video start datetime.
            time_offset_seconds: Offset between video start and GPX start.
            embed_images_as_base64: If True, encodes crop images into base64 data URIs.
            
        Returns:
            Dict containing GeoJSON data, file path, interactive HTML map, and execution metrics.
        """
        warnings: List[str] = []

        # 1. Parse GPX trackpoints
        logger.info("Parsing GPX track: %s", gpx_path)
        gpx_proc = GPXProcessor(gpx_path)
        if gpx_proc.total_points == 0:
            raise ValueError(f"GPX file {gpx_path} contains no valid trackpoints.")

        # 2. Extract finalized defects from existing RDD outputs
        defects, detected_output_dir = extract_defects_from_source(summary_source)
        target_output_dir = output_dir or detected_output_dir or "."
        os.makedirs(target_output_dir, exist_ok=True)

        if not defects:
            raise ValueError("No finalized unique defects found to enrich with GIS.")

        # 3. Probe video for actual FPS and frame count
        search_dirs = [
            target_output_dir,
            detected_output_dir,
            os.path.dirname(video_path or ""),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "inputs")
        ]
        resolved_video = resolve_video_path(video_path, search_dirs)
        if not resolved_video:
            raise FileNotFoundError(
                f"Could not locate video file '{video_path}' or corresponding output_video_*.mp4 "
                f"in '{target_output_dir}' to probe stream FPS."
            )

        logger.info("Probing video properties: %s", resolved_video)
        video_meta = probe_video(resolved_video)
        fps = video_meta["fps"]

        # 4. Synchronize video time with GPX
        sync_cfg = SyncConfig(
            video_start_time=video_start_time,
            offset_seconds=time_offset_seconds,
            sync_mode="manual" if video_start_time else "auto_align_gpx_start"
        )
        matcher = GPSMatcher(
            gpx_processor=gpx_proc,
            fps=fps,
            sync_config=sync_cfg,
        )

        # 5. Resolve crop images and match defects to GPS
        search_dirs = [target_output_dir, detected_output_dir, os.path.dirname(video_path)]
        matched_defects, match_warnings = matcher.match_all_defects(defects)
        warnings.extend(match_warnings)

        image_uri_map: Dict[str, str] = {}
        for d in matched_defects:
            did = d["defect_id"]
            raw_crop = d.get("crop_path", "")
            resolved_crop = resolve_crop_image_path(raw_crop, search_dirs)
            if resolved_crop:
                d["crop_path"] = resolved_crop
                if embed_images_as_base64:
                    b64_uri = encode_image_to_base64(resolved_crop)
                    if b64_uri:
                        image_uri_map[did] = b64_uri
                    else:
                        image_uri_map[did] = resolved_crop
                else:
                    image_uri_map[did] = resolved_crop
            else:
                warnings.append(f"Representative crop image missing for defect '{did}' ({raw_crop})")

        # 6. Generate GeoJSON FeatureCollection
        route_coords = gpx_proc.get_route_coordinates()
        route_meta = {
            "total_distance_meters": round(gpx_proc.total_distance_meters, 2),
            "duration_seconds": round(gpx_proc.duration_seconds, 2),
            "start_time": gpx_proc.start_time.isoformat() if gpx_proc.start_time else None,
            "end_time": gpx_proc.end_time.isoformat() if gpx_proc.end_time else None,
        }

        geojson_data = GeoJSONGenerator.generate_feature_collection(
            matched_defects=matched_defects,
            route_coordinates=route_coords,
            route_metadata=route_meta,
            image_uri_map=image_uri_map,
        )

        # 7. Save GeoJSON output file
        geojson_filename = "defects.geojson"
        geojson_path = os.path.join(target_output_dir, geojson_filename)
        GeoJSONGenerator.save_to_file(geojson_data, geojson_path)

        # 8. Generate interactive Leaflet map HTML
        leaflet_map_html = generate_leaflet_map_html(geojson_data)

        # Defect counts breakdown
        counts: Dict[str, int] = {}
        for md in matched_defects:
            dtype = md.get("defect_type", "Unknown")
            counts[dtype] = counts.get(dtype, 0) + 1

        return {
            "success": True,
            "geojson": geojson_data,
            "geojson_path": geojson_path,
            "html_map": leaflet_map_html,
            "total_defects": len(matched_defects),
            "defect_counts": counts,
            "video_fps": round(fps, 3),
            "total_video_frames": video_meta["total_frames"],
            "survey_distance_meters": round(gpx_proc.total_distance_meters, 2),
            "survey_duration_seconds": round(gpx_proc.duration_seconds, 2),
            "sync_info": matcher.sync_description,
            "warnings": warnings,
        }


# Global convenience function
def enrich_inspection_with_gis(
    video_path: str,
    gpx_path: str,
    summary_source: Union[str, Dict[str, Any]],
    output_dir: Optional[str] = None,
    video_start_time: Optional[datetime] = None,
    time_offset_seconds: float = 0.0,
    embed_images_as_base64: bool = True,
) -> Dict[str, Any]:
    """Module-level function for GeoService.enrich_inspection_with_gis."""
    service = GeoService()
    return service.enrich_inspection_with_gis(
        video_path=video_path,
        gpx_path=gpx_path,
        summary_source=summary_source,
        output_dir=output_dir,
        video_start_time=video_start_time,
        time_offset_seconds=time_offset_seconds,
        embed_images_as_base64=embed_images_as_base64,
    )
