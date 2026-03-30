from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import requests
import supervisely as sly
from dotenv import load_dotenv
from PIL import Image
from pygmdl import save_image
from pygmdl.downloader import calc, top_left_from_center
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data"
IMAGES_DIR = OUTPUT_DIR / "images"
META_DIR = OUTPUT_DIR / "meta"
CONFIG_PATH = Path(__file__).resolve().parent / "osm_classes.json"

PROJECT_NAME = os.getenv("SLY_PROJECT_NAME", "pygmdl_osm_instances")
DATASET_NAME = os.getenv("SLY_DATASET_NAME", "train")
ANNOTATION_BATCH_SIZE = int(os.getenv("SLY_ANN_BATCH_SIZE", "200"))

# Higher priority classes keep their pixels; lower priority classes are clipped.
CLASS_MASK_DEFAULT_PRIORITY = 50
CLASS_MASK_PRIORITY: dict[str, int] = {
    "building": 100,
    "water": 95,
    "road_main": 90,
    "road_minor": 85,
    "field": 20,
    "forest": 10,
}

# Batch-ready input list. Add more scene definitions as needed.
SCENES: list[dict[str, Any]] = [
    {
        "id": "scene_0001",
        "center_lat": 47.975679348309754,
        "center_lon": 10.788837124189856,
        "size_m": 1024,
        "rotation_deg": 0,
        "zoom": 18,
    }
]


@dataclass(frozen=True)
class OSMClassSpec:
    name: str
    geometry: str
    tags: dict[str, Any]
    buffer_m: float
    color: list[int] | None


@dataclass
class SceneGeoContext:
    top_left_lat: float
    top_left_lon: float
    corners_lon_lat: list[tuple[float, float]]
    bbox_left_bottom_right_top: tuple[float, float, float, float]
    local_crs_wkt: str
    local_crs_projjson: dict[str, Any]
    scene_polygon_local: Polygon
    to_local: Transformer
    local_to_pixel_h: np.ndarray


def load_environment() -> None:
    # supervisely.env should contain SERVER_ADDRESS and API_TOKEN.
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    # local.env should contain TEAM_ID, WORKSPACE_ID, optional PROJECT_ID.
    load_dotenv(BASE_DIR / "local.env")


def load_osm_class_specs(config_path: Path) -> list[OSMClassSpec]:
    with config_path.open("r", encoding="utf-8") as file:
        raw_specs = json.load(file)

    specs: list[OSMClassSpec] = []
    for raw in raw_specs:
        specs.append(
            OSMClassSpec(
                name=raw["name"],
                geometry=raw["geometry"],
                tags=raw["tags"],
                buffer_m=float(raw.get("buffer_m", 0.0)),
                color=raw.get("color"),
            )
        )
    return specs


def ensure_output_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def ensure_project_and_dataset(
    api: sly.Api,
    workspace_id: int,
    class_specs: list[OSMClassSpec],
) -> tuple[Any, Any, sly.ProjectMeta]:
    project_id_env = os.getenv("PROJECT_ID")
    if project_id_env:
        project_info = api.project.get_info_by_id(int(project_id_env))
        if project_info is None:
            raise RuntimeError(
                f"Project with id={project_id_env} was not found in Supervisely."
            )
    else:
        project_info = api.project.get_or_create(workspace_id, PROJECT_NAME)

    dataset_info = api.dataset.get_or_create(project_info.id, DATASET_NAME)

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_info.id))
    changed = False
    for spec in class_specs:
        if project_meta.get_obj_class(spec.name) is None:
            obj_class = sly.ObjClass(spec.name, sly.Bitmap, color=spec.color)
            project_meta = project_meta.add_obj_class(obj_class)
            changed = True

    if changed:
        project_meta = api.project.update_meta(project_info.id, project_meta)

    return project_info, dataset_info, project_meta


def compute_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    if src_xy.shape != (4, 2) or dst_uv.shape != (4, 2):
        raise ValueError(
            "Homography requires exactly 4 source and 4 destination points."
        )

    rows = []
    for (x, y), (u, v) in zip(src_xy, dst_uv):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])

    matrix = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(matrix)
    h = vt[-1, :]
    h = h / h[-1]
    return h.reshape(3, 3)


def project_local_xy_to_pixels(
    points_xy: np.ndarray, homography: np.ndarray
) -> np.ndarray:
    points_h = np.hstack(
        [points_xy.astype(np.float64), np.ones((points_xy.shape[0], 1))]
    )
    projected = (homography @ points_h.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return projected[:, :2] / denom


def build_scene_geo_context(
    center_lat: float,
    center_lon: float,
    size_m: int,
    rotation_deg: float,
    image_width: int,
    image_height: int,
) -> SceneGeoContext:
    top_left_lat, top_left_lon = top_left_from_center(
        center_lat, center_lon, size_m, rotation_deg
    )
    lats, lons = calc(top_left_lat, top_left_lon, rotation_deg, size_m)
    corners_lon_lat = [
        (float(lons[0]), float(lats[0])),
        (float(lons[1]), float(lats[1])),
        (float(lons[2]), float(lats[2])),
        (float(lons[3]), float(lats[3])),
    ]

    crs_local = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} +lon_0={center_lon} +datum=WGS84 +units=m +no_defs"
    )
    to_local = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)

    local_corners = np.asarray(
        [to_local.transform(lon, lat) for lon, lat in corners_lon_lat],
        dtype=np.float64,
    )

    destination_pixels = np.asarray(
        [
            [0.0, 0.0],
            [float(image_width - 1), 0.0],
            [float(image_width - 1), float(image_height - 1)],
            [0.0, float(image_height - 1)],
        ],
        dtype=np.float64,
    )
    local_to_pixel_h = compute_homography(local_corners, destination_pixels)

    bbox = (
        float(min(lons)),
        float(min(lats)),
        float(max(lons)),
        float(max(lats)),
    )

    return SceneGeoContext(
        top_left_lat=float(top_left_lat),
        top_left_lon=float(top_left_lon),
        corners_lon_lat=corners_lon_lat,
        bbox_left_bottom_right_top=bbox,
        local_crs_wkt=crs_local.to_wkt(),
        local_crs_projjson=crs_local.to_json_dict(),
        scene_polygon_local=Polygon(local_corners),
        to_local=to_local,
        local_to_pixel_h=local_to_pixel_h,
    )


def iter_polygons(geometry: Any) -> list[Polygon]:
    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return [poly for poly in geometry.geoms if not poly.is_empty]

    if isinstance(geometry, GeometryCollection):
        result: list[Polygon] = []
        for geom in geometry.geoms:
            result.extend(iter_polygons(geom))
        return result

    return []


def geometry_to_polygons(
    geometry_local: Any,
    expected_geometry: str,
    buffer_m: float,
    clip_polygon: Polygon,
) -> list[Polygon]:
    geom = geometry_local
    geom_type = geom.geom_type

    if expected_geometry == "polygon":
        if geom_type not in {"Polygon", "MultiPolygon"}:
            return []
    elif expected_geometry == "line":
        if geom_type not in {"LineString", "MultiLineString"}:
            return []
        geom = geom.buffer(buffer_m, cap_style=2, join_style=2)
    elif expected_geometry == "point":
        if geom_type not in {"Point", "MultiPoint"}:
            return []
        geom = geom.buffer(buffer_m)
    else:
        raise ValueError(f"Unsupported geometry type: {expected_geometry}")

    if geom.is_empty:
        return []

    if not geom.is_valid:
        geom = geom.buffer(0)
    if geom.is_empty:
        return []

    clipped = geom.intersection(clip_polygon)
    if clipped.is_empty:
        return []

    polygons = [poly for poly in iter_polygons(clipped) if poly.area > 0]
    return polygons


def ring_to_supervisely_points(
    ring_uv: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    points_rc: list[tuple[int, int]] = []

    for col, row in ring_uv:
        # Keep boundary points stable when tiny floating-point errors push values
        # just outside the image bounds after clipping in pixel space.
        col_i = int(round(float(np.clip(col, -0.5, width - 0.5))))
        row_i = int(round(float(np.clip(row, -0.5, height - 0.5))))
        point = (row_i, col_i)
        if not points_rc or points_rc[-1] != point:
            points_rc.append(point)

    if len(points_rc) >= 2 and points_rc[0] == points_rc[-1]:
        points_rc.pop()

    if len(set(points_rc)) < 3:
        return []
    return points_rc


def project_polygon_to_pixel_geometry(
    polygon: Polygon,
    homography: np.ndarray,
) -> Any | None:
    exterior_xy = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if exterior_xy.shape[0] < 3:
        return None

    exterior_uv = project_local_xy_to_pixels(exterior_xy, homography)
    interior_uv_list: list[list[list[float]]] = []
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords[:-1], dtype=np.float64)
        if interior_xy.shape[0] < 3:
            continue
        interior_uv = project_local_xy_to_pixels(interior_xy, homography)
        if interior_uv.shape[0] >= 3:
            interior_uv_list.append(interior_uv.tolist())

    pixel_geometry: Any = Polygon(exterior_uv.tolist(), interior_uv_list)
    if pixel_geometry.is_empty:
        return None

    if not pixel_geometry.is_valid:
        pixel_geometry = pixel_geometry.buffer(0)
    if pixel_geometry.is_empty:
        return None

    return pixel_geometry


def polygon_to_bitmap(
    polygon: Polygon,
    homography: np.ndarray,
    width: int,
    height: int,
) -> sly.Bitmap | None:
    pixel_geometry = project_polygon_to_pixel_geometry(polygon, homography)
    if pixel_geometry is None:
        return None

    clip_rect = box(0.0, 0.0, float(width - 1), float(height - 1))
    clipped_geometry = pixel_geometry.intersection(clip_rect)
    if clipped_geometry.is_empty:
        return None

    clipped_polygons = [
        poly for poly in iter_polygons(clipped_geometry) if poly.area > 0
    ]
    if not clipped_polygons:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for clipped_polygon in clipped_polygons:
        exterior_uv = np.asarray(clipped_polygon.exterior.coords[:-1], dtype=np.float64)
        exterior_rc = ring_to_supervisely_points(exterior_uv, width, height)
        if len(exterior_rc) < 3:
            continue

        interior_rc_list: list[list[tuple[int, int]]] = []
        for interior in clipped_polygon.interiors:
            interior_uv = np.asarray(interior.coords[:-1], dtype=np.float64)
            interior_rc = ring_to_supervisely_points(interior_uv, width, height)
            if len(interior_rc) >= 3:
                interior_rc_list.append(interior_rc)

        sly_polygon = sly.Polygon(exterior=exterior_rc, interior=interior_rc_list)
        sly_polygon.draw(mask, color=1)

    binary_mask = mask.astype(np.bool_)
    if not np.any(binary_mask):
        return None

    return sly.Bitmap(binary_mask)


def is_positive_osm_tag_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)

    try:
        if np.isnan(value):
            return False
    except TypeError:
        pass

    text = str(value).strip().lower()
    if text in {"", "no", "false", "0", "nan", "none", "null"}:
        return False
    return True


def is_underground_location_value(value: Any) -> bool:
    if value is None:
        return False

    try:
        if np.isnan(value):
            return False
    except TypeError:
        pass

    text = str(value).strip().lower()
    if text == "":
        return False

    hidden_tokens = {
        "underground",
        "underwater",
        "subway",
        "indoor",
        "inside",
        "below_ground",
    }
    parts = [part.strip() for part in text.replace("|", ";").split(";")]
    return any(part in hidden_tokens for part in parts if part)


def is_negative_layer_value(value: Any) -> bool:
    if value is None:
        return False

    if isinstance(value, (int, float)):
        return float(value) < 0

    text = str(value).strip().lower()
    if text == "":
        return False

    for part in text.replace("|", ";").split(";"):
        item = part.strip()
        if item == "":
            continue
        try:
            if float(item) < 0:
                return True
        except ValueError:
            continue
    return False


def fetch_class_labels(
    class_spec: OSMClassSpec,
    obj_class: sly.ObjClass,
    scene_geo: SceneGeoContext,
    image_width: int,
    image_height: int,
) -> tuple[list[sly.Label], int]:
    left, bottom, right, top = scene_geo.bbox_left_bottom_right_top
    try:
        gdf = ox.features.features_from_bbox(
            (left, bottom, right, top), class_spec.tags
        )
    except Exception as exc:
        # OSMnx may raise "No matching features" for sparse classes in a tile.
        if "No matching features" in str(exc):
            return [], 0
        raise

    if class_spec.geometry == "line" and "highway" in class_spec.tags:
        excluded_total = 0
        hidden_mask = np.zeros(len(gdf), dtype=bool)

        if "tunnel" in gdf.columns:
            tunnel_mask = gdf["tunnel"].map(is_positive_osm_tag_value).to_numpy()
            tunnel_count = int(tunnel_mask.sum())
            excluded_total += tunnel_count
            hidden_mask |= tunnel_mask

        if "covered" in gdf.columns:
            covered_mask = gdf["covered"].map(is_positive_osm_tag_value).to_numpy()
            covered_count = int((covered_mask & ~hidden_mask).sum())
            excluded_total += covered_count
            hidden_mask |= covered_mask

        if "location" in gdf.columns:
            location_mask = (
                gdf["location"].map(is_underground_location_value).to_numpy()
            )
            location_count = int((location_mask & ~hidden_mask).sum())
            excluded_total += location_count
            hidden_mask |= location_mask

        if "layer" in gdf.columns:
            layer_mask = gdf["layer"].map(is_negative_layer_value).to_numpy()
            layer_count = int((layer_mask & ~hidden_mask).sum())
            excluded_total += layer_count
            hidden_mask |= layer_mask

        if "indoor" in gdf.columns:
            indoor_mask = gdf["indoor"].map(is_positive_osm_tag_value).to_numpy()
            indoor_count = int((indoor_mask & ~hidden_mask).sum())
            excluded_total += indoor_count
            hidden_mask |= indoor_mask

        if excluded_total > 0:
            print(
                f"[filter] {class_spec.name}: excluded non-top-visible features={excluded_total}",
                flush=True,
            )
            gdf = gdf.loc[~hidden_mask]

    if gdf.empty:
        return [], 0

    labels: list[sly.Label] = []
    raw_features = 0
    for geometry_wgs84 in gdf.geometry:
        if geometry_wgs84 is None or geometry_wgs84.is_empty:
            continue

        raw_features += 1
        geometry_local = shapely_transform(scene_geo.to_local.transform, geometry_wgs84)
        polygons = geometry_to_polygons(
            geometry_local=geometry_local,
            expected_geometry=class_spec.geometry,
            buffer_m=class_spec.buffer_m,
            clip_polygon=scene_geo.scene_polygon_local,
        )

        for polygon in polygons:
            bitmap = polygon_to_bitmap(
                polygon=polygon,
                homography=scene_geo.local_to_pixel_h,
                width=image_width,
                height=image_height,
            )
            if bitmap is None:
                continue
            labels.append(sly.Label(bitmap, obj_class))

    return labels, raw_features


def scene_metadata_payload(
    scene: dict[str, Any],
    scene_geo: SceneGeoContext,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    pixel_to_local_h = np.linalg.inv(scene_geo.local_to_pixel_h)
    return {
        "source": "pygmdl",
        "center": {
            "lat": float(scene["center_lat"]),
            "lon": float(scene["center_lon"]),
        },
        "top_left": {
            "lat": scene_geo.top_left_lat,
            "lon": scene_geo.top_left_lon,
        },
        "size_m": int(scene["size_m"]),
        "rotation_deg": float(scene["rotation_deg"]),
        "zoom": int(scene["zoom"]),
        "image_size_px": {"width": image_width, "height": image_height},
        "bbox_left_bottom_right_top": list(scene_geo.bbox_left_bottom_right_top),
        "corners_lon_lat": [list(point) for point in scene_geo.corners_lon_lat],
        "local_crs_wkt": scene_geo.local_crs_wkt,
        "local_crs_projjson": scene_geo.local_crs_projjson,
        "local_to_pixel_h": scene_geo.local_to_pixel_h.tolist(),
        "pixel_to_local_h": pixel_to_local_h.tolist(),
    }


def class_specs_export_payload(class_specs: list[OSMClassSpec]) -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "geometry": spec.geometry,
            "tags": spec.tags,
            "buffer_m": spec.buffer_m,
        }
        for spec in class_specs
    ]


def class_mask_priority(class_name: str) -> int:
    return int(CLASS_MASK_PRIORITY.get(class_name, CLASS_MASK_DEFAULT_PRIORITY))


def bitmap_to_full_mask(
    bitmap: sly.Bitmap,
    image_height: int,
    image_width: int,
) -> np.ndarray:
    full_mask = np.zeros((image_height, image_width), dtype=bool)
    local_mask = bitmap.data.astype(bool)
    if not np.any(local_mask):
        return full_mask

    origin_row = int(bitmap.origin.row)
    origin_col = int(bitmap.origin.col)

    row0 = max(0, origin_row)
    col0 = max(0, origin_col)
    row1 = min(image_height, origin_row + local_mask.shape[0])
    col1 = min(image_width, origin_col + local_mask.shape[1])
    if row0 >= row1 or col0 >= col1:
        return full_mask

    src_row0 = row0 - origin_row
    src_col0 = col0 - origin_col
    src_row1 = src_row0 + (row1 - row0)
    src_col1 = src_col0 + (col1 - col0)

    full_mask[row0:row1, col0:col1] = local_mask[src_row0:src_row1, src_col0:src_col1]
    return full_mask


def enforce_non_overlapping_labels(
    labels_by_class: dict[str, list[sly.Label]],
    class_specs: list[OSMClassSpec],
    image_height: int,
    image_width: int,
) -> tuple[dict[str, list[sly.Label]], dict[str, dict[str, int]]]:
    occupancy = np.zeros((image_height, image_width), dtype=bool)
    cleaned_by_class: dict[str, list[sly.Label]] = {spec.name: [] for spec in class_specs}
    stats: dict[str, dict[str, int]] = {
        spec.name: {"dropped_instances": 0, "overlap_pixels_removed": 0}
        for spec in class_specs
    }

    indexed_specs = list(enumerate(class_specs))
    ordered_specs = sorted(
        indexed_specs,
        key=lambda item: (-class_mask_priority(item[1].name), item[0]),
    )

    for _, spec in ordered_specs:
        class_name = spec.name
        for label in labels_by_class.get(class_name, []):
            if not isinstance(label.geometry, sly.Bitmap):
                cleaned_by_class[class_name].append(label)
                continue

            full_mask = bitmap_to_full_mask(
                label.geometry,
                image_height=image_height,
                image_width=image_width,
            )
            if not np.any(full_mask):
                stats[class_name]["dropped_instances"] += 1
                continue

            visible_mask = full_mask & ~occupancy
            removed_pixels = int(np.count_nonzero(full_mask) - np.count_nonzero(visible_mask))
            stats[class_name]["overlap_pixels_removed"] += removed_pixels

            if not np.any(visible_mask):
                stats[class_name]["dropped_instances"] += 1
                continue

            occupancy |= visible_mask
            cleaned_by_class[class_name].append(
                sly.Label(sly.Bitmap(visible_mask), label.obj_class)
            )

    return cleaned_by_class, stats


def upload_annotation_resilient(
    api: sly.Api,
    image_id: int,
    labels: list[sly.Label],
    initial_batch_size: int,
) -> None:
    if len(labels) == 0:
        return

    next_index = 0
    batch_size = max(1, int(initial_batch_size))
    while next_index < len(labels):
        current_batch = labels[next_index : next_index + batch_size]
        try:
            api.annotation.append_labels(image_id, current_batch)
            next_index += len(current_batch)
        except (requests.exceptions.RetryError, requests.exceptions.HTTPError) as exc:
            if batch_size == 1:
                raise
            new_batch_size = max(1, batch_size // 2)
            print(
                f"Annotation append failed at index={next_index} with batch_size={batch_size}: {exc}. "
                f"Retrying with batch_size={new_batch_size}..."
            )
            batch_size = new_batch_size


def process_scene(
    api: sly.Api,
    dataset_id: int,
    project_meta: sly.ProjectMeta,
    class_specs: list[OSMClassSpec],
    scene: dict[str, Any],
) -> None:
    scene_id = scene["id"]
    image_path = IMAGES_DIR / f"{scene_id}.png"

    print(f"[{scene_id}] Downloading image with pygmdl...")
    save_image(
        lat=float(scene["center_lat"]),
        lon=float(scene["center_lon"]),
        size=int(scene["size_m"]),
        output_path=str(image_path),
        rotation=int(scene["rotation_deg"]),
        zoom=int(scene["zoom"]),
        from_center=True,
        show_progress=True,
    )

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    scene_geo = build_scene_geo_context(
        center_lat=float(scene["center_lat"]),
        center_lon=float(scene["center_lon"]),
        size_m=int(scene["size_m"]),
        rotation_deg=float(scene["rotation_deg"]),
        image_width=image_width,
        image_height=image_height,
    )
    metadata = scene_metadata_payload(scene, scene_geo, image_width, image_height)
    metadata["osm_class_specs"] = class_specs_export_payload(class_specs)

    labels_by_class: dict[str, list[sly.Label]] = {}
    class_stats: dict[str, dict[str, Any]] = {}

    for class_spec in class_specs:
        obj_class = project_meta.get_obj_class(class_spec.name)
        try:
            labels, raw_feature_count = fetch_class_labels(
                class_spec=class_spec,
                obj_class=obj_class,
                scene_geo=scene_geo,
                image_width=image_width,
                image_height=image_height,
            )
        except Exception as exc:
            labels, raw_feature_count = [], 0
            class_stats[class_spec.name] = {
                "raw_features": 0,
                "instance_masks": 0,
                "failed": True,
                "error": str(exc),
            }
            print(
                f"[{scene_id}] {class_spec.name}: failed ({exc}); continuing with other classes",
                flush=True,
            )
            continue

        labels_by_class[class_spec.name] = labels
        class_stats[class_spec.name] = {
            "raw_features": raw_feature_count,
            "instance_masks": len(labels),
            "failed": False,
        }
        print(
            f"[{scene_id}] {class_spec.name}: "
            f"raw={raw_feature_count}, masks={len(labels)}"
        )

    cleaned_by_class, overlap_stats = enforce_non_overlapping_labels(
        labels_by_class=labels_by_class,
        class_specs=class_specs,
        image_height=image_height,
        image_width=image_width,
    )

    all_labels: list[sly.Label] = []
    for class_spec in class_specs:
        class_name = class_spec.name
        cleaned_labels = cleaned_by_class.get(class_name, [])
        all_labels.extend(cleaned_labels)

        if class_name in class_stats:
            class_stats[class_name]["instance_masks_non_overlap"] = len(cleaned_labels)
            class_stats[class_name]["dropped_instances_overlap"] = overlap_stats[
                class_name
            ]["dropped_instances"]
            class_stats[class_name]["overlap_pixels_removed"] = overlap_stats[
                class_name
            ]["overlap_pixels_removed"]

            if overlap_stats[class_name]["overlap_pixels_removed"] > 0:
                print(
                    f"[{scene_id}] {class_name}: non-overlap removed_px="
                    f"{overlap_stats[class_name]['overlap_pixels_removed']} "
                    f"dropped_instances={overlap_stats[class_name]['dropped_instances']}",
                    flush=True,
                )

    metadata["class_stats"] = class_stats
    metadata["instance_count"] = len(all_labels)

    image_info = api.image.upload_paths(
        dataset_id,
        [image_path.name],
        [str(image_path)],
        metas=[{"geo": metadata}],
        conflict_resolution="replace",
    )[0]

    print(
        f"[{scene_id}] Uploading {len(all_labels)} instance masks "
        f"with chunk_size={ANNOTATION_BATCH_SIZE}..."
    )
    upload_annotation_resilient(
        api=api,
        image_id=image_info.id,
        labels=all_labels,
        initial_batch_size=ANNOTATION_BATCH_SIZE,
    )

    metadata_path = META_DIR / f"{scene_id}.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(
        f"[{scene_id}] Uploaded image_id={image_info.id}, "
        f"instances={len(all_labels)}, metadata={metadata_path}"
    )


def main() -> None:
    load_environment()
    ensure_output_dirs()
    ox.settings.use_cache = True

    team_id = sly.env.team_id()
    workspace_id = sly.env.workspace_id()
    api: sly.Api = sly.Api.from_env()

    class_specs = load_osm_class_specs(CONFIG_PATH)
    _, dataset_info, project_meta = ensure_project_and_dataset(
        api, workspace_id, class_specs
    )

    print(f"API initialized for team_id={team_id}, workspace_id={workspace_id}")
    print(f"Processing {len(SCENES)} scene(s) into dataset_id={dataset_info.id}...")

    for scene in SCENES:
        process_scene(
            api=api,
            dataset_id=dataset_info.id,
            project_meta=project_meta,
            class_specs=class_specs,
            scene=scene,
        )

    print("Done.")


if __name__ == "__main__":
    main()
