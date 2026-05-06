from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import osmnx as ox
import requests
import supervisely as sly
from PIL import Image
from pydtmdl import DTMProvider, ImageryProvider
from shapely.ops import transform as shapely_transform

from import_osm.src.slyosm.geometry import (
    SceneGeoContext,
    SceneRequest,
    build_scene_geo_context,
    geometry_to_polygons,
    polygon_to_bitmap,
    polygon_to_supervisely_polygons,
    scene_metadata_payload,
)
from import_osm.src.slyosm.osm_config import (
    OSMClassSpec,
    class_specs_to_metadata_payload,
    ensure_project_meta_has_classes,
)
from import_osm.src.slyosm.settings import (
    ANNOTATION_BATCH_SIZE,
    IMAGES_DIR,
    META_DIR,
    build_generated_name,
    ensure_data_directories,
)

CLASS_MASK_DEFAULT_PRIORITY = 50
CLASS_MASK_PRIORITY = {
    "building": 100,
    "water": 95,
    "road_main": 90,
    "road_minor": 85,
    "field": 20,
    "forest": 10,
}

INTERFACE_MODE_MULTIVIEW = "multiview"
INTERFACE_MODE_OVERLAY = "overlay"
INTERFACE_MODE_ONLY_SATELLITE = "only_satellite"
INTERFACE_MODE_ONLY_DTM = "only_dtm"

SUPPORTED_INTERFACE_MODES = {
    INTERFACE_MODE_MULTIVIEW,
    INTERFACE_MODE_OVERLAY,
    INTERFACE_MODE_ONLY_SATELLITE,
    INTERFACE_MODE_ONLY_DTM,
}

DEFAULT_OUTPUT_RESOLUTION_PX = 1024
TARGET_GEOMETRY_POLYGON = "polygon"
TARGET_GEOMETRY_MASK = "mask"
SUPPORTED_TARGET_GEOMETRIES = {TARGET_GEOMETRY_POLYGON, TARGET_GEOMETRY_MASK}

if hasattr(Image, "Resampling"):
    _RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
    _RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
else:
    _RESAMPLE_LANCZOS = Image.LANCZOS
    _RESAMPLE_BICUBIC = Image.BICUBIC


@dataclass(frozen=True)
class SceneDownloadResult:
    """Successfully processed scene upload result."""

    scene_id: str
    image_id: int
    image_name: str
    metadata_path: Path
    instance_count: int


@dataclass(frozen=True)
class SceneDownloadFailure:
    """Per-scene failure captured during a batch run."""

    scene_id: str
    error: str


@dataclass(frozen=True)
class ResizedImageResult:
    """Result of aspect-preserving resize to square output."""

    image_rgb: np.ndarray
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    source_to_output_h: np.ndarray


def ensure_project_and_dataset(
    api: sly.Api,
    workspace_id: int,
    class_specs: List[OSMClassSpec],
    project_id: Optional[int] = None,
    project_name: Optional[str] = None,
    dataset_id: Optional[int] = None,
    dataset_name: Optional[str] = None,
    polygon_target_geometry: str = TARGET_GEOMETRY_POLYGON,
) -> Tuple[Any, Any, sly.ProjectMeta]:
    """Resolve or create the destination project and dataset.

    :param api: Supervisely API client.
    :type api: sly.Api
    :param workspace_id: Workspace identifier.
    :type workspace_id: int
    :param class_specs: OSM class specifications.
    :type class_specs: List[OSMClassSpec]
    :param project_id: Existing project identifier.
    :type project_id: Optional[int]
    :param project_name: New project name if a project must be created.
    :type project_name: Optional[str]
    :param dataset_id: Existing dataset identifier.
    :type dataset_id: Optional[int]
    :param dataset_name: New dataset name if a dataset must be created.
    :type dataset_name: Optional[str]
    :return: Project info, dataset info, and updated project meta.
    :rtype: Tuple[Any, Any, sly.ProjectMeta]
    :raises RuntimeError: If the requested destination can not be resolved.
    """

    if dataset_id is not None:
        dataset_info = api.dataset.get_info_by_id(int(dataset_id))
        if dataset_info is None:
            raise RuntimeError(
                "Dataset with id={dataset_id} was not found.".format(
                    dataset_id=dataset_id
                )
            )
        project_info = api.project.get_info_by_id(int(dataset_info.project_id))
        if project_info is None:
            raise RuntimeError(
                "Project with id={project_id} for dataset id={dataset_id} was not found.".format(
                    project_id=dataset_info.project_id,
                    dataset_id=dataset_id,
                )
            )
        if project_id is not None and int(project_id) != int(project_info.id):
            raise RuntimeError(
                "Dataset id={dataset_id} does not belong to project id={project_id}.".format(
                    dataset_id=dataset_id,
                    project_id=project_id,
                )
            )
    else:
        if project_id is not None:
            project_info = api.project.get_info_by_id(int(project_id))
            if project_info is None:
                raise RuntimeError(
                    "Project with id={project_id} was not found.".format(
                        project_id=project_id
                    )
                )
        else:
            resolved_project_name = project_name or build_generated_name(
                "import_osm"
            )
            project_info = api.project.create(
                workspace_id,
                resolved_project_name,
                change_name_if_conflict=True,
            )

        resolved_dataset_name = dataset_name or build_generated_name("download")
        dataset_info = api.dataset.create(
            project_info.id,
            resolved_dataset_name,
            change_name_if_conflict=True,
        )

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_info.id))
    updated_meta = ensure_project_meta_has_classes(
        project_meta,
        class_specs,
        polygon_target_geometry=polygon_target_geometry,
    )
    if updated_meta != project_meta:
        project_meta = api.project.update_meta(project_info.id, updated_meta)
    return project_info, dataset_info, project_meta


def save_dataset_custom_data(
    api: sly.Api, dataset_id: int, class_specs: List[OSMClassSpec]
) -> Any:
    """Persist class mapping to dataset custom data.

    :param api: Supervisely API client.
    :type api: sly.Api
    :param dataset_id: Dataset identifier.
    :type dataset_id: int
    :param class_specs: OSM class specifications.
    :type class_specs: List[OSMClassSpec]
    :return: Updated dataset info.
    :rtype: Any
    """

    dataset_info = api.dataset.get_info_by_id(dataset_id)
    custom_data = (
        dict(dataset_info.custom_data or {}) if dataset_info is not None else {}
    )
    custom_data["osm_class_specs"] = class_specs_to_metadata_payload(class_specs)
    custom_data["slyosm_schema_version"] = 1
    return api.dataset.update_custom_data(dataset_id, custom_data)


def is_positive_osm_tag_value(value: Any) -> bool:
    """Return ``True`` when an OSM attribute represents a positive flag value."""

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
    return text not in {"", "no", "false", "0", "nan", "none", "null"}


def is_underground_location_value(value: Any) -> bool:
    """Return ``True`` for OSM location values that should be hidden from export."""

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
    """Return ``True`` when the OSM layer attribute is negative."""

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
    polygon_target_geometry: str = TARGET_GEOMETRY_POLYGON,
) -> Tuple[List[sly.Label], int]:
    """Fetch matching OSM features for one class and convert them to labels.

    :param class_spec: Class specification.
    :type class_spec: OSMClassSpec
    :param obj_class: Supervisely object class.
    :type obj_class: sly.ObjClass
    :param scene_geo: Scene geometry context.
    :type scene_geo: SceneGeoContext
    :param image_width: Image width.
    :type image_width: int
    :param image_height: Image height.
    :type image_height: int
    :return: Labels and raw feature count.
    :rtype: Tuple[List[sly.Label], int]
    """

    left, bottom, right, top = scene_geo.bbox_left_bottom_right_top
    try:
        geodataframe = ox.features.features_from_bbox(
            (left, bottom, right, top), class_spec.tags
        )
    except Exception as exc:
        if "No matching features" in str(exc):
            return [], 0
        raise

    if class_spec.geometry == "line" and "highway" in class_spec.tags:
        excluded_total = 0
        hidden_mask = np.zeros(len(geodataframe), dtype=bool)

        if "tunnel" in geodataframe.columns:
            tunnel_mask = (
                geodataframe["tunnel"].map(is_positive_osm_tag_value).to_numpy()
            )
            excluded_total += int(tunnel_mask.sum())
            hidden_mask |= tunnel_mask

        if "covered" in geodataframe.columns:
            covered_mask = (
                geodataframe["covered"].map(is_positive_osm_tag_value).to_numpy()
            )
            excluded_total += int((covered_mask & ~hidden_mask).sum())
            hidden_mask |= covered_mask

        if "location" in geodataframe.columns:
            location_mask = (
                geodataframe["location"].map(is_underground_location_value).to_numpy()
            )
            excluded_total += int((location_mask & ~hidden_mask).sum())
            hidden_mask |= location_mask

        if "layer" in geodataframe.columns:
            layer_mask = geodataframe["layer"].map(is_negative_layer_value).to_numpy()
            excluded_total += int((layer_mask & ~hidden_mask).sum())
            hidden_mask |= layer_mask

        if "indoor" in geodataframe.columns:
            indoor_mask = (
                geodataframe["indoor"].map(is_positive_osm_tag_value).to_numpy()
            )
            excluded_total += int((indoor_mask & ~hidden_mask).sum())
            hidden_mask |= indoor_mask

        if excluded_total > 0:
            sly.logger.info(
                "Filtered %s non-top-visible features for class '%s'.",
                excluded_total,
                class_spec.name,
            )
            geodataframe = geodataframe.loc[~hidden_mask]

    if geodataframe.empty:
        return [], 0

    labels = []
    raw_features = 0
    for geometry_wgs84 in geodataframe.geometry:
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
            if polygon_target_geometry == TARGET_GEOMETRY_POLYGON:
                sly_polygons = polygon_to_supervisely_polygons(
                    polygon=polygon,
                    homography=scene_geo.local_to_pixel_h,
                    width=image_width,
                    height=image_height,
                )
                for sly_polygon in sly_polygons:
                    labels.append(sly.Label(sly_polygon, obj_class))
            else:
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


def class_mask_priority(class_name: str) -> int:
    """Return pixel priority for overlap removal.

    :param class_name: Class name.
    :type class_name: str
    :return: Priority value.
    :rtype: int
    """

    return int(CLASS_MASK_PRIORITY.get(class_name, CLASS_MASK_DEFAULT_PRIORITY))


def bitmap_to_full_mask(
    bitmap: sly.Bitmap, image_height: int, image_width: int
) -> np.ndarray:
    """Expand a Supervisely bitmap into a full image-sized boolean mask."""

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
    labels_by_class: Dict[str, List[sly.Label]],
    class_specs: List[OSMClassSpec],
    image_height: int,
    image_width: int,
) -> Tuple[Dict[str, List[sly.Label]], Dict[str, Dict[str, int]]]:
    """Clip label masks so higher-priority classes keep overlapping pixels.

    :param labels_by_class: Labels grouped by class name.
    :type labels_by_class: Dict[str, List[sly.Label]]
    :param class_specs: Class specifications.
    :type class_specs: List[OSMClassSpec]
    :param image_height: Image height.
    :type image_height: int
    :param image_width: Image width.
    :type image_width: int
    :return: Cleaned labels and overlap statistics.
    :rtype: Tuple[Dict[str, List[sly.Label]], Dict[str, Dict[str, int]]]
    """

    occupancy = np.zeros((image_height, image_width), dtype=bool)
    cleaned_by_class = dict((class_spec.name, []) for class_spec in class_specs)
    stats = dict(
        (class_spec.name, {"dropped_instances": 0, "overlap_pixels_removed": 0})
        for class_spec in class_specs
    )

    indexed_specs = list(enumerate(class_specs))
    ordered_specs = sorted(
        indexed_specs,
        key=lambda item: (-class_mask_priority(item[1].name), item[0]),
    )

    for _, class_spec in ordered_specs:
        class_name = class_spec.name
        for label in labels_by_class.get(class_name, []):
            if not isinstance(label.geometry, sly.Bitmap):
                cleaned_by_class[class_name].append(label)
                continue

            full_mask = bitmap_to_full_mask(
                label.geometry, image_height=image_height, image_width=image_width
            )
            if not np.any(full_mask):
                stats[class_name]["dropped_instances"] += 1
                continue

            visible_mask = full_mask & ~occupancy
            removed_pixels = int(
                np.count_nonzero(full_mask) - np.count_nonzero(visible_mask)
            )
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
    labels: List[sly.Label],
    initial_batch_size: int,
) -> None:
    """Append labels with automatic batch-size backoff on transient API failures.

    :param api: Supervisely API client.
    :type api: sly.Api
    :param image_id: Image identifier.
    :type image_id: int
    :param labels: Labels to append.
    :type labels: List[sly.Label]
    :param initial_batch_size: Initial chunk size.
    :type initial_batch_size: int
    """

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
            sly.logger.warning(
                "Annotation append failed at index=%s with batch_size=%s: %s. Retrying with %s.",
                next_index,
                batch_size,
                exc,
                new_batch_size,
            )
            batch_size = new_batch_size


def _normalize_to_uint8(data: np.ndarray) -> np.ndarray:
    """Convert numeric imagery arrays to uint8 for PNG export."""

    if data.dtype == np.uint8:
        return data

    data_float = data.astype(np.float32)
    min_value = float(np.min(data_float))
    max_value = float(np.max(data_float))
    if max_value <= min_value:
        return np.zeros_like(data_float, dtype=np.uint8)

    scaled = (data_float - min_value) * (255.0 / (max_value - min_value))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _to_hwc_array(data: Any) -> np.ndarray:
    """Convert pydtmdl output to an HxWxC array."""

    if np.ma.isMaskedArray(data):
        array = np.asarray(data.filled(0))
    else:
        array = np.asarray(data)

    if array.ndim == 2:
        array = array[:, :, None]
    elif array.ndim == 3:
        if array.shape[0] in {1, 3, 4} and array.shape[2] not in {1, 3, 4}:
            array = np.transpose(array, (1, 2, 0))
    else:
        raise RuntimeError(
            "Unexpected imagery data shape from pydtmdl: {shape}".format(
                shape=array.shape
            )
        )

    return array


def _visualize_array_rgb(data: Any) -> np.ndarray:
    """Convert numeric arrays to RGB uint8 visualization."""

    array = _to_hwc_array(data)
    channels = int(array.shape[2])
    if channels == 1:
        normalized = _normalize_to_uint8(array[:, :, 0])
        return np.repeat(normalized[:, :, None], 3, axis=2)

    if channels >= 3:
        rgb = array[:, :, :3]
        return _normalize_to_uint8(rgb)

    raise RuntimeError("Unsupported number of channels: {channels}".format(channels=channels))


def _resize_to_square_with_aspect(
    image_rgb: np.ndarray, output_size_px: int
) -> ResizedImageResult:
    """Resize image with aspect-ratio preservation and center-crop to a square canvas."""

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise RuntimeError("Expected RGB image array for resize.")

    target_size = int(output_size_px)
    if target_size <= 0:
        raise ValueError("Output resolution must be positive.")

    image = Image.fromarray(image_rgb, mode="RGB")
    source_width, source_height = image.size
    scale = max(target_size / float(source_width), target_size / float(source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    if scale < 1.0:
        resample = _RESAMPLE_LANCZOS
    else:
        resample = _RESAMPLE_BICUBIC
    resized = image.resize((resized_width, resized_height), resample=resample)

    left = max(0, (resized_width - target_size) // 2)
    top = max(0, (resized_height - target_size) // 2)
    right = left + target_size
    bottom = top + target_size
    cropped = resized.crop((left, top, right, bottom))

    # Projective geometry in this pipeline uses pixel-edge coordinates
    # ([-0.5, width-0.5] x [-0.5, height-0.5]).
    # Resize transform must preserve that convention before crop.
    scale_x = float(resized_width) / float(source_width) if source_width > 0 else 1.0
    scale_y = float(resized_height) / float(source_height) if source_height > 0 else 1.0
    bias_x = (scale_x - 1.0) * 0.5 - float(left)
    bias_y = (scale_y - 1.0) * 0.5 - float(top)
    source_to_output_h = np.asarray(
        [
            [float(scale_x), 0.0, float(bias_x)],
            [0.0, float(scale_y), float(bias_y)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return ResizedImageResult(
        image_rgb=np.asarray(cropped),
        source_width=int(source_width),
        source_height=int(source_height),
        output_width=int(target_size),
        output_height=int(target_size),
        source_to_output_h=source_to_output_h,
    )


def _download_satellite_layer(
    scene: SceneRequest,
    target_images_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """Download satellite imagery for scene and return processed RGB layer with metadata."""

    provider_code = None
    result = ImageryProvider.extract_area(
        center=(float(scene.center_lat), float(scene.center_lon)),
        width_m=int(scene.size_m),
        height_m=int(scene.size_m),
        rotation_deg=float(scene.rotation_deg),
        provider_code=provider_code,
        fallback_provider_code="sentinel2_l2a",
        directory=str(target_images_dir),
    )
    return (
        _visualize_array_rgb(result.data),
        result.metadata.model_dump(),
        "auto",
    )


def _download_dtm_layer(
    scene: SceneRequest,
    target_images_dir: Path,
) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """Download DTM data for scene and return processed RGB layer with metadata."""

    dtm_provider_code = None

    result = DTMProvider.extract_area(
        center=(float(scene.center_lat), float(scene.center_lon)),
        width_m=int(scene.size_m),
        height_m=int(scene.size_m),
        rotation_deg=float(scene.rotation_deg),
        provider_code=dtm_provider_code,
        fallback_provider_code="srtm30",
        directory=str(target_images_dir),
    )
    return _visualize_array_rgb(result.data), result.metadata.model_dump(), "auto"


def process_scene(
    api: sly.Api,
    dataset_id: int,
    project_meta: sly.ProjectMeta,
    class_specs: List[OSMClassSpec],
    scene: SceneRequest,
    interface_mode: str = INTERFACE_MODE_MULTIVIEW,
    output_resolution_px: int = DEFAULT_OUTPUT_RESOLUTION_PX,
    polygon_target_geometry: str = TARGET_GEOMETRY_POLYGON,
    download_osm: bool = True,
    images_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
) -> SceneDownloadResult:
    """Download, annotate, and upload one scene.

    :param api: Supervisely API client.
    :type api: sly.Api
    :param dataset_id: Target dataset identifier.
    :type dataset_id: int
    :param project_meta: Project metadata with object classes.
    :type project_meta: sly.ProjectMeta
    :param class_specs: OSM class specifications.
    :type class_specs: List[OSMClassSpec]
    :param scene: Scene request.
    :type scene: SceneRequest
    :param download_osm: If ``True``, also fetch and upload OSM labels.
    :type download_osm: bool
    :param images_dir: Optional directory for temporary images.
    :type images_dir: Optional[Path]
    :param meta_dir: Optional directory for metadata dumps.
    :type meta_dir: Optional[Path]
    :return: Upload result.
    :rtype: SceneDownloadResult
    """

    ensure_data_directories()
    target_images_dir = images_dir or IMAGES_DIR
    target_meta_dir = meta_dir or META_DIR
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_meta_dir.mkdir(parents=True, exist_ok=True)

    if polygon_target_geometry not in SUPPORTED_TARGET_GEOMETRIES:
        raise ValueError(
            "Unsupported polygon target geometry: {value}. Expected one of {supported}.".format(
                value=polygon_target_geometry,
                supported=sorted(SUPPORTED_TARGET_GEOMETRIES),
            )
        )

    if interface_mode not in SUPPORTED_INTERFACE_MODES:
        raise ValueError(
            "Unsupported interface mode: {mode}. Expected one of {supported}.".format(
                mode=interface_mode,
                supported=sorted(SUPPORTED_INTERFACE_MODES),
            )
        )

    should_download_satellite = interface_mode in {
        INTERFACE_MODE_MULTIVIEW,
        INTERFACE_MODE_OVERLAY,
        INTERFACE_MODE_ONLY_SATELLITE,
    }
    should_download_dtm = interface_mode in {
        INTERFACE_MODE_MULTIVIEW,
        INTERFACE_MODE_OVERLAY,
        INTERFACE_MODE_ONLY_DTM,
    }

    satellite_path = target_images_dir / "{scene_id}__satellite.png".format(
        scene_id=scene.identifier
    )
    dtm_path = target_images_dir / "{scene_id}__dtm.png".format(scene_id=scene.identifier)
    primary_path = target_images_dir / "{scene_id}.png".format(scene_id=scene.identifier)

    satellite_array = None
    dtm_array = None
    satellite_meta = None
    dtm_meta = None
    satellite_provider_requested = None
    dtm_provider_requested = None
    satellite_source_width = None
    satellite_source_height = None
    dtm_source_width = None
    dtm_source_height = None
    satellite_source_to_output_h = None
    dtm_source_to_output_h = None

    if should_download_satellite:
        sly.logger.info(
            "[%s] Downloading satellite imagery with pydtmdl provider=%s...",
            scene.identifier,
            "auto",
        )
        satellite_array, satellite_meta, satellite_provider_requested = _download_satellite_layer(
            scene, target_images_dir
        )
        satellite_resize = _resize_to_square_with_aspect(satellite_array, output_resolution_px)
        satellite_array = satellite_resize.image_rgb
        satellite_source_width = satellite_resize.source_width
        satellite_source_height = satellite_resize.source_height
        satellite_source_to_output_h = satellite_resize.source_to_output_h
        Image.fromarray(satellite_array, mode="RGB").save(satellite_path)

    if should_download_dtm:
        sly.logger.info("[%s] Downloading DTM data with pydtmdl...", scene.identifier)
        dtm_array, dtm_meta, dtm_provider_requested = _download_dtm_layer(
            scene, target_images_dir
        )
        dtm_resize = _resize_to_square_with_aspect(dtm_array, output_resolution_px)
        dtm_array = dtm_resize.image_rgb
        dtm_source_width = dtm_resize.source_width
        dtm_source_height = dtm_resize.source_height
        dtm_source_to_output_h = dtm_resize.source_to_output_h
        Image.fromarray(dtm_array, mode="RGB").save(dtm_path)

    if interface_mode in {INTERFACE_MODE_MULTIVIEW, INTERFACE_MODE_OVERLAY}:
        if satellite_array is None or dtm_array is None:
            raise RuntimeError(
                "Both satellite and DTM layers are required for mode '{mode}'.".format(
                    mode=interface_mode
                )
            )

    if interface_mode == INTERFACE_MODE_ONLY_DTM:
        if dtm_array is None:
            raise RuntimeError("DTM layer was not downloaded.")
        if dtm_source_width is None or dtm_source_height is None or dtm_source_to_output_h is None:
            raise RuntimeError("DTM source transform is missing.")
        Image.fromarray(dtm_array, mode="RGB").save(primary_path)
        image_width = int(dtm_array.shape[1])
        image_height = int(dtm_array.shape[0])
        source_width = int(dtm_source_width)
        source_height = int(dtm_source_height)
        source_to_output_h = dtm_source_to_output_h
    else:
        if satellite_array is None:
            raise RuntimeError("Satellite layer was not downloaded.")
        if (
            satellite_source_width is None
            or satellite_source_height is None
            or satellite_source_to_output_h is None
        ):
            raise RuntimeError("Satellite source transform is missing.")
        Image.fromarray(satellite_array, mode="RGB").save(primary_path)
        image_width = int(satellite_array.shape[1])
        image_height = int(satellite_array.shape[0])
        source_width = int(satellite_source_width)
        source_height = int(satellite_source_height)
        source_to_output_h = satellite_source_to_output_h

    scene_geo_base = build_scene_geo_context(
        center_lat=float(scene.center_lat),
        center_lon=float(scene.center_lon),
        size_m=int(scene.size_m),
        rotation_deg=float(scene.rotation_deg),
        image_width=source_width,
        image_height=source_height,
    )
    scene_geo = replace(
        scene_geo_base,
        local_to_pixel_h=np.asarray(source_to_output_h @ scene_geo_base.local_to_pixel_h, dtype=np.float64),
    )
    metadata = scene_metadata_payload(scene, scene_geo, image_width, image_height)
    metadata["interface_mode"] = interface_mode
    metadata["polygon_target_geometry"] = polygon_target_geometry
    metadata["output_resolution_px"] = int(output_resolution_px)
    metadata["render_transform"] = {
        "source_size_px": {"width": int(source_width), "height": int(source_height)},
        "output_size_px": {"width": int(image_width), "height": int(image_height)},
        "source_to_output_h": np.asarray(source_to_output_h, dtype=np.float64).tolist(),
        "policy": "cover_center_crop",
    }
    metadata["layers"] = {}
    if satellite_meta is not None:
        metadata["layers"]["satellite"] = {
            "provider_requested": satellite_provider_requested,
            "provider_used": satellite_meta.get("actual_provider"),
            "pydtmdl": satellite_meta,
            "image_name": satellite_path.name,
        }
    if dtm_meta is not None:
        metadata["layers"]["dtm"] = {
            "provider_requested": dtm_provider_requested,
            "provider_used": dtm_meta.get("actual_provider"),
            "pydtmdl": dtm_meta,
            "image_name": dtm_path.name,
        }
    metadata["osm_class_specs"] = class_specs_to_metadata_payload(class_specs)
    metadata["osm_download_enabled"] = bool(download_osm)

    labels_by_class = {}
    class_stats = {}

    if download_osm:
        for class_spec in class_specs:
            obj_class = project_meta.get_obj_class(class_spec.name)
            if obj_class is None:
                raise RuntimeError(
                    "Project meta is missing class '{name}'.".format(
                        name=class_spec.name
                    )
                )
            try:
                labels, raw_feature_count = fetch_class_labels(
                    class_spec=class_spec,
                    obj_class=obj_class,
                    scene_geo=scene_geo,
                    image_width=image_width,
                    image_height=image_height,
                    polygon_target_geometry=polygon_target_geometry,
                )
            except Exception as exc:
                labels = []
                raw_feature_count = 0
                class_stats[class_spec.name] = {
                    "raw_features": 0,
                    "instance_masks": 0,
                    "failed": True,
                    "error": str(exc),
                }
                sly.logger.warning(
                    "[%s] Failed to fetch class '%s': %s",
                    scene.identifier,
                    class_spec.name,
                    exc,
                )
                continue

            labels_by_class[class_spec.name] = labels
            class_stats[class_spec.name] = {
                "raw_features": raw_feature_count,
                "instance_masks": len(labels),
                "failed": False,
            }
            sly.logger.info(
                "[%s] %s: raw=%s masks=%s",
                scene.identifier,
                class_spec.name,
                raw_feature_count,
                len(labels),
            )

        cleaned_by_class, overlap_stats = enforce_non_overlapping_labels(
            labels_by_class=labels_by_class,
            class_specs=class_specs,
            image_height=image_height,
            image_width=image_width,
        )
    else:
        cleaned_by_class = dict((class_spec.name, []) for class_spec in class_specs)
        overlap_stats = dict(
            (class_spec.name, {"dropped_instances": 0, "overlap_pixels_removed": 0})
            for class_spec in class_specs
        )
        for class_spec in class_specs:
            class_stats[class_spec.name] = {
                "raw_features": 0,
                "instance_masks": 0,
                "instance_masks_non_overlap": 0,
                "dropped_instances_overlap": 0,
                "overlap_pixels_removed": 0,
                "failed": False,
                "skipped": True,
            }

    all_labels = []
    for class_spec in class_specs:
        class_name = class_spec.name
        cleaned_labels = cleaned_by_class.get(class_name, [])
        all_labels.extend(cleaned_labels)

        class_entry = class_stats.setdefault(
            class_name,
            {
                "raw_features": 0,
                "instance_masks": 0,
                "failed": False,
            },
        )
        class_entry["instance_masks_non_overlap"] = len(cleaned_labels)
        class_entry["dropped_instances_overlap"] = overlap_stats[class_name][
            "dropped_instances"
        ]
        class_entry["overlap_pixels_removed"] = overlap_stats[class_name][
            "overlap_pixels_removed"
        ]

    metadata["class_stats"] = class_stats
    metadata["instance_count"] = len(all_labels)

    if interface_mode in {INTERFACE_MODE_ONLY_SATELLITE, INTERFACE_MODE_ONLY_DTM}:
        image_info = api.image.upload_paths(
            dataset_id,
            [primary_path.name],
            [str(primary_path)],
            metas=[{"geo": metadata}],
            conflict_resolution="replace",
        )[0]
    elif interface_mode == INTERFACE_MODE_OVERLAY:
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        if dataset_info is not None:
            api.project.set_overlay_settings(dataset_info.project_id)

        parent_infos, _ = api.image.upload_overlay_images(
            dataset_id=dataset_id,
            names=[primary_path.name],
            paths=[str(primary_path)],
            overlay_names=[[dtm_path.name]],
            overlay_paths=[[str(dtm_path)]],
            conflict_resolution="replace",
        )
        image_info = parent_infos[0]
        api.image.update_meta(image_info.id, {"geo": metadata})
    else:
        dataset_info = api.dataset.get_info_by_id(dataset_id)
        if dataset_info is not None:
            api.project.set_multiview_settings(dataset_info.project_id)

        multiview_metas = [
            {
                "geo": {
                    **metadata,
                    "multiview_layer": "satellite",
                }
            },
            {
                "geo": {
                    **metadata,
                    "multiview_layer": "dtm",
                }
            },
        ]
        image_infos = api.image.upload_multiview_images(
            dataset_id=dataset_id,
            group_name=scene.identifier,
            paths=[str(satellite_path), str(dtm_path)],
            metas=multiview_metas,
            conflict_resolution="replace",
        )
        image_info = image_infos[0]

    if all_labels:
        sly.logger.info(
            "[%s] Uploading %s instance masks with chunk_size=%s...",
            scene.identifier,
            len(all_labels),
            ANNOTATION_BATCH_SIZE,
        )
        upload_annotation_resilient(
            api=api,
            image_id=image_info.id,
            labels=all_labels,
            initial_batch_size=ANNOTATION_BATCH_SIZE,
        )

    metadata_path = target_meta_dir / "{scene_id}.json".format(
        scene_id=scene.identifier
    )
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    sly.logger.info(
        "[%s] Uploaded image_id=%s instances=%s metadata=%s",
        scene.identifier,
        image_info.id,
        len(all_labels),
        metadata_path,
    )
    return SceneDownloadResult(
        scene_id=scene.identifier,
        image_id=image_info.id,
        image_name=image_info.name,
        metadata_path=metadata_path,
        instance_count=len(all_labels),
    )


def process_scenes(
    api: sly.Api,
    dataset_id: int,
    project_meta: sly.ProjectMeta,
    class_specs: List[OSMClassSpec],
    scenes: List[SceneRequest],
    interface_mode: str = INTERFACE_MODE_MULTIVIEW,
    output_resolution_px: int = DEFAULT_OUTPUT_RESOLUTION_PX,
    polygon_target_geometry: str = TARGET_GEOMETRY_POLYGON,
    download_osm: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int, SceneRequest], None]] = None,
    images_dir: Optional[Path] = None,
    meta_dir: Optional[Path] = None,
) -> Tuple[List[SceneDownloadResult], List[SceneDownloadFailure], bool]:
    """Process multiple scene requests and continue after per-scene failures.

    :param api: Supervisely API client.
    :type api: sly.Api
    :param dataset_id: Target dataset identifier.
    :type dataset_id: int
    :param project_meta: Project metadata with object classes.
    :type project_meta: sly.ProjectMeta
    :param class_specs: OSM class specifications.
    :type class_specs: List[OSMClassSpec]
    :param scenes: Scene requests.
    :type scenes: List[SceneRequest]
    :param download_osm: If ``True``, fetch and upload OSM labels.
    :type download_osm: bool
    :param should_stop: Optional callback checked between scenes.
    :type should_stop: Optional[Callable[[], bool]]
    :param progress_callback: Optional callback called after each scene attempt.
    :type progress_callback: Optional[Callable[[int, int, SceneRequest], None]]
    :param images_dir: Optional directory for temporary images.
    :type images_dir: Optional[Path]
    :param meta_dir: Optional directory for metadata dumps.
    :type meta_dir: Optional[Path]
    :return: Successful results, failures, and a stop flag.
    :rtype: Tuple[List[SceneDownloadResult], List[SceneDownloadFailure], bool]
    """

    results = []
    failures = []
    total = len(scenes)
    stopped = False

    for index, scene in enumerate(scenes, start=1):
        try:
            result = process_scene(
                api=api,
                dataset_id=dataset_id,
                project_meta=project_meta,
                class_specs=class_specs,
                scene=scene,
                interface_mode=interface_mode,
                output_resolution_px=output_resolution_px,
                polygon_target_geometry=polygon_target_geometry,
                download_osm=download_osm,
                images_dir=images_dir,
                meta_dir=meta_dir,
            )
            results.append(result)
        except Exception as exc:
            sly.logger.exception("[%s] Scene processing failed.", scene.identifier)
            failures.append(
                SceneDownloadFailure(scene_id=scene.identifier, error=str(exc))
            )

        if progress_callback is not None:
            progress_callback(index, total, scene)

        if should_stop is not None and should_stop():
            stopped = index < total
            break

    return results, failures, stopped
