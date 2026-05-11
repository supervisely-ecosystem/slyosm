from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import supervisely as sly
from pyproj import CRS, Transformer
from shapely.geometry import LineString as ShapelyLineString

from import_osm.src.slyosm.osm_config import (
    OSMClassSpec,
    load_osm_class_specs,
    load_osm_class_specs_from_payload,
    resolve_default_osm_tag,
)
from import_osm.src.slyosm.settings import OSM_CLASSES_PATH

LABELING_INTERFACE_OVERLAY = "overlay"
LABELING_INTERFACE_MULTIVIEW = "multi_view"

ANN_SOURCE_OSM = "osm"
ANN_SOURCE_JSON = "json"
ANN_SOURCE_NONE = "none"

MULTIVIEW_LAYER_SATELLITE = "satellite"
MULTIVIEW_LAYER_DTM = "dtm"


@dataclass
class ImageImportResult:
    image_name: str
    image_id: int
    annotation_source: str


@dataclass
class DatasetImportResult:
    dataset_name: str
    images: List[ImageImportResult] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)


@dataclass
class ProjectImportResult:
    project_id: int
    datasets: List[DatasetImportResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Inverse geo transform
# ---------------------------------------------------------------------------

def _build_lonlat_to_pixel_fn(
    pixel_to_local_h_list: List[List[float]],
    crs_wkt: str,
) -> Callable[[float, float], Tuple[float, float]]:
    """Return a callable (lon, lat) → (pixel_x, pixel_y) using the inverse geo transform."""
    h = np.array(pixel_to_local_h_list, dtype=np.float64)
    h_inv = np.linalg.inv(h)
    local_crs = CRS.from_wkt(crs_wkt)
    from_wgs84 = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)

    def lonlat_to_pixel(lon: float, lat: float) -> Tuple[float, float]:
        local_x, local_y = from_wgs84.transform(lon, lat)
        pt = h_inv @ np.array([local_x, local_y, 1.0], dtype=np.float64)
        return float(pt[0] / pt[2]), float(pt[1] / pt[2])

    return lonlat_to_pixel


# ---------------------------------------------------------------------------
# OSM file parsing
# ---------------------------------------------------------------------------

def _tag_value_str(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return str(value)


def _match_spec_to_tags(
    tags: Dict[str, str],
    class_specs: List[OSMClassSpec],
) -> Optional[OSMClassSpec]:
    """Find the first class spec whose resolved default_tag matches the given OSM tags."""
    for spec in class_specs:
        resolved = resolve_default_osm_tag(spec)
        if all(tags.get(k) == _tag_value_str(v) for k, v in resolved.items()):
            return spec
    return None


def _parse_osm_xml(
    osm_path: Path,
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[List[Tuple[float, float]], Dict[str, str]]],
    List[Tuple[List[int], List[int], Dict[str, str]]],
]:
    """Parse an OSM XML file into nodes, ways, and relations.

    :return: Tuple of (nodes dict, ways dict, relations list).
             nodes: {id: (lat, lon)}
             ways:  {id: ([(lat, lon), ...], {tag_key: tag_val})}
             relations: [(outer_way_ids, inner_way_ids, {tag_key: tag_val})]
    """
    tree = ET.parse(str(osm_path))
    root = tree.getroot()

    nodes: Dict[int, Tuple[float, float]] = {}
    for node in root.findall("node"):
        nid = int(node.get("id", "0"))
        nodes[nid] = (float(node.get("lat", "0")), float(node.get("lon", "0")))

    ways: Dict[int, Tuple[List[Tuple[float, float]], Dict[str, str]]] = {}
    for way in root.findall("way"):
        wid = int(way.get("id", "0"))
        coords = [nodes[int(nd.get("ref", "0"))] for nd in way.findall("nd") if int(nd.get("ref", "0")) in nodes]
        tags = {t.get("k", ""): t.get("v", "") for t in way.findall("tag")}
        ways[wid] = (coords, tags)

    relations: List[Tuple[List[int], List[int], Dict[str, str]]] = []
    for rel in root.findall("relation"):
        outer: List[int] = []
        inner: List[int] = []
        for member in rel.findall("member"):
            if member.get("type") != "way":
                continue
            ref = int(member.get("ref", "0"))
            role = member.get("role", "")
            (outer if role == "outer" else inner).append(ref)
        tags = {t.get("k", ""): t.get("v", "") for t in rel.findall("tag")}
        relations.append((outer, inner, tags))

    return nodes, ways, relations


def _buffer_line_latlon_to_pixel_pts(
    lat_lon_list: List[Tuple[float, float]],
    buffer_m: float,
    pixel_to_local_h_list: List[List[float]],
    crs_wkt: str,
) -> List[sly.PointLocation]:
    """Buffer a road centerline in local CRS and return pixel-space polygon exterior.

    Mirrors the downloader's geometry_to_polygons + buffer_m approach so that
    line-class features are reconstructed as correctly-shaped road polygons rather
    than thin node-sliver polygons.
    """
    h = np.array(pixel_to_local_h_list, dtype=np.float64)
    h_inv = np.linalg.inv(h)
    local_crs = CRS.from_wkt(crs_wkt)
    from_wgs84 = Transformer.from_crs(CRS.from_epsg(4326), local_crs, always_xy=True)

    local_coords = [from_wgs84.transform(lon, lat) for lat, lon in lat_lon_list]
    if len(local_coords) < 2:
        return []

    line = ShapelyLineString(local_coords)
    effective_buffer = max(float(buffer_m), 1.0)
    buffered = line.buffer(effective_buffer)
    if buffered.is_empty:
        return []

    pts = []
    for lx, ly in buffered.exterior.coords:
        pt = h_inv @ np.array([lx, ly, 1.0], dtype=np.float64)
        pts.append(sly.PointLocation(row=float(pt[1] / pt[2]), col=float(pt[0] / pt[2])))
    return pts


def _latlon_list_to_pixel_points(
    lat_lon_list: List[Tuple[float, float]],
    lonlat_to_pixel: Callable[[float, float], Tuple[float, float]],
) -> List[sly.PointLocation]:
    result = []
    for lat, lon in lat_lon_list:
        px, py = lonlat_to_pixel(lon, lat)
        result.append(sly.PointLocation(row=py, col=px))
    return result


def _ring_from_way_ids(
    way_ids: List[int],
    ways: Dict[int, Tuple[List[Tuple[float, float]], Dict[str, str]]],
    lonlat_to_pixel: Callable[[float, float], Tuple[float, float]],
) -> List[List[sly.PointLocation]]:
    rings = []
    for wid in way_ids:
        if wid not in ways:
            continue
        coords, _ = ways[wid]
        pts = _latlon_list_to_pixel_points(coords, lonlat_to_pixel)
        # Drop the closing duplicate (first == last) added during export
        if len(pts) > 3 and pts[0].row == pts[-1].row and pts[0].col == pts[-1].col:
            pts = pts[:-1]
        if len(pts) >= 3:
            rings.append(pts)
    return rings


def osm_file_to_annotation(
    osm_path: Path,
    class_specs: List[OSMClassSpec],
    geo_meta: Dict[str, Any],
    image_width: int,
    image_height: int,
    project_meta: sly.ProjectMeta,
) -> sly.Annotation:
    """Parse an exported OSM XML file into a Supervisely Annotation.

    Uses the inverse of the pixel-to-world homography stored in ``geo_meta`` to
    reproject geographic coordinates back to pixel space.  Tags in the OSM file
    are matched against the resolved ``default_tag`` of each class spec to
    identify the Supervisely class.

    :param osm_path: Path to the ``.osm`` file produced by the exporter.
    :param class_specs: Class specifications for tag-to-class matching.
    :param geo_meta: Image metadata dict (from ``meta/{image}.json``).
    :param image_width: Width of the target image in pixels.
    :param image_height: Height of the target image in pixels.
    :param project_meta: Project metadata containing class definitions.
    :return: Supervisely Annotation with all recovered labels.
    """
    img_size = (image_height, image_width)

    geo = geo_meta.get("geo") if isinstance(geo_meta, dict) else None
    if not isinstance(geo, dict):
        return sly.Annotation(img_size=img_size)

    h_list = geo.get("pixel_to_local_h")
    # The downloader stores the CRS under "local_crs_wkt"; support both names.
    crs_wkt = geo.get("local_crs_wkt") or geo.get("crs_wkt")
    if not crs_wkt:
        projjson = geo.get("local_crs_projjson")
        if projjson:
            try:
                crs_wkt = CRS.from_json_dict(projjson).to_wkt()
            except Exception:
                pass
    if not h_list or not crs_wkt:
        return sly.Annotation(img_size=img_size)

    try:
        lonlat_to_pixel = _build_lonlat_to_pixel_fn(h_list, crs_wkt)
    except Exception:
        sly.logger.exception("Failed to build inverse geo transform for '%s'.", osm_path.name)
        return sly.Annotation(img_size=img_size)

    try:
        _nodes, ways, relations = _parse_osm_xml(osm_path)
    except Exception:
        sly.logger.exception("Failed to parse OSM XML '%s'.", osm_path.name)
        return sly.Annotation(img_size=img_size)

    # Way IDs that belong to relations — don't add them as standalone features
    relation_way_ids: set = set()
    for outer_ids, inner_ids, _ in relations:
        relation_way_ids.update(outer_ids)
        relation_way_ids.update(inner_ids)

    labels: List[sly.Label] = []

    # Standalone ways (not part of any relation)
    for way_id, (coords, tags) in ways.items():
        if way_id in relation_way_ids or len(coords) < 2:
            continue
        spec = _match_spec_to_tags(tags, class_specs)
        if spec is None:
            continue
        obj_class = project_meta.get_obj_class(spec.name)
        if obj_class is None:
            continue

        if spec.geometry == "line":
            # The exporter skeletonizes buffered road polygons down to OSM centerlines.
            # Reconstruct the original polygon by re-buffering in local CRS, matching
            # what the downloader does with geometry_to_polygons + buffer_m.
            exterior = _buffer_line_latlon_to_pixel_pts(coords, spec.buffer_m, h_list, crs_wkt)
            if len(exterior) >= 3:
                labels.append(sly.Label(sly.Polygon(exterior=exterior, interior=[]), obj_class))
        else:
            pts = _latlon_list_to_pixel_points(coords, lonlat_to_pixel)
            is_closed = len(coords) >= 3 and coords[0] == coords[-1]
            if obj_class.geometry_type == sly.Polygon:
                exterior = pts[:-1] if is_closed else pts
                if len(exterior) >= 3:
                    labels.append(sly.Label(sly.Polygon(exterior=exterior, interior=[]), obj_class))
            elif obj_class.geometry_type == sly.Polyline:
                if len(pts) >= 2:
                    labels.append(sly.Label(sly.Polyline(pts), obj_class))

    # Multipolygon relations
    for outer_ids, inner_ids, tags in relations:
        spec = _match_spec_to_tags(tags, class_specs)
        if spec is None:
            continue
        obj_class = project_meta.get_obj_class(spec.name)
        if obj_class is None or obj_class.geometry_type != sly.Polygon:
            continue

        outer_rings = _ring_from_way_ids(outer_ids, ways, lonlat_to_pixel)
        inner_rings = _ring_from_way_ids(inner_ids, ways, lonlat_to_pixel)

        for outer_ring in outer_rings:
            labels.append(
                sly.Label(
                    sly.Polygon(exterior=outer_ring, interior=inner_rings[:1] if inner_rings else []),
                    obj_class,
                )
            )

    return sly.Annotation(img_size=img_size, labels=labels)


# ---------------------------------------------------------------------------
# Annotation resolution (OSM > JSON)
# ---------------------------------------------------------------------------

def _read_image_meta(meta_dir: Path, image_name: str) -> Dict[str, Any]:
    meta_file = meta_dir / "{name}.json".format(name=image_name)
    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _resolve_annotation(
    api: sly.Api,
    image_name: str,
    image_id: int,
    image_width: int,
    image_height: int,
    ann_dir: Path,
    osm_dir: Path,
    meta_dir: Path,
    class_specs: List[OSMClassSpec],
    project_meta: sly.ProjectMeta,
) -> str:
    """Upload the best available annotation for ``image_id`` and return the source used.

    OSM files take priority over annotation JSON when both are present.
    """
    osm_path = osm_dir / "{name}.osm".format(name=image_name)
    ann_path = ann_dir / "{name}.json".format(name=image_name)

    if osm_dir.is_dir() and osm_path.exists():
        try:
            geo_meta = _read_image_meta(meta_dir, image_name)
            annotation = osm_file_to_annotation(
                osm_path, class_specs, geo_meta, image_width, image_height, project_meta
            )
            api.annotation.upload_ann(image_id, annotation)
            return ANN_SOURCE_OSM
        except Exception:
            sly.logger.exception(
                "OSM annotation parse failed for '%s'; falling back to JSON.", image_name
            )

    if ann_dir.is_dir() and ann_path.exists():
        try:
            ann_json = json.loads(ann_path.read_text(encoding="utf-8"))
            api.annotation.upload_json(image_id, ann_json)
            return ANN_SOURCE_JSON
        except Exception:
            sly.logger.exception("JSON annotation upload failed for '%s'.", image_name)

    return ANN_SOURCE_NONE


# ---------------------------------------------------------------------------
# Class spec loading
# ---------------------------------------------------------------------------

def _load_class_specs(dataset_dir: Path) -> List[OSMClassSpec]:
    """Load class specs from the dataset's custom_data.json, falling back to built-in defaults."""
    custom_data_file = dataset_dir / "custom_data.json"
    if custom_data_file.exists():
        try:
            payload = json.loads(custom_data_file.read_text(encoding="utf-8"))
            specs = load_osm_class_specs_from_payload(payload)
            if specs:
                return specs
        except Exception:
            pass
    return load_osm_class_specs(OSM_CLASSES_PATH)


# ---------------------------------------------------------------------------
# Per-mode image import helpers
# ---------------------------------------------------------------------------

def _import_standard(
    api: sly.Api,
    dataset_id: int,
    image_files: List[Path],
    meta_dir: Path,
    ann_dir: Path,
    osm_dir: Path,
    class_specs: List[OSMClassSpec],
    project_meta: sly.ProjectMeta,
    result: DatasetImportResult,
    progress_callback: Optional[Callable[[int, int], None]],
    should_stop: Optional[Callable[[], bool]],
) -> None:
    total = len(image_files)
    for index, img_file in enumerate(image_files, start=1):
        if should_stop and should_stop():
            break
        image_name = img_file.name
        try:
            image_meta = _read_image_meta(meta_dir, image_name)
            image_info = api.image.upload_paths(
                dataset_id,
                names=[image_name],
                paths=[str(img_file)],
                metas=[image_meta if image_meta else {}],
                conflict_resolution="replace",
            )[0]
            ann_source = _resolve_annotation(
                api, image_name, image_info.id, image_info.width, image_info.height,
                ann_dir, osm_dir, meta_dir, class_specs, project_meta,
            )
            result.images.append(
                ImageImportResult(image_name=image_name, image_id=image_info.id, annotation_source=ann_source)
            )
        except Exception as exc:
            sly.logger.exception("Failed to import image '%s'.", image_name)
            result.failures.append("{name}: {error}".format(name=image_name, error=exc))
        if progress_callback:
            progress_callback(index, total)


def _import_overlay(
    api: sly.Api,
    dataset_id: int,
    image_files: List[Path],
    meta_dir: Path,
    overlay_dir: Path,
    ann_dir: Path,
    osm_dir: Path,
    class_specs: List[OSMClassSpec],
    project_meta: sly.ProjectMeta,
    result: DatasetImportResult,
    progress_callback: Optional[Callable[[int, int], None]],
    should_stop: Optional[Callable[[], bool]],
) -> None:
    total = len(image_files)
    for index, img_file in enumerate(image_files, start=1):
        if should_stop and should_stop():
            break
        image_name = img_file.name
        try:
            image_meta = _read_image_meta(meta_dir, image_name)
            image_stem = Path(image_name).stem
            this_overlay_dir = overlay_dir / image_stem if overlay_dir.is_dir() else None
            overlay_names: List[str] = []
            overlay_paths: List[str] = []
            if this_overlay_dir and this_overlay_dir.is_dir():
                for ov in sorted(this_overlay_dir.iterdir()):
                    if ov.is_file():
                        overlay_names.append(ov.name)
                        overlay_paths.append(str(ov))

            if overlay_names:
                parent_infos, _ = api.image.upload_overlay_images(
                    dataset_id=dataset_id,
                    names=[image_name],
                    paths=[str(img_file)],
                    overlay_names=[overlay_names],
                    overlay_paths=[overlay_paths],
                    conflict_resolution="replace",
                )
                image_info = parent_infos[0]
                if image_meta:
                    api.image.update_meta(image_info.id, image_meta)
            else:
                image_info = api.image.upload_paths(
                    dataset_id,
                    names=[image_name],
                    paths=[str(img_file)],
                    metas=[image_meta if image_meta else {}],
                    conflict_resolution="replace",
                )[0]

            ann_source = _resolve_annotation(
                api, image_name, image_info.id, image_info.width, image_info.height,
                ann_dir, osm_dir, meta_dir, class_specs, project_meta,
            )
            result.images.append(
                ImageImportResult(image_name=image_name, image_id=image_info.id, annotation_source=ann_source)
            )
        except Exception as exc:
            sly.logger.exception("Failed to import overlay image '%s'.", image_name)
            result.failures.append("{name}: {error}".format(name=image_name, error=exc))
        if progress_callback:
            progress_callback(index, total)


def _import_multiview(
    api: sly.Api,
    dataset_id: int,
    image_files: List[Path],
    meta_dir: Path,
    ann_dir: Path,
    osm_dir: Path,
    class_specs: List[OSMClassSpec],
    project_meta: sly.ProjectMeta,
    result: DatasetImportResult,
    progress_callback: Optional[Callable[[int, int], None]],
    should_stop: Optional[Callable[[], bool]],
) -> None:
    # Group images by scene_id using the multiview_layer field in geo meta.
    # Images without multiview_layer are imported as standalone.
    groups: Dict[str, Dict[str, Path]] = {}
    standalone: List[Path] = []

    for img_file in image_files:
        geo_meta = _read_image_meta(meta_dir, img_file.name)
        geo = geo_meta.get("geo", {}) if isinstance(geo_meta, dict) else {}
        scene_id = geo.get("scene_id") if isinstance(geo, dict) else None
        layer = geo.get("multiview_layer") if isinstance(geo, dict) else None
        if scene_id and layer in (MULTIVIEW_LAYER_SATELLITE, MULTIVIEW_LAYER_DTM):
            groups.setdefault(str(scene_id), {})[layer] = img_file
        else:
            standalone.append(img_file)

    total = len(image_files)
    processed = 0

    for scene_id, layers in groups.items():
        if should_stop and should_stop():
            break
        ordered = [
            (MULTIVIEW_LAYER_SATELLITE, layers.get(MULTIVIEW_LAYER_SATELLITE)),
            (MULTIVIEW_LAYER_DTM, layers.get(MULTIVIEW_LAYER_DTM)),
        ]
        names_to_upload = [name for _, f in ordered if f for name in [f.name]]
        paths_to_upload = [str(f) for _, f in ordered if f]
        metas_to_upload = [
            _read_image_meta(meta_dir, f.name) or {}
            for _, f in ordered if f
        ]
        if not names_to_upload:
            continue
        try:
            image_infos = api.image.upload_multiview_images(
                dataset_id=dataset_id,
                group_name=scene_id,
                paths=paths_to_upload,
                metas=metas_to_upload,
                conflict_resolution="replace",
            )
            primary_info = image_infos[0]
            ann_source = _resolve_annotation(
                api, names_to_upload[0], primary_info.id,
                primary_info.width, primary_info.height,
                ann_dir, osm_dir, meta_dir, class_specs, project_meta,
            )
            for img_info, img_name in zip(image_infos, names_to_upload):
                result.images.append(
                    ImageImportResult(image_name=img_name, image_id=img_info.id, annotation_source=ann_source)
                )
        except Exception as exc:
            sly.logger.exception("Failed to import multiview group '%s'.", scene_id)
            for _, f in ordered:
                if f:
                    result.failures.append("{name}: {error}".format(name=f.name, error=exc))
        processed += len(layers)
        if progress_callback:
            progress_callback(min(processed, total), total)

    for img_file in standalone:
        if should_stop and should_stop():
            break
        image_name = img_file.name
        try:
            image_meta = _read_image_meta(meta_dir, image_name)
            image_info = api.image.upload_paths(
                dataset_id,
                names=[image_name],
                paths=[str(img_file)],
                metas=[image_meta if image_meta else {}],
                conflict_resolution="replace",
            )[0]
            ann_source = _resolve_annotation(
                api, image_name, image_info.id, image_info.width, image_info.height,
                ann_dir, osm_dir, meta_dir, class_specs, project_meta,
            )
            result.images.append(
                ImageImportResult(image_name=image_name, image_id=image_info.id, annotation_source=ann_source)
            )
        except Exception as exc:
            sly.logger.exception("Failed to import standalone image '%s'.", image_name)
            result.failures.append("{name}: {error}".format(name=image_name, error=exc))
        processed += 1
        if progress_callback:
            progress_callback(processed, total)


# ---------------------------------------------------------------------------
# Dataset-level import
# ---------------------------------------------------------------------------

def import_dataset_dir(
    api: sly.Api,
    dataset_dir: Path,
    project_id: int,
    project_meta: sly.ProjectMeta,
    labeling_interface: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> DatasetImportResult:
    """Import one dataset directory into a Supervisely project.

    :param api: Supervisely API client.
    :param dataset_dir: Dataset directory (must contain ``img/``; optionally ``ann/``,
        ``meta/``, ``osm/``, ``overlay/``).
    :param project_id: Target Supervisely project ID.
    :param project_meta: Project metadata with class definitions.
    :param labeling_interface: ``"overlay"``, ``"multi_view"``, or ``""`` for default.
    :param progress_callback: Optional ``(current, total)`` progress callback.
    :param should_stop: Optional callable returning ``True`` to abort early.
    :return: Dataset import result.
    """
    dataset_name = dataset_dir.name
    result = DatasetImportResult(dataset_name=dataset_name)

    img_dir = dataset_dir / "img"
    ann_dir = dataset_dir / "ann"
    meta_dir = dataset_dir / "meta"
    osm_dir = dataset_dir / "osm"
    overlay_dir = dataset_dir / "overlay"

    if not img_dir.is_dir():
        result.failures.append(
            "Dataset '{name}' has no img/ subdirectory.".format(name=dataset_name)
        )
        return result

    image_files = sorted(
        [p for p in img_dir.iterdir() if p.is_file()],
        key=lambda p: p.name,
    )
    if not image_files:
        return result

    class_specs = _load_class_specs(dataset_dir)

    dataset_info = api.dataset.create(
        project_id, dataset_name, change_name_if_conflict=True
    )

    kwargs = dict(
        api=api,
        dataset_id=dataset_info.id,
        image_files=image_files,
        meta_dir=meta_dir,
        ann_dir=ann_dir,
        osm_dir=osm_dir,
        class_specs=class_specs,
        project_meta=project_meta,
        result=result,
        progress_callback=progress_callback,
        should_stop=should_stop,
    )

    if labeling_interface == LABELING_INTERFACE_MULTIVIEW:
        _import_multiview(**kwargs)
    elif labeling_interface == LABELING_INTERFACE_OVERLAY:
        _import_overlay(overlay_dir=overlay_dir, **kwargs)
    else:
        _import_standard(**kwargs)

    return result


# ---------------------------------------------------------------------------
# Project-level import
# ---------------------------------------------------------------------------

def import_from_local_dir(
    api: sly.Api,
    source_dir: Path,
    workspace_id: int,
    project_name_override: str = "",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> ProjectImportResult:
    """Import a complete Supervisely export directory into a new project.

    Reads ``meta.json`` from the export root to restore class definitions, tag
    schemas, and interface settings (multiview or overlay).  For each dataset
    subdirectory the images, per-image geo metadata, and annotations are
    re-uploaded.  When both an OSM file and a JSON annotation exist for the same
    image the **OSM file takes priority**.

    :param api: Supervisely API client.
    :param source_dir: Root directory of the export (contains ``meta.json`` and
        one subdirectory per dataset).
    :param workspace_id: Target Supervisely workspace ID.
    :param project_name_override: If non-empty, use this as the project name
        instead of the source directory name.
    :param progress_callback: Optional ``(current, total)`` callback per image.
    :param should_stop: Optional callable returning ``True`` to abort.
    :return: Project import result with per-dataset breakdowns.
    """
    meta_json_path = source_dir / "meta.json"
    if not meta_json_path.exists():
        raise RuntimeError(
            "No meta.json found in '{dir}'. "
            "The source must be a valid Supervisely project export.".format(dir=source_dir)
        )

    meta_json = json.loads(meta_json_path.read_text(encoding="utf-8"))
    project_meta = sly.ProjectMeta.from_json(meta_json)

    project_settings = meta_json.get("projectSettings", {})
    labeling_interface_raw = project_settings.get("labelingInterface", "")
    multiview_cfg = project_settings.get("multiView", {})
    multiview_enabled = (
        multiview_cfg.get("enabled", False) if isinstance(multiview_cfg, dict) else False
    )

    if labeling_interface_raw == "overlay":
        effective_interface = LABELING_INTERFACE_OVERLAY
    elif multiview_enabled:
        effective_interface = LABELING_INTERFACE_MULTIVIEW
    else:
        effective_interface = ""

    project_name = project_name_override.strip() or source_dir.name
    project_info = api.project.create(
        workspace_id, project_name, sly.ProjectType.IMAGES, change_name_if_conflict=True
    )
    project_id = project_info.id

    api.project.update_meta(project_id, project_meta.to_json())

    if effective_interface == LABELING_INTERFACE_OVERLAY:
        api.project.set_overlay_settings(project_id)
    elif effective_interface == LABELING_INTERFACE_MULTIVIEW:
        api.project.set_multiview_settings(project_id)

    dataset_dirs = sorted(
        [d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.name,
    )

    import_result = ProjectImportResult(project_id=project_id)
    for dataset_dir in dataset_dirs:
        if should_stop and should_stop():
            break
        sly.logger.info("Importing dataset '%s'.", dataset_dir.name)
        dataset_result = import_dataset_dir(
            api=api,
            dataset_dir=dataset_dir,
            project_id=project_id,
            project_meta=project_meta,
            labeling_interface=effective_interface,
            progress_callback=progress_callback,
            should_stop=should_stop,
        )
        import_result.datasets.append(dataset_result)
        sly.logger.info(
            "Dataset '%s': %s image(s) imported, %s failure(s).",
            dataset_dir.name,
            len(dataset_result.images),
            len(dataset_result.failures),
        )

    return import_result
