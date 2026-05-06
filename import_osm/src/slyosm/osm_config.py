from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import supervisely as sly


SUPPORTED_GEOMETRIES = {"polygon", "line", "point"}


@dataclass(frozen=True)
class OSMClassSpec:
    """Mapping between a Supervisely class and OSM tags.

    :param name: Supervisely class name.
    :type name: str
    :param geometry: Geometry type expected by the class.
    :type geometry: str
    :param tags: OSM tag filters used for import.
    :type tags: Dict[str, Any]
    :param default_tag: Single OSM tag used for export.
    :type default_tag: Dict[str, Any]
    :param buffer_m: Buffer used for lines and points during import.
    :type buffer_m: float
    :param color: Optional RGB color for the Supervisely class.
    :type color: Optional[List[int]]
    """

    name: str
    geometry: str
    tags: Dict[str, Any]
    default_tag: Dict[str, Any]
    buffer_m: float = 0.0
    color: Optional[List[int]] = None


def normalize_osm_tags(class_name: str, tags: Dict[str, Any]) -> Dict[str, str]:
    """Convert raw OSM tag declarations to a single concrete string mapping.

    :param class_name: Class name used for highway fallback resolution.
    :type class_name: str
    :param tags: Raw tag mapping from configuration.
    :type tags: Dict[str, Any]
    :return: Normalized OSM tag mapping.
    :rtype: Dict[str, str]
    """

    normalized = {}
    for key, value in tags.items():
        if isinstance(value, bool):
            if value:
                normalized[key] = "yes"
            continue

        if isinstance(value, list):
            if len(value) == 0:
                continue
            if key == "highway" and len(value) > 1:
                if "main" in class_name:
                    normalized[key] = "secondary"
                elif "minor" in class_name:
                    normalized[key] = "residential"
                else:
                    normalized[key] = str(value[0])
            else:
                normalized[key] = str(value[0])
            continue

        if value is None:
            continue

        normalized[key] = str(value)

    return normalized


def resolve_default_osm_tag(class_spec: OSMClassSpec) -> Dict[str, str]:
    """Resolve the single export tag for a class specification.

    :param class_spec: Class specification.
    :type class_spec: OSMClassSpec
    :return: Normalized single-tag mapping.
    :rtype: Dict[str, str]
    """

    normalized = normalize_osm_tags(class_spec.name, class_spec.default_tag)
    if not normalized:
        normalized = normalize_osm_tags(class_spec.name, class_spec.tags)

    if len(normalized) <= 1:
        return normalized

    first_key = sorted(normalized.keys())[0]
    return {first_key: normalized[first_key]}


def load_osm_class_specs(config_path: Path) -> List[OSMClassSpec]:
    """Load OSM class specifications from a JSON file.

    :param config_path: Path to the JSON file.
    :type config_path: Path
    :return: Parsed class specifications.
    :rtype: List[OSMClassSpec]
    """

    return parse_osm_class_specs(config_path.read_text(encoding="utf-8"))


def parse_osm_class_specs(raw_text: str) -> List[OSMClassSpec]:
    """Parse OSM class specifications from JSON text.

    :param raw_text: JSON payload.
    :type raw_text: str
    :return: Parsed class specifications.
    :rtype: List[OSMClassSpec]
    :raises ValueError: If the JSON payload is invalid.
    """

    try:
        raw_specs = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse osm_classes JSON: {error}".format(error=exc))

    if not isinstance(raw_specs, list):
        raise ValueError("OSM classes configuration must be a JSON array.")

    return _parse_raw_specs(raw_specs)


def load_osm_class_specs_from_payload(payload: Any) -> List[OSMClassSpec]:
    """Load class specifications from image or dataset custom data payload.

    :param payload: Dataset custom data or image geo payload.
    :type payload: Any
    :return: Parsed class specifications. Empty list means the payload has no mapping.
    :rtype: List[OSMClassSpec]
    """

    if isinstance(payload, dict):
        payload = payload.get("osm_class_specs")

    if payload is None:
        return []

    if not isinstance(payload, list):
        raise ValueError("OSM class specs payload must be a list.")

    return _parse_raw_specs(payload)


def serialize_osm_class_specs(class_specs: List[OSMClassSpec]) -> str:
    """Serialize class specifications to human-readable JSON.

    :param class_specs: Class specifications.
    :type class_specs: List[OSMClassSpec]
    :return: JSON text.
    :rtype: str
    """

    return json.dumps(class_specs_to_metadata_payload(class_specs), indent=2)


def class_specs_to_metadata_payload(class_specs: List[OSMClassSpec]) -> List[Dict[str, Any]]:
    """Convert class specifications to a JSON-serializable payload.

    :param class_specs: Class specifications.
    :type class_specs: List[OSMClassSpec]
    :return: JSON-serializable payload.
    :rtype: List[Dict[str, Any]]
    """

    payload = []
    for class_spec in class_specs:
        payload.append(
            {
                "name": class_spec.name,
                "geometry": class_spec.geometry,
                "tags": class_spec.tags,
                "default_tag": class_spec.default_tag,
                "buffer_m": class_spec.buffer_m,
                "color": class_spec.color,
            }
        )
    return payload


def ensure_project_meta_has_classes(
    project_meta: sly.ProjectMeta,
    class_specs: List[OSMClassSpec],
    polygon_target_geometry: str = "polygon",
) -> sly.ProjectMeta:
    """Add missing bitmap object classes to project metadata.

    :param project_meta: Existing project metadata.
    :type project_meta: sly.ProjectMeta
    :param class_specs: Class specifications.
    :type class_specs: List[OSMClassSpec]
    :return: Updated project metadata.
    :rtype: sly.ProjectMeta
    """

    if polygon_target_geometry not in {"polygon", "mask"}:
        raise ValueError(
            "Unsupported polygon target geometry: {value}".format(
                value=polygon_target_geometry
            )
        )

    updated_meta = project_meta
    for class_spec in class_specs:
        desired_geometry_type = (
            sly.Polygon if polygon_target_geometry == "polygon" else sly.Bitmap
        )

        existing_class = updated_meta.get_obj_class(class_spec.name)
        if existing_class is not None:
            if existing_class.geometry_type != desired_geometry_type:
                raise RuntimeError(
                    "Existing class '{name}' has geometry '{existing}', expected '{expected}'. "
                    "Use another project/dataset or align the target geometry mode.".format(
                        name=class_spec.name,
                        existing=existing_class.geometry_type.geometry_name(),
                        expected=desired_geometry_type.geometry_name(),
                    )
                )
            continue
        updated_meta = updated_meta.add_obj_class(
            sly.ObjClass(class_spec.name, desired_geometry_type, color=class_spec.color)
        )
    return updated_meta


def _parse_raw_specs(raw_specs: List[Any]) -> List[OSMClassSpec]:
    class_specs = []
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("Every class specification must be a JSON object.")

        name = str(raw_spec.get("name", "")).strip()
        geometry = str(raw_spec.get("geometry", "")).strip().lower()
        tags = raw_spec.get("tags")
        default_tag = raw_spec.get("default_tag")
        buffer_m = float(raw_spec.get("buffer_m", 0.0))
        color = _parse_color(raw_spec.get("color"))

        if name == "":
            raise ValueError("Every class specification must have a non-empty 'name'.")
        if geometry not in SUPPORTED_GEOMETRIES:
            raise ValueError(
                "Unsupported geometry '{geometry}' for class '{name}'.".format(
                    geometry=geometry,
                    name=name,
                )
            )
        if not isinstance(tags, dict) or len(tags) == 0:
            raise ValueError(
                "Class '{name}' must define a non-empty 'tags' object.".format(name=name)
            )

        resolved_default_tag = _parse_default_tag(name, default_tag, tags)

        class_specs.append(
            OSMClassSpec(
                name=name,
                geometry=geometry,
                tags=dict(tags),
                default_tag=resolved_default_tag,
                buffer_m=buffer_m,
                color=color,
            )
        )

    return class_specs


def _parse_color(raw_color: Any) -> Optional[List[int]]:
    if raw_color is None:
        return None

    if not isinstance(raw_color, list) or len(raw_color) != 3:
        raise ValueError("'color' must be a list with three RGB values.")

    color = []
    for value in raw_color:
        channel = int(value)
        if channel < 0 or channel > 255:
            raise ValueError("RGB values must be within [0, 255].")
        color.append(channel)
    return color


def _parse_default_tag(name: str, default_tag: Any, tags: Dict[str, Any]) -> Dict[str, Any]:
    if default_tag is None:
        fallback = normalize_osm_tags(name, tags)
        if not fallback:
            raise ValueError(
                "Class '{name}' does not define an exportable 'default_tag'.".format(name=name)
            )
        first_key = sorted(fallback.keys())[0]
        return {first_key: fallback[first_key]}

    if not isinstance(default_tag, dict) or len(default_tag) != 1:
        raise ValueError(
            "Class '{name}' must define 'default_tag' as a single-entry JSON object.".format(
                name=name
            )
        )

    return dict(default_tag)
