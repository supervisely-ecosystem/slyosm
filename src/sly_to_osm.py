from __future__ import annotations

import importlib
import json
import os
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervisely as sly
from dotenv import load_dotenv
from pyproj import CRS, Transformer

try:
    skimage_skeletonize = importlib.import_module("skimage.morphology").skeletonize
except Exception:
    skimage_skeletonize = None

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "osm_classes.json"

# Fill these values directly here.
TEAM_ID = 144
WORKSPACE_ID = 270
PROJECT_ID = 6621
IMAGE_ID = 6122656
OUTPUT_PATH = ""
PROGRESS_EVERY_LABELS = 25
PROGRESS_EVERY_SECONDS = 10.0
LINE_MASK_TARGET_MAX_DIM = 2200
POLYGON_SIMPLIFY_EPSILON_PX = 1.25
LINE_MIN_EXPORT_LENGTH_PX = 4.0
LINE_MIN_COMPONENT_PIXELS = 8
LINE_MIN_MASK_COMPONENT_AREA_PX = 24
LINE_MERGE_GAP_PX = 22.0
LINE_MERGE_LATERAL_TOL_PX = 7.0
LINE_MERGE_ANGLE_DEG = 12.0
LINE_ENDPOINT_SNAP_PX = 16.0
LINE_ENDPOINT_TO_SEGMENT_SNAP_PX = 18.0
LINE_NODE_CLUSTER_TOL_PX = 2.0
LINE_PAIRWISE_MAX_PATHS = 2500
LINE_SPUR_MIN_LENGTH_PX = 10.0
LINE_SPUR_PRUNE_PASSES = 2

OFFSETS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


@dataclass(frozen=True)
class OSMClassSpec:
    name: str
    geometry: str
    tags: dict[str, Any]


@dataclass(frozen=True)
class GeoTransformContext:
    pixel_to_local_h: np.ndarray
    to_wgs84: Transformer
    width: int
    height: int


@dataclass
class OSMNodeRecord:
    node_id: int
    lat: float
    lon: float
    tags: dict[str, str]


@dataclass
class OSMWayRecord:
    way_id: int
    node_refs: list[int]
    tags: dict[str, str]


@dataclass
class OSMRelationRecord:
    relation_id: int
    members: list[tuple[str, int, str]]
    tags: dict[str, str]


class OSMBuilder:
    def __init__(self) -> None:
        self._next_node_id = -1
        self._next_way_id = -1
        self._next_relation_id = -1
        self.nodes: list[OSMNodeRecord] = []
        self.ways: list[OSMWayRecord] = []
        self.relations: list[OSMRelationRecord] = []

    def _alloc_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id -= 1
        return node_id

    def _alloc_way_id(self) -> int:
        way_id = self._next_way_id
        self._next_way_id -= 1
        return way_id

    def _alloc_relation_id(self) -> int:
        relation_id = self._next_relation_id
        self._next_relation_id -= 1
        return relation_id

    def add_node(
        self, lon: float, lat: float, tags: dict[str, str] | None = None
    ) -> int:
        node_id = self._alloc_node_id()
        self.nodes.append(
            OSMNodeRecord(
                node_id=node_id,
                lat=float(lat),
                lon=float(lon),
                tags=dict(tags or {}),
            )
        )
        return node_id

    def add_way(
        self,
        coords_lon_lat: list[tuple[float, float]],
        tags: dict[str, str] | None,
        closed: bool,
    ) -> int | None:
        cleaned: list[tuple[float, float]] = []
        for lon, lat in coords_lon_lat:
            if not cleaned:
                cleaned.append((float(lon), float(lat)))
                continue
            prev_lon, prev_lat = cleaned[-1]
            if abs(prev_lon - lon) < 1e-12 and abs(prev_lat - lat) < 1e-12:
                continue
            cleaned.append((float(lon), float(lat)))

        if closed and len(cleaned) > 1:
            first_lon, first_lat = cleaned[0]
            last_lon, last_lat = cleaned[-1]
            if abs(first_lon - last_lon) < 1e-12 and abs(first_lat - last_lat) < 1e-12:
                cleaned.pop()

        min_vertices = 3 if closed else 2
        if len(cleaned) < min_vertices:
            return None

        node_ids = [self.add_node(lon=lon, lat=lat) for lon, lat in cleaned]
        if closed:
            node_ids.append(node_ids[0])

        way_id = self._alloc_way_id()
        self.ways.append(
            OSMWayRecord(way_id=way_id, node_refs=node_ids, tags=dict(tags or {}))
        )
        return way_id

    def add_polygon(
        self,
        exterior_lon_lat: list[tuple[float, float]],
        holes_lon_lat: list[list[tuple[float, float]]],
        tags: dict[str, str],
    ) -> None:
        if not holes_lon_lat:
            self.add_way(coords_lon_lat=exterior_lon_lat, tags=tags, closed=True)
            return

        outer_way_id = self.add_way(
            coords_lon_lat=exterior_lon_lat, tags=None, closed=True
        )
        if outer_way_id is None:
            return

        inner_way_ids: list[int] = []
        for hole in holes_lon_lat:
            inner_way_id = self.add_way(coords_lon_lat=hole, tags=None, closed=True)
            if inner_way_id is not None:
                inner_way_ids.append(inner_way_id)

        relation_id = self._alloc_relation_id()
        relation_tags = {"type": "multipolygon"}
        relation_tags.update(tags)

        members: list[tuple[str, int, str]] = [("way", outer_way_id, "outer")]
        members.extend(("way", inner_id, "inner") for inner_id in inner_way_ids)

        self.relations.append(
            OSMRelationRecord(
                relation_id=relation_id,
                members=members,
                tags=relation_tags,
            )
        )

    def add_line(
        self, coords_lon_lat: list[tuple[float, float]], tags: dict[str, str]
    ) -> None:
        closed = False
        if len(coords_lon_lat) >= 3:
            first_lon, first_lat = coords_lon_lat[0]
            last_lon, last_lat = coords_lon_lat[-1]
            closed = (
                abs(first_lon - last_lon) < 1e-12 and abs(first_lat - last_lat) < 1e-12
            )

        self.add_way(coords_lon_lat=coords_lon_lat, tags=tags, closed=closed)

    def write(
        self,
        output_path: Path,
        bounds: tuple[float, float, float, float] | None,
    ) -> None:
        root = ET.Element(
            "osm", attrib={"version": "0.6", "generator": "maps4fssly-sly_to_osm"}
        )

        if bounds is not None:
            min_lon, min_lat, max_lon, max_lat = bounds
            ET.SubElement(
                root,
                "bounds",
                attrib={
                    "minlat": f"{min_lat:.12f}",
                    "minlon": f"{min_lon:.12f}",
                    "maxlat": f"{max_lat:.12f}",
                    "maxlon": f"{max_lon:.12f}",
                },
            )

        for node in self.nodes:
            node_el = ET.SubElement(
                root,
                "node",
                attrib={
                    "id": str(node.node_id),
                    "action": "modify",
                    "visible": "true",
                    "lat": f"{node.lat:.12f}",
                    "lon": f"{node.lon:.12f}",
                },
            )
            for key, value in sorted(node.tags.items()):
                ET.SubElement(node_el, "tag", attrib={"k": key, "v": value})

        for way in self.ways:
            way_el = ET.SubElement(
                root,
                "way",
                attrib={"id": str(way.way_id), "action": "modify", "visible": "true"},
            )
            for ref in way.node_refs:
                ET.SubElement(way_el, "nd", attrib={"ref": str(ref)})
            for key, value in sorted(way.tags.items()):
                ET.SubElement(way_el, "tag", attrib={"k": key, "v": value})

        for relation in self.relations:
            relation_el = ET.SubElement(
                root,
                "relation",
                attrib={
                    "id": str(relation.relation_id),
                    "action": "modify",
                    "visible": "true",
                },
            )
            for member_type, member_ref, role in relation.members:
                ET.SubElement(
                    relation_el,
                    "member",
                    attrib={"type": member_type, "ref": str(member_ref), "role": role},
                )
            for key, value in sorted(relation.tags.items()):
                ET.SubElement(relation_el, "tag", attrib={"k": key, "v": value})

        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)


def load_environment() -> None:
    load_dotenv(os.path.expanduser("~/supervisely.env"))
    load_dotenv(BASE_DIR / "local.env")


def load_specs_from_json(config_path: Path) -> dict[str, OSMClassSpec]:
    with config_path.open("r", encoding="utf-8") as file:
        raw_specs = json.load(file)

    specs: dict[str, OSMClassSpec] = {}
    for raw in raw_specs:
        spec = OSMClassSpec(
            name=raw["name"],
            geometry=raw["geometry"],
            tags=raw["tags"],
        )
        specs[spec.name] = spec
    return specs


def load_specs_from_metadata(image_geo: dict[str, Any]) -> dict[str, OSMClassSpec]:
    raw_specs = image_geo.get("osm_class_specs")
    if not isinstance(raw_specs, list):
        return {}

    specs: dict[str, OSMClassSpec] = {}
    for raw in raw_specs:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        geometry = raw.get("geometry")
        tags = raw.get("tags")
        if (
            not isinstance(name, str)
            or not isinstance(geometry, str)
            or not isinstance(tags, dict)
        ):
            continue
        specs[name] = OSMClassSpec(name=name, geometry=geometry, tags=tags)
    return specs


def normalize_osm_tags(class_name: str, tags: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
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


def validate_required_geo_metadata(image_geo: dict[str, Any]) -> None:
    required = ["pixel_to_local_h", "image_size_px"]
    missing = [key for key in required if key not in image_geo]
    if missing:
        missing_keys = ", ".join(missing)
        raise RuntimeError(
            "Missing required image geo metadata keys: "
            f"{missing_keys}. "
            "Forward upload must include geo metadata with pixel_to_local_h and image_size_px."
        )

    if "local_crs_wkt" not in image_geo and "local_crs_projjson" not in image_geo:
        raise RuntimeError(
            "Missing required local CRS metadata. "
            "Expected local_crs_wkt or local_crs_projjson in image meta.geo."
        )


def make_geo_context(image_geo: dict[str, Any]) -> GeoTransformContext:
    validate_required_geo_metadata(image_geo)

    pixel_to_local_h = np.asarray(image_geo["pixel_to_local_h"], dtype=np.float64)
    if pixel_to_local_h.shape != (3, 3):
        raise RuntimeError("image geo metadata pixel_to_local_h must be a 3x3 matrix")

    if "local_crs_wkt" in image_geo:
        local_crs = CRS.from_wkt(image_geo["local_crs_wkt"])
    else:
        local_crs = CRS.from_json_dict(image_geo["local_crs_projjson"])
    to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    image_size = image_geo["image_size_px"]
    width = int(image_size["width"])
    height = int(image_size["height"])

    return GeoTransformContext(
        pixel_to_local_h=pixel_to_local_h,
        to_wgs84=to_wgs84,
        width=width,
        height=height,
    )


def apply_homography(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points_h = np.hstack(
        [points_xy.astype(np.float64), np.ones((points_xy.shape[0], 1))]
    )
    projected = (homography @ points_h.T).T
    denom = projected[:, 2:3]
    denom[np.abs(denom) < 1e-12] = 1e-12
    return projected[:, :2] / denom


def project_xy_points_to_lon_lat(
    points_xy: np.ndarray,
    geo_context: GeoTransformContext,
) -> list[tuple[float, float]]:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return []

    cols = points_xy[:, 0].astype(np.float64)
    rows = points_xy[:, 1].astype(np.float64)
    pixel_xy = np.column_stack([cols, rows])

    local_xy = apply_homography(pixel_xy, geo_context.pixel_to_local_h)
    lon_lat = [
        tuple(map(float, geo_context.to_wgs84.transform(x, y))) for x, y in local_xy
    ]
    return lon_lat


def ring_points_from_polygon(
    geometry: sly.Polygon,
) -> tuple[np.ndarray, list[np.ndarray]]:
    points = geometry.to_json().get("points", {})
    exterior = np.asarray(points.get("exterior", []), dtype=np.float64)
    interiors = [
        np.asarray(ring, dtype=np.float64) for ring in points.get("interior", [])
    ]
    return exterior, interiors


def clean_ring_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)

    cleaned: list[tuple[float, float]] = []
    for col, row in points_xy:
        point = (float(col), float(row))
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)

    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()

    if len(cleaned) < 3:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(cleaned, dtype=np.float64)


def simplify_ring_xy(points_xy: np.ndarray, epsilon_px: float) -> np.ndarray:
    if points_xy.shape[0] < 3 or epsilon_px <= 0:
        return points_xy

    approx_xy = cv2.approxPolyDP(
        points_xy.astype(np.float32).reshape(-1, 1, 2),
        float(epsilon_px),
        True,
    ).reshape(-1, 2)
    if approx_xy.shape[0] < 3:
        return points_xy
    return approx_xy.astype(np.float64)


def clean_line_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return np.empty((0, 2), dtype=np.float64)

    cleaned: list[tuple[float, float]] = []
    for col, row in points_xy:
        point = (float(col), float(row))
        if not cleaned or cleaned[-1] != point:
            cleaned.append(point)

    if len(cleaned) < 2:
        return np.empty((0, 2), dtype=np.float64)

    return np.asarray(cleaned, dtype=np.float64)


def bitmap_centroid_xy(bitmap: sly.Bitmap) -> np.ndarray | None:
    mask = bitmap.data.astype(np.uint8)
    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return None

    row = float(rows.mean() + bitmap.origin.row)
    col = float(cols.mean() + bitmap.origin.col)
    return np.asarray([col, row], dtype=np.float64)


def skeletonize_mask(mask_bool: np.ndarray, log_prefix: str = "") -> np.ndarray:
    if skimage_skeletonize is not None:
        started_at = time.monotonic()
        skeleton = skimage_skeletonize(mask_bool.astype(bool))
        elapsed_s = time.monotonic() - started_at
        if elapsed_s >= 1.0:
            print(
                f"{log_prefix}[skeleton] method=skimage time={elapsed_s:.1f}s",
                flush=True,
            )
        return skeleton.astype(bool)

    mask_u8 = (mask_bool > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(mask_u8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iterations = max(64, int(mask_u8.shape[0] + mask_u8.shape[1] + 64))

    working = mask_u8
    started_at = time.monotonic()
    last_log_at = started_at
    iteration = 0
    while True:
        iteration += 1
        previous = working
        eroded = cv2.erode(
            previous,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        temp = cv2.dilate(
            eroded,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        temp = cv2.subtract(working, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        working = eroded

        remaining = cv2.countNonZero(working)
        now = time.monotonic()
        if now - last_log_at >= PROGRESS_EVERY_SECONDS:
            print(
                f"{log_prefix}[skeleton] iter={iteration} remaining_px={remaining}",
                flush=True,
            )
            last_log_at = now

        if np.array_equal(eroded, previous):
            print(
                f"{log_prefix}[skeleton] stagnated at iter={iteration} remaining_px={remaining}; stopping early",
                flush=True,
            )
            break

        if iteration >= max_iterations:
            print(
                f"{log_prefix}[skeleton] hit iteration limit ({max_iterations}) remaining_px={remaining}; stopping early",
                flush=True,
            )
            break

        if remaining == 0:
            break

    elapsed_s = time.monotonic() - started_at
    if elapsed_s >= 1.0:
        print(
            f"{log_prefix}[skeleton] done in {elapsed_s:.1f}s iters={iteration}",
            flush=True,
        )

    return skeleton > 0


def remove_small_components(mask_bool: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 1:
        return mask_bool.astype(bool)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_bool.astype(np.uint8),
        connectivity=8,
    )
    if labels_count <= 1:
        return mask_bool.astype(bool)

    cleaned = np.zeros_like(mask_bool, dtype=bool)
    for label_index in range(1, labels_count):
        area = int(stats[label_index, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned[labels == label_index] = True
    return cleaned


def preprocess_line_mask(mask_bool: np.ndarray, scale: int) -> np.ndarray:
    close_kernel = 5 if scale > 1 else 3
    open_kernel = 3 if scale > 1 else 2

    work = cv2.morphologyEx(
        mask_bool.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((close_kernel, close_kernel), dtype=np.uint8),
        iterations=1,
    )
    work = cv2.morphologyEx(
        work,
        cv2.MORPH_OPEN,
        np.ones((open_kernel, open_kernel), dtype=np.uint8),
        iterations=1,
    )
    min_area = max(4, int(LINE_MIN_MASK_COMPONENT_AREA_PX / max(1, scale * scale)))
    return remove_small_components(work > 0, min_area_px=min_area)


def build_neighbors(
    pixels: set[tuple[int, int]],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row, col in pixels:
        pixel_neighbors: list[tuple[int, int]] = []
        for d_row, d_col in OFFSETS_8:
            candidate = (row + d_row, col + d_col)
            if candidate in pixels:
                pixel_neighbors.append(candidate)
        neighbors[(row, col)] = pixel_neighbors
    return neighbors


def split_connected_components(
    pixels: set[tuple[int, int]],
) -> list[set[tuple[int, int]]]:
    if not pixels:
        return []

    remaining = set(pixels)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        start = next(iter(remaining))
        queue: deque[tuple[int, int]] = deque([start])
        remaining.remove(start)
        component = {start}

        while queue:
            row, col = queue.popleft()
            for d_row, d_col in OFFSETS_8:
                candidate = (row + d_row, col + d_col)
                if candidate in remaining:
                    remaining.remove(candidate)
                    component.add(candidate)
                    queue.append(candidate)

        components.append(component)

    return components


def undirected_edge_key(
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def cluster_endpoint_indexes(
    endpoint_points: list[np.ndarray],
    tolerance_px: float,
) -> list[list[int]]:
    if len(endpoint_points) < 2 or tolerance_px <= 0:
        return [[index] for index in range(len(endpoint_points))]

    parent = list(range(len(endpoint_points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    cell_size = max(float(tolerance_px), 1e-6)
    grid: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(endpoint_points):
        cell = (
            int(np.floor(float(point[0]) / cell_size)),
            int(np.floor(float(point[1]) / cell_size)),
        )
        grid.setdefault(cell, []).append(index)

    for (cell_x, cell_y), indexes in grid.items():
        candidates: list[int] = list(indexes)
        for d_x in (-1, 0, 1):
            for d_y in (-1, 0, 1):
                neighbor_cell = (cell_x + d_x, cell_y + d_y)
                if neighbor_cell == (cell_x, cell_y):
                    continue
                candidates.extend(grid.get(neighbor_cell, []))

        unique_candidates = sorted(set(candidates))
        for i in indexes:
            for j in unique_candidates:
                if j <= i:
                    continue
                if (
                    np.linalg.norm(endpoint_points[i] - endpoint_points[j])
                    <= tolerance_px
                ):
                    union(i, j)

    clusters: dict[int, list[int]] = {}
    for index in range(len(endpoint_points)):
        root = find(index)
        clusters.setdefault(root, []).append(index)
    return list(clusters.values())


def collapse_key_node_regions(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], list[np.ndarray]]:
    key_pixels = {pixel for pixel, ns in neighbors.items() if len(ns) != 2}
    if not key_pixels:
        return {}, []

    key_components = split_connected_components(key_pixels)
    pixel_to_node: dict[tuple[int, int], int] = {}
    node_centers_rc: list[np.ndarray] = []

    for node_id, component in enumerate(key_components):
        points_rc = np.asarray(list(component), dtype=np.float64)
        center_rc = np.asarray(
            [float(points_rc[:, 0].mean()), float(points_rc[:, 1].mean())],
            dtype=np.float64,
        )
        node_centers_rc.append(center_rc)
        for pixel in component:
            pixel_to_node[pixel] = node_id

    return pixel_to_node, node_centers_rc


def trace_path_between_nodes(
    start_pixel: tuple[int, int],
    next_pixel: tuple[int, int],
    start_node_id: int,
    pixel_to_node: dict[tuple[int, int], int],
    node_centers_rc: list[np.ndarray],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray | None:
    path_points: list[np.ndarray] = [node_centers_rc[start_node_id].copy()]
    previous = start_pixel
    current = next_pixel

    while True:
        current_node_id = pixel_to_node.get(current)
        if current_node_id is not None:
            path_points.append(node_centers_rc[current_node_id].copy())
            break

        path_points.append(
            np.asarray([float(current[0]), float(current[1])], dtype=np.float64)
        )
        candidates = [
            candidate for candidate in neighbors[current] if candidate != previous
        ]
        if not candidates:
            break

        next_candidate: tuple[int, int] | None = None
        for candidate in candidates:
            edge_key = undirected_edge_key(current, candidate)
            if edge_key not in visited_edges:
                next_candidate = candidate
                break
        if next_candidate is None:
            break

        visited_edges.add(undirected_edge_key(current, next_candidate))
        previous = current
        current = next_candidate

    path_rc = clean_line_xy(np.asarray(path_points, dtype=np.float64))
    if path_rc.shape[0] < 2:
        return None
    return path_rc


def trace_pure_loop_component(
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> list[np.ndarray]:
    if not neighbors:
        return []

    start = next(iter(neighbors))
    previous: tuple[int, int] | None = None
    current = start
    path_points = [np.asarray([float(start[0]), float(start[1])], dtype=np.float64)]
    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()

    while True:
        candidates = neighbors[current]
        if previous is not None:
            candidates = [
                candidate for candidate in candidates if candidate != previous
            ]
        if not candidates:
            break

        next_pixel = candidates[0]
        edge_key = undirected_edge_key(current, next_pixel)
        if edge_key in visited_edges:
            break
        visited_edges.add(edge_key)

        previous, current = current, next_pixel
        path_points.append(
            np.asarray([float(current[0]), float(current[1])], dtype=np.float64)
        )
        if current == start:
            break

    path_rc = clean_line_xy(np.asarray(path_points, dtype=np.float64))
    if path_rc.shape[0] >= 3:
        if np.linalg.norm(path_rc[0] - path_rc[-1]) > 1e-9:
            path_rc = np.vstack([path_rc, path_rc[0]])
        return [path_rc]
    return []


def prune_short_spurs_rc(
    paths_rc: list[np.ndarray], min_length_px: float
) -> list[np.ndarray]:
    paths = [clean_line_xy(path) for path in paths_rc if path.shape[0] >= 2]
    if not paths:
        return []

    for _ in range(LINE_SPUR_PRUNE_PASSES):
        endpoint_degree: dict[tuple[int, int], int] = {}
        endpoint_refs: list[tuple[tuple[int, int], tuple[int, int]]] = []

        for path in paths:
            start_key = (
                int(round(float(path[0, 0]) * 2.0)),
                int(round(float(path[0, 1]) * 2.0)),
            )
            end_key = (
                int(round(float(path[-1, 0]) * 2.0)),
                int(round(float(path[-1, 1]) * 2.0)),
            )
            endpoint_degree[start_key] = endpoint_degree.get(start_key, 0) + 1
            endpoint_degree[end_key] = endpoint_degree.get(end_key, 0) + 1
            endpoint_refs.append((start_key, end_key))

        filtered: list[np.ndarray] = []
        removed_any = False
        for path, (start_key, end_key) in zip(paths, endpoint_refs):
            length = polyline_length_px(path)
            is_short = length < min_length_px
            touches_leaf = (
                endpoint_degree.get(start_key, 0) <= 1
                or endpoint_degree.get(end_key, 0) <= 1
            )
            if is_short and touches_leaf:
                removed_any = True
                continue
            filtered.append(path)

        paths = filtered
        if not removed_any:
            break

    return paths


def extract_main_paths_from_component(
    component_pixels: set[tuple[int, int]],
) -> list[np.ndarray]:
    if len(component_pixels) < LINE_MIN_COMPONENT_PIXELS:
        return []

    neighbors = build_neighbors(component_pixels)
    if not neighbors:
        return []

    pixel_to_node, node_centers_rc = collapse_key_node_regions(neighbors)
    if not pixel_to_node:
        return trace_pure_loop_component(neighbors)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    result: list[np.ndarray] = []

    for start_pixel, start_node_id in pixel_to_node.items():
        for next_pixel in neighbors[start_pixel]:
            edge_key = undirected_edge_key(start_pixel, next_pixel)
            if edge_key in visited_edges:
                continue
            visited_edges.add(edge_key)

            path_rc = trace_path_between_nodes(
                start_pixel=start_pixel,
                next_pixel=next_pixel,
                start_node_id=start_node_id,
                pixel_to_node=pixel_to_node,
                node_centers_rc=node_centers_rc,
                neighbors=neighbors,
                visited_edges=visited_edges,
            )
            if path_rc is not None:
                result.append(path_rc)

    return prune_short_spurs_rc(result, min_length_px=LINE_SPUR_MIN_LENGTH_PX)


def simplify_polyline_rc(points_rc: np.ndarray, epsilon_px: float = 0.8) -> np.ndarray:
    if points_rc.shape[0] <= 2:
        return points_rc

    is_closed_loop = bool(
        points_rc.shape[0] >= 4 and np.linalg.norm(points_rc[0] - points_rc[-1]) <= 1e-9
    )

    points_xy = np.column_stack([points_rc[:, 1], points_rc[:, 0]]).astype(np.float32)
    approx_xy = cv2.approxPolyDP(
        points_xy.reshape(-1, 1, 2), epsilon_px, is_closed_loop
    ).reshape(-1, 2)
    if approx_xy.shape[0] < 2:
        return np.vstack([points_rc[0], points_rc[-1]])

    approx_rc = np.column_stack([approx_xy[:, 1], approx_xy[:, 0]]).astype(np.float64)
    if approx_rc.shape[0] >= 2:
        approx_rc[0] = points_rc[0]
        approx_rc[-1] = points_rc[-1]
    return approx_rc


def simplify_polyline_xy(points_xy: np.ndarray, epsilon_px: float = 1.0) -> np.ndarray:
    if points_xy.shape[0] <= 2:
        return points_xy

    is_closed_loop = bool(
        points_xy.shape[0] >= 4 and np.linalg.norm(points_xy[0] - points_xy[-1]) <= 1e-9
    )

    approx_xy = cv2.approxPolyDP(
        points_xy.astype(np.float32).reshape(-1, 1, 2),
        float(epsilon_px),
        is_closed_loop,
    ).reshape(-1, 2)
    if approx_xy.shape[0] < 2:
        return np.vstack([points_xy[0], points_xy[-1]])

    simplified = approx_xy.astype(np.float64)
    if simplified.shape[0] >= 2:
        simplified[0] = points_xy[0]
        simplified[-1] = points_xy[-1]
    return simplified


def point_to_segment_distance_px(
    point_xy: np.ndarray,
    seg_start_xy: np.ndarray,
    seg_end_xy: np.ndarray,
) -> float:
    segment = seg_end_xy - seg_start_xy
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-12:
        return float(np.linalg.norm(point_xy - seg_start_xy))

    t = float(np.dot(point_xy - seg_start_xy, segment) / seg_len_sq)
    t = min(1.0, max(0.0, t))
    projection = seg_start_xy + t * segment
    return float(np.linalg.norm(point_xy - projection))


def path_direction_xy(points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.shape[0] < 2:
        return None
    direction = points_xy[-1] - points_xy[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    return direction / norm


def straighten_nearly_linear_xy(points_xy: np.ndarray, scale: int) -> np.ndarray:
    if points_xy.shape[0] < 3:
        return points_xy

    length = polyline_length_px(points_xy)
    if length < 60.0:
        return points_xy

    start = points_xy[0]
    end = points_xy[-1]
    chord = float(np.linalg.norm(end - start))
    if chord <= 1e-9:
        return points_xy

    if chord / (length + 1e-9) < 0.988:
        return points_xy

    max_dev = max(
        point_to_segment_distance_px(point, start, end) for point in points_xy
    )
    tolerance = max(1.2, float(scale) * 0.5)
    if max_dev > tolerance:
        return points_xy

    return np.vstack([start, end]).astype(np.float64)


def projection_interval_overlap_ratio(
    a_points_xy: np.ndarray,
    b_points_xy: np.ndarray,
    axis_xy: np.ndarray,
) -> float:
    a_proj = np.dot(a_points_xy, axis_xy)
    b_proj = np.dot(b_points_xy, axis_xy)

    a_min = float(np.min(a_proj))
    a_max = float(np.max(a_proj))
    b_min = float(np.min(b_proj))
    b_max = float(np.max(b_proj))

    overlap = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    shorter = max(1e-9, min(a_max - a_min, b_max - b_min))
    return float(overlap / shorter)


def merge_collinear_paths_xy(
    paths_xy: list[np.ndarray],
    gap_px: float,
    lateral_tol_px: float,
    angle_deg: float,
) -> list[np.ndarray]:
    if len(paths_xy) < 2:
        return paths_xy

    merged = [path.copy() for path in paths_xy]
    cos_tol = float(np.cos(np.deg2rad(angle_deg)))
    changed = True

    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                a = merged[i]
                b = merged[j]
                if a.shape[0] < 2 or b.shape[0] < 2:
                    j += 1
                    continue

                dir_a = path_direction_xy(a)
                dir_b = path_direction_xy(b)
                if dir_a is None or dir_b is None:
                    j += 1
                    continue

                if abs(float(np.dot(dir_a, dir_b))) < cos_tol:
                    j += 1
                    continue

                a_endpoints = np.vstack([a[0], a[-1]])
                b_endpoints = np.vstack([b[0], b[-1]])
                endpoint_dist = np.linalg.norm(
                    a_endpoints[:, None, :] - b_endpoints[None, :, :], axis=2
                )
                min_gap = float(np.min(endpoint_dist))

                lateral_a = max(
                    point_to_segment_distance_px(point, a[0], a[-1])
                    for point in b_endpoints
                )
                lateral_b = max(
                    point_to_segment_distance_px(point, b[0], b[-1])
                    for point in a_endpoints
                )
                lateral = min(lateral_a, lateral_b)

                axis = (
                    dir_a if polyline_length_px(a) >= polyline_length_px(b) else dir_b
                )
                overlap_ratio = projection_interval_overlap_ratio(a, b, axis)

                can_merge = (min_gap <= gap_px and lateral <= lateral_tol_px) or (
                    overlap_ratio >= 0.65 and lateral <= lateral_tol_px
                )
                if not can_merge:
                    j += 1
                    continue

                candidates = np.vstack([a_endpoints, b_endpoints])
                pairwise = np.linalg.norm(
                    candidates[:, None, :] - candidates[None, :, :], axis=2
                )
                first_idx, second_idx = np.unravel_index(
                    int(np.argmax(pairwise)), pairwise.shape
                )
                merged[i] = np.vstack(
                    [candidates[first_idx], candidates[second_idx]]
                ).astype(np.float64)
                del merged[j]
                changed = True

            i += 1

    return merged


def snap_path_endpoints_xy(
    paths_xy: list[np.ndarray],
    snap_px: float,
) -> list[np.ndarray]:
    if len(paths_xy) < 2:
        return paths_xy

    paths = [path.copy() for path in paths_xy]
    endpoint_refs: list[tuple[int, int, np.ndarray]] = []
    for path_index, path in enumerate(paths):
        if path.shape[0] < 2:
            continue
        endpoint_refs.append((path_index, 0, path[0].copy()))
        endpoint_refs.append((path_index, -1, path[-1].copy()))

    count = len(endpoint_refs)
    if count < 2:
        return paths

    endpoint_points = [endpoint_refs[i][2] for i in range(count)]
    clusters = cluster_endpoint_indexes(endpoint_points, tolerance_px=snap_px)

    for members in clusters:
        if len(members) < 2:
            continue
        center = np.mean([endpoint_refs[idx][2] for idx in members], axis=0)
        center = center.astype(np.float64)
        for idx in members:
            path_index, endpoint_index, _ = endpoint_refs[idx]
            if endpoint_index == 0:
                paths[path_index][0] = center
            else:
                paths[path_index][-1] = center

    return [clean_line_xy(path) for path in paths]


def closest_point_on_segment_xy(
    point_xy: np.ndarray,
    seg_start_xy: np.ndarray,
    seg_end_xy: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    segment = seg_end_xy - seg_start_xy
    seg_len_sq = float(np.dot(segment, segment))
    if seg_len_sq <= 1e-12:
        return seg_start_xy.copy(), float(np.linalg.norm(point_xy - seg_start_xy)), 0.0

    t = float(np.dot(point_xy - seg_start_xy, segment) / seg_len_sq)
    t = min(1.0, max(0.0, t))
    projection = seg_start_xy + t * segment
    distance = float(np.linalg.norm(point_xy - projection))
    return projection, distance, t


def snap_endpoints_to_nearby_segments_xy(
    paths_xy: list[np.ndarray],
    snap_px: float,
) -> list[np.ndarray]:
    if len(paths_xy) < 2:
        return paths_xy

    paths = [path.copy() for path in paths_xy]

    for path_index, path in enumerate(paths):
        if path.shape[0] < 2:
            continue

        for endpoint_index in (0, -1):
            endpoint = path[endpoint_index]
            best_dist = float("inf")
            best_projection: np.ndarray | None = None

            for other_index, other in enumerate(paths):
                if other_index == path_index or other.shape[0] < 2:
                    continue
                for seg_index in range(other.shape[0] - 1):
                    projection, dist, t = closest_point_on_segment_xy(
                        endpoint,
                        other[seg_index],
                        other[seg_index + 1],
                    )
                    if t <= 0.03 or t >= 0.97:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_projection = projection

            if best_projection is not None and best_dist <= snap_px:
                path[endpoint_index] = best_projection

    return [clean_line_xy(path) for path in paths]


def stitch_paths_at_degree_two_nodes_xy(
    paths_xy: list[np.ndarray],
    cluster_tol_px: float,
) -> list[np.ndarray]:
    paths = [clean_line_xy(path) for path in paths_xy if path.shape[0] >= 2]
    if len(paths) < 2:
        return paths

    endpoint_points: list[np.ndarray] = []
    endpoint_refs: list[tuple[int, int]] = []
    for path_index, path in enumerate(paths):
        endpoint_refs.append((path_index, 0))
        endpoint_points.append(path[0].copy())
        endpoint_refs.append((path_index, -1))
        endpoint_points.append(path[-1].copy())

    clusters = cluster_endpoint_indexes(endpoint_points, tolerance_px=cluster_tol_px)
    endpoint_to_cluster: dict[int, int] = {}
    cluster_centers: dict[int, np.ndarray] = {}
    for cluster_id, members in enumerate(clusters):
        center = np.mean([endpoint_points[index] for index in members], axis=0).astype(
            np.float64
        )
        cluster_centers[cluster_id] = center
        for endpoint_index in members:
            endpoint_to_cluster[endpoint_index] = cluster_id

    @dataclass
    class PathEdge:
        start_node: int
        end_node: int
        geometry: np.ndarray

    edges: list[PathEdge] = []
    adjacency: dict[int, list[int]] = {}
    for path_index, path in enumerate(paths):
        start_endpoint_idx = 2 * path_index
        end_endpoint_idx = 2 * path_index + 1
        start_node = endpoint_to_cluster[start_endpoint_idx]
        end_node = endpoint_to_cluster[end_endpoint_idx]

        geometry = path.copy()
        geometry[0] = cluster_centers[start_node]
        geometry[-1] = cluster_centers[end_node]
        geometry = clean_line_xy(geometry)
        if geometry.shape[0] < 2:
            continue

        edge_id = len(edges)
        edges.append(
            PathEdge(start_node=start_node, end_node=end_node, geometry=geometry)
        )
        adjacency.setdefault(start_node, []).append(edge_id)
        adjacency.setdefault(end_node, []).append(edge_id)

    if len(edges) < 2:
        return [edge.geometry for edge in edges]

    visited_edges: set[int] = set()

    def edge_other_node(edge_id: int, node_id: int) -> int:
        edge = edges[edge_id]
        return edge.end_node if edge.start_node == node_id else edge.start_node

    def oriented_geometry(edge_id: int, from_node: int) -> np.ndarray:
        edge = edges[edge_id]
        if edge.start_node == from_node:
            return edge.geometry
        return edge.geometry[::-1]

    result: list[np.ndarray] = []
    node_degree = {node_id: len(edge_ids) for node_id, edge_ids in adjacency.items()}
    anchor_nodes = [node_id for node_id, degree in node_degree.items() if degree != 2]

    def walk_path(start_node: int, first_edge_id: int) -> np.ndarray:
        visited_edges.add(first_edge_id)
        geometry = oriented_geometry(first_edge_id, start_node).copy()
        current_node = edge_other_node(first_edge_id, start_node)

        while node_degree.get(current_node, 0) == 2:
            candidate_edges = [
                edge_id
                for edge_id in adjacency.get(current_node, [])
                if edge_id not in visited_edges
            ]
            if not candidate_edges:
                break
            next_edge_id = candidate_edges[0]
            next_geometry = oriented_geometry(next_edge_id, current_node)
            visited_edges.add(next_edge_id)
            geometry = np.vstack([geometry, next_geometry[1:]])
            current_node = edge_other_node(next_edge_id, current_node)

        return clean_line_xy(geometry)

    for node_id in anchor_nodes:
        for edge_id in adjacency.get(node_id, []):
            if edge_id in visited_edges:
                continue
            merged = walk_path(node_id, edge_id)
            if merged.shape[0] >= 2:
                result.append(merged)

    for edge_id in range(len(edges)):
        if edge_id in visited_edges:
            continue
        edge = edges[edge_id]
        merged = walk_path(edge.start_node, edge_id)
        if merged.shape[0] >= 3:
            if np.linalg.norm(merged[0] - merged[-1]) > 1e-9:
                merged = np.vstack([merged, merged[0]])
        if merged.shape[0] >= 2:
            result.append(merged)

    return result


def regularize_final_path_xy(path_xy: np.ndarray) -> np.ndarray:
    path_xy = clean_line_xy(path_xy)
    if path_xy.shape[0] < 2:
        return path_xy

    length = polyline_length_px(path_xy)
    epsilon = min(6.0, max(0.7, length * 0.0045))
    path_xy = simplify_polyline_xy(path_xy, epsilon_px=epsilon)
    path_xy = clean_line_xy(path_xy)
    if path_xy.shape[0] < 2:
        return path_xy

    path_xy = straighten_nearly_linear_xy(path_xy, scale=2)
    path_xy = clean_line_xy(path_xy)
    return path_xy


def polyline_length_px(points_rc: np.ndarray) -> float:
    if points_rc.shape[0] < 2:
        return 0.0
    diffs = np.diff(points_rc, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def line_paths_from_mask_xy(
    mask: np.ndarray,
    origin_row: int,
    origin_col: int,
    log_prefix: str = "",
) -> list[np.ndarray]:
    mask = mask.astype(bool)
    if not np.any(mask):
        return []

    active_pixels = int(np.count_nonzero(mask))
    print(
        f"{log_prefix}[line] mask_shape={mask.shape[0]}x{mask.shape[1]} active_px={active_pixels}",
        flush=True,
    )

    non_zero_rows, non_zero_cols = np.where(mask)
    min_row = int(non_zero_rows.min())
    max_row = int(non_zero_rows.max())
    min_col = int(non_zero_cols.min())
    max_col = int(non_zero_cols.max())

    work_mask = mask[min_row : max_row + 1, min_col : max_col + 1]
    row_offset = int(origin_row) + min_row
    col_offset = int(origin_col) + min_col

    scale = 1
    max_dim = max(work_mask.shape)
    if max_dim > LINE_MASK_TARGET_MAX_DIM:
        scale = int(np.ceil(max_dim / LINE_MASK_TARGET_MAX_DIM))
        target_h = max(1, int(np.ceil(work_mask.shape[0] / scale)))
        target_w = max(1, int(np.ceil(work_mask.shape[1] / scale)))
        resized = cv2.resize(
            work_mask.astype(np.float32),
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        work_mask = resized >= 0.2
        print(
            f"{log_prefix}[line] downscaled mask by {scale}x -> {target_h}x{target_w}",
            flush=True,
        )

    work_mask = preprocess_line_mask(work_mask, scale=scale)
    cleaned_pixels = int(np.count_nonzero(work_mask))
    print(f"{log_prefix}[line] cleaned_active_px={cleaned_pixels}", flush=True)

    skeleton = skeletonize_mask(work_mask, log_prefix=log_prefix)
    rows, cols = np.where(skeleton)
    if rows.size == 0:
        return []

    print(f"{log_prefix}[line] skeleton_px={rows.size}", flush=True)

    pixels = {(int(row), int(col)) for row, col in zip(rows.tolist(), cols.tolist())}

    print(f"{log_prefix}[line] tracing graph nodes={len(pixels)}", flush=True)
    components = split_connected_components(pixels)
    print(f"{log_prefix}[line] components={len(components)}", flush=True)

    raw_paths: list[np.ndarray] = []
    for component in components:
        if len(component) < LINE_MIN_COMPONENT_PIXELS:
            continue
        raw_paths.extend(extract_main_paths_from_component(component))

    print(f"{log_prefix}[line] raw_paths={len(raw_paths)}", flush=True)
    result: list[np.ndarray] = []

    for path_rc in raw_paths:
        # Trace in work-mask coordinates, then map to full image coordinates.
        path_rc = path_rc.astype(np.float64)
        if scale == 1:
            path_rc[:, 0] += row_offset
            path_rc[:, 1] += col_offset
        else:
            path_rc[:, 0] = (path_rc[:, 0] + 0.5) * scale - 0.5 + row_offset
            path_rc[:, 1] = (path_rc[:, 1] + 0.5) * scale - 0.5 + col_offset

        if polyline_length_px(path_rc) < LINE_MIN_EXPORT_LENGTH_PX:
            continue

        epsilon_px = min(
            3.2,
            max(0.45, float(scale) * 0.45, polyline_length_px(path_rc) * 0.0025),
        )
        simplified_rc = simplify_polyline_rc(path_rc, epsilon_px=epsilon_px)
        if polyline_length_px(simplified_rc) < LINE_MIN_EXPORT_LENGTH_PX:
            continue

        points_xy = np.column_stack([simplified_rc[:, 1], simplified_rc[:, 0]])
        points_xy = clean_line_xy(points_xy)
        if points_xy.shape[0] < 2:
            continue

        points_xy = straighten_nearly_linear_xy(points_xy, scale=scale)
        points_xy = clean_line_xy(points_xy)
        if points_xy.shape[0] < 2:
            continue

        result.append(points_xy)

    result = [
        path
        for path in result
        if path.shape[0] >= 2 and polyline_length_px(path) >= LINE_MIN_EXPORT_LENGTH_PX
    ]

    print(f"{log_prefix}[line] output_paths={len(result)}", flush=True)

    return result


def bitmap_to_line_paths_rc(
    bitmap: sly.Bitmap, log_prefix: str = ""
) -> list[np.ndarray]:
    return line_paths_from_mask_xy(
        mask=bitmap.data.astype(bool),
        origin_row=int(bitmap.origin.row),
        origin_col=int(bitmap.origin.col),
        log_prefix=log_prefix,
    )


def resolve_class_specs(
    image_geo: dict[str, Any],
    fallback_config_path: Path,
) -> dict[str, OSMClassSpec]:
    specs = load_specs_from_metadata(image_geo)
    if specs:
        return specs

    print(
        "Warning: image metadata has no osm_class_specs; "
        f"falling back to config file at {fallback_config_path}",
        flush=True,
    )
    return load_specs_from_json(fallback_config_path)


def convert_annotation_to_osm(
    annotation: sly.Annotation,
    class_specs: dict[str, OSMClassSpec],
    geo_context: GeoTransformContext,
) -> tuple[OSMBuilder, dict[str, int]]:
    builder = OSMBuilder()

    stats = {
        "labels_total": len(annotation.labels),
        "labels_skipped_unknown_class": 0,
        "polygon_features": 0,
        "line_features": 0,
        "point_features": 0,
    }

    total_labels = len(annotation.labels)
    if total_labels == 0:
        print("[convert] No labels found in annotation.", flush=True)
        return builder, stats

    started_at = time.monotonic()
    last_log_at = started_at
    print(f"[convert] Processing {total_labels} label(s)...", flush=True)
    pending_line_features: list[tuple[np.ndarray, dict[str, str]]] = []
    pending_line_bitmap_groups: dict[
        tuple[str, tuple[tuple[str, str], ...]],
        dict[str, Any],
    ] = {}

    for index, label in enumerate(annotation.labels, start=1):
        class_name = label.obj_class.name
        spec = class_specs.get(class_name)
        if spec is None:
            stats["labels_skipped_unknown_class"] += 1
        else:
            tags = normalize_osm_tags(class_name=class_name, tags=spec.tags)

            if spec.geometry == "polygon":
                if isinstance(label.geometry, sly.Bitmap):
                    polygons = label.geometry.to_contours()
                elif isinstance(label.geometry, sly.Polygon):
                    polygons = [label.geometry]
                else:
                    polygons = []

                for polygon in polygons:
                    exterior_xy, interiors_xy = ring_points_from_polygon(polygon)
                    exterior_xy = clean_ring_xy(exterior_xy)
                    exterior_xy = simplify_ring_xy(
                        exterior_xy, POLYGON_SIMPLIFY_EPSILON_PX
                    )
                    exterior_xy = clean_ring_xy(exterior_xy)
                    if exterior_xy.shape[0] < 3:
                        continue

                    holes_xy: list[np.ndarray] = []
                    for interior_xy in interiors_xy:
                        cleaned_hole = clean_ring_xy(interior_xy)
                        cleaned_hole = simplify_ring_xy(
                            cleaned_hole, POLYGON_SIMPLIFY_EPSILON_PX
                        )
                        cleaned_hole = clean_ring_xy(cleaned_hole)
                        if cleaned_hole.shape[0] >= 3:
                            holes_xy.append(cleaned_hole)

                    exterior_lon_lat = project_xy_points_to_lon_lat(
                        exterior_xy, geo_context
                    )
                    holes_lon_lat = [
                        project_xy_points_to_lon_lat(hole_xy, geo_context)
                        for hole_xy in holes_xy
                    ]

                    builder.add_polygon(
                        exterior_lon_lat=exterior_lon_lat,
                        holes_lon_lat=holes_lon_lat,
                        tags=tags,
                    )
                    stats["polygon_features"] += 1

            elif spec.geometry == "line":
                if isinstance(label.geometry, sly.Bitmap):
                    mask = label.geometry.data.astype(bool)
                    if np.any(mask):
                        key = (class_name, tuple(sorted(tags.items())))
                        group = pending_line_bitmap_groups.setdefault(
                            key,
                            {
                                "class_name": class_name,
                                "tags": tags,
                                "chunks": [],
                            },
                        )
                        group["chunks"].append(
                            (
                                mask,
                                int(label.geometry.origin.row),
                                int(label.geometry.origin.col),
                            )
                        )
                    continue

                if isinstance(label.geometry, sly.Polyline):
                    points = np.asarray(
                        label.geometry.to_json().get("points", {}).get("exterior", []),
                        dtype=np.float64,
                    )
                    points = clean_line_xy(points)
                    if points.shape[0] >= 2:
                        pending_line_features.append((points, tags))

            elif spec.geometry == "point":
                point_rc_list: list[np.ndarray] = []

                if isinstance(label.geometry, sly.Bitmap):
                    centroid = bitmap_centroid_xy(label.geometry)
                    if centroid is not None:
                        point_rc_list = [centroid]
                elif isinstance(label.geometry, sly.Point):
                    location = (
                        label.geometry.to_json().get("points", {}).get("exterior", [])
                    )
                    if len(location) == 2 and not isinstance(location[0], list):
                        point_rc_list = [np.asarray(location, dtype=np.float64)]
                    elif len(location) == 1 and isinstance(location[0], list):
                        point_rc_list = [np.asarray(location[0], dtype=np.float64)]

                for point_rc in point_rc_list:
                    lon_lat = project_xy_points_to_lon_lat(
                        point_rc.reshape(1, 2), geo_context
                    )[0]
                    lon, lat = lon_lat
                    builder.add_node(lon=lon, lat=lat, tags=tags)
                    stats["point_features"] += 1

            else:
                raise ValueError(
                    f"Unsupported geometry type in class spec: {spec.geometry}"
                )

        now = time.monotonic()
        should_log = (
            index == total_labels
            or index == 1
            or (index % PROGRESS_EVERY_LABELS == 0)
            or (now - last_log_at >= PROGRESS_EVERY_SECONDS)
        )
        if should_log:
            elapsed_s = max(now - started_at, 1e-6)
            rate = index / elapsed_s
            remaining = total_labels - index
            eta_s = remaining / rate if rate > 0 else 0.0
            print(
                "[convert] "
                f"{index}/{total_labels} ({(index / total_labels) * 100:.1f}%) "
                f"elapsed={elapsed_s:.1f}s eta={eta_s:.1f}s "
                f"polygon={stats['polygon_features']} line={stats['line_features']} "
                f"point={stats['point_features']} skipped={stats['labels_skipped_unknown_class']}",
                flush=True,
            )
            last_log_at = now

    if pending_line_bitmap_groups:
        for group in pending_line_bitmap_groups.values():
            chunks: list[tuple[np.ndarray, int, int]] = group["chunks"]
            if not chunks:
                continue

            min_row = min(row for _, row, _ in chunks)
            min_col = min(col for _, _, col in chunks)
            max_row = max(row + chunk.shape[0] for chunk, row, _ in chunks)
            max_col = max(col + chunk.shape[1] for chunk, _, col in chunks)

            merged_mask = np.zeros((max_row - min_row, max_col - min_col), dtype=bool)
            for chunk, row, col in chunks:
                row0 = row - min_row
                col0 = col - min_col
                row1 = row0 + chunk.shape[0]
                col1 = col0 + chunk.shape[1]
                merged_mask[row0:row1, col0:col1] |= chunk

            log_prefix = (
                f"[convert line-merged class={group['class_name']} "
                f"parts={len(chunks)}] "
            )
            merged_paths = line_paths_from_mask_xy(
                mask=merged_mask,
                origin_row=min_row,
                origin_col=min_col,
                log_prefix=log_prefix,
            )
            for path_xy in merged_paths:
                pending_line_features.append((path_xy, group["tags"]))

    if pending_line_features:
        grouped_line_features: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
        for path_xy, tags in pending_line_features:
            key = tuple(sorted(tags.items()))
            entry = grouped_line_features.setdefault(
                key,
                {"tags": tags, "paths": []},
            )
            entry["paths"].append(path_xy)

        raw_line_count = stats["line_features"]
        raw_line_count = len(pending_line_features)
        stats["line_features"] = 0

        for entry in grouped_line_features.values():
            paths = stitch_paths_at_degree_two_nodes_xy(
                entry["paths"],
                cluster_tol_px=LINE_ENDPOINT_SNAP_PX,
            )
            if len(paths) <= LINE_PAIRWISE_MAX_PATHS:
                paths = snap_endpoints_to_nearby_segments_xy(
                    paths,
                    snap_px=LINE_ENDPOINT_TO_SEGMENT_SNAP_PX,
                )
                paths = merge_collinear_paths_xy(
                    paths,
                    gap_px=LINE_MERGE_GAP_PX * 1.25,
                    lateral_tol_px=LINE_MERGE_LATERAL_TOL_PX,
                    angle_deg=LINE_MERGE_ANGLE_DEG,
                )
            else:
                print(
                    "[convert] skipping expensive pairwise line ops for "
                    f"{len(paths)} paths (threshold={LINE_PAIRWISE_MAX_PATHS})",
                    flush=True,
                )
            paths = [regularize_final_path_xy(path) for path in paths]
            paths = stitch_paths_at_degree_two_nodes_xy(
                paths,
                cluster_tol_px=max(1.0, LINE_NODE_CLUSTER_TOL_PX),
            )
            if len(paths) <= LINE_PAIRWISE_MAX_PATHS:
                paths = merge_collinear_paths_xy(
                    paths,
                    gap_px=LINE_MERGE_GAP_PX,
                    lateral_tol_px=LINE_MERGE_LATERAL_TOL_PX,
                    angle_deg=LINE_MERGE_ANGLE_DEG,
                )

            for path_xy in paths:
                if (
                    path_xy.shape[0] < 2
                    or polyline_length_px(path_xy) < LINE_MIN_EXPORT_LENGTH_PX
                ):
                    continue

                lon_lat = project_xy_points_to_lon_lat(path_xy, geo_context)
                builder.add_line(coords_lon_lat=lon_lat, tags=entry["tags"])
                stats["line_features"] += 1

        print(
            "[convert] line postprocess "
            f"raw={raw_line_count} final={stats['line_features']} ",
            flush=True,
        )

    return builder, stats


def resolve_output_path(image_name: str, image_id: int) -> Path:
    if OUTPUT_PATH:
        return Path(OUTPUT_PATH)

    stem = Path(image_name).stem if image_name else f"image_{image_id}"
    return BASE_DIR / "data" / "osm" / f"{stem}.osm"


def main() -> None:
    load_environment()
    print(
        f"[init] Starting reverse export for image_id={IMAGE_ID}, project_id={PROJECT_ID}",
        flush=True,
    )

    if IMAGE_ID <= 0:
        raise RuntimeError("IMAGE_ID must be set at the top of this file")

    api = sly.Api.from_env()

    print("[api] Fetching image info...", flush=True)
    image_info = api.image.get_info_by_id(IMAGE_ID)
    if image_info is None:
        raise RuntimeError(f"Image with id={IMAGE_ID} was not found")

    print("[api] Fetching dataset info...", flush=True)
    dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
    if dataset_info is None:
        raise RuntimeError(f"Dataset with id={image_info.dataset_id} was not found")

    project_id = PROJECT_ID if PROJECT_ID > 0 else int(dataset_info.project_id)
    print(f"[api] Resolving project_id={project_id}...", flush=True)
    project_info = api.project.get_info_by_id(project_id)
    if project_info is None:
        raise RuntimeError(f"Project with id={project_id} was not found")

    if WORKSPACE_ID > 0 and int(project_info.workspace_id) != int(WORKSPACE_ID):
        raise RuntimeError(
            f"workspace_id mismatch: project belongs to workspace_id={project_info.workspace_id}, "
            f"but got workspace_id={WORKSPACE_ID}"
        )

    if TEAM_ID > 0:
        workspace_info = api.workspace.get_info_by_id(project_info.workspace_id)
        if workspace_info is not None and int(workspace_info.team_id) != int(TEAM_ID):
            raise RuntimeError(
                f"team_id mismatch: workspace belongs to team_id={workspace_info.team_id}, "
                f"but got team_id={TEAM_ID}"
            )

    image_meta = image_info.meta if isinstance(image_info.meta, dict) else {}
    image_geo = image_meta.get("geo") if isinstance(image_meta, dict) else None
    if not isinstance(image_geo, dict):
        raise RuntimeError(
            "Image meta has no geo payload. "
            "Forward upload must include metas=[{'geo': ...}] for reverse export."
        )

    print("[meta] Resolving class specs and geo transform context...", flush=True)
    class_specs = resolve_class_specs(
        image_geo=image_geo, fallback_config_path=CONFIG_PATH
    )
    geo_context = make_geo_context(image_geo)

    print("[api] Downloading project meta...", flush=True)
    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
    print("[api] Downloading annotation JSON...", flush=True)
    annotation_json = api.annotation.download_json(IMAGE_ID)
    object_count = len(annotation_json.get("objects", []))
    print(f"[api] Annotation JSON received: objects={object_count}", flush=True)
    print("[parse] Parsing annotation JSON...", flush=True)
    annotation = sly.Annotation.from_json(annotation_json, project_meta)
    print(f"[parse] Parsed labels={len(annotation.labels)}", flush=True)

    print("[convert] Converting annotation to OSM entities...", flush=True)
    builder, stats = convert_annotation_to_osm(
        annotation=annotation,
        class_specs=class_specs,
        geo_context=geo_context,
    )

    bounds = None
    bbox = image_geo.get("bbox_left_bottom_right_top")
    if isinstance(bbox, list) and len(bbox) == 4:
        bounds = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    output_path = resolve_output_path(image_info.name, IMAGE_ID)
    print(f"[write] Writing OSM XML to {output_path}...", flush=True)
    builder.write(output_path=output_path, bounds=bounds)

    print(
        "Export complete: "
        f"nodes={len(builder.nodes)}, ways={len(builder.ways)}, relations={len(builder.relations)}, "
        f"output={output_path}",
        flush=True,
    )
    print(
        "Conversion stats: "
        f"labels_total={stats['labels_total']}, "
        f"labels_skipped_unknown_class={stats['labels_skipped_unknown_class']}, "
        f"polygon_features={stats['polygon_features']}, "
        f"line_features={stats['line_features']}, "
        f"point_features={stats['point_features']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
