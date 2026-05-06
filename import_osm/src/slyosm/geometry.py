from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import supervisely as sly
from pyproj import CRS, Geod, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box


GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True)
class SceneRequest:
    """Parameters needed to download and upload one satellite image.

    :param identifier: Unique scene identifier.
    :type identifier: str
    :param center_lat: Scene center latitude.
    :type center_lat: float
    :param center_lon: Scene center longitude.
    :type center_lon: float
    :param size_m: Tile size in meters.
    :type size_m: int
    :param rotation_deg: Tile rotation in degrees.
    :type rotation_deg: float
    :param imagery_provider: Optional imagery provider code used by pydtmdl.
    :type imagery_provider: Optional[str]
    :param grid_row: Optional grid row index.
    :type grid_row: Optional[int]
    :param grid_col: Optional grid column index.
    :type grid_col: Optional[int]
    :param metadata: Additional metadata merged into image geo payload.
    :type metadata: Dict[str, Any]
    """

    identifier: str
    center_lat: float
    center_lon: float
    size_m: int
    rotation_deg: float
    imagery_provider: Optional[str] = None
    grid_row: Optional[int] = None
    grid_col: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGeoContext:
    """Derived geometry and transforms for one downloaded image."""

    top_left_lat: float
    top_left_lon: float
    corners_lon_lat: List[Tuple[float, float]]
    bbox_left_bottom_right_top: Tuple[float, float, float, float]
    local_crs_wkt: str
    local_crs_projjson: Dict[str, Any]
    scene_polygon_local: Polygon
    to_local: Transformer
    local_to_pixel_h: np.ndarray


def compute_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    """Compute a projective transform from local coordinates to image pixels.

    :param src_xy: Four source points.
    :type src_xy: np.ndarray
    :param dst_uv: Four destination points.
    :type dst_uv: np.ndarray
    :return: Homography matrix.
    :rtype: np.ndarray
    :raises ValueError: If point counts do not match the required shape.
    """

    if src_xy.shape != (4, 2) or dst_uv.shape != (4, 2):
        raise ValueError("Homography requires exactly 4 source and 4 destination points.")

    rows = []
    for (x_coord, y_coord), (u_coord, v_coord) in zip(src_xy, dst_uv):
        rows.append([x_coord, y_coord, 1.0, 0.0, 0.0, 0.0, -u_coord * x_coord, -u_coord * y_coord, -u_coord])
        rows.append([0.0, 0.0, 0.0, x_coord, y_coord, 1.0, -v_coord * x_coord, -v_coord * y_coord, -v_coord])

    matrix = np.asarray(rows, dtype=np.float64)
    _, _, vt_matrix = np.linalg.svd(matrix)
    homography = vt_matrix[-1, :]
    homography = homography / homography[-1]
    return homography.reshape(3, 3)


def project_local_xy_to_pixels(points_xy: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project local XY coordinates to image pixel coordinates.

    :param points_xy: Input local XY coordinates.
    :type points_xy: np.ndarray
    :param homography: Local-to-pixel homography.
    :type homography: np.ndarray
    :return: Pixel coordinates.
    :rtype: np.ndarray
    """

    points_h = np.hstack([points_xy.astype(np.float64), np.ones((points_xy.shape[0], 1))])
    projected = (homography @ points_h.T).T
    denominator = projected[:, 2:3]
    denominator[np.abs(denominator) < 1e-12] = 1e-12
    return projected[:, :2] / denominator


def _scene_corners_from_center(
    center_lat: float,
    center_lon: float,
    size_m: int,
    rotation_deg: float,
) -> List[Tuple[float, float]]:
    """Compute scene corner coordinates in lon-lat order from center and rotation."""

    half_size = float(size_m) / 2.0
    top_center_lat, top_center_lon = move_point(
        center_lat, center_lon, rotation_deg, half_size
    )
    bottom_center_lat, bottom_center_lon = move_point(
        center_lat, center_lon, 180.0 + rotation_deg, half_size
    )

    top_left_lat, top_left_lon = move_point(
        top_center_lat, top_center_lon, 270.0 + rotation_deg, half_size
    )
    top_right_lat, top_right_lon = move_point(
        top_center_lat, top_center_lon, 90.0 + rotation_deg, half_size
    )
    bottom_right_lat, bottom_right_lon = move_point(
        bottom_center_lat, bottom_center_lon, 90.0 + rotation_deg, half_size
    )
    bottom_left_lat, bottom_left_lon = move_point(
        bottom_center_lat, bottom_center_lon, 270.0 + rotation_deg, half_size
    )

    return [
        (top_left_lon, top_left_lat),
        (top_right_lon, top_right_lat),
        (bottom_right_lon, bottom_right_lat),
        (bottom_left_lon, bottom_left_lat),
    ]


def build_scene_geo_context(
    center_lat: float,
    center_lon: float,
    size_m: int,
    rotation_deg: float,
    image_width: int,
    image_height: int,
) -> SceneGeoContext:
    """Build projection context and scene polygon for one downloaded image.

    :param center_lat: Scene center latitude.
    :type center_lat: float
    :param center_lon: Scene center longitude.
    :type center_lon: float
    :param size_m: Tile size in meters.
    :type size_m: int
    :param rotation_deg: Rotation in degrees.
    :type rotation_deg: float
    :param image_width: Image width in pixels.
    :type image_width: int
    :param image_height: Image height in pixels.
    :type image_height: int
    :return: Derived scene geometry context.
    :rtype: SceneGeoContext
    """

    corners_lon_lat = _scene_corners_from_center(
        center_lat=center_lat,
        center_lon=center_lon,
        size_m=int(size_m),
        rotation_deg=float(rotation_deg),
    )
    top_left_lon, top_left_lat = corners_lon_lat[0]

    crs_local = CRS.from_proj4(
        "+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs".format(
            lat=center_lat,
            lon=center_lon,
        )
    )
    to_local = Transformer.from_crs("EPSG:4326", crs_local, always_xy=True)

    local_corners = np.asarray(
        [to_local.transform(lon_value, lat_value) for lon_value, lat_value in corners_lon_lat],
        dtype=np.float64,
    )

    destination_pixels = np.asarray(
        [
            [-0.5, -0.5],
            [float(image_width) - 0.5, -0.5],
            [float(image_width) - 0.5, float(image_height) - 0.5],
            [-0.5, float(image_height) - 0.5],
        ],
        dtype=np.float64,
    )
    local_to_pixel_h = compute_homography(local_corners, destination_pixels)

    lons = [corner[0] for corner in corners_lon_lat]
    lats = [corner[1] for corner in corners_lon_lat]
    bbox = (float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats)))

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


def iter_polygons(geometry: Any) -> List[Polygon]:
    """Extract polygons from a Shapely geometry collection.

    :param geometry: Any Shapely geometry.
    :type geometry: Any
    :return: Non-empty polygons.
    :rtype: List[Polygon]
    """

    if geometry.is_empty:
        return []

    if isinstance(geometry, Polygon):
        return [geometry]

    if isinstance(geometry, MultiPolygon):
        return [polygon for polygon in geometry.geoms if not polygon.is_empty]

    if isinstance(geometry, GeometryCollection):
        polygons = []
        for item in geometry.geoms:
            polygons.extend(iter_polygons(item))
        return polygons

    return []


def geometry_to_polygons(
    geometry_local: Any,
    expected_geometry: str,
    buffer_m: float,
    clip_polygon: Polygon,
) -> List[Polygon]:
    """Convert a local OSM geometry to clipped polygons ready for mask rasterization.

    :param geometry_local: OSM geometry in the local scene CRS.
    :type geometry_local: Any
    :param expected_geometry: Expected geometry type from class configuration.
    :type expected_geometry: str
    :param buffer_m: Buffer size for line and point classes.
    :type buffer_m: float
    :param clip_polygon: Scene polygon used for clipping.
    :type clip_polygon: Polygon
    :return: Polygons inside the scene footprint.
    :rtype: List[Polygon]
    """

    geometry = geometry_local
    geometry_type = geometry.geom_type

    if expected_geometry == "polygon":
        if geometry_type not in {"Polygon", "MultiPolygon"}:
            return []
    elif expected_geometry == "line":
        if geometry_type not in {"LineString", "MultiLineString"}:
            return []
        geometry = geometry.buffer(buffer_m, cap_style=2, join_style=2)
    elif expected_geometry == "point":
        if geometry_type not in {"Point", "MultiPoint"}:
            return []
        geometry = geometry.buffer(buffer_m)
    else:
        raise ValueError("Unsupported geometry type: {geometry}".format(geometry=expected_geometry))

    if geometry.is_empty:
        return []

    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return []

    clipped = geometry.intersection(clip_polygon)
    if clipped.is_empty:
        return []

    return [polygon for polygon in iter_polygons(clipped) if polygon.area > 0]


def ring_to_supervisely_points(ring_uv: np.ndarray, width: int, height: int) -> List[Tuple[int, int]]:
    """Convert polygon ring points to Supervisely row-column tuples.

    :param ring_uv: Polygon ring in image pixel coordinates.
    :type ring_uv: np.ndarray
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: Row-column points.
    :rtype: List[Tuple[int, int]]
    """

    points_rc = []
    for col_value, row_value in ring_uv:
        col_index = int(
            np.clip(np.floor(float(col_value) + 0.5), 0.0, float(width - 1))
        )
        row_index = int(
            np.clip(np.floor(float(row_value) + 0.5), 0.0, float(height - 1))
        )
        point = (row_index, col_index)
        if not points_rc or points_rc[-1] != point:
            points_rc.append(point)

    if len(points_rc) >= 2 and points_rc[0] == points_rc[-1]:
        points_rc.pop()

    if len(set(points_rc)) < 3:
        return []
    return points_rc


def ring_to_supervisely_polygon_points(
    ring_uv: np.ndarray, width: int, height: int
) -> List[Tuple[int, int]]:
    """Convert ring points to edge-inclusive vertices for polygon geometry upload."""

    points_rc = []
    for col_value, row_value in ring_uv:
        # For vector polygons, allow right/bottom boundary vertices at width/height.
        col_index = int(np.clip(np.rint(float(col_value)), 0.0, float(width)))
        row_index = int(np.clip(np.rint(float(row_value)), 0.0, float(height)))
        point = (row_index, col_index)
        if not points_rc or points_rc[-1] != point:
            points_rc.append(point)

    if len(points_rc) >= 2 and points_rc[0] == points_rc[-1]:
        points_rc.pop()

    if len(set(points_rc)) < 3:
        return []
    return points_rc


def project_polygon_to_pixel_geometry(polygon: Polygon, homography: np.ndarray) -> Optional[Any]:
    """Project a polygon from local scene space into image pixel space.

    :param polygon: Local polygon.
    :type polygon: Polygon
    :param homography: Local-to-pixel homography.
    :type homography: np.ndarray
    :return: Projected pixel polygon or ``None``.
    :rtype: Optional[Any]
    """

    exterior_xy = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if exterior_xy.shape[0] < 3:
        return None

    exterior_uv = project_local_xy_to_pixels(exterior_xy, homography)
    interior_uv_list = []
    for interior in polygon.interiors:
        interior_xy = np.asarray(interior.coords[:-1], dtype=np.float64)
        if interior_xy.shape[0] < 3:
            continue
        interior_uv = project_local_xy_to_pixels(interior_xy, homography)
        if interior_uv.shape[0] >= 3:
            interior_uv_list.append(interior_uv.tolist())

    pixel_geometry = Polygon(exterior_uv.tolist(), interior_uv_list)
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
) -> Optional[sly.Bitmap]:
    """Rasterize a polygon to a Supervisely bitmap.

    :param polygon: Polygon to rasterize.
    :type polygon: Polygon
    :param homography: Local-to-pixel homography.
    :type homography: np.ndarray
    :param width: Image width.
    :type width: int
    :param height: Image height.
    :type height: int
    :return: Rasterized bitmap or ``None``.
    :rtype: Optional[sly.Bitmap]
    """

    pixel_geometry = project_polygon_to_pixel_geometry(polygon, homography)
    if pixel_geometry is None:
        return None

    clip_rect = box(-0.5, -0.5, float(width) - 0.5, float(height) - 0.5)
    clipped_geometry = pixel_geometry.intersection(clip_rect)
    if clipped_geometry.is_empty:
        return None

    clipped_polygons = [polygon_item for polygon_item in iter_polygons(clipped_geometry) if polygon_item.area > 0]
    if not clipped_polygons:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    for clipped_polygon in clipped_polygons:
        exterior_uv = np.asarray(clipped_polygon.exterior.coords[:-1], dtype=np.float64)
        exterior_rc = ring_to_supervisely_points(exterior_uv, width, height)
        if len(exterior_rc) < 3:
            continue

        interior_rc_list = []
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


def polygon_to_supervisely_polygons(
    polygon: Polygon,
    homography: np.ndarray,
    width: int,
    height: int,
) -> List[sly.Polygon]:
    """Project and clip a polygon to image bounds and return Supervisely polygon geometries."""

    pixel_geometry = project_polygon_to_pixel_geometry(polygon, homography)
    if pixel_geometry is None:
        return []

    clip_rect = box(-0.5, -0.5, float(width) - 0.5, float(height) - 0.5)
    clipped_geometry = pixel_geometry.intersection(clip_rect)
    if clipped_geometry.is_empty:
        return []

    clipped_polygons = [
        polygon_item
        for polygon_item in iter_polygons(clipped_geometry)
        if polygon_item.area > 0
    ]
    if not clipped_polygons:
        return []

    result = []
    for clipped_polygon in clipped_polygons:
        exterior_uv = np.asarray(clipped_polygon.exterior.coords[:-1], dtype=np.float64)
        exterior_rc = ring_to_supervisely_polygon_points(exterior_uv, width, height)
        if len(exterior_rc) < 3:
            continue

        interior_rc_list = []
        for interior in clipped_polygon.interiors:
            interior_uv = np.asarray(interior.coords[:-1], dtype=np.float64)
            interior_rc = ring_to_supervisely_polygon_points(
                interior_uv, width, height
            )
            if len(interior_rc) >= 3:
                interior_rc_list.append(interior_rc)

        result.append(sly.Polygon(exterior=exterior_rc, interior=interior_rc_list))

    return result


def move_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    """Move a WGS84 coordinate by distance and bearing.

    :param lat: Start latitude.
    :type lat: float
    :param lon: Start longitude.
    :type lon: float
    :param bearing_deg: Bearing in degrees.
    :type bearing_deg: float
    :param distance_m: Distance in meters.
    :type distance_m: float
    :return: Destination latitude and longitude.
    :rtype: Tuple[float, float]
    """

    dst_lon, dst_lat, _ = GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return float(dst_lat), float(dst_lon)


def center_from_top_left(top_left_lat: float, top_left_lon: float, size_m: int, rotation_deg: float) -> Tuple[float, float]:
    """Compute the scene center from its top-left corner.

    :param top_left_lat: Top-left latitude.
    :type top_left_lat: float
    :param top_left_lon: Top-left longitude.
    :type top_left_lon: float
    :param size_m: Tile size in meters.
    :type size_m: int
    :param rotation_deg: Rotation in degrees.
    :type rotation_deg: float
    :return: Center latitude and longitude.
    :rtype: Tuple[float, float]
    """

    half_size = float(size_m) / 2.0
    top_middle_lat, top_middle_lon = move_point(top_left_lat, top_left_lon, 90.0 + rotation_deg, half_size)
    return move_point(top_middle_lat, top_middle_lon, 180.0 + rotation_deg, half_size)


def build_coordinate_scenes(
    coordinates: Sequence[Tuple[float, float]],
    size_m: int,
    rotation_deg: float,
    imagery_provider: Optional[str] = None,
    prefix: str = "scene",
) -> List[SceneRequest]:
    """Build scene requests for explicit center coordinates.

    :param coordinates: Latitude-longitude pairs.
    :type coordinates: Sequence[Tuple[float, float]]
    :param size_m: Tile size in meters.
    :type size_m: int
    :param rotation_deg: Rotation in degrees.
    :type rotation_deg: float
    :param imagery_provider: Optional imagery provider code used by pydtmdl.
    :type imagery_provider: Optional[str]
    :param prefix: Scene identifier prefix.
    :type prefix: str
    :return: Scene requests.
    :rtype: List[SceneRequest]
    """

    scenes = []
    for index, (lat_value, lon_value) in enumerate(coordinates, start=1):
        scenes.append(
            SceneRequest(
                identifier="{prefix}_{index:04d}".format(prefix=prefix, index=index),
                center_lat=float(lat_value),
                center_lon=float(lon_value),
                size_m=int(size_m),
                rotation_deg=float(rotation_deg),
                imagery_provider=imagery_provider,
            )
        )
    return scenes


def build_grid_scenes(
    prefix: str,
    top_left_lat: float,
    top_left_lon: float,
    rows: int,
    cols: int,
    size_m: int,
    rotation_deg: float,
    imagery_provider: Optional[str] = None,
) -> List[SceneRequest]:
    """Build contiguous scene requests for a rotated grid.

    :param prefix: Scene identifier prefix.
    :type prefix: str
    :param top_left_lat: Top-left latitude of the whole grid footprint.
    :type top_left_lat: float
    :param top_left_lon: Top-left longitude of the whole grid footprint.
    :type top_left_lon: float
    :param rows: Number of rows.
    :type rows: int
    :param cols: Number of columns.
    :type cols: int
    :param size_m: Tile size in meters.
    :type size_m: int
    :param rotation_deg: Rotation in degrees.
    :type rotation_deg: float
    :param imagery_provider: Optional imagery provider code used by pydtmdl.
    :type imagery_provider: Optional[str]
    :return: Scene requests.
    :rtype: List[SceneRequest]
    """

    scenes = []
    for row_index in range(rows):
        for col_index in range(cols):
            row_start_lat, row_start_lon = move_point(
                top_left_lat,
                top_left_lon,
                180.0 + rotation_deg,
                float(row_index * size_m),
            )
            cell_top_left_lat, cell_top_left_lon = move_point(
                row_start_lat,
                row_start_lon,
                90.0 + rotation_deg,
                float(col_index * size_m),
            )
            center_lat, center_lon = center_from_top_left(
                cell_top_left_lat,
                cell_top_left_lon,
                size_m,
                rotation_deg,
            )
            scenes.append(
                SceneRequest(
                    identifier="{prefix}_r{row:03d}_c{col:03d}".format(
                        prefix=prefix,
                        row=row_index,
                        col=col_index,
                    ),
                    center_lat=center_lat,
                    center_lon=center_lon,
                    size_m=int(size_m),
                    rotation_deg=float(rotation_deg),
                    imagery_provider=imagery_provider,
                    grid_row=row_index,
                    grid_col=col_index,
                    metadata={
                        "grid": {
                            "row": row_index,
                            "col": col_index,
                            "rows": int(rows),
                            "cols": int(cols),
                            "global_top_left": {
                                "lat": float(top_left_lat),
                                "lon": float(top_left_lon),
                            },
                        }
                    },
                )
            )
    return scenes


def scene_from_dict(raw_scene: Dict[str, Any]) -> SceneRequest:
    """Convert a raw dictionary scene declaration to :class:`SceneRequest`.

    :param raw_scene: Raw scene dictionary.
    :type raw_scene: Dict[str, Any]
    :return: Scene request.
    :rtype: SceneRequest
    """

    metadata = raw_scene.get("metadata") if isinstance(raw_scene.get("metadata"), dict) else {}
    return SceneRequest(
        identifier=str(raw_scene["id"]),
        center_lat=float(raw_scene["center_lat"]),
        center_lon=float(raw_scene["center_lon"]),
        size_m=int(raw_scene["size_m"]),
        rotation_deg=float(raw_scene["rotation_deg"]),
        imagery_provider=(
            str(raw_scene.get("imagery_provider")).strip() or None
            if raw_scene.get("imagery_provider") is not None
            else None
        ),
        grid_row=_optional_int(raw_scene.get("grid_row")),
        grid_col=_optional_int(raw_scene.get("grid_col")),
        metadata=metadata,
    )


def scene_metadata_payload(
    scene: SceneRequest,
    scene_geo: SceneGeoContext,
    image_width: int,
    image_height: int,
) -> Dict[str, Any]:
    """Create geospatial metadata attached to each uploaded image.

    :param scene: Scene request.
    :type scene: SceneRequest
    :param scene_geo: Derived geometry context.
    :type scene_geo: SceneGeoContext
    :param image_width: Image width in pixels.
    :type image_width: int
    :param image_height: Image height in pixels.
    :type image_height: int
    :return: Metadata payload.
    :rtype: Dict[str, Any]
    """

    pixel_to_local_h = np.linalg.inv(scene_geo.local_to_pixel_h)
    metadata = {
        "source": "pydtmdl",
        "scene_id": scene.identifier,
        "center": {"lat": float(scene.center_lat), "lon": float(scene.center_lon)},
        "top_left": {"lat": scene_geo.top_left_lat, "lon": scene_geo.top_left_lon},
        "size_m": int(scene.size_m),
        "rotation_deg": float(scene.rotation_deg),
        "image_size_px": {"width": int(image_width), "height": int(image_height)},
        "bbox_left_bottom_right_top": list(scene_geo.bbox_left_bottom_right_top),
        "corners_lon_lat": [list(point) for point in scene_geo.corners_lon_lat],
        "local_crs_wkt": scene_geo.local_crs_wkt,
        "local_crs_projjson": scene_geo.local_crs_projjson,
        "local_to_pixel_h": scene_geo.local_to_pixel_h.tolist(),
        "pixel_to_local_h": pixel_to_local_h.tolist(),
    }
    if scene.imagery_provider:
        metadata["imagery_provider"] = scene.imagery_provider
    if scene.grid_row is not None and scene.grid_col is not None:
        metadata["grid"] = {"row": int(scene.grid_row), "col": int(scene.grid_col)}
    if scene.metadata:
        metadata.update(scene.metadata)
    return metadata


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)
