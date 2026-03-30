from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import supervisely as sly
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep

from main import (
    BASE_DIR,
    CONFIG_PATH,
    OSMClassSpec,
    ensure_output_dirs,
    load_environment,
    load_osm_class_specs,
    process_scene,
)

# Production job settings.
PROJECT_NAME = "Training Data (RAW)"
RANDOM_SEED = 20260330
COORD_KEY_DECIMALS = 6
MAX_SAMPLE_ATTEMPTS_PER_IMAGE = 300

# Scene defaults.
DEFAULT_SCENE_SIZE_M = 1024
DEFAULT_ZOOM = 18
ROTATION_MIN_DEG = -90
ROTATION_MAX_DEG = 90

# Extend this list for other countries.
COUNTRY_RUNS: list[dict[str, Any]] = [
    {
        "key": "germany",
        "query": "Germany",
        "dataset_name": "germany",
        "target_images": 1000,
        "size_m": DEFAULT_SCENE_SIZE_M,
        "zoom": DEFAULT_ZOOM,
    }
]

SAMPLE_STATE_DIR = BASE_DIR / "data" / "meta" / "sampling"


@dataclass(frozen=True)
class CountryRun:
    key: str
    query: str
    dataset_name: str
    target_images: int
    size_m: int
    zoom: int


def parse_country_runs(raw_runs: list[dict[str, Any]]) -> list[CountryRun]:
    result: list[CountryRun] = []
    for raw in raw_runs:
        result.append(
            CountryRun(
                key=str(raw["key"]).strip().lower(),
                query=str(raw["query"]),
                dataset_name=str(raw["dataset_name"]),
                target_images=int(raw["target_images"]),
                size_m=int(raw.get("size_m", DEFAULT_SCENE_SIZE_M)),
                zoom=int(raw.get("zoom", DEFAULT_ZOOM)),
            )
        )
    return result


def ensure_project_with_classes(
    api: sly.Api,
    workspace_id: int,
    class_specs: list[OSMClassSpec],
) -> tuple[Any, sly.ProjectMeta]:
    project_info = api.project.get_or_create(workspace_id, PROJECT_NAME)

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_info.id))
    changed = False
    for spec in class_specs:
        if project_meta.get_obj_class(spec.name) is None:
            obj_class = sly.ObjClass(spec.name, sly.Bitmap, color=spec.color)
            project_meta = project_meta.add_obj_class(obj_class)
            changed = True

    if changed:
        project_meta = api.project.update_meta(project_info.id, project_meta)

    return project_info, project_meta


def load_state(path: Path) -> tuple[set[str], int]:
    if not path.exists():
        return set(), 1

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    used_keys = set(payload.get("used_keys", []))
    next_index = int(payload.get("next_index", len(used_keys) + 1))
    next_index = max(1, next_index)
    return used_keys, next_index


def save_state(path: Path, used_keys: set[str], next_index: int) -> None:
    payload = {
        "used_keys": sorted(used_keys),
        "next_index": int(next_index),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_country_polygon_wgs84(country_query: str):
    gdf = ox.geocode_to_gdf(country_query)
    if gdf.empty:
        raise RuntimeError(f"Country query returned no geometry: {country_query}")

    polygon = unary_union([geom for geom in gdf.geometry if geom is not None])
    if polygon.is_empty:
        raise RuntimeError(f"Country geometry is empty: {country_query}")

    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        raise RuntimeError(
            f"Country geometry is invalid after cleanup: {country_query}"
        )

    return polygon


def coordinate_key(lat: float, lon: float) -> str:
    return f"{lat:.{COORD_KEY_DECIMALS}f},{lon:.{COORD_KEY_DECIMALS}f}"


def generate_random_scenes(
    country: CountryRun,
    country_polygon,
    used_keys: set[str],
    next_index: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[str]]:
    if count <= 0:
        return [], []

    prepared = prep(country_polygon)
    min_lon, min_lat, max_lon, max_lat = country_polygon.bounds

    scenes: list[dict[str, Any]] = []
    keys: list[str] = []
    planned_keys: set[str] = set()

    max_attempts = count * MAX_SAMPLE_ATTEMPTS_PER_IMAGE
    attempts = 0

    while len(scenes) < count and attempts < max_attempts:
        attempts += 1

        lon = float(rng.uniform(min_lon, max_lon))
        lat = float(rng.uniform(min_lat, max_lat))
        point = Point(lon, lat)
        if not prepared.contains(point):
            continue

        key = coordinate_key(lat, lon)
        if key in used_keys or key in planned_keys:
            continue

        planned_keys.add(key)

        scene_index = next_index + len(scenes)
        scene = {
            "id": f"{country.key}_{scene_index:06d}",
            "center_lat": lat,
            "center_lon": lon,
            "size_m": country.size_m,
            "rotation_deg": int(rng.integers(ROTATION_MIN_DEG, ROTATION_MAX_DEG + 1)),
            "zoom": country.zoom,
            "coord_key": key,
        }
        scenes.append(scene)
        keys.append(key)

    if len(scenes) < count:
        raise RuntimeError(
            f"Failed to sample enough unique points for {country.key}: "
            f"requested={count}, sampled={len(scenes)}, attempts={attempts}"
        )

    return scenes, keys


def process_country_run(
    api: sly.Api,
    project_id: int,
    project_meta: sly.ProjectMeta,
    class_specs: list[OSMClassSpec],
    country: CountryRun,
    rng: np.random.Generator,
) -> None:
    dataset_info = api.dataset.get_or_create(project_id, country.dataset_name)

    state_path = SAMPLE_STATE_DIR / f"{country.key}.json"
    used_keys, next_index = load_state(state_path)

    remaining = max(0, country.target_images - len(used_keys))
    print(
        f"[{country.key}] dataset_id={dataset_info.id} "
        f"existing={len(used_keys)} target={country.target_images} remaining={remaining}",
        flush=True,
    )
    if remaining == 0:
        print(f"[{country.key}] Target already reached. Skipping.", flush=True)
        return

    print(f"[{country.key}] Resolving country boundary for sampling...", flush=True)
    country_polygon = build_country_polygon_wgs84(country.query)

    scenes, _ = generate_random_scenes(
        country=country,
        country_polygon=country_polygon,
        used_keys=used_keys,
        next_index=next_index,
        count=remaining,
        rng=rng,
    )

    uploaded = 0
    for scene in scenes:
        scene_key = str(scene["coord_key"])
        try:
            process_scene(
                api=api,
                dataset_id=dataset_info.id,
                project_meta=project_meta,
                class_specs=class_specs,
                scene=scene,
            )
        except Exception as exc:
            print(
                f"[{country.key}] Failed scene_id={scene['id']} "
                f"lat={scene['center_lat']:.6f} lon={scene['center_lon']:.6f}: {exc}",
                flush=True,
            )
            continue

        uploaded += 1
        used_keys.add(scene_key)
        next_index += 1
        save_state(state_path, used_keys, next_index)

        if uploaded % 25 == 0:
            print(
                f"[{country.key}] progress uploaded={uploaded}/{remaining}",
                flush=True,
            )

    print(
        f"[{country.key}] Completed uploaded={uploaded}/{remaining}. "
        f"Total unique coords tracked={len(used_keys)}",
        flush=True,
    )


def main() -> None:
    load_environment()
    ensure_output_dirs()
    SAMPLE_STATE_DIR.mkdir(parents=True, exist_ok=True)

    ox.settings.use_cache = True

    api = sly.Api.from_env()
    workspace_id = sly.env.workspace_id()

    class_specs = load_osm_class_specs(CONFIG_PATH)
    country_runs = parse_country_runs(COUNTRY_RUNS)

    project_info, project_meta = ensure_project_with_classes(
        api=api,
        workspace_id=workspace_id,
        class_specs=class_specs,
    )

    rng = np.random.default_rng(RANDOM_SEED)

    print(
        f"Production generation started for project='{PROJECT_NAME}' "
        f"countries={len(country_runs)}",
        flush=True,
    )

    for country in country_runs:
        process_country_run(
            api=api,
            project_id=project_info.id,
            project_meta=project_meta,
            class_specs=class_specs,
            country=country,
            rng=rng,
        )

    print("Production generation finished.", flush=True)


if __name__ == "__main__":
    main()
