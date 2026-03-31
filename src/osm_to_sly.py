from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import osmnx as ox
import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from satellite_osm_downloader.src.slyosm.geometry import SceneRequest
from satellite_osm_downloader.src.slyosm.osm_config import (
    OSMClassSpec,
    ensure_project_meta_has_classes,
    load_osm_class_specs,
)
from satellite_osm_downloader.src.slyosm.sampling import (
    build_area_polygon_wgs84,
    coordinate_key,
    generate_random_coordinates,
)
from satellite_osm_downloader.src.slyosm.scene_downloader import (
    process_scene,
    save_dataset_custom_data,
)
from satellite_osm_downloader.src.slyosm.settings import (
    OSM_CLASSES_PATH,
    SAMPLING_STATE_DIR,
    ensure_data_directories,
    load_environment,
    read_optional_int_env,
)

PROJECT_NAME = "Training Data (RAW)"
RANDOM_SEED = 20260330
ROTATION_MIN_DEG = -90
ROTATION_MAX_DEG = 90
DEFAULT_SCENE_SIZE_M = 1024
DEFAULT_ZOOM = 18

COUNTRY_RUNS: List[Dict[str, Any]] = [
    {
        "key": "germany",
        "query": "Germany",
        "dataset_name": "germany",
        "target_images": 1000,
        "size_m": DEFAULT_SCENE_SIZE_M,
        "zoom": DEFAULT_ZOOM,
    }
]


@dataclass(frozen=True)
class CountryRun:
    """Legacy production sampling configuration for one area query."""

    key: str
    query: str
    dataset_name: str
    target_images: int
    size_m: int
    zoom: int


def parse_country_runs(raw_runs: List[Dict[str, Any]]) -> List[CountryRun]:
    """Parse legacy country-run dictionaries into typed records."""

    runs = []
    for raw_run in raw_runs:
        runs.append(
            CountryRun(
                key=str(raw_run["key"]).strip().lower(),
                query=str(raw_run["query"]),
                dataset_name=str(raw_run["dataset_name"]),
                target_images=int(raw_run["target_images"]),
                size_m=int(raw_run.get("size_m", DEFAULT_SCENE_SIZE_M)),
                zoom=int(raw_run.get("zoom", DEFAULT_ZOOM)),
            )
        )
    return runs


def load_state(path: Path) -> Tuple[Set[str], int]:
    """Load persisted random-sampling state from disk."""

    if not path.exists():
        return set(), 1

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    used_keys = set(payload.get("used_keys", []))
    next_index = int(payload.get("next_index", len(used_keys) + 1))
    return used_keys, max(1, next_index)


def save_state(path: Path, used_keys: Set[str], next_index: int) -> None:
    """Persist random-sampling state to disk."""

    payload = {"used_keys": sorted(used_keys), "next_index": int(next_index)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def ensure_project(
    api: sly.Api, workspace_id: int, class_specs: List[OSMClassSpec]
) -> Tuple[Any, sly.ProjectMeta]:
    """Resolve or create the legacy production project."""

    project_id = read_optional_int_env("PROJECT_ID")
    if project_id is not None:
        project_info = api.project.get_info_by_id(project_id)
        if project_info is None:
            raise RuntimeError(
                "Project with id={project_id} was not found.".format(
                    project_id=project_id
                )
            )
    else:
        project_info = api.project.create(
            workspace_id, PROJECT_NAME, change_name_if_conflict=True
        )

    project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_info.id))
    updated_meta = ensure_project_meta_has_classes(project_meta, class_specs)
    if updated_meta != project_meta:
        project_meta = api.project.update_meta(project_info.id, updated_meta)
    return project_info, project_meta


def process_country_run(
    api: sly.Api,
    project_id: int,
    project_meta: sly.ProjectMeta,
    class_specs: List[OSMClassSpec],
    country: CountryRun,
    rng: np.random.Generator,
) -> None:
    """Run the legacy random sampling workflow for one configured area."""

    dataset_info = api.dataset.get_or_create(project_id, country.dataset_name)
    save_dataset_custom_data(api, dataset_info.id, class_specs)

    state_path = SAMPLING_STATE_DIR / "{key}.json".format(key=country.key)
    used_keys, next_index = load_state(state_path)
    remaining = max(0, country.target_images - len(used_keys))

    print(
        "[{key}] dataset_id={dataset_id} existing={existing} target={target} remaining={remaining}".format(
            key=country.key,
            dataset_id=dataset_info.id,
            existing=len(used_keys),
            target=country.target_images,
            remaining=remaining,
        ),
        flush=True,
    )
    if remaining == 0:
        print(
            "[{key}] Target already reached. Skipping.".format(key=country.key),
            flush=True,
        )
        return

    area_polygon = build_area_polygon_wgs84(country.query)
    coordinates = generate_random_coordinates(
        area_polygon, remaining, rng, used_keys=used_keys
    )

    scene_key_by_id = {}
    scenes = []
    for lat, lon in coordinates:
        key = coordinate_key(lat, lon)
        scene_id = "{key}_{index:06d}".format(
            key=country.key, index=next_index + len(scenes)
        )
        scene_key_by_id[scene_id] = key
        scenes.append(
            SceneRequest(
                identifier=scene_id,
                center_lat=float(lat),
                center_lon=float(lon),
                size_m=country.size_m,
                rotation_deg=int(rng.integers(ROTATION_MIN_DEG, ROTATION_MAX_DEG + 1)),
                zoom=country.zoom,
            )
        )

    uploaded = 0
    for scene in scenes:
        try:
            process_scene(
                api=api,
                dataset_id=dataset_info.id,
                project_meta=project_meta,
                class_specs=class_specs,
                scene=scene,
                download_osm=True,
            )
        except Exception as exc:
            print(
                "[{key}] Failed scene_id={scene_id} lat={lat:.6f} lon={lon:.6f}: {error}".format(
                    key=country.key,
                    scene_id=scene.identifier,
                    lat=scene.center_lat,
                    lon=scene.center_lon,
                    error=exc,
                ),
                flush=True,
            )
            continue

        uploaded += 1
        used_keys.add(scene_key_by_id[scene.identifier])
        next_index += 1
        save_state(state_path, used_keys, next_index)

        if uploaded % 25 == 0:
            print(
                "[{key}] progress uploaded={uploaded}/{remaining}".format(
                    key=country.key,
                    uploaded=uploaded,
                    remaining=remaining,
                ),
                flush=True,
            )

    print(
        "[{key}] Completed uploaded={uploaded}/{remaining}. Total unique coords tracked={total}".format(
            key=country.key,
            uploaded=uploaded,
            remaining=remaining,
            total=len(used_keys),
        ),
        flush=True,
    )


def main() -> None:
    """Run the legacy random import workflow using the refactored modules."""

    load_environment()
    ensure_data_directories()
    SAMPLING_STATE_DIR.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True

    api = sly.Api.from_env()
    workspace_id = sly.env.workspace_id()
    class_specs = load_osm_class_specs(OSM_CLASSES_PATH)
    country_runs = parse_country_runs(COUNTRY_RUNS)

    project_info, project_meta = ensure_project(api, workspace_id, class_specs)
    rng = np.random.default_rng(RANDOM_SEED)

    print(
        "Production generation started for project='{name}' areas={count}".format(
            name=project_info.name,
            count=len(country_runs),
        ),
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
