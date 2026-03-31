from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import osmnx as ox
import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from satellite_osm_downloader.src.slyosm.geometry import scene_from_dict
from satellite_osm_downloader.src.slyosm.osm_config import load_osm_class_specs
from satellite_osm_downloader.src.slyosm.scene_downloader import (
    ensure_project_and_dataset,
    process_scenes,
    save_dataset_custom_data,
)
from satellite_osm_downloader.src.slyosm.settings import (
    DEFAULT_DATASET_NAME,
    DEFAULT_PROJECT_NAME,
    OSM_CLASSES_PATH,
    ensure_data_directories,
    load_environment,
    read_optional_int_env,
)

SCENES: List[Dict[str, Any]] = [
    {
        "id": "scene_0001",
        "center_lat": 47.975679348309754,
        "center_lon": 10.788837124189856,
        "size_m": 1024,
        "rotation_deg": 0,
        "zoom": 18,
    }
]


def main() -> None:
    """Run the legacy explicit-scene import workflow using the refactored modules."""

    load_environment()
    ensure_data_directories()
    ox.settings.use_cache = True

    api = sly.Api.from_env()
    workspace_id = sly.env.workspace_id()
    project_id = read_optional_int_env("PROJECT_ID")
    dataset_id = read_optional_int_env("DATASET_ID")

    class_specs = load_osm_class_specs(OSM_CLASSES_PATH)
    _, dataset_info, project_meta = ensure_project_and_dataset(
        api=api,
        workspace_id=workspace_id,
        class_specs=class_specs,
        project_id=project_id,
        project_name=DEFAULT_PROJECT_NAME,
        dataset_id=dataset_id,
        dataset_name=DEFAULT_DATASET_NAME,
    )
    save_dataset_custom_data(api, dataset_info.id, class_specs)

    scenes = [scene_from_dict(raw_scene) for raw_scene in SCENES]
    results, failures, stopped = process_scenes(
        api=api,
        dataset_id=dataset_info.id,
        project_meta=project_meta,
        class_specs=class_specs,
        scenes=scenes,
        download_osm=True,
    )

    print(
        "Processed scenes into dataset_id={dataset_id}: uploaded={uploaded}, failures={failures}, stopped={stopped}".format(
            dataset_id=dataset_info.id,
            uploaded=len(results),
            failures=len(failures),
            stopped=stopped,
        ),
        flush=True,
    )
    for failure in failures:
        print(
            "[{scene_id}] {error}".format(
                scene_id=failure.scene_id, error=failure.error
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
