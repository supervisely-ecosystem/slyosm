from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from import_osm.src.slyosm.osm_export import export_image_to_osm
from import_osm.src.slyosm.settings import (
    OSM_EXPORT_DIR,
    load_environment,
    read_optional_int_env,
)


def main() -> None:
    """Export one image annotation to OSM XML using the refactored export module."""

    load_environment()

    image_id = read_optional_int_env("IMAGE_ID") or 0
    output_path_value = os.getenv("OUTPUT_PATH", "").strip()
    if image_id <= 0:
        raise RuntimeError(
            "Set IMAGE_ID in the environment before running this script."
        )

    api = sly.Api.from_env()
    image_info = api.image.get_info_by_id(image_id)
    if image_info is None:
        raise RuntimeError(
            "Image with id={image_id} was not found.".format(image_id=image_id)
        )

    dataset_info = api.dataset.get_info_by_id(image_info.dataset_id)
    if dataset_info is None:
        raise RuntimeError(
            "Dataset with id={dataset_id} was not found.".format(
                dataset_id=image_info.dataset_id
            )
        )

    project_meta = sly.ProjectMeta.from_json(
        api.project.get_meta(dataset_info.project_id)
    )
    dataset_custom_data = (
        dataset_info.custom_data if isinstance(dataset_info.custom_data, dict) else {}
    )

    target_output_path: Optional[Path]
    if output_path_value:
        target_output_path = Path(output_path_value)
        target_output_dir = target_output_path.parent
    else:
        target_output_dir = OSM_EXPORT_DIR
        target_output_path = None

    result = export_image_to_osm(
        api=api,
        image_info=image_info,
        dataset_custom_data=dataset_custom_data,
        project_meta=project_meta,
        output_dir=target_output_dir,
        output_path=target_output_path,
    )

    print(
        "Export complete: image_id={image_id} nodes={nodes} ways={ways} relations={relations} output={output}".format(
            image_id=result.image_id,
            nodes=result.nodes,
            ways=result.ways,
            relations=result.relations,
            output=result.output_path,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
