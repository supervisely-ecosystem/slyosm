from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from import_osm.src.slyosm.osm_export import (
    SuperviselyDatasetExportResult,
    export_dataset_to_supervisely_dir,
)
from import_osm.src.slyosm.settings import (
    OSM_EXPORT_DIR,
    ensure_data_directories,
    load_environment,
    sanitize_filename,
)

APP_DIR = Path(__file__).resolve().parents[1]

load_environment(APP_DIR / "local.env")
ensure_data_directories()


class ProgressReporter:
    def __init__(self) -> None:
        self._last_logged_at = 0.0

    def __call__(self, index: int, total: int, image_info: Any) -> None:
        now = time.monotonic()
        if (
            index == total
            or index == 1
            or index % 10 == 0
            or now - self._last_logged_at >= 10.0
        ):
            image_name = getattr(image_info, "name", str(image_info))
            sly.logger.info(
                "Export progress: %s/%s image(s). Latest: %s",
                index,
                total,
                image_name,
            )
            self._last_logged_at = now


def _resolve_context(api: sly.Api) -> Tuple[int, List[Any], str]:
    """Resolve export targets from environment.

    Prefers DATASET_ID over PROJECT_ID when both are set.

    :return: ``(project_id, list_of_dataset_infos, export_name)``
    """
    dataset_id_raw = sly.env.dataset_id(raise_not_found=False)
    if dataset_id_raw is not None:
        dataset_info = api.dataset.get_info_by_id(int(dataset_id_raw))
        if dataset_info is None:
            raise RuntimeError(
                "Dataset with id={id} was not found.".format(id=dataset_id_raw)
            )
        return dataset_info.project_id, [dataset_info], dataset_info.name

    project_id_raw = sly.env.project_id(raise_not_found=False)
    if project_id_raw is not None:
        project_info = api.project.get_info_by_id(int(project_id_raw))
        if project_info is None:
            raise RuntimeError(
                "Project with id={id} was not found.".format(id=project_id_raw)
            )
        datasets = api.dataset.get_list(int(project_id_raw))
        if not datasets:
            raise RuntimeError(
                "Project '{name}' contains no datasets.".format(name=project_info.name)
            )
        return int(project_id_raw), datasets, project_info.name

    raise RuntimeError(
        "No export target found. Launch the app from a dataset or project context, "
        "or set DATASET_ID / PROJECT_ID in the environment."
    )


def _log_mapping_source(dataset_info: Any) -> None:
    custom_data = (
        dataset_info.custom_data if isinstance(dataset_info.custom_data, dict) else {}
    )
    specs = custom_data.get("osm_class_specs")
    if isinstance(specs, list) and len(specs) > 0:
        sly.logger.info(
            "Dataset '%s': using custom OSM mapping with %s class entries.",
            dataset_info.name,
            len(specs),
        )
    else:
        sly.logger.info(
            "Dataset '%s': no custom OSM mapping found, using shared default.",
            dataset_info.name,
        )


def _log_dataset_result(result: SuperviselyDatasetExportResult) -> None:
    osm_count = sum(1 for img in result.images if img.osm_path is not None)
    if result.failures:
        sly.logger.warning(
            "Dataset '%s': %s image(s) exported (%s with OSM), %s failure(s).",
            result.dataset_name,
            len(result.images),
            osm_count,
            len(result.failures),
        )
    else:
        sly.logger.info(
            "Dataset '%s': %s image(s) exported (%s with OSM).",
            result.dataset_name,
            len(result.images),
            osm_count,
        )


@sly.handle_exceptions
def main() -> None:
    api = sly.Api.from_env()

    project_id, datasets, export_name = _resolve_context(api)

    sly.logger.info(
        "Starting OSM export for %s dataset(s) (export name: '%s').",
        len(datasets),
        export_name,
    )

    anchor_id = datasets[0].id if len(datasets) == 1 else project_id
    export_slug = sanitize_filename(
        "{name}_{id}".format(name=export_name, id=anchor_id)
    )
    export_dir = OSM_EXPORT_DIR / export_slug
    export_dir.mkdir(parents=True, exist_ok=True)

    project_meta_json = api.project.get_meta(project_id)
    project_info = api.project.get_info_by_id(project_id)
    if project_info is not None and project_info.settings:
        project_meta_json["projectSettings"] = project_info.settings
    meta_path = export_dir / "meta.json"
    meta_path.write_text(json.dumps(project_meta_json, indent=2), encoding="utf-8")
    sly.logger.info("Written meta.json to '%s'.", meta_path)

    reporter = ProgressReporter()
    all_results: List[SuperviselyDatasetExportResult] = []

    for dataset_info in datasets:
        _log_mapping_source(dataset_info)
        dataset_output_dir = export_dir / sanitize_filename(dataset_info.name)
        result = export_dataset_to_supervisely_dir(
            api=api,
            dataset_id=dataset_info.id,
            output_dir=dataset_output_dir,
            progress_callback=reporter,
        )
        _log_dataset_result(result)
        all_results.append(result)

    total_images = sum(len(r.images) for r in all_results)
    total_failures = sum(len(r.failures) for r in all_results)
    total_osm = sum(
        sum(1 for img in r.images if img.osm_path is not None) for r in all_results
    )

    if total_failures == 0:
        sly.logger.info(
            "Finished export. %s image(s) exported (%s with OSM).",
            total_images,
            total_osm,
        )
    else:
        sly.logger.warning(
            "Finished export with %s success(es) (%s with OSM) and %s failure(s).",
            total_images,
            total_osm,
            total_failures,
        )
        sly.logger.warning(
            "Some images were skipped. See the failure logs above for details."
        )

    sly.output.set_download(str(export_dir))


if __name__ == "__main__":
    main()
