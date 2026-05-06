from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from import_osm.src.slyosm.osm_export import export_dataset_to_archive
from import_osm.src.slyosm.settings import (
    ensure_data_directories,
    load_environment,
)

APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_DIR = "/slyosm/osm_exports"

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


def _resolve_dataset_id() -> int:
    if os.environ.get("DATASET_ID", "").strip() == "":
        os.environ.pop("DATASET_ID", None)

    dataset_id = sly.env.dataset_id(raise_not_found=False)
    if dataset_id is None:
        raise RuntimeError(
            "Dataset id is not available. Launch the app from an images dataset or set DATASET_ID in the environment."
        )
    return int(dataset_id)


def _resolve_remote_dir() -> str:
    folder = sly.env.folder(raise_not_found=False)
    if folder is None:
        return DEFAULT_REMOTE_DIR

    normalized = folder.strip()
    if normalized == "":
        return DEFAULT_REMOTE_DIR
    if normalized == "/":
        return "/"
    return "/" + normalized.strip("/")


def _build_remote_path(remote_dir: str, file_name: str) -> str:
    if remote_dir == "/":
        return "/" + file_name
    return remote_dir.rstrip("/") + "/" + file_name


def _log_mapping_source(dataset_info: Any) -> None:
    custom_data = (
        dataset_info.custom_data if isinstance(dataset_info.custom_data, dict) else {}
    )
    specs = custom_data.get("osm_class_specs")
    if isinstance(specs, list) and len(specs) > 0:
        sly.logger.info(
            "Using dataset custom OSM mapping with %s class entries.",
            len(specs),
        )
    else:
        sly.logger.info(
            "Dataset custom OSM mapping is missing, the shared default mapping will be used."
        )


@sly.handle_exceptions
def main() -> None:
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    dataset_id = _resolve_dataset_id()
    remote_dir = _resolve_remote_dir()

    dataset_info = api.dataset.get_info_by_id(dataset_id)
    if dataset_info is None:
        raise RuntimeError(
            "Dataset with id={dataset_id} was not found.".format(dataset_id=dataset_id)
        )

    sly.logger.info(
        "Starting OSM export for dataset '%s' (%s).",
        dataset_info.name,
        dataset_id,
    )
    sly.logger.info("Resolved Team Files output directory: %s", remote_dir)
    _log_mapping_source(dataset_info)

    result = export_dataset_to_archive(
        api=api,
        dataset_id=dataset_id,
        progress_callback=ProgressReporter(),
    )

    remote_path = _build_remote_path(remote_dir, result.archive_path.name)
    api.file.upload(team_id, str(result.archive_path), remote_path)

    if len(result.failures) == 0:
        sly.logger.info(
            "Finished export. Exported %s image(s). Archive uploaded to %s",
            len(result.images),
            remote_path,
        )
    else:
        sly.logger.warning(
            "Finished export with %s success(es) and %s failure(s). Archive uploaded to %s",
            len(result.images),
            len(result.failures),
            remote_path,
        )
        sly.logger.warning(
            "Some images were skipped. See the failure logs above for details."
        )


if __name__ == "__main__":
    main()
