from __future__ import annotations

import sys
import tarfile
import tempfile
from pathlib import Path

import supervisely as sly

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from import_images_with_osm.src.importer import import_from_local_dir
from import_osm.src.slyosm.settings import load_environment

APP_DIR = Path(__file__).resolve().parents[1]
load_environment(APP_DIR / "local.env")


def _download_source(api: sly.Api, team_id: int, remote_path: str, local_dir: Path) -> Path:
    """Download a Team Files path (directory or .tar) to local_dir and return the project root."""
    remote_path = remote_path.strip().rstrip("/")
    if not remote_path:
        raise ValueError("Team Files path must not be empty.")

    if remote_path.endswith(".tar"):
        local_archive = local_dir / "archive.tar"
        sly.logger.info("Downloading archive '%s' from Team Files.", remote_path)
        api.file.download(team_id, remote_path, str(local_archive))
        with tarfile.open(local_archive) as tar:
            tar.extractall(str(local_dir))
        local_archive.unlink(missing_ok=True)
        subdirs = [p for p in local_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and not (local_dir / "meta.json").exists():
            return subdirs[0]
        return local_dir
    else:
        sly.logger.info("Downloading directory '%s' from Team Files.", remote_path)
        api.file.download_directory(team_id, remote_path, str(local_dir))
        subdirs = [p for p in local_dir.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and not (local_dir / "meta.json").exists():
            return subdirs[0]
        return local_dir


@sly.handle_exceptions
def main():
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    workspace_id = sly.env.workspace_id()

    remote_path = sly.env.file(raise_not_found=False) or sly.env.folder(raise_not_found=False)
    if not remote_path:
        raise RuntimeError(
            "No input path provided. Run the app from the Team Files context menu "
            "on a directory or .tar archive exported by the Export to OSM Format app."
        )

    sly.logger.info("Starting import from Team Files path: '%s'.", remote_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_dir = Path(tmp_dir)
        source_dir = _download_source(api, team_id, remote_path, local_dir)

        sly.logger.info("Download complete. Importing from '%s'.", source_dir)
        import_result = import_from_local_dir(
            api=api,
            source_dir=source_dir,
            workspace_id=workspace_id,
        )

    total_images = sum(len(d.images) for d in import_result.datasets)
    total_failures = sum(len(d.failures) for d in import_result.datasets)
    sly.logger.info(
        "Import complete. Project ID: %s. Images: %s. Failures: %s.",
        import_result.project_id,
        total_images,
        total_failures,
    )


if __name__ == "__main__":
    main()
