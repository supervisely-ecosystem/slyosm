# Satellite, DTM & OSM — Supervisely Apps Collection

Download georeferenced satellite imagery, DTM elevation tiles, and OpenStreetMap vector annotations directly into Supervisely, then export annotated datasets back to OSM XML format.

1. [Satellite, DTM & OSM Downloader](./import_osm/README.md)

<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/import-poster.png"/>

2. [Export to OSM Format](./export_to_osm/README.md)

<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/export-poster.png"/>

---

## Repository Layout

- `import_osm/` — Supervisely interactive app: UI + downloader pipeline (satellite, DTM, OSM).
- `export_to_osm/` — Supervisely headless export app: annotations → OSM XML + Supervisely format archive.
- `.github/workflows/` — Docker build and Supervisely release workflows.
- `Dockerfile` — Custom image used by both apps.

## Build and Release

1. Build and push a Docker image from GitHub Actions:
   - Run workflow: `Docker Image Build`
   - Required secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
   - Set `DOCKERHUB_USERNAME=supervisely` to publish as `supervisely/slyosm`
2. Release apps to Supervisely:
   - Tagged releases: `Release` workflow
   - Branch releases: `Release branch` workflow
   - Manual production publish: `Publish app to production` workflow

Required repository secrets: `SUPERVISELY_DEV_API_TOKEN`, `SUPERVISELY_PRIVATE_DEV_API_TOKEN`, `SUPERVISELY_PROD_API_TOKEN`

Required repository variables: `SUPERVISELY_DEV_SERVER_ADDRESS`, `SUPERVISELY_PROD_SERVER_ADDRESS`
