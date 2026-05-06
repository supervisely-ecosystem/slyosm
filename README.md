# SlyOSM Apps Collection

Collection of Supervisely apps for importing and exporting geospatial image data with OSM metadata.

## Apps

1. [Import OSM](./import_osm/README.md)
2. [Export To OSM](./export_to_osm/README.md)

## Repository Layout

- `import_osm/`: Supervisely import app (UI + downloader pipeline).
- `export_to_osm/`: Supervisely export app (dataset to OSM XML).
- `.github/workflows/`: Docker build and Supervisely release workflows.
- `Dockerfile`: Custom image used by both apps.

## Build And Release

1. Build and push a Docker image from GitHub Actions:
   - Run workflow: `Docker Image Build`
   - Required secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`
   - Set `DOCKERHUB_USERNAME=supervisely` to publish as `supervisely/slyosm`
2. Release apps to Supervisely:
   - For tagged releases: `Release` workflow
   - For branch releases: `Release branch` workflow
   - For manual production publish: `Publish app to production` workflow

Required repository secrets:

- `SUPERVISELY_DEV_API_TOKEN`
- `SUPERVISELY_PRIVATE_DEV_API_TOKEN`
- `SUPERVISELY_PROD_API_TOKEN`

Required repository variables:

- `SUPERVISELY_DEV_SERVER_ADDRESS`
- `SUPERVISELY_PROD_SERVER_ADDRESS`