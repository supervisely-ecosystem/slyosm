<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/poster-import-osm-annotations.2.jpg"/>

# Import Images with OSM

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#input-format">Input Format</a> •
  <a href="#annotation-priority">Annotation Priority</a> •
  <a href="#geo-metadata">Geo Metadata</a> •
  <a href="#interface-modes">Interface Modes</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#license-and-attribution">License & Attribution</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/slyosm/import_images_with_osm)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/slyosm)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/slyosm/import_images_with_osm.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/slyosm/import_images_with_osm.png)](https://supervisely.com)

</div>

## Overview

This app imports images together with OSM vector annotations into Supervisely. It accepts a structured directory (or `.tar` archive) in Team Files that contains images, OSM XML files, per-image geo metadata, and optional Supervisely JSON annotations, then creates a new project with all class definitions, interface settings, and geo context restored.

The expected input format is produced automatically by the **[Export to OSM Format](../export_to_osm/README.md)** app, which makes the two apps a natural pair for a round-trip workflow:

1. **[Satellite, DTM & OSM Downloader](../import_osm/README.md)** — downloads satellite imagery, elevation tiles, and OpenStreetMap vector annotations into Supervisely.
2. Annotate or review in the Supervisely labeling toolbox.
3. **[Export to OSM Format](../export_to_osm/README.md)** — exports the annotated project to a structured directory in Team Files.
4. Edit the exported OSM files externally (e.g., in [JOSM](https://josm.openstreetmap.de/)) and re-import with this app.

You can also prepare the input directory manually — the app only requires that the layout described in [Input Format](#input-format) is followed.

---

## Input Format

The app expects a Supervisely project export with the following structure:

```
📂 export_root/
├── 📄 meta.json              ← required: project classes, tags, interface settings
└── 📁 dataset_name/
    ├── 📂 img/
    │   ├── 🖼️ image_001.png
    │   └── 🖼️ ...
    ├── 📂 meta/              ← per-image geo metadata (required for OSM import)
    │   ├── 📝 image_001.png.json
    │   └── 📝 ...
    ├── 📂 ann/               ← optional: Supervisely annotation JSONs
    │   ├── 📝 image_001.png.json
    │   └── 📝 ...
    ├── 📂 osm/               ← optional: OSM XML files (higher priority than ann/)
    │   ├── 🗺️ image_001.png.osm
    │   └── 🗺️ ...
    └── 📂 overlay/           ← optional: overlay images (overlay projects only)
        └── 📁 image_001/
            └── 🖼️ image_001_dtm.png
```

**Rules:**
- `meta.json` at the project root is **required**. It contains the project class definitions, tag schemas, and interface settings. Without it the app raises an error.
- `img/` is the only required subdirectory inside each dataset directory.
- `meta/` contains per-image geo metadata. Without it OSM files cannot be reprojected to pixel coordinates and OSM import is skipped for those images.
- At least one of `ann/` or `osm/` must be present for annotations to be imported. If neither exists, images are uploaded with no labels.

The same structure supports exporting a project with **multiple datasets**:

```
📂 export_root/
├── 📄 meta.json
├── 📁 dataset_name_1/
│   ├── 📂 img/
│   ├── 📂 ann/
│   ├── 📂 meta/
│   └── 📂 osm/
└── 📁 dataset_name_2/
    └── ...
```

`.tar` and `.zip` archives as well as unpacked directories are accepted as the source path.

---

## Annotation Priority

When **both** an OSM file and a Supervisely JSON annotation exist for the same image, **the OSM file always wins**.

| `osm/image.osm` present | `ann/image.json` present | Used source |
|---|---|---|
| Yes | Yes | **OSM** (higher priority) |
| Yes | No | OSM |
| No | Yes | JSON |
| No | No | No annotations uploaded |

### OSM → Supervisely conversion

When the OSM file is used, the app converts OSM geometry back to pixel space:

1. The per-image geo metadata from `meta/image.png.json` provides the `pixel_to_local_h` homography matrix and the `crs_wkt` coordinate reference system.
2. The inverse of that homography is applied to reproject each OSM node's latitude/longitude back to pixel coordinates.
3. OSM way tags are matched against the class mapping (see below) to identify the correct Supervisely class for each feature.
4. Closed ways become `Polygon` labels; open ways become `Polyline` labels; multipolygon relations become `Polygon` labels with interior holes.

### Class mapping for OSM import

To identify which Supervisely class an OSM feature belongs to, the app looks for a `custom_data.json` file inside each dataset directory. This file is written automatically by both the **Satellite, DTM & OSM Downloader** and the **Export to OSM Format** apps. If none is found (e.g., when importing a manually prepared dataset), the built-in default OSM class mapping is used.

The matching rule: for each OSM way or relation, the app checks whether its tags contain the key-value pair defined in that class's `default_tag`. The first match wins.

---

## Geo Metadata

Each file in `meta/{image_name}.json` contains the geospatial context stored on that image in Supervisely. The app re-attaches this metadata to the uploaded image so that:

- The **Export to OSM Format** app can still generate OSM files from the re-imported project.
- The coordinate transform (tile center, projection, homography) is preserved for downstream geographic analysis.

Key fields in the `geo` object:

| Field | Description |
|---|---|
| `crs_wkt` | Coordinate reference system in WKT format |
| `pixel_to_local_h` | 3×3 homography matrix: pixel → local CRS coordinates |
| `bbox_lonlat` | Geographic bounding box (lon/lat of all four corners) |
| `scene_id` | Scene identifier used for multiview grouping |
| `rotation_deg`, `tile_size_m` | Acquisition parameters |
| `multiview_layer` | `"satellite"` or `"dtm"` — identifies the image's role in a multiview pair |

If a `meta/` file does not exist for a given image, the image is uploaded without geo metadata and the OSM export for that image will be skipped in future exports.

---

## Interface Modes

The labeling interface (multiview, overlay, or default) is read from `meta.json` and applied to the recreated project automatically — no manual configuration is required.

| `meta.json` setting | Result |
|---|---|
| `projectSettings.labelingInterface = "overlay"` | Overlay mode enabled; DTM images are uploaded as overlay layers |
| `projectSettings.multiView.enabled = true` | Multiview mode enabled; satellite and DTM images are uploaded as linked pairs |
| Neither | Default interface; images are uploaded individually |

### Overlay

For overlay projects, the app looks for overlay images in `overlay/{image_stem}/` and uploads them as linked overlay layers using the Supervisely overlay image API. Only the primary image (from `img/`) receives annotations and geo metadata; the overlay image is attached as a transparent layer.

### Multiview

For multiview projects, images are grouped by `scene_id` from their geo metadata. Images with `multiview_layer = "satellite"` and `multiview_layer = "dtm"` sharing the same `scene_id` are uploaded as a linked multiview pair using the Supervisely multiview image API.

---

## How To Run

This app is **headless** — it has no interactive UI and runs entirely from the task log.

**Step 1:** In Team Files, navigate to the project directory or `.tar` archive you want to import. The path must contain (or resolve to) `meta.json` at its root — see [Input Format](#input-format) for the expected layout.

**Step 2:** Right-click the directory or `.tar` file and select **Run App → Import Images with OSM** from the context menu.

The app starts immediately. It downloads the source from Team Files to a temporary local directory, creates a new project in the current workspace, and uploads all images, geo metadata, and annotations. Progress is logged in the task output.

When the import is complete, the task log shows the total number of images imported and the count of any failures.

---

## License and Attribution

### OpenStreetMap

OSM XML files that may be re-imported by this app contain data derived from [OpenStreetMap](https://www.openstreetmap.org/). The underlying OSM data is © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and is licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

**You are required to credit OpenStreetMap** in any public-facing product, publication, or dataset that includes data processed by this app. The standard attribution is:

> © OpenStreetMap contributors

Full requirements are described at [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright). Compliance is your responsibility.
