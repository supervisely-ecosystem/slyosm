<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/poster-export-osm-format-res.jpg"/>

# Export to OSM Format

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#output-format">Output Format</a> •
  <a href="#osm-class-mapping">OSM Class Mapping</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#license-and-attribution">License & Attribution</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/slyosm/export_to_osm)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/slyosm)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/slyosm/export_to_osm.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/slyosm/export_to_osm.png)](https://supervisely.com)

</div>

## Overview

This headless app exports a Supervisely dataset — or all datasets in a project — to a structured archive containing:

- **Original images** as downloaded from the imagery provider.
- **Supervisely annotation JSON files** in the standard Supervisely format.
- **Per-image metadata JSON files** preserving the embedded geospatial information required for geographic reprojection and re-import.
- **Project-level `meta.json`** with class definitions, tag schemas, and interface settings (multiview, overlay, or default).
- **OSM XML files** with each annotation projected back to geographic coordinates (longitude/latitude) and written as OSM ways, relations, and nodes with the appropriate OSM tags.

The app runs entirely in the background — no UI is shown. When finished, it uploads the archive to Team Files automatically.

This app is designed as the companion to the [Satellite, DTM & OSM Downloader](https://ecosystem.supervisely.com/apps/slyosm/import_osm). Images created by that app carry embedded geo metadata required for the coordinate projection. Images without geo metadata are still exported to `img/` and `ann/`; the OSM export is skipped and no `meta/` file is written for those images, with a warning in the task log.

---

## Output Format

The export always produces a valid Supervisely project directory. A project-level `meta.json` sits at the root, and each dataset gets its own named subdirectory containing `img/`, `ann/`, `meta/`, and `osm/`. For overlay projects an additional `overlay/` subdirectory is created.

```
📂 export_root/
├── 📄 meta.json
└── 📁 dataset_name/
    ├── 📂 img/
    │   ├── 🖼️ image_001.png
    │   └── 🖼️ ...
    ├── 📂 ann/
    │   ├── 📝 image_001.png.json
    │   └── 📝 ...
    ├── 📂 meta/
    │   ├── 📝 image_001.png.json
    │   └── 📝 ...
    └── 📂 osm/
        ├── 🗺️ image_001.png.osm
        └── 🗺️ ...
```

When the project uses the **overlay** interface, each dataset also contains:

```
    └── 📂 overlay/
        └── 📁 image_001/
            └── 🖼️ image_001_dtm.png
```

When exporting a **project with multiple datasets**, each dataset gets its own subdirectory under the same root:

```
📂 export_root/
├── 📄 meta.json
├── 📁 dataset_name_1/
│   ├── 📂 img/
│   ├── 📂 ann/
│   ├── 📂 meta/
│   └── 📂 osm/
└── 📁 dataset_name_2/
    ├── 📂 img/
    ├── 📂 ann/
    ├── 📂 meta/
    └── 📂 osm/
```

The archive is a `.tar` file uploaded to Team Files under `/slyosm/osm_exports/` by default. The output path can be changed by setting the `FOLDER` environment variable before launching the app.

### meta.json

The top-level `meta.json` is the standard Supervisely project metadata file. It contains:

- **`classes`** — all object class definitions (name, shape, color) used in the annotations.
- **`tags`** — all tag schemas defined for the project (value type, color, allowed values).
- **`projectType`** — always `"images"` for datasets produced by the downloader.
- **`projectSettings`** — interface configuration for the labeling tool:
  - For **multiview** projects: `multiView.enabled`, `multiView.tagName`, and `multiView.tagId` record which tag links paired satellite and DTM images together.
  - For **overlay** projects: `labelingInterface` is set to `"overlay"`, telling Supervisely to render the DTM layer on top of the satellite image.
  - For **default** projects (satellite-only or DTM-only): `labelingInterface` is `"default"`.

This file is required for a valid Supervisely project import. Without it, Supervisely cannot reconstruct class colors, tag schemas, or the correct labeling interface.

### meta/ — Per-Image Geospatial Metadata

Each file in `meta/` is named `{image_name}.json` and contains the metadata dictionary stored on that image in Supervisely. For images produced by the downloader, this always includes a `geo` object with:

- **`crs_wkt`** — the coordinate reference system of the tile in WKT format, used to convert pixel coordinates to a local projected CRS.
- **`pixel_to_local_h`** — the 3×3 homography matrix that maps pixel coordinates to the local CRS.
- **`bbox_lonlat`** — the geographic bounding box of the tile (longitude/latitude of all four corners).
- **`scene_id`**, **`rotation_deg`**, **`tile_size_m`** — scene identity and acquisition parameters.
- **`multiview_layer`** — present in multiview projects; value is `"satellite"` or `"dtm"`, identifying each image's role in the pair.

This metadata is what makes OSM export possible: the coordinate transform encoded here is what reprojects annotated pixel polygons back to real-world longitude/latitude. Preserving it in the export means the archive is self-contained — it can be re-imported into Supervisely and the OSM export can be run again without any data loss.

### OSM File Format

Each `.osm` file is a standard [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML) file compatible with [JOSM](https://josm.openstreetmap.de/), Overpass API, osmium, and other OSM tooling. Polygon annotations are written as closed ways or multipolygon relations (when holes are present). Line annotations are written as open ways. Point annotations are written as nodes. All IDs are negative (following OSM convention for locally-generated data).

---

## OSM Class Mapping

To project pixel-space annotations back to geographic coordinates correctly, the app needs to know which Supervisely class corresponds to which OSM tag. This mapping is read automatically from the **dataset's custom metadata** — it was written there by the [Satellite, DTM & OSM Downloader](https://ecosystem.supervisely.com/apps/slyosm/import_osm) at import time.

**Fallback order:**
1. Dataset custom metadata (`osm_class_specs` key) — used when the dataset was created by the import app.
2. Per-image metadata — used for images uploaded with an embedded class spec.
3. Built-in default mapping — used when neither of the above is present.

This means you can export any dataset created by the import app without any extra configuration. If you are exporting a manually annotated dataset, ensure your class names match the built-in defaults or embed a custom mapping in the dataset custom data under the key `osm_class_specs`.

---

## How To Run

The app runs headlessly and requires no user interaction after launch.

### From a Dataset

In the Supervisely workspace, right-click the target dataset → **Run App** → **Export to OSM Format**.

### From a Project

Right-click the project → **Run App** → **Export to OSM Format**. All datasets inside the project are exported into a single archive.

The task log shows per-dataset progress, which images have OSM output, and any skipped images with reasons. When the task completes, the archive path in Team Files is printed to the log.

---

## License and Attribution

### OpenStreetMap

OSM XML files produced by this app contain data derived from [OpenStreetMap](https://www.openstreetmap.org/). The underlying OSM data is © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and is licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

**You are required to credit OpenStreetMap** in any public-facing product, publication, or dataset that includes files exported by this app. The standard attribution is:

> © OpenStreetMap contributors

Full requirements are described at [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright). Compliance is your responsibility.
