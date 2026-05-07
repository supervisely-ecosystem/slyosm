<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/export-poster.png"/>

# Export to OSM Format

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#output-format">Output Format</a> •
  <a href="#osm-class-mapping">OSM Class Mapping</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#license-and-attribution">License & Attribution</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/slyosm/export_to_osm)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/slyosm)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/slyosm/export_to_osm.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/slyosm/export_to_osm.png)](https://supervisely.com)

</div>

## Overview

This headless app exports a Supervisely dataset — or all datasets in a project — to a structured archive containing:

- **Original images** as downloaded from the imagery provider.
- **Supervisely annotation JSON files** in the standard Supervisely format.
- **OSM XML files** with each annotation projected back to geographic coordinates (longitude/latitude) and written as OSM ways, relations, and nodes with the appropriate OSM tags.

The app runs entirely in the background — no UI is shown. When finished, it uploads the archive to Team Files automatically.

This app is designed as the companion to the [Satellite, DTM & OSM Downloader](../import_osm/README.md). Images created by that app carry embedded geo metadata required for the coordinate projection. Images without geo metadata are still exported to `img/` and `ann/`; the OSM export is skipped for those files with a warning in the task log.

---

## Output Format

For each dataset the archive contains three subdirectories, following standard Supervisely download conventions with one additional folder:

```
img/
    image_001.png
    image_002.png
    ...
ann/
    image_001.png.json
    image_002.png.json
    ...
osm/
    image_001.png.osm
    image_002.png.osm
    ...
```

When exporting at **project level**, each dataset gets its own subdirectory:

```
dataset_name_1/
    img/
    ann/
    osm/
dataset_name_2/
    img/
    ann/
    osm/
```

The archive is a `.tar` file uploaded to Team Files under `/slyosm/osm_exports/` by default. The output path can be changed by setting the `FOLDER` environment variable before launching the app.

### OSM File Format

Each `.osm` file is a standard [OSM XML](https://wiki.openstreetmap.org/wiki/OSM_XML) file compatible with [JOSM](https://josm.openstreetmap.de/), Overpass API, osmium, and other OSM tooling. Polygon annotations are written as closed ways or multipolygon relations (when holes are present). Line annotations are written as open ways. Point annotations are written as nodes. All IDs are negative (following OSM convention for locally-generated data).

---

## OSM Class Mapping

To project pixel-space annotations back to geographic coordinates correctly, the app needs to know which Supervisely class corresponds to which OSM tag. This mapping is read automatically from the **dataset's custom metadata** — it was written there by the [Satellite, DTM & OSM Downloader](../import_osm/README.md) at import time.

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
