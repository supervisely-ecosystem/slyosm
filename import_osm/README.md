<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/import-poster.png"/>

# Satellite, DTM & OSM Downloader

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#download-modes">Download Modes</a> •
  <a href="#settings">Settings</a> •
  <a href="#interface-modes">Interface Modes</a> •
  <a href="#osm-class-mapping">OSM Class Mapping</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#license-and-attribution">License & Attribution</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/slyosm/import_osm)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/slyosm)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/slyosm/import_osm.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/slyosm/import_osm.png)](https://supervisely.com)

</div>

## Overview

This app downloads three types of geospatial data for a set of geographic locations and uploads them as an annotated Supervisely dataset:

1. **Satellite imagery** — optical tiles from the provider of your choice.
2. **DTM (Digital Terrain Model)** — elevation tiles for the same area.
3. **OpenStreetMap vector features** — roads, buildings, water, forests, and more, fetched from the [OpenStreetMap](https://www.openstreetmap.org/) API and stored as Supervisely object annotations.

All downloaded images carry embedded geo metadata (center coordinates, projection, homography matrix) that the companion [Export to OSM Format](https://ecosystem.supervisely.com/apps/slyosm/export_to_osm) app uses to project annotations back to geographic coordinates and produce `.osm` files ready to open in [JOSM](https://josm.openstreetmap.de/) or any other OSM editor.

### Imagery and Elevation Coverage

Satellite and DTM tiles are downloaded via the **[pydtmdl](https://github.com/iwatkot/pydtmdl)** library. The available providers, their geographic coverage, resolution, and update frequency are determined entirely by pydtmdl and its upstream data sources — see the pydtmdl documentation for details. Coverage questions should be directed there, not to Supervisely.

### OSM Data Coverage

Vector annotation coverage matches whatever OpenStreetMap contributors have mapped for the requested area. Dense urban areas are typically very complete; rural or remote areas may have sparse or missing features. OSM is a community-maintained dataset — gaps are normal.

---

## Download Modes

Select a download mode with the **Download Mode** radio selector at the top of the UI.

### Coordinates

Provide an explicit list of tile center coordinates — one `lat, lon` pair per line in the text area. Each coordinate becomes the center of one downloaded tile.

**Example:**
```
48.8566, 2.3522
51.5074, -0.1278
40.7128, -74.0060
```

Use this mode when you have a specific list of locations (e.g., from a previous sampling run, a field survey, or a customer-provided list).

### Random

Provide a **place name** (any city, region, country, or address that [Nominatim](https://nominatim.openstreetmap.org/) can geocode) and a **tile count**. The app:

1. Geocodes the place name to a boundary polygon.
2. Randomly samples that many tile centers uniformly inside the polygon.
3. Applies a **minimum distance** filter — any candidate closer than N meters to an already-accepted tile is discarded, preventing near-duplicate coverage.

The app persists sampled coordinates between runs so you can incrementally extend an existing dataset by running again with a higher count — previously sampled locations are not re-downloaded.

**Example place names:** `Berlin`, `Tokyo`, `São Paulo`, `New South Wales, Australia`

### Grid

Provide a **top-left corner** (lat/lon), the number of **rows** and **columns**, and the tile size. Tiles are placed contiguously with no gap, forming a seamless mosaic.

Use this mode for complete, structured coverage of a rectangular area — for example, mapping an entire city block by block or building a validation set with known spatial layout.

---

## Settings

### Tile Size (meters)

Ground coverage of each tile in meters (width × height on the ground). Larger tiles cover more area per download but reduce effective spatial resolution for the fixed output pixel size.

**Tip:** For 1024 px output and typical satellite resolution, 500–1000 m tiles give usable feature detail.

### Rotation

Angle in degrees to rotate the tile bounding box before downloading. Two options:

- **Fixed value** — every tile is rotated by the same angle. Useful when you want consistent orientation (e.g., aligning tiles to a road network).
- **`random`** — each tile gets an independent uniformly random rotation angle. This increases geometric diversity in the training set and prevents models from overfitting to north-up imagery.

### Output Resolution

All downloaded images are resized to a fixed pixel dimension on the long side (default: **1024 px**) to keep file sizes and upload times consistent regardless of tile size or provider native resolution.

### OSM Download

Toggle whether OSM vector annotations are fetched alongside imagery. Disable to download imagery only (faster, useful for initial area exploration).

### Target Geometry

Controls how OSM vector features are stored in Supervisely:

| Option | Description |
|---|---|
| **Polygon** (default) | Vector polygons and polylines. Fast, lightweight, editable. |
| **Bitmap (Mask)** | Rasterized binary masks. Overlap between classes is resolved by a fixed priority order (building > water > road_main > road_minor > field > forest). |

---

## Interface Modes

The app can upload satellite and DTM data in four configurations that correspond to different Supervisely labeling toolbox modes:

| Mode | What is uploaded | Supervisely toolbox |
|---|---|---|
| **Multiview** (default) | Satellite image + DTM image as a linked pair | [Multi-view](https://docs.supervisely.com/labeling/labeling-toolbox/multi-view-images) — view both side-by-side in one session |
| **Overlay** | Single image with DTM blended on top of satellite | [Overlay](https://docs.supervisely.com/labeling/labeling-toolbox/overlay) — adjust DTM transparency on the fly |
| **Satellite only** | Satellite image | Standard labeling toolbox |
| **DTM only** | DTM image | Standard labeling toolbox |

In **Multiview** mode you annotate both layers simultaneously — one set of labels is shared across the pair. In **Overlay** mode you work on a single composited image and can tune how much elevation data bleeds through. Use **Satellite only** or **DTM only** when you need a clean single-channel dataset.

---

## OSM Class Mapping

The OSM class mapping defines which OpenStreetMap features are downloaded, how they are named in Supervisely, and how they are projected back to OSM tags on export.

### Structure

The mapping is a JSON array. Each entry describes one Supervisely object class:

```json
[
  {
    "name": "building",
    "geometry": "polygon",
    "tags": { "building": true },
    "default_tag": { "building": "yes" },
    "color": [180, 180, 180]
  },
  {
    "name": "road_main",
    "geometry": "line",
    "tags": { "highway": ["motorway", "trunk", "primary", "secondary", "tertiary"] },
    "default_tag": { "highway": "secondary" },
    "buffer_m": 5.0,
    "color": [240, 93, 66]
  }
]
```

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Supervisely object class name |
| `geometry` | Yes | `"polygon"`, `"line"`, or `"point"` |
| `tags` | Yes | OSM tag filters. Value can be a string, boolean, or list of strings |
| `default_tag` | Yes | Single OSM tag written when re-exporting to `.osm` files |
| `buffer_m` | No | Expand lines/points by this many meters to create a polygon footprint |
| `color` | No | RGB color `[R, G, B]` for the Supervisely class |

### Default Classes

The app ships with six built-in classes:

| Class | Geometry | OSM Tags |
|---|---|---|
| `forest` | polygon | `natural=wood`, `landuse=forest`, `landcover=trees` |
| `field` | polygon | `landuse=farmland/meadow/grass` |
| `road_main` | line (5 m buffer) | `highway=motorway/trunk/primary/secondary/tertiary` |
| `road_minor` | line (3 m buffer) | `highway=residential/unclassified/service/track/road` |
| `water` | polygon | `natural=water` |
| `building` | polygon | `building=*` |

You can edit the JSON in the app UI to add, remove, or modify classes before starting a download.

### How the Mapping is Saved

When the app uploads images to Supervisely, it **saves the active OSM class mapping to the dataset's custom metadata** under the key `osm_class_specs`. This happens automatically — you do not need to do anything.

The companion **Export to OSM Format** app reads this mapping back from the dataset when you export. This means annotations always round-trip using the same class definitions they were created with, even if the default mapping changes in a future app version.

---

## How To Run

**Step 1:** Run the app from the Supervisely Ecosystem or from the workspace Apps panel.<br>

**Step 2:** Select a **Download Mode** and fill in the corresponding parameters (coordinates, place name, or grid settings).<br>

<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/scr1e.png"/><br>

**Step 3:** Configure **Tile Size**, **Rotation**, **Interface Mode**, **Target Geometry**, and the **OSM class mapping** JSON as needed.<br>

<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/scr2e.png"/><br>

**Step 4:** Click **Run** to start the download. The app uploads each tile as it finishes — you can open the dataset and start labeling while the download is still in progress.<br>

<img src="https://github.com/supervisely-ecosystem/slyosm/releases/download/0.0.1/scr3e.png"/><br>

---

## License and Attribution

### OpenStreetMap

OSM vector data used by this app is © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) and is made available under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

**You are required to credit OpenStreetMap** in any public-facing product, publication, or dataset that includes data downloaded by this app. The standard attribution is:

> © OpenStreetMap contributors

Full requirements are described at [openstreetmap.org/copyright](https://www.openstreetmap.org/copyright). Supervisely is not a party to that license — compliance is your responsibility.

### Satellite and DTM Imagery (pydtmdl)

Satellite and DTM tiles are fetched via the **[pydtmdl](https://github.com/iwatkot/pydtmdl)** library. **Supervisely has no affiliation with pydtmdl and takes no responsibility whatsoever for the imagery sources it connects to.** Coverage, data quality, licensing, and terms of use vary by provider and are determined solely by pydtmdl and its upstream sources.

Before using imagery downloaded through this app for any purpose beyond personal experimentation, it is entirely **your responsibility** to:

- Identify which data source the selected provider uses.
- Verify that you have the legal right to download and use that data for your intended purpose.
- Comply with the data source's license, attribution requirements, and usage restrictions.

Questions about imagery coverage, licensing, or provider behavior should be directed to pydtmdl and the relevant upstream providers — not to Supervisely.
