# maps4fssly

End-to-end script to:
1. Download satellite imagery with `pygmdl` (zoom 18).
2. Fetch OSM features with `osmnx`.
3. Convert them to instance masks.
4. Upload image + annotation + geospatial metadata to Supervisely.

## Setup

1. Install dependencies:

	```
	pip install -r requirements.txt
	```

2. Configure Supervisely auth in `~/supervisely.env`:
	- `SERVER_ADDRESS`
	- `API_TOKEN`

3. Configure workspace info in `local.env`:
	- `TEAM_ID`
	- `WORKSPACE_ID`
	- optional `PROJECT_ID` to upload into an existing project

4. Edit class-to-OSM mapping in `src/osm_classes.json` if needed.

5. Edit scene list in `src/main.py` (`SCENES`) for batch processing.

## Run

```
python src/main.py
```

Generated files:
- Downloaded images: `data/images/`
- Per-image metadata JSON: `data/meta/`

## Production Generator (OSM -> Supervisely)

Use `src/osm_to_sly.py` for production pre-generation of training data.

This script is non-CLI by design. Configure values at the top of the file:
- `PROJECT_NAME` (default: `Training Data (RAW)`)
- `COUNTRY_RUNS` (for example, Germany -> dataset `germany`, `target_images=1000`)
- sampling/randomization constants

Behavior:
- Randomly samples unique coordinates inside each configured country boundary.
- Prevents duplicate sampled coordinates with persistent state files in `data/meta/sampling/`.
- Uploads each sample as image + masks + metadata into the target Supervisely dataset.

Run:

```
python src/osm_to_sly.py
```

## Reverse Export (Supervisely -> OSM)

Use `src/sly_to_osm.py` to reconstruct an OSM XML from a Supervisely image annotation.

It reads:
- image annotation (instance masks / predictions)
- image geo metadata (`meta.geo`) uploaded with the image
- class mapping (`osm_class_specs`) from image metadata, or fallback to `src/osm_classes.json`

Run:

```
python src/sly_to_osm.py
```

Before running, set values at the top of `src/sly_to_osm.py`:
- `TEAM_ID`
- `WORKSPACE_ID`
- `PROJECT_ID`
- `IMAGE_ID`
- optional `OUTPUT_PATH` (leave empty for auto path in `data/osm/`)