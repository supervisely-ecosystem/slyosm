from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import osmnx as ox
import supervisely as sly
from pydtmdl import ImageryProvider
from supervisely.app.widgets import (
    Button,
    Card,
    Container,
    DestinationProject,
    Editor,
    Field,
    Input,
    InputNumber,
    Progress,
    RadioTabs,
    SelectString,
    Switch,
    Text,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from import_osm.src.slyosm.geometry import SceneRequest, build_grid_scenes
from import_osm.src.slyosm.osm_config import (
    load_osm_class_specs,
    parse_osm_class_specs,
    serialize_osm_class_specs,
)
from import_osm.src.slyosm.sampling import (
    build_area_polygon_wgs84,
    generate_random_coordinates,
)
from import_osm.src.slyosm.scene_downloader import (
    INTERFACE_MODE_MULTIVIEW,
    INTERFACE_MODE_ONLY_DTM,
    INTERFACE_MODE_ONLY_SATELLITE,
    INTERFACE_MODE_OVERLAY,
    TARGET_GEOMETRY_MASK,
    TARGET_GEOMETRY_POLYGON,
    ensure_project_and_dataset,
    process_scenes,
    save_dataset_custom_data,
)
from import_osm.src.slyosm.settings import (
    OSM_CLASSES_PATH,
    build_generated_name,
    ensure_data_directories,
    load_environment,
    sanitize_filename,
)

APP_DIR = Path(__file__).resolve().parents[1]
MODE_COORDINATES = "Coordinates"
MODE_RANDOM = "Random"
MODE_GRID = "Grid"
AUTO_PROVIDER_VALUE = "auto"
INTERFACE_MODE_LABELS = {
    INTERFACE_MODE_MULTIVIEW: "Multiview (default)",
    INTERFACE_MODE_OVERLAY: "Overlay",
    INTERFACE_MODE_ONLY_SATELLITE: "Only Satellite",
    INTERFACE_MODE_ONLY_DTM: "Only DTM",
}
TARGET_GEOMETRY_LABELS = {
    TARGET_GEOMETRY_POLYGON: "Polygons (default)",
    TARGET_GEOMETRY_MASK: "Masks",
}


class DownloaderState(object):
    """Mutable app state shared between button callbacks."""

    def __init__(self) -> None:
        self.is_running = False
        self.stop_requested = False


load_environment(APP_DIR / "local.env")
ensure_data_directories()
ox.settings.use_cache = True

api = sly.Api.from_env()
workspace_id = sly.env.workspace_id()
default_osm_specs_text = serialize_osm_class_specs(
    load_osm_class_specs(OSM_CLASSES_PATH)
)
state = DownloaderState()


coordinates_input = Input(
    value="",
    placeholder="47.975679, 10.788837; 47.980000, 10.790000",
    type="textarea",
)
coordinates_field = Field(
    title="Center coordinates",
    description="Enter one or multiple center coordinates separated by ';'.",
    content=coordinates_input,
)
coordinates_container = Container([coordinates_field])

random_query_input = Input(value="Germany", placeholder="Germany")
random_count_input = InputNumber(value=10, min=1, max=10000, step=1)
random_query_field = Field(
    title="Area query",
    description="Geocoder query used to resolve the sampling polygon.",
    content=random_query_input,
)
random_count_field = Field(
    title="Images count",
    description="Number of random image centers to sample inside the resolved area.",
    content=random_count_input,
)
random_container = Container([random_query_field, random_count_field])

grid_top_left_input = Input(value="", placeholder="48.000000, 11.000000")
grid_rows_input = InputNumber(value=2, min=1, max=1000, step=1)
grid_cols_input = InputNumber(value=2, min=1, max=1000, step=1)
grid_top_left_field = Field(
    title="Global top-left coordinate",
    description="Top-left corner of the whole contiguous grid footprint.",
    content=grid_top_left_input,
)
grid_rows_field = Field(title="Rows", content=grid_rows_input)
grid_cols_field = Field(title="Columns", content=grid_cols_input)
grid_container = Container(
    [
        grid_top_left_field,
        Container([grid_rows_field, grid_cols_field], direction="horizontal"),
    ]
)

mode_tabs = RadioTabs(
    titles=[MODE_COORDINATES, MODE_RANDOM, MODE_GRID],
    descriptions=[
        "Download one image or a list of explicit center coordinates.",
        "Sample random centers inside an area resolved from an OSM geocoder query.",
        "Build a contiguous grid from a global top-left corner.",
    ],
    contents=[coordinates_container, random_container, grid_container],
)

imagery_provider_classes = sorted(
    ImageryProvider.get_non_base_providers(), key=lambda provider: provider.code()
)
imagery_provider_values = [AUTO_PROVIDER_VALUE] + [
    provider.code() for provider in imagery_provider_classes
]
imagery_provider_labels = ["Auto (best for location)"] + [
    "{name} ({code})".format(name=provider.name(), code=provider.code())
    for provider in imagery_provider_classes
]
imagery_provider_select = SelectString(
    values=imagery_provider_values,
    labels=imagery_provider_labels,
)
imagery_provider_select.set_value(AUTO_PROVIDER_VALUE)
imagery_provider_field = Field(
    title="Imagery provider",
    description="Select a pydtmdl provider or let pydtmdl auto-pick the best provider for each scene.",
    content=imagery_provider_select,
)

interface_mode_values = [
    INTERFACE_MODE_MULTIVIEW,
    INTERFACE_MODE_OVERLAY,
    INTERFACE_MODE_ONLY_SATELLITE,
    INTERFACE_MODE_ONLY_DTM,
]
interface_mode_select = SelectString(
    values=interface_mode_values,
    labels=[INTERFACE_MODE_LABELS[value] for value in interface_mode_values],
)
interface_mode_select.set_value(INTERFACE_MODE_MULTIVIEW)
interface_mode_field = Field(
    title="Supervisely Interface Mode",
    description="Choose how downloaded layers are uploaded and visualized in Supervisely.",
    content=interface_mode_select,
)

target_geometry_values = [TARGET_GEOMETRY_POLYGON, TARGET_GEOMETRY_MASK]
target_geometry_select = SelectString(
    values=target_geometry_values,
    labels=[TARGET_GEOMETRY_LABELS[value] for value in target_geometry_values],
)
target_geometry_select.set_value(TARGET_GEOMETRY_POLYGON)
target_geometry_field = Field(
    title="Target Geometry",
    description="Choose how imported OSM features are uploaded: as polygon objects (default, including buffered roads/points) or as bitmap masks.",
    content=target_geometry_select,
)

tile_size_value = InputNumber(value=1024, min=200, max=20000, step=50, width=220)
tile_size_min = InputNumber(value=600, min=200, max=20000, step=50, width=220)
tile_size_max = InputNumber(value=1400, min=200, max=20000, step=50, width=220)
tile_size_switch = Switch(
    switched=False,
    width=104,
    on_text="Random",
    off_text="Fixed",
)
tile_size_fixed_field = Field(title="Tile size (m)", content=tile_size_value)
tile_size_random_fields = Container(
    [
        Field(title="Random min tile size (m)", content=tile_size_min),
        Field(title="Random max tile size (m)", content=tile_size_max),
    ],
    direction="horizontal",
    fractions=[1, 1],
)
tile_size_controls = Container(
    [
        tile_size_switch,
        tile_size_fixed_field,
        tile_size_random_fields,
    ]
)
tile_size_field = Field(
    title="Tile size",
    description="Use one tile size for all images or randomize it per scene. Grid mode always uses a fixed size.",
    content=tile_size_controls,
)

rotation_single = InputNumber(value=0, min=-90, max=90, step=1, width=220)
rotation_min = InputNumber(value=-20, min=-90, max=90, step=1, width=220)
rotation_max = InputNumber(value=20, min=-90, max=90, step=1, width=220)
rotation_switch = Switch(
    switched=False,
    width=104,
    on_text="Random",
    off_text="Fixed",
)
rotation_fixed_field = Field(title="Rotation (deg)", content=rotation_single)
rotation_random_fields = Container(
    [
        Field(title="Random min rotation (deg)", content=rotation_min),
        Field(title="Random max rotation (deg)", content=rotation_max),
    ],
    direction="horizontal",
    fractions=[1, 1],
)
rotation_controls = Container(
    [
        rotation_switch,
        rotation_fixed_field,
        rotation_random_fields,
    ]
)
rotation_field = Field(
    title="Rotation",
    description="Use one rotation for all images or randomize it per scene. Grid mode always uses a fixed value.",
    content=rotation_controls,
)

download_osm_switch = Switch(
    switched=True, width=96, on_text="OSM ON", off_text="OSM OFF"
)
download_osm_field = Field(
    title="OSM download",
    description="If disabled, only satellite images and geo metadata are uploaded.",
    content=download_osm_switch,
)

osm_editor = Editor(
    initial_text=default_osm_specs_text,
    height_lines=24,
    language_mode="json",
    auto_format=True,
)
osm_editor_field = Field(
    title="osm_classes.json",
    description="Edit the mapping between Supervisely classes and OSM tags. The same mapping is saved to dataset custom data.",
    content=osm_editor,
)

destination = DestinationProject(
    workspace_id=workspace_id, project_type=sly.ProjectType.IMAGES
)
destination_field = Field(
    title="Destination",
    description="Select an existing project or dataset, or create new ones. Blank names are generated automatically.",
    content=destination,
)

start_button = Button("Start download", icon="zmdi zmdi-play")
stop_button = Button(
    "Stop after current image", icon="zmdi zmdi-stop", button_type="danger"
)
stop_button.hide()
progress = Progress(message="Downloading scenes", show_percents=True)
status_text = Text()
status_text.hide()


def _set_status(message: str, status: str = "info") -> None:
    status_text.set(message, status)
    status_text.show()


def _parse_coordinate_pair(raw_value: str) -> Tuple[float, float]:
    parts = [item.strip() for item in raw_value.split(",")]
    if len(parts) != 2:
        raise ValueError(
            "Coordinate '{value}' must contain exactly one comma.".format(
                value=raw_value
            )
        )
    return float(parts[0]), float(parts[1])


def _parse_coordinates(raw_text: str) -> List[Tuple[float, float]]:
    values = [
        item.strip() for item in raw_text.replace("\n", ";").split(";") if item.strip()
    ]
    if len(values) == 0:
        raise ValueError("Provide at least one coordinate pair.")
    return [_parse_coordinate_pair(item) for item in values]


def _sample_tile_sizes(count: int) -> List[int]:
    if not tile_size_switch.is_on():
        return [int(tile_size_value.get_value())] * count

    minimum = int(tile_size_min.get_value())
    maximum = int(tile_size_max.get_value())
    if minimum > maximum:
        raise ValueError("Tile size minimum must be less than or equal to the maximum.")

    rng = np.random.default_rng()
    return [int(value) for value in rng.integers(minimum, maximum + 1, size=count)]


def _sample_rotations(count: int) -> List[int]:
    if not rotation_switch.is_on():
        return [int(rotation_single.get_value())] * count

    minimum = int(rotation_min.get_value())
    maximum = int(rotation_max.get_value())
    if minimum > maximum:
        raise ValueError("Rotation minimum must be less than or equal to the maximum.")

    rng = np.random.default_rng()
    return [int(value) for value in rng.integers(minimum, maximum + 1, size=count)]


def _build_scenes_from_ui() -> List[SceneRequest]:
    active_mode = mode_tabs.get_active_tab()
    selected_provider = imagery_provider_select.get_value()
    imagery_provider = (
        None
        if selected_provider in {None, "", AUTO_PROVIDER_VALUE}
        else selected_provider
    )

    if active_mode == MODE_GRID:
        top_left_lat, top_left_lon = _parse_coordinate_pair(
            grid_top_left_input.get_value()
        )
        rows = int(grid_rows_input.get_value())
        cols = int(grid_cols_input.get_value())
        size_m = int(tile_size_value.get_value())
        rotation_deg = int(rotation_single.get_value())
        prefix = sanitize_filename(build_generated_name("grid"))
        return build_grid_scenes(
            prefix,
            top_left_lat,
            top_left_lon,
            rows,
            cols,
            size_m,
            rotation_deg,
            imagery_provider,
        )

    if active_mode == MODE_COORDINATES:
        coordinates = _parse_coordinates(coordinates_input.get_value())
        prefix = sanitize_filename(build_generated_name("manual"))
    elif active_mode == MODE_RANDOM:
        area_query = random_query_input.get_value().strip()
        if area_query == "":
            raise ValueError("Area query is required for random mode.")
        count = int(random_count_input.get_value())
        coordinates = generate_random_coordinates(
            build_area_polygon_wgs84(area_query), count, np.random.default_rng()
        )
        prefix = sanitize_filename(build_generated_name(area_query))
    else:
        raise ValueError("Unsupported download mode: {mode}".format(mode=active_mode))

    tile_sizes = _sample_tile_sizes(len(coordinates))
    rotations = _sample_rotations(len(coordinates))
    scenes = []
    for index, (lat, lon) in enumerate(coordinates, start=1):
        scenes.append(
            SceneRequest(
                identifier="{prefix}_{index:04d}".format(prefix=prefix, index=index),
                center_lat=float(lat),
                center_lon=float(lon),
                size_m=int(tile_sizes[index - 1]),
                rotation_deg=float(rotations[index - 1]),
                imagery_provider=imagery_provider,
            )
        )
    return scenes


def _resolve_destination(class_specs, polygon_target_geometry: str):
    selected_project_id = destination.get_selected_project_id()
    selected_dataset_id = destination.get_selected_dataset_id()
    project_name = destination.get_project_name().strip() or build_generated_name(
        "import_osm"
    )
    dataset_name = destination.get_dataset_name().strip() or build_generated_name(
        "download"
    )
    return ensure_project_and_dataset(
        api=api,
        workspace_id=workspace_id,
        class_specs=class_specs,
        project_id=selected_project_id,
        project_name=project_name,
        dataset_id=selected_dataset_id,
        dataset_name=dataset_name,
        polygon_target_geometry=polygon_target_geometry,
    )


def _update_tile_size_ui() -> None:
    if tile_size_switch.is_on():
        tile_size_fixed_field.hide()
        tile_size_random_fields.show()
    else:
        tile_size_fixed_field.show()
        tile_size_random_fields.hide()


def _update_rotation_ui() -> None:
    if rotation_switch.is_on():
        rotation_fixed_field.hide()
        rotation_random_fields.show()
    else:
        rotation_fixed_field.show()
        rotation_random_fields.hide()


def _update_mode_ui() -> None:
    is_grid_mode = mode_tabs.get_active_tab() == MODE_GRID
    if is_grid_mode:
        tile_size_switch.off()
        tile_size_switch.disable()
        rotation_switch.off()
        rotation_switch.disable()
    else:
        tile_size_switch.enable()
        rotation_switch.enable()

    _update_tile_size_ui()
    _update_rotation_ui()


@mode_tabs.value_changed
def on_mode_changed(_: str) -> None:
    _update_mode_ui()


@tile_size_switch.value_changed
def on_tile_size_mode_changed(_: bool) -> None:
    _update_tile_size_ui()


@rotation_switch.value_changed
def on_rotation_mode_changed(_: bool) -> None:
    _update_rotation_ui()


@stop_button.click
def request_stop() -> None:
    if not state.is_running:
        return
    state.stop_requested = True
    _set_status(
        "Stop requested. The app will stop after the current image finishes.", "warning"
    )


@start_button.click
def start_download() -> None:
    if state.is_running:
        return

    try:
        state.is_running = True
        state.stop_requested = False
        start_button.hide()
        stop_button.show()
        _set_status("Preparing download job...", "info")

        class_specs = parse_osm_class_specs(osm_editor.get_text())
        scenes = _build_scenes_from_ui()
        download_osm = download_osm_switch.is_on()
        interface_mode = interface_mode_select.get_value()
        polygon_target_geometry = target_geometry_select.get_value()

        _, dataset_info, project_meta = _resolve_destination(
            class_specs,
            polygon_target_geometry,
        )
        save_dataset_custom_data(api, dataset_info.id, class_specs)

        with progress(message="Downloading scenes", total=len(scenes)) as progress_bar:
            results, failures, stopped = process_scenes(
                api=api,
                dataset_id=dataset_info.id,
                project_meta=project_meta,
                class_specs=class_specs,
                scenes=scenes,
                interface_mode=interface_mode,
                polygon_target_geometry=polygon_target_geometry,
                download_osm=download_osm,
                should_stop=lambda: state.stop_requested,
                progress_callback=lambda _index, _total, _scene: progress_bar.update(1),
            )

        if len(failures) == 0 and not stopped:
            _set_status(
                "Uploaded {count} scene(s) to dataset '{dataset}'.".format(
                    count=len(results),
                    dataset=dataset_info.name,
                ),
                "success",
            )
        else:
            _set_status(
                "Uploaded {uploaded}/{planned} scene(s) to dataset '{dataset}'. Failures: {failures}. Stopped: {stopped}.".format(
                    uploaded=len(results),
                    planned=len(scenes),
                    dataset=dataset_info.name,
                    failures=len(failures),
                    stopped=stopped,
                ),
                "warning",
            )
    except Exception as exc:
        sly.logger.exception("Downloader app failed.")
        _set_status(str(exc), "error")
    finally:
        state.is_running = False
        state.stop_requested = False
        stop_button.hide()
        start_button.show()


mode_card = Card(
    title="1. Download Mode",
    description="Choose how scene centers should be produced.",
    content=mode_tabs,
)
settings_card = Card(
    title="2. Settings",
    description="Configure image size, rotation, OSM classes, and whether annotations should be downloaded.",
    content=Container(
        [
            imagery_provider_field,
            interface_mode_field,
            target_geometry_field,
            tile_size_field,
            rotation_field,
            download_osm_field,
            osm_editor_field,
        ]
    ),
)
run_card = Card(
    title="3. Destination And Run",
    description="Choose where images should be uploaded and start or stop the download process.",
    content=Container(
        [
            destination_field,
            Container([start_button, stop_button], direction="horizontal"),
            progress,
            status_text,
        ]
    ),
)

layout = Container([mode_card, settings_card, run_card])
app = sly.Application(layout=layout)

_update_mode_ui()
