"""Interactive Viser workspace for dataset selection and publication figures."""
from __future__ import annotations

import copy
import io
import logging
import math
import threading
import zipfile
from pathlib import Path

import numpy as np

from visualization.data import VisualizationRepository, load_result, save_result
from visualization.export import export_figure, keyframe_constraints, method_color, method_label, render_preview, scene_bounds

LOGGER = logging.getLogger(__name__)
MODEL_NAMES = {"lens_craft": "LensCraft", "et": "E.T.", "ccdm": "CCDM", "gendop": "GenDoP"}
MODES = {"Text only": "prompt_generation", "Text + keyframes": "key_framing+prompt",
         "Keyframes only": "key_framing", "Reconstruction": "reconstruction"}


def keyframe_numbers(value: str) -> list[int] | None:
    try:
        numbers = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        raise ValueError("Use comma-separated frame numbers, e.g. 0,14,29.") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("Frame numbers must be non-negative.")
    return numbers or None


def rgb(name: str) -> tuple[int, int, int]:
    color = method_color(name).lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class VisualizationApp:
    def __init__(self, samples, *, repository=None, host="127.0.0.1", port=8080, output_dir=Path("qualitative")):
        import viser
        self.server = viser.ViserServer(host=host, port=port, label="LensCraft · Qualitative Studio")
        self.server.gui.configure_theme(control_layout="fixed", control_width="large", dark_mode=False,
                                        show_logo=False, show_share_button=False, brand_color=(21, 128, 122))
        self.server.scene.set_up_direction("+y")
        self.repository = repository or VisualizationRepository()
        self.samples = list(samples)
        self.sample = self.samples[0]
        self.collection = []
        self.output_dir = Path(output_dir)
        self.lock = threading.RLock()
        self.busy = False
        self.syncing = False
        self.closed = threading.Event()
        self.method_controls = []
        self.dynamic_handles = []
        self.center = np.zeros(3)
        self.extent = 5.0
        self._make_controls()
        self._select(self.sample)
        self._fit(self.server.initial_camera)

        @self.server.on_client_connect
        def connected(client):
            self._fit(client)

    def _make_controls(self):
        gui = self.server.gui
        gui.add_markdown("# LensCraft\n**Qualitative Studio** · Explore, compare, compose.")
        self.status = gui.add_markdown("Ready")
        tabs = gui.add_tab_group()
        with tabs.add_tab("Input"):
            self.summary = gui.add_markdown("")
            with gui.add_folder("Dataset", expand_by_default=True):
                self.dataset = gui.add_dropdown("Dataset", ("simulation", "et", "ccdm"),
                                                initial_value=self.sample.dataset if self.sample.dataset in ("simulation", "et", "ccdm") else "simulation")
                self.split = gui.add_dropdown("Split", ("test", "val", "train", "all"))
                self.data_path = gui.add_text("Dataset path", "", hint="Optional. Leave empty to use project configuration.")
                self.sample_index = gui.add_number("Sample index", initial_value=0, min=0, step=1)
                self.sample_id = gui.add_text("Sample ID", "", hint="Optional exact ID or filename; overrides index.")
                gui.add_button("Load sample", color="teal").on_click(lambda _: self._action(self._load_dataset))
                navigation = gui.add_button_group("Browse", ("Previous", "Next"))
                navigation.on_click(lambda e: self._action(lambda: self._navigate(-1 if e.target.value == "Previous" else 1)))
                self.dataset_info = gui.add_markdown("Indices start at 0. Use All to select a file outside the test split.")
            with gui.add_folder("Saved results"):
                self.result_path = gui.add_text("Result file", "", hint="Comparison bundle or inference_result.json")
                self.result_index = gui.add_number("Result index", initial_value=0, min=0, step=1)
                gui.add_button("Open result").on_click(lambda _: self._action(self._load_result))
                self.opened = gui.add_dropdown("Opened samples", [self._sample_label(i, s) for i, s in enumerate(self.samples)])
                self.opened.on_update(lambda e: self._choose_opened(e.target.value))
            with gui.add_folder("Generate model outputs", expand_by_default=True):
                self.models = {key: gui.add_checkbox(label, initial_value=(key == "lens_craft")) for key, label in MODEL_NAMES.items()}
                self.mode = gui.add_dropdown("Conditioning", tuple(MODES))
                self.input_keyframes = gui.add_text("Input keyframes", "", hint="Comma-separated input frame indices. Empty uses default keyframes.", visible=False)
                self.mode.on_update(lambda _: self._conditioning_changed())
                self.seed = gui.add_number("Seed", initial_value=self.repository.seed, min=0, step=1)
                gui.add_button("Generate selected models", color="teal").on_click(lambda _: self._action(self._generate))
                gui.add_markdown("Checkpoints use your project configuration. Baselines accept text conditioning.")
        with tabs.add_tab("Compare"):
            self.prompt = gui.add_markdown("")
            self.layout = gui.add_dropdown("Layout", ("Side by side", "Overlay"))
            self.show_subject = gui.add_checkbox("Subject and motion", initial_value=True)
            self.show_keyframes = gui.add_checkbox("Input keyframes", initial_value=True)
            self.show_grid = gui.add_checkbox("Ground grid", initial_value=True)
            self.camera_count = gui.add_slider("Camera poses", min=0, max=16, step=1, initial_value=6)
            self.methods_folder = gui.add_folder("Visible trajectories", expand_by_default=True)
            self.frame = gui.add_slider("Time (%)", min=0, max=100, step=1, initial_value=0)
            self.play = gui.add_checkbox("Play", initial_value=False)
            self.frame_info = gui.add_markdown("")
            self.frame.on_update(lambda _: self._update_frame())
            gui.add_button("Fit view").on_click(lambda _: self._fit_all())
            gui.add_button("Download viewport PNG").on_click(lambda e: self._action(lambda: self._capture(e.client)))
            for control in (self.layout, self.show_subject, self.show_keyframes, self.show_grid, self.camera_count):
                control.on_update(lambda _: self._redraw())
        with tabs.add_tab("Paper"):
            gui.add_markdown("**Compose your figure**\nAdd samples as rows. Every method in a row uses the same view and scale.")
            gui.add_button("Add current sample to figure", color="teal").on_click(lambda _: self._add_row())
            gui.add_button("Clear figure rows").on_click(lambda _: self._clear_rows())
            self.rows = gui.add_markdown("No saved rows. The current sample will be exported.")
            self.figure_name = gui.add_text("Filename", "comparison")
            self.figure_title = gui.add_text("Figure title", "")
            self.include_input = gui.add_checkbox("Separate input keyframe panel", initial_value=True)
            self.elev = gui.add_slider("Elevation", min=-85, max=85, step=1, initial_value=24)
            self.azim = gui.add_slider("Azimuth", min=-180, max=180, step=1, initial_value=-58)
            self.dpi = gui.add_dropdown("PNG resolution", ("300", "600", "150"))
            gui.add_button("Preview figure").on_click(lambda _: self._action(self._preview))
            self.preview = gui.add_image(np.full((8, 16, 3), 255, dtype=np.uint8), label="Figure preview", visible=False)
            gui.add_button("Export PNG + PDF + SVG", color="teal").on_click(lambda e: self._action(lambda: self._export(e.client)))
            gui.add_button("Save comparison data").on_click(lambda e: self._action(lambda: self._save(e.client)))
            self.export_info = gui.add_markdown(f"Output folder: `{self.output_dir.resolve()}`")

    @staticmethod
    def _sample_label(index, sample):
        return f"{index + 1}. {sample.dataset} · {sample.sample_id}"

    def _action(self, callback):
        with self.lock:
            if self.busy:
                return
            self.busy = True
        self.status.content = "**Working…**"
        self.play.value = False
        try:
            callback()
            errors = self.sample.metadata.get("errors")
            failed = self.sample.metadata.get("last_run", {}).get("generated_methods") == []
            self.status.content = "**Ready**" if not errors else ("**Generation failed**" if failed else "**Some outputs are unavailable**") + "\n\n" + str(errors)
        except Exception as exc:
            LOGGER.exception("Visualizer operation failed")
            self.status.content = f"**Could not complete the action**\n\n{exc}"
        finally:
            self.busy = False

    def _load_dataset(self):
        data_path = self.data_path.value.strip() or None
        sample = self.repository.load_sample(self.dataset.value, self.split.value, index=int(self.sample_index.value),
                                             sample_id=self.sample_id.value.strip() or None, data_path=data_path)
        count = len(self.repository.list_samples(self.dataset.value, self.split.value, data_path=data_path))
        self.dataset_info.content = f"**{count} samples** · {self.dataset.value} / {self.split.value}"
        self._remember(sample)

    def _navigate(self, delta):
        self.sample_id.value = ""
        self.sample_index.value = max(0, int(self.sample_index.value) + delta)
        self._load_dataset()

    def _load_result(self):
        self._remember(load_result(Path(self.result_path.value).expanduser(), index=int(self.result_index.value)))

    def _remember(self, sample):
        self.samples.append(sample)
        self.syncing = True
        self.opened.options = [self._sample_label(i, s) for i, s in enumerate(self.samples)]
        self.opened.value = self.opened.options[-1]
        self.syncing = False
        self._select(sample)

    def _choose_opened(self, value):
        if not self.syncing and not self.busy:
            self._select(self.samples[int(value.split(".", 1)[0]) - 1])

    def _conditioning_changed(self):
        self.input_keyframes.visible = "key_framing" in MODES[self.mode.value]

    def _generate(self):
        models = [key for key, control in self.models.items() if control.value]
        if not models:
            raise ValueError("Select at least one model.")
        sample = self.repository.generate(self.sample, models=models, mode=MODES[self.mode.value],
                                          keyframes=keyframe_numbers(self.input_keyframes.value), seed=int(self.seed.value))
        self._remember(sample)

    def _select(self, sample):
        with self.lock:
            self.sample = sample
            if sample.dataset in ("simulation", "et", "ccdm"):
                self.dataset.value = sample.dataset
                self.split.value = sample.metadata.get("split", "test")
                self.sample_index.value = sample.metadata.get("index", 0)
                self.sample_id.value = ""
                self.data_path.value = str(sample._context[1].get("data_path") or "") if sample._context else ""
            mode = sample.metadata.get("mode")
            if mode in MODES.values():
                self.mode.value = next(label for label, key in MODES.items() if key == mode)
            self.input_keyframes.value = ",".join(map(str, sample.keyframes))
            self.seed.value = sample.metadata.get("seed", self.repository.seed)
            self.play.value = False
            self.frame.value = 0
            self.prompt.content = f"**{sample.dataset} · {sample.sample_id}**\n\n{sample.prompt}"
            demo = "\n\n**Synthetic demo — illustrative paths, not model results.**" if sample.dataset == "demo" or sample.metadata.get("demo") else ""
            self.summary.content = f"**{sample.sample_id}**\n\n{sample.prompt}{demo}"
            for _, control in self.method_controls:
                control.remove()
            self.method_controls = []
            with self.methods_folder:
                for name in sample.trajectories:
                    checkbox = self.server.gui.add_checkbox(method_label(name), initial_value=True)
                    checkbox.on_update(lambda _: self._redraw())
                    self.method_controls.append((name, checkbox))
            self._redraw()
            self._fit_all()

    def _visible(self):
        return [name for name, control in self.method_controls if control.value]

    def _line(self, name, points, color, width=2.0):
        if len(points) > 1:
            self.server.scene.add_line_segments(name, np.stack((points[:-1], points[1:]), axis=1).astype(np.float32),
                                                colors=color, thickness=width, thickness_units="screen")

    def _pose(self, name, pose, color, scale, offset):
        from viser.transforms import SO3
        return self.server.scene.add_camera_frustum(name, fov=math.radians(50), aspect=16 / 9, scale=scale,
                                                   color=color, wxyz=SO3.from_matrix(pose[:3, :3]).wxyz,
                                                   position=pose[:3, 3] + offset, thickness=1.5,
                                                   thickness_units="screen", cast_shadow=False)

    def _redraw(self):
        with self.lock:
            self.server.scene.reset()
            self.dynamic_handles = []
            names = self._visible()
            if not names:
                return
            sample = self.sample
            bounds = scene_bounds(sample, names, show_keyframes=self.show_keyframes.value)
            lo, hi = bounds[:, 0], bounds[:, 1]
            extent = max(float(np.linalg.norm(hi - lo)), 1.0)
            spacing = extent * 1.15 if self.layout.value == "Side by side" else 0.0
            scale = extent * 0.045
            self.extent = max(extent, spacing * len(names) * 0.55)
            self.center = (lo + hi) / 2 + np.array([spacing * (len(names) - 1) / 2, 0, 0])
            if self.show_grid.value:
                self.server.scene.add_grid("/ground", width=max(10, spacing * len(names) + extent), height=max(10, extent * 2),
                                           plane="xz", position=(self.center[0], lo[1] - extent * 0.1, self.center[2]),
                                           cell_size=extent / 6, section_size=extent / 3,
                                           cell_color=(236, 240, 243), section_color=(219, 228, 233),
                                           fade_distance=extent * 3)
            from viser.transforms import SO3
            for column, name in enumerate(names):
                path = sample.trajectories[name]
                offset = np.array([column * spacing, 0, 0])
                base = f"/comparison/m{column}"
                color = rgb(name)
                self._line(base + "/path", path[:, :3, 3] + offset, color, 3)
                for index in np.unique(np.linspace(0, len(path) - 1, int(self.camera_count.value), dtype=int)):
                    self._pose(f"{base}/poses/{index}", path[index], color, scale, offset)
                self.server.scene.add_icosphere(base + "/start", radius=scale * .18, color=color, position=path[0, :3, 3] + offset)
                with self.server.scene.add_3d_gui_container(base + "/title", visible=self.layout.value == "Side by side", position=((lo[0] + hi[0]) / 2 + offset[0], hi[1] + extent * .2, self.center[2])):
                    self.server.gui.add_markdown(f"**{method_label(name)}**")
                cursor = self._pose(base + "/current", path[0], color, scale * 1.4, offset)
                self.dynamic_handles.append((cursor, path, offset))
                if self.show_subject.value and sample.subject is not None:
                    self._line(base + "/subject-path", sample.subject[:, :3, 3] + offset, (168, 177, 185), 1)
                    dims = np.array([.5, 1.7, .35]) if sample.volume is None else np.asarray(sample.volume).reshape(-1)[:3]
                    subject = self.server.scene.add_box(base + "/subject", dimensions=tuple(np.maximum(dims, .01)),
                                                        color=(135, 153, 168), opacity=.8,
                                                        position=sample.subject[0, :3, 3] + offset,
                                                        wxyz=SO3.from_matrix(sample.subject[0, :3, :3]).wxyz)
                    self.dynamic_handles.append((subject, sample.subject, offset))
                if self.show_keyframes.value:
                    indices, poses, _ = keyframe_constraints(sample, name)
                    for k, pose in zip(indices, poses if poses is not None else []):
                        self._pose(f"{base}/keyframes/{k}", pose, (221, 169, 37), scale * 1.3, offset)
                        self.server.scene.add_icosphere(f"{base}/constraints/{k}", radius=scale * .24,
                                                       color=(238, 185, 46), position=pose[:3, 3] + offset)
            self._update_frame()

    def _update_frame(self):
        from viser.transforms import SO3
        with self.lock:
            for handle, poses, offset in self.dynamic_handles:
                index = round(float(self.frame.value) / 100 * (len(poses) - 1))
                handle.position = poses[index, :3, 3] + offset
                handle.wxyz = SO3.from_matrix(poses[index, :3, :3]).wxyz
            frames = ", ".join(f"{method_label(name)}: {round(self.frame.value / 100 * (len(path) - 1))}/{len(path) - 1}"
                               for name, path in self.sample.trajectories.items() if name in self._visible())
            self.frame_info.content = frames + ("\n\nInput keyframes: " + ", ".join(map(str, self.sample.keyframes)) if self.sample.keyframes else "")

    def _fit(self, client):
        camera = getattr(client, "camera", client)
        if hasattr(camera, "up_direction"):
            camera.up_direction = (0, 1, 0)
        else:
            camera.up = (0, 1, 0)
        # Viser translates look_at together with position; set the target last.
        camera.position = self.center + self.extent * np.array([0, .75, 1.05])
        camera.look_at = self.center

    def _fit_all(self):
        for client in self.server.get_clients().values():
            self._fit(client)

    def _capture(self, client):
        if client is None:
            raise ValueError("Open the viewer in a browser before taking a viewport screenshot.")
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(client.get_render(height=1440, width=2560)).save(output, format="PNG")
        client.send_file_download("lenscraft-viewport.png", output.getvalue(), save_immediately=True)

    def _add_row(self):
        visible = self._visible()
        if not visible:
            self.status.content = "Select at least one visible trajectory before adding a row."
            return
        sample = copy.copy(self.sample)
        sample._context = None
        sample.metadata = copy.deepcopy(sample.metadata)
        _, poses, _ = keyframe_constraints(sample)
        if poses is not None:
            sample.metadata["keyframe_poses"] = poses.tolist()
        sample.metadata["figure_methods"] = visible
        self.collection.append(copy.deepcopy(sample))
        self.rows.content = "\n\n".join(f"**{i + 1}.** {row.dataset} · {row.sample_id}" for i, row in enumerate(self.collection))

    def _clear_rows(self):
        self.collection.clear()
        self.rows.content = "No saved rows. The current sample will be exported."

    def _figure_options(self):
        names = self._visible()
        if not names and not self.collection:
            raise ValueError("Select at least one visible trajectory in Compare.")
        if self.collection:
            names = list(dict.fromkeys(name for row in self.collection for name in row.metadata["figure_methods"]))
        return dict(methods=names, show_keyframes=self.show_keyframes.value, include_input=self.include_input.value,
                    elev=self.elev.value, azim=self.azim.value, camera_count=int(self.camera_count.value),
                    title=self.figure_title.value or None)

    def _figure_samples(self):
        if not self.collection:
            return [self.sample]
        rows = []
        for row in self.collection:
            selected = copy.copy(row)
            selected.trajectories = {name: row.trajectories[name] for name in row.metadata["figure_methods"]}
            rows.append(selected)
        return rows

    def _preview(self):
        self.preview.image = render_preview(self._figure_samples(), **self._figure_options())
        self.preview.visible = True

    def _export(self, client):
        name = Path(self.figure_name.value.strip()).name
        if not name or name in (".", ".."):
            raise ValueError("Enter a filename for the figure.")
        files = export_figure(self._figure_samples(), self.output_dir / name,
                              dpi=int(self.dpi.value), **self._figure_options())
        self.export_info.content = "Saved:\n\n" + "\n\n".join(f"`{path.resolve()}`" for path in files.values())
        if client is not None:
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in files.values():
                    archive.write(path, path.name)
            client.send_file_download(f"{Path(name).stem}.zip", output.getvalue(), save_immediately=True)

    def _save(self, client):
        path = save_result(self.collection or [self.sample], self.output_dir / "comparison-data.json")
        self.export_info.content = f"Comparison saved to `{path.resolve()}`."
        if client is not None:
            client.send_file_download(path.name, path.read_bytes(), save_immediately=True)

    def run(self):
        try:
            while not self.closed.wait(0.05):
                if self.play.value and not self.busy:
                    self.frame.value = (self.frame.value + 1) % 101
        except KeyboardInterrupt:
            pass
        finally:
            self.closed.set()
            self.server.stop()
