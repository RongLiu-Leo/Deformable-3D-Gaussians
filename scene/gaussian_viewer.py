#
# NeRF-style viewer for Deformable 3D Gaussians (viser + nerfview).
# Simplified from Neural-Gaussian: no per-attribute base/residual visualization.
#

import math
import threading
import time
from typing import Callable, Literal, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import viser
from nerfview import Viewer, RenderTabState
from matplotlib import colormaps
from utils.graphics_utils import getProjectionMatrix


class GaussianRenderTabState(RenderTabState):
    """Render tab state for Gaussian Viewer (no attr visualization)."""

    # Read-only display
    total_count_number: int = 0
    rendered_count_number: int = 0

    # Controllable
    timestamp: float = 0.0
    near_plane: float = 0.01
    far_plane: float = 100.0
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Depth", "Normal"] = "RGB"


def build_viewpoint_camera(
    camera_state,
    width: int,
    height: int,
    timestamp: float = 0.0,
    znear: float = 0.01,
    zfar: float = 100.0,
):
    """
    Build a minimal viewpoint camera from nerfview CameraState for use with
    gaussian_renderer.render(). Returns an object with the same interface
    as scene.Camera (image_width, image_height, FoVx, FoVy, world_view_transform,
    full_proj_transform, camera_center) and .fid for time.
    """
    c2w = np.array(camera_state.c2w, dtype=np.float32)
    fov = float(camera_state.fov)
    aspect = float(camera_state.aspect)

    fovy = fov
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * (width / height)) if height else fovy

    w2c = np.linalg.inv(c2w).astype(np.float32)

    world_view_transform = torch.tensor(w2c, dtype=torch.float32).transpose(0, 1).cuda()
    projection_matrix = (
        getProjectionMatrix(znear, zfar, fovx, fovy).transpose(0, 1).cuda()
    )
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).float()
    camera_center = torch.tensor(c2w[:3, 3], dtype=torch.float32, device="cuda")

    class ViewCam:
        pass

    cam = ViewCam()
    cam.image_width = width
    cam.image_height = height
    cam.FoVx = fovx
    cam.FoVy = fovy
    cam.znear = znear
    cam.zfar = zfar
    cam.world_view_transform = world_view_transform
    cam.projection_matrix = projection_matrix
    cam.full_proj_transform = full_proj_transform
    cam.camera_center = camera_center
    cam.fid = torch.tensor([timestamp], dtype=torch.float32, device="cuda")
    return cam


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img."""
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis."""
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img


def depth_to_points(
    depths: torch.Tensor, camtoworlds: torch.Tensor, Ks: torch.Tensor, z_depth: bool = True
) -> torch.Tensor:
    """Convert depth maps to 3D points."""
    assert depths.shape[-1] == 1, f"Invalid depth shape: {depths.shape}"
    device = depths.device
    height, width = depths.shape[-3:-1]

    x, y = torch.meshgrid(
        torch.arange(width, device=device),
        torch.arange(height, device=device),
        indexing="xy",
    )

    fx = Ks[..., 0, 0]
    fy = Ks[..., 1, 1]
    cx = Ks[..., 0, 2]
    cy = Ks[..., 1, 2]

    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cx[..., None, None] + 0.5) / fx[..., None, None],
                (y - cy[..., None, None] + 0.5) / fy[..., None, None],
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )

    directions = torch.einsum(
        "...ij,...hwj->...hwi", camtoworlds[..., :3, :3], camera_dirs
    )
    origins = camtoworlds[..., :3, -1]

    if not z_depth:
        directions = F.normalize(directions, dim=-1)

    points = origins[..., None, None, :] + depths * directions
    return points


def depth_to_normal(
    depths: torch.Tensor, camtoworlds: torch.Tensor, Ks: torch.Tensor, z_depth: bool = True
) -> torch.Tensor:
    """Convert depth maps to surface normals."""
    points = depth_to_points(depths, camtoworlds, Ks, z_depth=z_depth)
    dx = torch.cat(
        [points[..., 2:, 1:-1, :] - points[..., :-2, 1:-1, :]], dim=-3
    )
    dy = torch.cat(
        [points[..., 1:-1, 2:, :] - points[..., 1:-1, :-2, :]], dim=-2
    )
    normals = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    normals = F.pad(normals, (0, 0, 1, 1, 1, 1), value=0.0)
    return normals


class GaussianViewer(Viewer):
    """Viewer for Deformable 3D Gaussian Splatting (no attribute visualization)."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        is_dynamic: bool = True,
        mode: Literal["rendering", "training"] = "training",
        share_url: bool = False,
    ):
        self.is_dynamic = is_dynamic
        super().__init__(server, render_fn, mode=mode)
        server.gui.set_panel_label("Deformable 3D Gaussians Viewer")
        self._playing_time = False
        self._play_thread = None
        self._play_thread_lock = threading.Lock()
        if share_url:
            server.request_share_url()

    def _init_rendering_tab(self):
        self.render_tab_state = GaussianRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _populate_rendering_tab(self):
        with self._rendering_folder:
            if self.is_dynamic:
                self.gui_slider_time = self.server.gui.add_slider(
                    "Time",
                    min=0.0,
                    max=1.0,
                    step=0.0001,
                    initial_value=self.render_tab_state.timestamp,
                    hint="Timestamp for dynamic scene (0–1).",
                )

                @self.gui_slider_time.on_update
                def _(_) -> None:
                    self.render_tab_state.timestamp = self.gui_slider_time.value
                    self.rerender(_)

                self.gui_play_speed = self.server.gui.add_slider(
                    "Play speed (×)",
                    min=0.1,
                    max=4.0,
                    step=0.1,
                    initial_value=1.0,
                    hint="Playback speed for Time slider.",
                )
                self.gui_btn_toggle = self.server.gui.add_button("▶ Play")

                @self.gui_btn_toggle.on_click
                def _toggle(_) -> None:
                    with self._play_thread_lock:
                        self._playing_time = not self._playing_time
                        self._set_play_button_ui(self._playing_time)
                        if self._playing_time:
                            self._play_thread = threading.Thread(
                                target=self._autoplay_time_loop, daemon=True
                            )
                            self._play_thread.start()

            with self.server.gui.add_folder("Render Mode"):
                self.render_mode_dropdown = self.server.gui.add_dropdown(
                    "Mode",
                    ["RGB", "Depth", "Normal"],
                    initial_value=self.render_tab_state.render_mode,
                    hint="RGB, Depth colormap, or Normal map.",
                )

                @self.render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = self.render_mode_dropdown.value
                    self.rerender(_)

                self.total_count_number = self.server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_count_number,
                    disabled=True,
                    hint="Total number of Gaussians.",
                )
                self.rendered_count_number = self.server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_count_number,
                    disabled=True,
                    hint="Number of Gaussians rendered.",
                )
                self.near_far_plane_vec2 = self.server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane.",
                )

                @self.near_far_plane_vec2.on_update
                def _(_) -> None:
                    (
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ) = self.near_far_plane_vec2.value
                    self.rerender(_)

                self.backgrounds_slider = self.server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color (0–255).",
                )

                @self.backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = self.backgrounds_slider.value
                    self.rerender(_)

        self._rendering_tab_handles = {
            "total_count_number": self.total_count_number,
            "rendered_count_number": self.rendered_count_number,
            "near_far_plane_vec2": self.near_far_plane_vec2,
            "render_mode_dropdown": self.render_mode_dropdown,
            "backgrounds_slider": self.backgrounds_slider,
            "play_speed": getattr(self, "gui_play_speed", None),
            "btn_toggle": getattr(self, "gui_btn_toggle", None),
        }
        if self.is_dynamic:
            self._rendering_tab_handles["timestamp"] = self.gui_slider_time
        super()._populate_rendering_tab()

    def _after_render(self):
        self._rendering_tab_handles[
            "total_count_number"
        ].value = self.render_tab_state.total_count_number
        self._rendering_tab_handles[
            "rendered_count_number"
        ].value = self.render_tab_state.rendered_count_number

    def _set_play_button_ui(self, playing: bool):
        label = "⏸ Pause" if playing else "▶ Play"
        btn = getattr(self, "gui_btn_toggle", None)
        if not btn:
            return
        for attr in ("label", "name", "title", "text"):
            if hasattr(btn, attr):
                try:
                    setattr(btn, attr, label)
                    return
                except Exception:
                    pass
        if hasattr(btn, "set_label"):
            try:
                btn.set_label(label)
                return
            except Exception:
                pass

    def _autoplay_time_loop(self):
        dt = 0.01
        while True:
            with self._play_thread_lock:
                if not self._playing_time:
                    self._set_play_button_ui(False)
                    break
            speed = (
                float(getattr(self, "gui_play_speed", 1.0).value)
                if hasattr(self, "gui_play_speed")
                else 1.0
            )
            curr_t = float(self.gui_slider_time.value)
            new_t = (curr_t + speed * dt) % 1.0
            self.gui_slider_time.value = new_t
            self.render_tab_state.timestamp = new_t
            self.rerender(None)
            time.sleep(dt)
