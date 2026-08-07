#!/usr/bin/env python3
"""3D Gaussian Splat viewer with click-to-measure distance tool.

Loads a 3D Gaussian Splatting .ply (INRIA/gsplat format: xyz, opacity,
scale_0-2, rot_0-3, f_dc_0-2, ...) into a viser scene. Clicking a Gaussian
drops a small opaque red sphere exactly at that Gaussian's true 3D center
(not a screen-space overlay), so the marker stays pixel-perfect and
occlusion-consistent as the camera moves. After two clicks, the straight-line
distance between the two picked centers is shown in the GUI.

Picking works by intersecting the click ray (given by viser in world
coordinates) against every Gaussian center directly in numpy -- no GPU
readback or per-point click handlers needed, so it stays instant even for
scenes with millions of Gaussians. Among all centers within the pick-angle
cone of the ray, the one closest to the camera wins (mimics normal
occlusion-aware picking).

Minimal usage:
    python viewer.py --scene-path splat_30000.ply

More options:
    python viewer.py --scene-path /abs/path/to/splat_30000.ply \\
        --port 8080 --pick-angle-deg 0.5 --marker-radius 0.02
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser
from plyfile import PlyData

SH_C0 = 0.28209479177387814  # Y_0^0 spherical-harmonics constant, DC -> RGB.


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_3dgs_ply(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a 3D Gaussian Splatting .ply into render-ready arrays.

    Returns:
        centers (N, 3), rgbs (N, 3) in [0, 1], opacities (N, 1) in [0, 1],
        covariances (N, 3, 3).
    """
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = v.data.dtype.names
    n = len(v.data)

    centers = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if "f_dc_0" in names:
        f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=-1)
        rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0).astype(np.float32)
    elif "red" in names:
        rgb = (
            np.stack([v["red"], v["green"], v["blue"]], axis=-1).astype(np.float32)
            / 255.0
        )
    else:
        rgb = np.ones((n, 3), dtype=np.float32)

    if "opacity" in names:
        opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float64)))
        opacity = opacity.astype(np.float32)
    else:
        opacity = np.ones(n, dtype=np.float32)

    if all(f"scale_{i}" in names for i in range(3)):
        scales = np.exp(
            np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float64)
        ).astype(np.float32)
    else:
        # Not a real splat file (e.g. a plain point cloud) -- use a small
        # fixed splat size so it still renders as something.
        extent = float(np.linalg.norm(centers.max(0) - centers.min(0)))
        scales = np.full((n, 3), max(extent, 1.0) * 1e-4, dtype=np.float32)

    if all(f"rot_{i}" in names for i in range(4)):
        quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
        quats = quats / (np.linalg.norm(quats, axis=-1, keepdims=True) + 1e-8)
    else:
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    covariances = _quat_scale_to_covariance(quats, scales)
    return centers, rgb, opacity[:, None], covariances


def _quat_scale_to_covariance(quats: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Vectorized per-Gaussian covariance = R @ diag(scale^2) @ R^T.

    quats: (N, 4) normalized, (w, x, y, z). scales: (N, 3).
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    n = quats.shape[0]
    r = np.empty((n, 3, 3), dtype=np.float32)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)

    s2 = scales.astype(np.float32) ** 2  # (N, 3)
    r_scaled = r * s2[:, None, :]  # scale columns of R by s^2
    cov = np.einsum("nij,nkj->nik", r_scaled, r)  # R diag(s^2) R^T
    return cov.astype(np.float32)


# --------------------------------------------------------------------------
# Picking
# --------------------------------------------------------------------------


def rotate_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector `v` by `angle_rad` around unit `axis` (Rodrigues' formula)."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return (
        v * np.cos(angle_rad)
        + np.cross(axis, v) * np.sin(angle_rad)
        + axis * np.dot(axis, v) * (1.0 - np.cos(angle_rad))
    )


def pick_gaussian(
    centers: np.ndarray,
    ray_origin: tuple[float, float, float],
    ray_direction: tuple[float, float, float],
    angle_threshold_rad: float,
) -> int | None:
    """Find the Gaussian center closest to the camera within a narrow cone
    around the click ray. Returns the point index, or None if nothing is
    close enough to the ray.

    Using an angular threshold (rather than a fixed world-space distance)
    keeps the pick cone a constant size in screen space regardless of how
    far the clicked surface is from the camera or how big the scene is.
    """
    o = np.asarray(ray_origin, dtype=np.float64)
    d = np.asarray(ray_direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)

    v = centers.astype(np.float64) - o
    t = v @ d  # distance along the ray to each point's closest approach
    in_front = t > 1e-6
    if not np.any(in_front):
        return None

    perp2 = np.clip(np.sum(v * v, axis=1) - t**2, 0.0, None)
    angle = np.arctan2(np.sqrt(perp2), t)
    candidates = in_front & (angle < angle_threshold_rad)
    if not np.any(candidates):
        return None

    idxs = np.nonzero(candidates)[0]
    return int(idxs[np.argmin(t[idxs])])


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View a 3D Gaussian Splat scene and measure distances by clicking points.",
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        required=True,
        help="Path (relative or absolute) to a 3DGS .ply file, e.g. splat_30000.ply",
    )
    parser.add_argument("--port", type=int, default=8080, help="viser server port.")
    parser.add_argument(
        "--pick-angle-deg",
        type=float,
        default=0.5,
        help="Half-angle (degrees) of the click picking cone. Larger = easier to click, less precise.",
    )
    parser.add_argument(
        "--marker-radius",
        type=float,
        default=None,
        help="Radius of the red highlight markers, in scene units. Default: auto-sized from the scene's bounding box.",
    )
    args = parser.parse_args()

    scene_path = Path(args.scene_path).expanduser().resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"--scene-path does not exist: {scene_path}")

    print(f"Loading {scene_path} ...")
    t0 = time.time()
    centers, rgbs, opacities, covariances = load_3dgs_ply(scene_path)
    print(f"Loaded {centers.shape[0]:,} Gaussians in {time.time() - t0:.2f}s")

    # Real 3DGS reconstructions often contain a handful of stray "floater"
    # Gaussians far outside the actual scene (bad COLMAP points, junk from
    # training). A naive min/max bbox is dominated by those outliers -- e.g.
    # on a real office scan this blew the diagonal up ~1000x, which silently
    # made the auto-sized marker so large the camera ended up *inside* it
    # (invisible, since sphere backfaces aren't rendered). Using the 1st/99th
    # percentile per axis instead gives the extent of the actual scene.
    bbox_lo = np.percentile(centers, 1, axis=0)
    bbox_hi = np.percentile(centers, 99, axis=0)
    bbox_diag = float(np.linalg.norm(bbox_hi - bbox_lo))
    default_marker_radius = args.marker_radius or max(bbox_diag * 0.0006, 1e-5)

    server = viser.ViserServer(port=args.port)
    server.scene.add_gaussian_splats("/scene", centers=centers, covariances=covariances, rgbs=rgbs, opacities=opacities)

    # ---- GUI ------------------------------------------------------------
    server.gui.add_markdown(
        f"**{scene_path.name}**\n\n"
        f"{centers.shape[0]:,} Gaussians · central-98% extent {bbox_diag:.3g}"
    )
    server.gui.add_markdown(
        "### Measure tool\nClick a Gaussian to drop a red marker on it. "
        "After 2 clicks, the distance between them is shown below."
    )

    point_a_text = server.gui.add_text("Point A", "-", disabled=True)
    point_b_text = server.gui.add_text("Point B", "-", disabled=True)
    distance_text = server.gui.add_text("Distance", "-", disabled=True)
    clear_button = server.gui.add_button("Clear points")

    with server.gui.add_folder("View"):
        roll_left_button = server.gui.add_button("⟲ Roll -20°")
        roll_right_button = server.gui.add_button("⟳ Roll +20°")

    with server.gui.add_folder("Settings"):
        marker_radius_slider = server.gui.add_slider(
            "Marker radius",
            min=bbox_diag * 0.00005,
            max=bbox_diag * 0.01,
            step=bbox_diag * 0.00005,
            initial_value=default_marker_radius,
        )
        pick_angle_slider = server.gui.add_slider(
            "Pick sensitivity (deg)",
            min=0.05,
            max=3.0,
            step=0.05,
            initial_value=args.pick_angle_deg,
        )

    # ---- Picking state ----------------------------------------------------
    picked_indices: list[int] = []
    marker_handles: list[viser.SceneNodeHandle] = []
    line_handles: list[viser.SceneNodeHandle] = []

    def make_marker(position: np.ndarray) -> viser.SceneNodeHandle:
        return server.scene.add_icosphere(
            f"/markers/m{len(marker_handles)}_{time.time_ns()}",
            radius=marker_radius_slider.value,
            color=(255, 0, 0),
            subdivisions=1,
            position=tuple(position.tolist()),
            cast_shadow=False,
        )

    def clear_points() -> None:
        for h in marker_handles:
            h.remove()
        marker_handles.clear()
        for h in line_handles:
            h.remove()
        line_handles.clear()
        picked_indices.clear()
        point_a_text.value = "-"
        point_b_text.value = "-"
        distance_text.value = "-"

    @clear_button.on_click
    def _(_) -> None:
        clear_points()

    @marker_radius_slider.on_update
    def _(_) -> None:
        for h in marker_handles:
            h.radius = marker_radius_slider.value

    def roll_viewport(degrees: float) -> None:
        """Roll every connected client's camera about its own viewing axis
        (rotates `up_direction` around the look_at - position forward
        vector), leaving position and look_at untouched."""
        angle_rad = float(np.deg2rad(degrees))
        for client in server.get_clients().values():
            cam = client.camera
            forward = cam.look_at - cam.position
            cam.up_direction = rotate_about_axis(cam.up_direction, forward, angle_rad)

    @roll_left_button.on_click
    def _(_) -> None:
        roll_viewport(-20.0)

    @roll_right_button.on_click
    def _(_) -> None:
        roll_viewport(20.0)

    @server.scene.on_click()
    def _(event: viser.SceneClickEvent) -> None:
        angle_rad = float(np.deg2rad(pick_angle_slider.value))
        idx = pick_gaussian(centers, event.ray_origin, event.ray_direction, angle_rad)
        if idx is None:
            return  # click didn't land near any Gaussian; ignore

        # A 3rd click starts a new pair.
        if len(picked_indices) >= 2:
            clear_points()

        picked_indices.append(idx)
        pos = centers[idx]
        marker_handles.append(make_marker(pos))

        if len(picked_indices) == 1:
            point_a_text.value = f"{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}"
        else:
            point_b_text.value = f"{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}"
            pt_a, pt_b = centers[picked_indices[0]], centers[picked_indices[1]]
            dist = float(np.linalg.norm(pt_a - pt_b))
            distance_text.value = f"{dist:.4f}"
            line_handles.append(
                server.scene.add_line_segments(
                    "/markers/line",
                    points=np.array([[pt_a, pt_b]], dtype=np.float32),
                    colors=(255, 0, 0),
                    line_width=4,
                )
            )

    print(f"Viser server running: http://localhost:{args.port}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
