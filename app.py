import io
import tempfile
import shutil

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle


# =========================================================
# Page setup
# =========================================================
st.set_page_config(page_title="MASBot Analysis App", layout="wide")
st.title("MASBot Analysis App")
st.write("Upload circle and dot trajectory files to generate plots and playback.")


# =========================================================
# Helpers
# =========================================================
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


@st.cache_data(show_spinner=False)
def load_tracking_data(file_bytes: bytes, filename: str):
    data = loadmat(io.BytesIO(file_bytes))

    if "Xmat" not in data or "Ymat" not in data:
        raise ValueError(f"{filename} must contain Xmat and Ymat.")

    X = np.asarray(data["Xmat"], dtype=float)
    Y = np.asarray(data["Ymat"], dtype=float)
    return X, Y


def apply_row_mapping(Xd: np.ndarray, Yd: np.ndarray, mapping_option: str):
    if mapping_option == "swap rows 1 and 2" and Xd.shape[0] >= 2:
        Xd = Xd[::-1]
        Yd = Yd[::-1]
    return Xd, Yd


@st.cache_data(show_spinner=False)
def compute_base_quantities(Xc: np.ndarray, Yc: np.ndarray, Xd: np.ndarray, Yd: np.ndarray, fps: int):
    dt = 1.0 / fps
    bots = Xc.shape[0]
    n_frames = Xc.shape[1]

    distance = None
    if bots >= 2:
        dx = Xc[0] - Xc[1]
        dy = Yc[0] - Yc[1]
        distance = np.sqrt(dx**2 + dy**2)

    raw_angles = []
    unwrapped_angles = []
    omega_raw = []

    for i in range(bots):
        theta = np.arctan2(Yd[i] - Yc[i], Xd[i] - Xc[i])
        theta_unwrap = np.unwrap(theta)
        omega = np.gradient(theta_unwrap, dt)

        raw_angles.append(theta)
        unwrapped_angles.append(theta_unwrap)
        omega_raw.append(omega)

    pair_orientation = None
    if bots >= 2:
        pair_orientation = np.arctan2(Yc[1] - Yc[0], Xc[1] - Xc[0])

    return {
        "bots": bots,
        "n_frames": n_frames,
        "distance": distance,
        "raw_angles": raw_angles,
        "unwrapped_angles": unwrapped_angles,
        "omega_raw": omega_raw,
        "pair_orientation": pair_orientation,
    }


def maybe_smooth(x: np.ndarray | None, enabled: bool, window: int):
    if x is None:
        return None
    if enabled:
        return moving_average(x, window)
    return x


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def file_to_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def add_direction_arrows(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    n_arrows: int,
    arrow_size: int,
    linewidth: float,
):
    if len(x) < 3 or n_arrows <= 0:
        return

    indices = np.linspace(1, len(x) - 2, n_arrows, dtype=int)
    used = set()

    for idx in indices:
        if idx in used:
            continue
        used.add(idx)

        x0, y0 = x[idx - 1], y[idx - 1]
        x1, y1 = x[idx], y[idx]

        dx = x1 - x0
        dy = y1 - y0

        if dx == 0 and dy == 0:
            continue

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=linewidth,
                mutation_scale=arrow_size,
                shrinkA=0,
                shrinkB=0,
            ),
        )


def plot_trajectory(
    Xc: np.ndarray,
    Yc: np.ndarray,
    bot_line_colors: list[str],
    linewidth: int,
    start_frame: int,
    end_frame: int,
    use_time_coloring: list[bool],
    cmap_names: list[str],
    cmins: list[float],
    cmaxs: list[float],
    show_start_marker: bool,
    show_end_marker: bool,
    start_marker_colors: list[str],
    end_marker_colors: list[str],
    start_marker_shape: str,
    end_marker_shape: str,
    start_marker_size: int,
    end_marker_size: int,
    show_arrows: bool,
    n_arrows: int,
    arrow_size: int,
):
    fig, ax = plt.subplots(figsize=(7, 7))
    bots = Xc.shape[0]

    legend_handles = []
    colorbar_info = []

    for i in range(bots):
        x = Xc[i, start_frame:end_frame + 1]
        y = Yc[i, start_frame:end_frame + 1]

        if len(x) < 2:
            continue

        if use_time_coloring[i]:
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            tvals = np.linspace(0, 1, len(segments))
            norm = mcolors.Normalize(vmin=cmins[i], vmax=cmaxs[i])

            lc = LineCollection(
                segments,
                cmap=cm.get_cmap(cmap_names[i]),
                norm=norm,
                linewidth=linewidth,
            )
            lc.set_array(tvals)
            ax.add_collection(lc)

            handle, = ax.plot([], [], color=bot_line_colors[i], linewidth=linewidth, label=f"Bot {i+1}")
            legend_handles.append(handle)
            colorbar_info.append((i, cmap_names[i], cmins[i], cmaxs[i]))
        else:
            handle, = ax.plot(x, y, color=bot_line_colors[i], linewidth=linewidth, label=f"Bot {i+1}")
            legend_handles.append(handle)

        if show_arrows:
            add_direction_arrows(
                ax=ax,
                x=x,
                y=y,
                color=bot_line_colors[i],
                n_arrows=n_arrows,
                arrow_size=arrow_size,
                linewidth=max(1.0, linewidth * 0.8),
            )

        if show_start_marker:
            ax.plot(
                x[0],
                y[0],
                marker=start_marker_shape,
                color=start_marker_colors[i],
                markersize=start_marker_size,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linestyle="None",
            )

        if show_end_marker:
            ax.plot(
                x[-1],
                y[-1],
                marker=end_marker_shape,
                color=end_marker_colors[i],
                markersize=end_marker_size,
                markeredgecolor="black",
                markeredgewidth=0.7,
                linestyle="None",
            )

    if len(colorbar_info) == 1:
        i, cmap_name, cmin, cmax = colorbar_info[0]
        sm = cm.ScalarMappable(
            cmap=cm.get_cmap(cmap_name),
            norm=mcolors.Normalize(vmin=cmin, vmax=cmax),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f"Normalized time (Bot {i+1})")

    elif len(colorbar_info) >= 2:
        i1, cmap1, cmin1, cmax1 = colorbar_info[0]
        sm1 = cm.ScalarMappable(
            cmap=cm.get_cmap(cmap1),
            norm=mcolors.Normalize(vmin=cmin1, vmax=cmax1),
        )
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label(f"Bot {i1+1} time")

        i2, cmap2, cmin2, cmax2 = colorbar_info[1]
        sm2 = cm.ScalarMappable(
            cmap=cm.get_cmap(cmap2),
            norm=mcolors.Normalize(vmin=cmin2, vmax=cmax2),
        )
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, ax=ax, fraction=0.046, pad=0.12)
        cbar2.set_label(f"Bot {i2+1} time")

    ax.set_title("Trajectory Plot")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.autoscale()

    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")

    plt.tight_layout()
    return fig


def plot_distance(
    distance: np.ndarray,
    color: str,
    smooth_enabled: bool,
    smooth_window: int,
    start_frame: int,
    end_frame: int,
):
    y = distance[start_frame:end_frame + 1]
    y_plot = maybe_smooth(y, smooth_enabled, smooth_window)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_plot, color=color, linewidth=2)
    ax.set_title("Distance Between Bots")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Distance")
    plt.tight_layout()
    return fig


def plot_phase(
    base: dict,
    bot_colors: list[str],
    unwrap_angle: bool,
    smooth_enabled: bool,
    smooth_window: int,
    start_frame: int,
    end_frame: int,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    bots = base["bots"]

    for i in range(bots):
        y = base["unwrapped_angles"][i] if unwrap_angle else base["raw_angles"][i]
        y = y[start_frame:end_frame + 1]
        y_plot = maybe_smooth(y, smooth_enabled, smooth_window)
        ax.plot(y_plot, color=bot_colors[i], linewidth=2, label=f"Bot {i+1}")

    ax.set_title("Angle / Phase")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (rad)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_omega(
    base: dict,
    bot_colors: list[str],
    smooth_enabled: bool,
    smooth_window: int,
    raw_alpha: float,
    start_frame: int,
    end_frame: int,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    bots = base["bots"]

    for i in range(bots):
        y_raw = base["omega_raw"][i][start_frame:end_frame + 1]
        ax.plot(y_raw, color=bot_colors[i], alpha=raw_alpha, linewidth=1)

        y_smooth = maybe_smooth(y_raw, smooth_enabled, smooth_window)
        ax.plot(y_smooth, color=bot_colors[i], linewidth=2.5, label=f"Bot {i+1}")

    ax.set_title("Angular Velocity")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angular velocity (rad/s)")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_pair_orientation(
    pair_orientation: np.ndarray,
    color: str,
    smooth_enabled: bool,
    smooth_window: int,
    start_frame: int,
    end_frame: int,
):
    y = pair_orientation[start_frame:end_frame + 1]
    y_plot = maybe_smooth(y, smooth_enabled, smooth_window)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y_plot, color=color, linewidth=2)
    ax.set_title("Particle-Particle Orientation")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (rad)")
    plt.tight_layout()
    return fig


def _compute_square_limits(
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    start_frame: int,
    end_frame: int,
    circle_radius: float,
):
    xmin = min(np.min(Xc[:, start_frame:end_frame + 1]), np.min(Xd[:, start_frame:end_frame + 1]))
    xmax = max(np.max(Xc[:, start_frame:end_frame + 1]), np.max(Xd[:, start_frame:end_frame + 1]))
    ymin = min(np.min(Yc[:, start_frame:end_frame + 1]), np.min(Yd[:, start_frame:end_frame + 1]))
    ymax = max(np.max(Yc[:, start_frame:end_frame + 1]), np.max(Yd[:, start_frame:end_frame + 1]))

    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    span = max(xmax - xmin, ymax - ymin)
    half = 0.5 * span + circle_radius + 40
    return xmid, ymid, half


def _build_playback_animation(
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    start_frame: int,
    end_frame: int,
    circle_radius: int,
    dot_radius: int,
    frame_step: int,
    interval: int,
    colors: list[str],
    line_width: int,
    fps_data: int,
):
    fig, ax = plt.subplots(figsize=(6, 6))

    xmid, ymid, half = _compute_square_limits(
        Xc, Yc, Xd, Yd, start_frame, end_frame, circle_radius
    )

    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid + half, ymid - half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    bots = Xc.shape[0]
    circle_artists = []
    dot_artists = []

    for i in range(bots):
        big_circle = Circle(
            (Xc[i, start_frame], Yc[i, start_frame]),
            radius=circle_radius,
            fill=False,
            edgecolor=colors[i],
            linewidth=line_width
        )
        small_dot = Circle(
            (Xd[i, start_frame], Yd[i, start_frame]),
            radius=dot_radius,
            fill=True,
            facecolor=colors[i],
            edgecolor="black",
            linewidth=0.8
        )

        ax.add_patch(big_circle)
        ax.add_patch(small_dot)

        circle_artists.append(big_circle)
        dot_artists.append(small_dot)

    frame_indices = list(range(start_frame, end_frame + 1, frame_step))
    if frame_indices[-1] != end_frame:
        frame_indices.append(end_frame)

    def update(frame_idx):
        idx = frame_indices[frame_idx]

        for i in range(bots):
            circle_artists[i].center = (Xc[i, idx], Yc[i, idx])
            dot_artists[i].center = (Xd[i, idx], Yd[i, idx])

        ax.set_title(f"time: {idx / fps_data:.2f}s   |   frame: {idx}")
        return circle_artists + dot_artists

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=interval,
        blit=False
    )
    return fig, anim


def generate_playback_file(
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xd: np.ndarray,
    Yd: np.ndarray,
    start_frame: int,
    end_frame: int,
    circle_radius: int,
    dot_radius: int,
    frame_step: int,
    interval: int,
    colors: list[str],
    line_width: int,
    fps_data: int,
    output_format: str,
):
    fig, anim = _build_playback_animation(
        Xc, Yc, Xd, Yd,
        start_frame, end_frame,
        circle_radius, dot_radius,
        frame_step, interval,
        colors, line_width,
        fps_data
    )

    if output_format == "GIF":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
        fps_out = max(1, int(1000 / interval))
        anim.save(tmp.name, writer=PillowWriter(fps=fps_out))
        plt.close(fig)
        return tmp.name, "image/gif"

    if output_format == "MP4":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        fps_out = max(1, int(1000 / interval))
        try:
            writer = FFMpegWriter(fps=fps_out, bitrate=1800)
            anim.save(tmp.name, writer=writer)
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(
                f"MP4 export failed. This usually means ffmpeg is not installed or not available. Original error: {e}"
            )
        plt.close(fig)
        return tmp.name, "video/mp4"

    plt.close(fig)
    raise ValueError("Unsupported output format.")


# =========================================================
# Shared sidebar inputs
# =========================================================
st.sidebar.header("Input Files")
circle_file = st.sidebar.file_uploader("Upload circle file (.mat)", type=["mat"], key="circle")
dot_file = st.sidebar.file_uploader("Upload dot file (.mat)", type=["mat"], key="dot")

st.sidebar.header("Shared Settings")
mapping_option = st.sidebar.selectbox(
    "Dot-to-circle row mapping",
    ["same order", "swap rows 1 and 2"]
)
fps = st.sidebar.number_input("Frames per second (fps)", min_value=1, value=30)

if circle_file is None or dot_file is None:
    st.info("Please upload both the circle file and the dot file.")
    st.stop()

try:
    circle_bytes = circle_file.getvalue()
    dot_bytes = dot_file.getvalue()

    Xc, Yc = load_tracking_data(circle_bytes, circle_file.name)
    Xd, Yd = load_tracking_data(dot_bytes, dot_file.name)

    Xd, Yd = apply_row_mapping(Xd, Yd, mapping_option)

    if Xc.shape != Yc.shape:
        st.error("Circle file: Xmat and Ymat must have the same shape.")
        st.stop()

    if Xd.shape != Yd.shape:
        st.error("Dot file: Xmat and Ymat must have the same shape.")
        st.stop()

    if Xc.shape != Xd.shape:
        st.error("Circle and dot data must have matching shapes.")
        st.stop()

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

base = compute_base_quantities(Xc, Yc, Xd, Yd, fps)
bots = base["bots"]
n_frames = base["n_frames"]

default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# =========================================================
# Section 1: Trajectory
# =========================================================
st.header("1. Trajectory Plot")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")

    traj_color_1 = st.color_picker("Bot 1 line color", default_colors[0], key="traj_c1")
    traj_color_2 = st.color_picker("Bot 2 line color", default_colors[1], key="traj_c2")

    start_marker_color_1 = st.color_picker("Bot 1 start-marker color", "#000000", key="traj_start_c1")
    start_marker_color_2 = st.color_picker("Bot 2 start-marker color", "#000000", key="traj_start_c2")

    end_marker_color_1 = st.color_picker("Bot 1 end-marker color", "#000000", key="traj_end_c1")
    end_marker_color_2 = st.color_picker("Bot 2 end-marker color", "#000000", key="traj_end_c2")

    traj_linewidth = st.slider("Line width", 1, 5, 2, key="traj_lw")

    marker_shape_options = {
        "circle": "o",
        "square": "s",
        "triangle": "^",
    }

    traj_show_start_marker = st.checkbox("Show start marker", value=True, key="traj_start_marker")
    traj_start_marker_shape_name = st.selectbox(
        "Start-marker shape",
        ["circle", "square", "triangle"],
        index=0,
        key="traj_start_shape_name"
    )
    traj_start_marker_shape = marker_shape_options[traj_start_marker_shape_name]
    traj_start_marker_size = st.slider("Start-marker size", 4, 16, 8, key="traj_start_size")

    traj_show_end_marker = st.checkbox("Show end marker", value=True, key="traj_end_marker")
    traj_end_marker_shape_name = st.selectbox(
        "End-marker shape",
        ["circle", "square", "triangle"],
        index=1,
        key="traj_end_shape_name"
    )
    traj_end_marker_shape = marker_shape_options[traj_end_marker_shape_name]
    traj_end_marker_size = st.slider("End-marker size", 4, 16, 8, key="traj_end_size")

    traj_show_arrows = st.checkbox("Show direction arrows", value=True, key="traj_arrows")
    traj_num_arrows = st.slider("Number of arrows", 0, 30, 8, key="traj_num_arrows")
    traj_arrow_size = st.slider("Arrow size", 6, 30, 14, key="traj_arrow_size")

    st.markdown("**Bot 1 time-coloring**")
    traj_use_time_coloring_1 = st.checkbox("Use time-varying color for Bot 1", value=False, key="traj_timecolor_1")
    if traj_use_time_coloring_1:
        traj_cmap_1 = st.selectbox(
            "Bot 1 colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
            index=0,
            key="traj_cmap_1"
        )
        traj_cmin_1, traj_cmax_1 = st.slider(
            "Bot 1 colorbar range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            key="traj_crange_1"
        )
    else:
        traj_cmap_1 = "viridis"
        traj_cmin_1, traj_cmax_1 = 0.0, 1.0

    st.markdown("**Bot 2 time-coloring**")
    traj_use_time_coloring_2 = st.checkbox("Use time-varying color for Bot 2", value=False, key="traj_timecolor_2")
    if traj_use_time_coloring_2:
        traj_cmap_2 = st.selectbox(
            "Bot 2 colormap",
            ["viridis", "plasma", "inferno", "magma", "cividis", "turbo"],
            index=2,
            key="traj_cmap_2"
        )
        traj_cmin_2, traj_cmax_2 = st.slider(
            "Bot 2 colorbar range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            key="traj_crange_2"
        )
    else:
        traj_cmap_2 = "inferno"
        traj_cmin_2, traj_cmax_2 = 0.0, 1.0

    traj_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="traj_range")

with plot_col:
    traj_line_colors = [traj_color_1, traj_color_2] + default_colors[2:]
    traj_start_colors = [start_marker_color_1, start_marker_color_2] + ["#000000"] * max(0, bots - 2)
    traj_end_colors = [end_marker_color_1, end_marker_color_2] + ["#000000"] * max(0, bots - 2)

    traj_use_time = [traj_use_time_coloring_1, traj_use_time_coloring_2] + [False] * max(0, bots - 2)
    traj_cmaps = [traj_cmap_1, traj_cmap_2] + ["viridis"] * max(0, bots - 2)
    traj_cmins = [traj_cmin_1, traj_cmin_2] + [0.0] * max(0, bots - 2)
    traj_cmaxs = [traj_cmax_1, traj_cmax_2] + [1.0] * max(0, bots - 2)

    fig_traj = plot_trajectory(
        Xc=Xc,
        Yc=Yc,
        bot_line_colors=traj_line_colors,
        linewidth=traj_linewidth,
        start_frame=traj_range[0],
        end_frame=traj_range[1],
        use_time_coloring=traj_use_time,
        cmap_names=traj_cmaps,
        cmins=traj_cmins,
        cmaxs=traj_cmaxs,
        show_start_marker=traj_show_start_marker,
        show_end_marker=traj_show_end_marker,
        start_marker_colors=traj_start_colors,
        end_marker_colors=traj_end_colors,
        start_marker_shape=traj_start_marker_shape,
        end_marker_shape=traj_end_marker_shape,
        start_marker_size=traj_start_marker_size,
        end_marker_size=traj_end_marker_size,
        show_arrows=traj_show_arrows,
        n_arrows=traj_num_arrows,
        arrow_size=traj_arrow_size,
    )

    st.pyplot(fig_traj)
    st.download_button(
        label="Download trajectory plot (PNG)",
        data=fig_to_png_bytes(fig_traj),
        file_name="trajectory_plot.png",
        mime="image/png",
        key="dl_traj"
    )


# =========================================================
# Section 2: Generated Playback
# =========================================================
st.header("2. Generated Playback")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")
    play_color_1 = st.color_picker("Bot 1 ring color", default_colors[0], key="play_c1")
    play_color_2 = st.color_picker("Bot 2 ring color", default_colors[1], key="play_c2")
    circle_radius = st.slider("Big circle radius", 10, 120, 40, key="play_radius")
    dot_radius = st.slider("Small dot radius", 2, 20, 5, key="play_dot_radius")
    play_linewidth = st.slider("Ring line width", 1, 6, 2, key="play_lw")
    play_frame_step = st.slider("Frame step", 1, 200, 20, key="play_step")
    play_interval = st.slider("Playback speed (ms/frame)", 20, 300, 80, key="play_interval")
    play_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="play_range")

    available_formats = ["GIF", "MP4"] if ffmpeg_available() else ["GIF"]
    play_format = st.selectbox("Playback format", available_formats, index=0, key="play_format")

    if not ffmpeg_available():
        st.caption("MP4 hidden because ffmpeg is not available.")

with plot_col:
    play_colors = [play_color_1, play_color_2] + default_colors[2:]

    generate_playback_btn = st.button("Generate Playback", key="generate_playback_btn")

    if generate_playback_btn:
        try:
            with st.spinner(f"Generating {play_format} playback..."):
                playback_path, playback_mime = generate_playback_file(
                    Xc, Yc, Xd, Yd,
                    start_frame=play_range[0],
                    end_frame=play_range[1],
                    circle_radius=circle_radius,
                    dot_radius=dot_radius,
                    frame_step=play_frame_step,
                    interval=play_interval,
                    colors=play_colors,
                    line_width=play_linewidth,
                    fps_data=fps,
                    output_format=play_format
                )

            st.session_state["playback_path"] = playback_path
            st.session_state["playback_mime"] = playback_mime
            st.session_state["playback_format"] = play_format

        except Exception as e:
            st.error(str(e))

    if "playback_path" in st.session_state:
        playback_path = st.session_state["playback_path"]
        playback_mime = st.session_state["playback_mime"]
        play_format_saved = st.session_state["playback_format"]

        if play_format_saved == "GIF":
            st.image(playback_path)
            download_name = "generated_playback.gif"
        else:
            st.video(playback_path)
            download_name = "generated_playback.mp4"

        st.download_button(
            label=f"Download playback ({play_format_saved})",
            data=file_to_bytes(playback_path),
            file_name=download_name,
            mime=playback_mime,
            key="dl_playback"
        )
    else:
        st.info("Click 'Generate Playback' to create the animation.")


# =========================================================
# Section 3: Distance
# =========================================================
st.header("3. Distance Between Bots")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")
    dist_color = st.color_picker("Line color", default_colors[0], key="dist_color")
    dist_smooth = st.checkbox("Enable smoothing", value=False, key="dist_smooth")
    dist_window = st.slider("Smoothing window", 1, 301, 31, step=2, key="dist_window")
    dist_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="dist_range")

with plot_col:
    if base["distance"] is not None:
        fig_dist = plot_distance(
            base["distance"],
            color=dist_color,
            smooth_enabled=dist_smooth,
            smooth_window=dist_window,
            start_frame=dist_range[0],
            end_frame=dist_range[1]
        )
        st.pyplot(fig_dist)
        st.download_button(
            label="Download distance plot (PNG)",
            data=fig_to_png_bytes(fig_dist),
            file_name="distance_plot.png",
            mime="image/png",
            key="dl_dist"
        )
    else:
        st.warning("Need at least 2 bots for distance plot.")


# =========================================================
# Section 4: Angle / Phase
# =========================================================
st.header("4. Angle / Phase")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")
    phase_color_1 = st.color_picker("Bot 1 color", default_colors[0], key="phase_c1")
    phase_color_2 = st.color_picker("Bot 2 color", default_colors[1], key="phase_c2")
    phase_unwrap = st.checkbox("Use unwrapped angle", value=True, key="phase_unwrap")
    phase_smooth = st.checkbox("Enable smoothing", value=False, key="phase_smooth")
    phase_window = st.slider("Smoothing window", 1, 301, 31, step=2, key="phase_window")
    phase_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="phase_range")

with plot_col:
    phase_colors = [phase_color_1, phase_color_2] + default_colors[2:]
    fig_phase = plot_phase(
        base,
        bot_colors=phase_colors,
        unwrap_angle=phase_unwrap,
        smooth_enabled=phase_smooth,
        smooth_window=phase_window,
        start_frame=phase_range[0],
        end_frame=phase_range[1]
    )
    st.pyplot(fig_phase)
    st.download_button(
        label="Download phase plot (PNG)",
        data=fig_to_png_bytes(fig_phase),
        file_name="phase_plot.png",
        mime="image/png",
        key="dl_phase"
    )


# =========================================================
# Section 5: Angular Velocity
# =========================================================
st.header("5. Angular Velocity")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")
    omega_color_1 = st.color_picker("Bot 1 color", default_colors[0], key="omega_c1")
    omega_color_2 = st.color_picker("Bot 2 color", default_colors[1], key="omega_c2")
    omega_smooth = st.checkbox("Enable smoothing", value=True, key="omega_smooth")
    omega_window = st.slider("Smoothing window", 1, 301, 31, step=2, key="omega_window")
    omega_raw_alpha = st.slider("Raw line alpha", 0.0, 1.0, 0.25, key="omega_alpha")
    omega_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="omega_range")

with plot_col:
    omega_colors = [omega_color_1, omega_color_2] + default_colors[2:]
    fig_omega = plot_omega(
        base,
        bot_colors=omega_colors,
        smooth_enabled=omega_smooth,
        smooth_window=omega_window,
        raw_alpha=omega_raw_alpha,
        start_frame=omega_range[0],
        end_frame=omega_range[1]
    )
    st.pyplot(fig_omega)
    st.download_button(
        label="Download angular velocity plot (PNG)",
        data=fig_to_png_bytes(fig_omega),
        file_name="angular_velocity_plot.png",
        mime="image/png",
        key="dl_omega"
    )


# =========================================================
# Section 6: Particle-Particle Orientation
# =========================================================
st.header("6. Particle-Particle Orientation")
plot_col, ctrl_col = st.columns([3, 1])

with ctrl_col:
    st.markdown("**Controls**")
    orient_color = st.color_picker("Line color", "#2ca02c", key="orient_color")
    orient_smooth = st.checkbox("Enable smoothing", value=False, key="orient_smooth")
    orient_window = st.slider("Smoothing window", 1, 301, 31, step=2, key="orient_window")
    orient_range = st.slider("Frame range", 0, n_frames - 1, (0, n_frames - 1), key="orient_range")

with plot_col:
    if base["pair_orientation"] is not None:
        fig_orient = plot_pair_orientation(
            base["pair_orientation"],
            color=orient_color,
            smooth_enabled=orient_smooth,
            smooth_window=orient_window,
            start_frame=orient_range[0],
            end_frame=orient_range[1]
        )
        st.pyplot(fig_orient)
        st.download_button(
            label="Download orientation plot (PNG)",
            data=fig_to_png_bytes(fig_orient),
            file_name="particle_particle_orientation.png",
            mime="image/png",
            key="dl_orient"
        )
    else:
        st.warning("Need at least 2 bots for particle-particle orientation.")


# =========================================================
# Footer summary
# =========================================================
st.divider()
st.subheader("Data Summary")
st.write(f"Number of bots: {bots}")
st.write(f"Number of frames: {n_frames}")
st.write(f"Data shape: {Xc.shape}")
