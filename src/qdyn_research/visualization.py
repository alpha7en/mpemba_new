import matplotlib.pyplot as plt
import numpy as np


# Figure helpers: same visual encoding reused in fig_1/fig_2 blocks.
def draw_population_mode_on_axis(
    ax,
    population_map,
    height,
    width,
    k_scaler,
    title_text,
    radius=0.35,
    grid_linewidth=0.5,
    hide_spines=True,
    circle_edgecolor="black",
    circle_linewidth=0.5,
    title_fontsize=9,
    title_y=-0.3,
    title_fontweight=None,
):
    """Render mode populations with sign-encoded color and amplitude-encoded opacity."""
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=grid_linewidth)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    if hide_spines:
        for spine in ax.spines.values():
            spine.set_visible(False)

    for y in range(height):
        for x in range(width):
            value = population_map[y, x]
            if value > 1e-6:
                color = "red"
            elif value < -1e-6:
                color = "blue"
            else:
                color = "white"
            alpha = min(1.0, np.abs(value) * k_scaler)
            circle = plt.Circle(
                (x, y),
                radius=radius,
                facecolor=color,
                alpha=alpha,
                edgecolor=circle_edgecolor,
                linewidth=circle_linewidth,
            )
            ax.add_patch(circle)

    ax.set_xlim(-0.6, width - 0.4)
    ax.set_ylim(-0.6, height - 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    kwargs = {"fontsize": title_fontsize}
    if title_y is not None:
        kwargs["y"] = title_y
    if title_fontweight is not None:
        kwargs["fontweight"] = title_fontweight
    ax.set_title(title_text, **kwargs)


def generate_relative_amplitude_colorbar(base_filename: str, label: str):
    """Render standalone colorbar for the alpha-blended red/blue amplitude encoding."""
    num_points = 512
    values = np.linspace(-1.0, 1.0, num_points)

    pixels = np.ones((1, num_points, 3), dtype=np.float64)
    for idx, v in enumerate(values):
        alpha = min(1.0, abs(v))
        if v > 1e-9:
            rgb = np.array([1.0, 0.0, 0.0])
        elif v < -1e-9:
            rgb = np.array([0.0, 0.0, 1.0])
        else:
            alpha = 0.0
            rgb = np.array([1.0, 1.0, 1.0])
        pixels[0, idx, :] = alpha * rgb + (1.0 - alpha) * np.array([1.0, 1.0, 1.0])

    fig, ax = plt.subplots(figsize=(16, 2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.imshow(pixels, aspect="auto", extent=[-1, 1, 0, 1], origin="lower")
    ax.set_yticks([])
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["-1.0", "-0.75", "-0.50", "-0.25", "0", "0.25", "0.50", "0.75", "1.0"], fontsize=16)
    ax.set_xlim(-1, 1)
    ax.set_title(label, fontsize=22, pad=15)

    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(axis="x", which="both", direction="out", length=5)

    filename = base_filename[:-4] if base_filename.endswith(".png") else base_filename
    out_filename = filename + "_bar_visualisation.png"
    fig.tight_layout()
    fig.savefig(out_filename, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out_filename

