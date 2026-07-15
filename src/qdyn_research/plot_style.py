import matplotlib as mpl
import matplotlib.pyplot as plt

# physical widths (inches) matched to the sn-jnl text block.
# The real sn-jnl \textwidth is 372 pt = 131.2 mm (single column), measured from the class.
# Figures MUST be built at the on-page width they will occupy (= fraction * \linewidth), so LaTeX
# includes them at natural size and the point sizes are WYSIWYG (no scaling that shrinks fonts).
MM = 1.0 / 25.4
WIDTH_FULL = 131.2 * MM           # = \linewidth  (\includegraphics[width=\linewidth])
WIDTH_HALF = 0.48 * WIDTH_FULL    # 2-up subfigures (0.48\linewidth): fig6, fig12
WIDTH_THIRD = 0.32 * WIDTH_FULL   # 3-up subfigures (0.32\linewidth): fig7, fig8
WIDTH_COL = WIDTH_HALF            # alias (~half width) used by fig6/fig12

SEQ_CMAP = "viridis"
LINE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "legend.frameon": False,
    "axes.prop_cycle": mpl.cycler(color=LINE_COLORS),
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
}
# graph edges on network/excitability maps: thin, so the graph shows without competing with
# the node colours (author feedback: thick edges read as visual noise at the true figure size).
EDGE_WIDTH = 0.8
EDGE_COLOR = "0.55"
EDGE_ALPHA = 0.8


def apply_style():
    """Install the shared rcParams. Call once at the top of every figure script."""
    mpl.rcParams.update(_RC)


def new_fig(width_in, aspect=0.70, **kw):
    """Create a (fig, ax) sized to a *final* on-page width. aspect = height/width."""
    kw.setdefault("layout", "constrained")
    return plt.subplots(figsize=(width_in, width_in * aspect), **kw)


def save_pdf(fig, path):
    """Save as vector PDF at the exact figsize (no bbox='tight' width drift)."""
    fig.savefig(path, format="pdf")
    plt.close(fig)
