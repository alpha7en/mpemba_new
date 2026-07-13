import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import MpembaValidator
from qdyn_research.plot_style import apply_style, save_pdf, WIDTH_FULL

FIG_DIR = Path(__file__).resolve().parent.parent / "paper" / "figures"
# same 6x6 realization used for the published algorithm figures (migrated legacy checkpoint)
CHECKPOINT = str(Path(__file__).resolve().parent / "precalc" / "validator_6x6_p015.pkl")


def main():
    apply_style()
    os.makedirs("res_gap", exist_ok=True)

    # ---------------------------
    # System and benchmark parameters
    # ---------------------------
    h, w = 6, 6
    p_rewire = 0.15
    num_random_trials = 10000
    # horizon must reach the algorithmic crossing (t* ~ 45 on this graph), else it is missed
    time_horizon = np.linspace(0, 120, 1200)

    # Load the exact illustrative graph from the article instead of drawing a new one.
    validator = MpembaValidator.load_state(CHECKPOINT)

    # Compare targeted state search against random sampling of admissible pairs.
    smart_ok, smart_time, smart_adv, smart_gap = validator.run_smart_strategy_score(time_horizon)

    if smart_ok:
        t_max = max(30, int(10 * smart_time))
        time_horizon = np.linspace(0, t_max, t_max * 10)

    # Use the published 10^4 random crossing times if available (the exact Fig.13 data),
    # otherwise recompute the benchmark from scratch.
    precomp = Path(__file__).resolve().parent / "precalc" / "pivoprosto.txt"
    if precomp.exists():
        import re
        rnd_times = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", precomp.read_text(encoding="utf-8", errors="ignore"))]
        num_random_trials = len(rnd_times)
        rnd_ok_count = len(rnd_times)
    else:
        rnd_ok_count, rnd_times = validator.run_random_pull_strategy(
            num_random_trials,
            time_horizon,
            metric_gap_min=min(np.log(validator.n), smart_gap) / 10,
        )

    rnd_rate = (rnd_ok_count / num_random_trials) * 100

    with open("res_gap/benchmark.txt", "w", encoding="utf-8") as out:
        out.write(f"max metric gap {np.log(validator.n)}\n")
        out.write(f"tau_sys {validator.tau_sys}\n")
        out.write(f"smart_ok={smart_ok}, smart_time={smart_time}, smart_adv={smart_adv}, smart_gap={smart_gap}\n")
        out.write(f"random_success={rnd_ok_count}/{num_random_trials} ({rnd_rate:.2f}%)\n")

    fig, ax = plt.subplots(figsize=(WIDTH_FULL * 0.75, WIDTH_FULL * 0.75 * 0.6), layout="constrained")
    ax.hist(rnd_times, bins=100, color="0.6", edgecolor="0.35", linewidth=0.3,
            label="Random search ($10^4$ pairs)")
    if smart_ok:
        ax.axvline(smart_time, color="#D55E00", linestyle="--", linewidth=1.6,
                   label=f"Algorithmic pair ($t^*={smart_time:.1f}$)")
    ax.set_xlabel("crossing time $t^*$")
    ax.set_ylabel("number of pairs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.4)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    save_pdf(fig, str(FIG_DIR / "alg_compare.pdf"))


if __name__ == "__main__":
    main()


