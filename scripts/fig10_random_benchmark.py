import os

import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from qdyn_research import MpembaValidator

def main():
    os.makedirs("res_gap", exist_ok=True)

    # ---------------------------
    # System and benchmark parameters
    # ---------------------------
    h, w = 6, 6
    p_rewire = 0.15
    num_random_trials = 5000
    time_horizon = np.linspace(0, 30, 300)

    validator = MpembaValidator(height=h, width=w, p=p_rewire, J=1.0, gamma=0.1)
    validator.save_state("res_gap/benchmark.pkl")

    # Compare targeted state search against random sampling of admissible pairs.
    smart_ok, smart_time, smart_adv, smart_gap = validator.run_smart_strategy_score(time_horizon)
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rnd_times, bins=100, color="gray", alpha=0.6, label="Random search successes")
    if smart_ok:
        ax.axvline(smart_time, color="red", linestyle="--", linewidth=3, label="Smart strategy")
    ax.set_title(f"Mpemba search benchmark ({h}x{w}, p={p_rewire})")
    ax.set_xlabel("Crossing time t*")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig("res_gap/benchmark.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()


