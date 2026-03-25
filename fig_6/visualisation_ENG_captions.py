import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# --- ============================================= ---
# --- MAIN BLOCK: Analysis and Plotting             ---
# --- ============================================= ---
if __name__ == "__main__":
    # 1. SEARCH AND LOAD DATA FILE
    list_of_files = glob.glob('*.npz')
    if not list_of_files:
        print("ERROR: No .npz data files found in the current directory.")
        exit()

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from file: '{latest_file}'\n")

    try:
        data = np.load(latest_file)
    except Exception as e:
        print(f"ERROR: Failed to load file. {e}")
        exit()

    # 2. DATA PARSING AND GROUPING
    grouped_lambdas = defaultdict(list)
    for key in data.keys():
        if key.endswith('_lambdas'):
            try:
                p_value = float(key.split('_')[1])
                grouped_lambdas[p_value].append(data[key])
            except (IndexError, ValueError):
                print(f"Warning: Failed to parse key '{key}'. Skipping.")

    if not grouped_lambdas:
        print("ERROR: No valid eigenvalue data found in the file.")
        exit()

    p_values = sorted(grouped_lambdas.keys())
    num_modes_k = len(next(iter(grouped_lambdas.values()))[0])

    print(f"Data successfully loaded and grouped.")
    print(f"Data found for p = {p_values}")
    print(f"Number of modes in each set (k) = {num_modes_k}\n")

    # 3. STATISTICS CALCULATION AND PLOTTING
    print("--- Plot generation ---")
    output_dir = 'visualisation_plots_ENG_captions'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to the folder: '{output_dir}'\n")

    plt.style.use('seaborn-v0_8-whitegrid')

    # Prepare data for logarithmic p scale (excluding p=0)
    p_values_log = [p for p in p_values if p > 0]
    p_indices_log = [p_values.index(p) for p in p_values_log]


    print("--- Generating plots for each mode ---")
    for mode_idx in range(num_modes_k):
        avg_re_lambdas = []
        std_re_lambdas = []

        for p in p_values:
            re_lambdas_for_current_mode = [
                run_lambdas[mode_idx].real for run_lambdas in grouped_lambdas[p]
            ]
            avg_re_lambdas.append(np.mean(re_lambdas_for_current_mode))
            # Error calculation (standard deviation)
            std_re_lambdas.append(np.std(re_lambdas_for_current_mode))

        # Check if all errors are zero
        if all(s == 0 for s in std_re_lambdas):
            print(f"Warning for mode {mode_idx}: All standard deviations are zero.")
            print("  Error bars will not be visible on the plot. Possibly only one run exists for each 'p'.")

        # --- Plot in linear scale for p ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(
            p_values, avg_re_lambdas, yerr=std_re_lambdas, fmt='-o',
            capsize=5, label=r'Average Re($\lambda$)', color='darkblue', alpha=0.8
        )
        ax.set_title(f"Behavior of mode $\lambda_{{{mode_idx}}}$ (Linear scale for p)", fontsize=16)
        ax.set_xlabel("Rewiring probability (p)", fontsize=12)
        ax.set_ylabel(f"Average Re($\lambda_{{{mode_idx}}}$)", fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        if mode_idx == 0:
            ax.set_ylim(-0.1, 0.1)
        plt.tight_layout()
        linear_filename = os.path.join(output_dir, f"mode_{mode_idx}_p_linear_ENG.png")
        plt.savefig(linear_filename)
        plt.close(fig)
        print(f"Plot saved: {linear_filename}")

        # --- Plot in logarithmic scale for p ---
        if p_values_log: # Plot only if there are p > 0 values
            avg_re_lambdas_log = [avg_re_lambdas[i] for i in p_indices_log]
            std_re_lambdas_log = [std_re_lambdas[i] for i in p_indices_log]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(
                p_values_log, avg_re_lambdas_log, yerr=std_re_lambdas_log, fmt='-o',
                capsize=5, label=r'Average Re($\lambda$)', color='darkred', alpha=0.8
            )
            ax.set_title(f"Behavior of mode $\lambda_{{{mode_idx}}}$ (Logarithmic scale for p)", fontsize=16)
            ax.set_xlabel("Rewiring probability (p)", fontsize=12)
            ax.set_ylabel(f"Average Re($\lambda_{{{mode_idx}}}$)", fontsize=12)
            ax.set_xscale('log') # Logarithmic scale for X axis
            ax.grid(True, which='both', linestyle='--')
            if mode_idx == 0:
                ax.set_ylim(-0.1, 0.1)
            plt.tight_layout()
            log_filename = os.path.join(output_dir, f"mode_{mode_idx}_p_log_ENG.png")
            plt.savefig(log_filename)
            plt.close(fig)
            print(f"Plot saved: {log_filename}")

    # --- PLOTTING GAP ---
    print("\n--- Generating plot for 'gap' (Re(λ₁) - Re(λ₂)) ---")
    if num_modes_k < 3:
        print("Warning: Not enough modes to calculate 'gap' (need >= 3). Skipping.")
    else:
        avg_gap = []
        std_gap = []
        for p in p_values:
            gaps_for_p = [
                run_lambdas[1].real - run_lambdas[2].real for run_lambdas in grouped_lambdas[p] if len(run_lambdas) > 2
            ]
            if gaps_for_p:
                avg_gap.append(np.mean(gaps_for_p))
                std_gap.append(np.std(gaps_for_p))
            else:
                avg_gap.append(np.nan)
                std_gap.append(np.nan)

        if all(s == 0 for s in std_gap if not np.isnan(s)):
            print("Warning for 'gap': All standard deviations are zero.")
            print("  Error bars will not be visible on the plot. Possibly only one run exists for each 'p'.")

        # --- Gap in linear scale for p ---
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.errorbar(
            p_values, avg_gap, yerr=std_gap, fmt='-o',
            capsize=5, label='Average Gap', color='darkgreen', alpha=0.8
        )
        ax.set_title("Gap: Re($\lambda_1$) - Re($\lambda_2$) vs. p (Linear scale for p)", fontsize=16)
        ax.set_xlabel("Rewiring probability (p)", fontsize=12)
        ax.set_ylabel("Average (Re($\lambda_1$) - Re($\lambda_2$))", fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        plt.tight_layout()
        gap_linear_filename = os.path.join(output_dir, "gap_p_linear_ENG.png")
        plt.savefig(gap_linear_filename)
        plt.close(fig)
        print(f"Plot saved: {gap_linear_filename}")

        # --- Gap in logarithmic scale for p ---
        if p_values_log:
            avg_gap_log = [avg_gap[i] for i in p_indices_log]
            std_gap_log = [std_gap[i] for i in p_indices_log]

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.errorbar(
                p_values_log, avg_gap_log, yerr=std_gap_log, fmt='-o',
                capsize=5, label='Average Gap', color='purple', alpha=0.8
            )
            ax.set_title("Gap: Re($\lambda_1$) - Re($\lambda_2$) vs. p (Logarithmic scale for p)", fontsize=16)
            ax.set_xlabel("Rewiring probability (p)", fontsize=12)
            ax.set_ylabel("Average (Re($\lambda_1$) - Re($\lambda_2$))", fontsize=12)
            ax.set_xscale('log') # Logarithmic scale for X axis
            ax.grid(True, which='both', linestyle='--')
            plt.tight_layout()
            gap_log_filename = os.path.join(output_dir, "gap_p_log_ENG.png")
            plt.savefig(gap_log_filename)
            plt.close(fig)
            print(f"Plot saved: {gap_log_filename}")


    print("\n--- All plots generated. ---")