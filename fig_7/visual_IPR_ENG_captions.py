import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict


# --- ============================================= ---
# --- ANALYSIS CORE: IPR Calculation Function       ---
# --- ============================================= ---

def calculate_ipr(eigenvector: np.ndarray, N: int) -> float:
    """
    Calculates the Inverse Participation Ratio (IPR) for the population distribution
    of a given Liouvillian eigenvector.

    Args:
        eigenvector (np.ndarray): Eigenvector v_k of dimension (N*N,).
        N (int): Number of sites in the original network (e.g., 25 for 5x5).

    Returns:
        float: IPR value (dimensionless quantity).
    """
    # 1. Restore the mode matrix ρ_k (NxN) from vector v_k
    # Use 'F' (Fortran/column-major), as it matches our Liouvillian construction
    rho_k = eigenvector.reshape((N, N), order='F')

    # 2. Extract diagonal elements (populations)
    population_distribution = np.diag(rho_k)

    # 3. Work with magnitudes, as mode populations can be complex
    abs_population = np.abs(population_distribution)

    # 4. Calculate IPR using the formula
    # IPR = sum(|ψ_i|^4) / (sum(|ψ_i|^2))^2
    sum_sq = np.sum(abs_population ** 2)

    # Protection against division by zero for trivial vectors
    if np.isclose(sum_sq, 0):
        return 0.0

    sum_quad = np.sum(abs_population ** 4)

    return sum_quad / (sum_sq ** 2)


# --- ============================================= ---
# --- MAIN BLOCK: Loading, Processing, Visualization ---
# --- ============================================= ---
if __name__ == "__main__":
    # 1. DATA SEARCH AND LOADING
    list_of_files = glob.glob('rewiring_spectrum_data_10x10_20251008_122853.npz')
    if not list_of_files:
        print("ERROR: .npz data files not found in the current directory.")
        exit()

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Loading data from file: '{latest_file}'\n")
    data = np.load(latest_file)

    # 2. DATA PARSING AND GROUPING
    grouped_vectors = defaultdict(list)
    p_values_from_file = set()

    for key in data.keys():
        if key.endswith('_vectors'):
            base_key = key.replace('_vectors', '')
            p_val = data[f"{base_key}_p_value"].item()
            p_values_from_file.add(p_val)
            grouped_vectors[p_val].append(data[key])

    if not grouped_vectors:
        print("ERROR: Correct eigenvector data not found in the file.")
        exit()

    p_values = sorted(list(p_values_from_file))

    # 3. DETERMINING PARAMETERS FROM DATA
    first_vector_set = next(iter(grouped_vectors.values()))[0]
    N_squared, num_modes_k = first_vector_set.shape
    print(N_squared)
    N = int(np.sqrt(N_squared))

    print(f"Data successfully loaded:")
    print(f"  - Network dimension (N): {N} (calculated from vectors)")
    print(f"  - Number of saved modes (k): {num_modes_k}")
    print(f"  - Data found for p = {p_values}\n")

    # 4. IPR AND STATISTICS CALCULATION
    # We want to analyze modes 1, 2, and 3 (indices 1, 2, 3)
    modes_to_analyze = [1, 2, 3]

    # Check if enough modes are saved
    if num_modes_k <= max(modes_to_analyze):
        print(
            f"ERROR: Only {num_modes_k} modes saved in the file, but {max(modes_to_analyze) + 1} are required for analysis.")
        exit()

    ipr_results = defaultdict(dict)

    for mode_idx in modes_to_analyze:
        avg_iprs = []
        std_iprs = []
        for p in p_values:
            iprs_for_current_p = [
                calculate_ipr(run_vectors[:, mode_idx], N)
                for run_vectors in grouped_vectors[p]
            ]
            avg_iprs.append(np.mean(iprs_for_current_p))
            std_iprs.append(np.std(iprs_for_current_p))

        ipr_results[mode_idx]['avg'] = avg_iprs
        ipr_results[mode_idx]['std'] = std_iprs

    # 5. PLOT GENERATION
    print("--- Generating IPR vs p plots ---")

    for mode_idx in modes_to_analyze:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))

        avg_data = ipr_results[mode_idx]['avg']
        std_data = ipr_results[mode_idx]['std']

        ax.errorbar(
            p_values,
            avg_data,
            yerr=std_data,
            fmt='-o',
            capsize=5,
            label=f'Average IPR',
            color='darkred' if mode_idx == 1 else 'darkgreen' if mode_idx == 2 else 'darkblue',
            alpha=0.9
        )

        ax.set_title(f"Localization of mode λ_{mode_idx} (Inverse Participation Ratio)", fontsize=16)
        ax.set_xlabel("Rewiring probability (p)", fontsize=12)
        ax.set_ylabel(f"<IPR(λ_{mode_idx})>", fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        ax.set_xscale('log')  # Logarithmic scale for p is often more representative
        ax.set_xticks([0.001, 0.01, 0.1, 1.0])  # Set convenient ticks
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout()
        plt.savefig(f"IPR from p images ENG/{N//10}x{N//10} IPR graph mode k={mode_idx}.png")
        plt.show()