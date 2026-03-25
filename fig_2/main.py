from core import *

# --- 1. Параметры эксперимента ---
J_val = 1.0
gamma_val = 0.1
import numpy as np
np.__config__.show()

initial_state_generators = {
    "1. Центр (Смесь)": create_localized_state,
    "2. Противоп. углы (Смесь)": create_opposite_corners_state,
    "3. Четыре угла (Смесь)": create_four_corners_state,
    "4. Смешанная диагональ": create_mixed_diagonal_state,
    "5. Запутанная диагональ": create_entangled_diagonal_state,
    "6. Внутр. углы (Смесь)": create_inner_corners_state,
    "7. Края (верх/низ) (Смесь)": create_top_bottom_edges_state,
    "8. Шахматка (Смесь)": create_checkerboard_state,
    "9. Граница (Смесь)": create_boundary_state,
}

# --- 2. Основной цикл исследования ---
for n in range(10, 11):
    visualize_experiment_for_n(n, J_val, gamma_val, initial_state_generators)