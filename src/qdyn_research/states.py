import numpy as np


# Initial-state factories used in fig_2 and fig_3.
def _create_mixed_state_rho(size: int, indices: list[int]):
    count = len(indices)
    if count == 0:
        return np.zeros((size, size), dtype=np.complex128), []
    rho = np.zeros((size, size), dtype=np.complex128)
    for idx in indices:
        rho[idx, idx] = 1.0 / count
    return rho, indices


def _create_pure_state_rho(size: int, indices: list[int]):
    count = len(indices)
    if count == 0:
        return np.zeros((size, size), dtype=np.complex128), []
    psi = np.zeros(size, dtype=np.complex128)
    psi[indices] = 1.0 / np.sqrt(count)
    return np.outer(psi, psi.conj()), indices


def create_localized_state(h: int, w: int):
    return _create_mixed_state_rho(h * w, [(h // 2) * w + (w // 2)])


def create_opposite_corners_state(h: int, w: int):
    return _create_mixed_state_rho(h * w, [0, h * w - 1])


def create_four_corners_state(h: int, w: int):
    return _create_mixed_state_rho(h * w, [0, w - 1, (h - 1) * w, h * w - 1])


def create_mixed_diagonal_state(h: int, w: int):
    return _create_mixed_state_rho(h * w, [i * w + i for i in range(min(h, w))])


def create_entangled_diagonal_state(h: int, w: int):
    return _create_pure_state_rho(h * w, [i * w + i for i in range(min(h, w))])


def create_inner_corners_state(h: int, w: int):
    if h <= 2 or w <= 2:
        return None, None
    return _create_mixed_state_rho(
        h * w,
        [1 * w + 1, 1 * w + (w - 2), (h - 2) * w + 1, (h - 2) * w + (w - 2)],
    )


def create_top_bottom_edges_state(h: int, w: int):
    top = list(range(w))
    bottom = list(range((h - 1) * w, h * w))
    return _create_mixed_state_rho(h * w, sorted(list(set(top + bottom))))


def create_checkerboard_state(h: int, w: int):
    return _create_mixed_state_rho(h * w, [i * w + j for i in range(h) for j in range(w) if (i + j) % 2 == 0])


def create_boundary_state(h: int, w: int):
    if h < 3 or w < 3:
        return None, None
    top = list(range(w))
    bottom = list(range((h - 1) * w, h * w))
    left = [i * w for i in range(1, h - 1)]
    right = [i * w + w - 1 for i in range(1, h - 1)]
    return _create_mixed_state_rho(h * w, sorted(list(set(top + bottom + left + right))))

