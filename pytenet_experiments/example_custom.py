import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pytenet as ptn
from scipy.linalg import svd

warnings.filterwarnings("ignore", category=RuntimeWarning)

images_folderpath = Path("pytenet_experiments").joinpath("generated")


def compute_entanglement_entropy(wave_func):
    """
    Compute the entanglement entropy at each bond of the MPS.
    """
    entanglement_entropy = []
    for site in range(len(wave_func.A) - 1):  # Exclude the last site
        A = wave_func.A[site]  # Shape: (d, D_left, D_right)
        d, D_left, D_right = A.shape

        # Reshape into a matrix (merge physical index and left bond index)
        A_matrix = A.reshape(d * D_left, D_right)

        # Perform SVD
        U, singular_values, Vh = svd(A_matrix, full_matrices=False)

        # Compute entanglement entropy S = - sum(λ^2 log λ^2)
        singular_values_sq = singular_values**2
        entropy = -np.sum(
            singular_values_sq * np.log(singular_values_sq + 1e-12)
        )  # Avoid log(0)
        entanglement_entropy.append(entropy)

    return np.array(entanglement_entropy)


def compute_local_magnetization(wave_func):
    """
    Compute local magnetization ⟨Sz⟩ at each site.
    """
    Sz = np.array([[0.5, 0], [0, -0.5]])  # Spin-1/2 Sz operator
    local_magnetization = []

    for site in range(len(wave_func.A)):
        A = wave_func.A[site]  # Shape: (d, D_left, D_right)
        d, D_left, D_right = A.shape

        # Compute expectation value <A| Sz |A>
        Sz_expectation = np.tensordot(
            A.conj(), np.tensordot(Sz, A, axes=([1], [0])), axes=([0, 0])
        )
        Sz_expectation = Sz_expectation.sum()  # Sum over all remaining indices

        local_magnetization.append(np.real(Sz_expectation))  # Ensure real output

    return np.array(local_magnetization)


def plot_wavefunc(wave_func: ptn.MPS, filename: Optional[str] = None) -> None:
    entanglement_entropy = compute_entanglement_entropy(wave_func)
    local_magnetization = compute_local_magnetization(wave_func)

    # Create a combined plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(16, 6))

    # Plot Entanglement Entropy on the left y-axis
    color1 = "tab:blue"
    ax1.set_xlabel("Lattice Site")
    ax1.set_ylabel("Entanglement Entropy", color=color1)
    ax1.plot(
        range(len(entanglement_entropy)),
        entanglement_entropy,
        marker="o",
        linestyle="-",
        color=color1,
        label="Entanglement Entropy",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create a second y-axis for Local Magnetization
    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Local Magnetization ⟨Sz⟩", color=color2)
    ax2.plot(
        range(len(local_magnetization)),
        local_magnetization,
        marker="s",
        linestyle="--",
        color=color2,
        label="Local Magnetization",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Title and Grid
    fig.suptitle("Entanglement Entropy and Local Magnetization Across Lattice Sites")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if filename is not None:
        plt.savefig(
            images_folderpath.joinpath(f"example_custom_{filename}.png"), dpi=300
        )
    plt.show()


param_num_lattices = 10
param_J = 1.0
param_Delta = 0.8
param_field_strength = -0.1

heisenberg_kwargs = {
    "L": param_num_lattices,
    "J": param_J,
    "D": param_Delta,
    "h": param_field_strength,
}

print(f"Creating Heisenberg XXZ MPO with parameters {heisenberg_kwargs}")
mpo = ptn.heisenberg_xxz_mpo(**heisenberg_kwargs)
print("MPO as a matrix:")
print(mpo.as_matrix())

print("MPO qd:")
print(mpo.qd)
print("MPO qD:")
print(mpo.qD)

print("Zeroing MPO qNumbers")
mpo.zero_qnumbers()
print("MPO qd:")
print(mpo.qd)
print("MPO qD:")
print(mpo.qD)

virtual_bond_quantum_numbers = [1, 2, 4, 8, 16, 28, 16, 8, 4, 2, 1]
wave_func = ptn.MPS(
    mpo.qd, [Di * [0] for Di in virtual_bond_quantum_numbers], fill="random"
)

Dinit = 8
for lattice_idx in range(param_num_lattices):
    wave_func.A[lattice_idx][:, Dinit:, :] = 0
    wave_func.A[lattice_idx][:, :, Dinit:] = 0
wave_func.orthonormalize(mode="left")

dt = 0.01 - 0.05j
numsteps = 101

plot_wavefunc(wave_func)
ees = [np.array(compute_entanglement_entropy(wave_func))]
lms = [np.array(compute_local_magnetization(wave_func))]
for i in range(numsteps):
    ptn.integrate_local_singlesite(mpo, wave_func, dt, 1, numiter_lanczos=5)
    ees.append(np.array(compute_entanglement_entropy(wave_func)))
    lms.append(np.array(compute_local_magnetization(wave_func)))
    if i % 50 == 0:
        plot_wavefunc(wave_func, filename=f"step_{i}")

ees_arr = np.vstack(ees)
lms_arr = np.vstack(lms)

plt.figure(figsize=(8, 5))
plt.imshow(ees_arr.T, aspect="auto", cmap="plasma", origin="lower")
plt.colorbar(label="Entanglement Entropy")
plt.xlabel("Time Step")
plt.ylabel("Lattice Site")
plt.title("Entanglement Entropy Evolution")
plt.savefig(
    images_folderpath.joinpath("example_custom_entanglement_entropy_evolution.png"),
    dpi=300,
)
plt.show()

plt.figure(figsize=(8, 5))
plt.imshow(lms_arr.T, aspect="auto", cmap="coolwarm", origin="lower")
plt.colorbar(label="Local Magnetization ⟨Sz⟩")
plt.xlabel("Time Step")
plt.ylabel("Lattice Site")
plt.title("Local Magnetization Evolution")
plt.savefig(
    images_folderpath.joinpath("example_custom_local_magnetization_evolution.png"),
    dpi=300,
)
plt.show()
