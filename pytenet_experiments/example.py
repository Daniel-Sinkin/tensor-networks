"""
From the README of the pytenet repo

Extended example to visualize the time evolution with a manual computation
of the energy expectation value <psi|H|psi>.

This implementation assumes that the MPO object (returned by
ptn.heisenberg_xxz_mpo) stores its site tensors in the attribute `A`
with shape (d, d, D_left, D_right) at each site.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytenet as ptn


def compute_energy(psi: ptn.MPS, mpo: ptn.MPO) -> float:
    """
    Compute the energy expectation value <psi|H|psi> by contracting the MPS with the MPO.

    The contraction is performed site-by-site via:

        L_new[x,i,j] = sum_{w,m,n,a,b} L[w,m,n] * conj(A)[a,m,i] * W[a,b,w,x] * A[b,n,j]

    where:
      - L is the left environment (initially shape (1,1,1)),
      - A is the MPS tensor at the current site (shape (d, m, n)),
      - W is the corresponding MPO tensor (shape (d, d, w, x)).

    Returns the real part of the energy.
    """
    # Initialize left environment L with shape (w, m, n) = (1, 1, 1)
    L_env = np.array([[[1.0]]], dtype=complex)

    # Loop over each site and update the left environment.
    for i in range(psi.nsites):
        A = psi.A[i]  # MPS tensor at site i with shape (d, m, n)
        A_conj = np.conjugate(A)  # complex conjugate of A
        W = mpo.A[i]  # MPO tensor at site i with shape (d, d, w, x)
        # Contract L_env, A_conj, W, and A.
        # Indices:
        #   L_env: (w, m, n)
        #   A_conj: (a, m, i) where a is physical index and i is new (bra) right index
        #   W: (a, b, w, x) with a, b physical and w, x MPO bond indices
        #   A: (b, n, j) where b is physical and j is new (ket) right index
        # The output L_new will have indices (x, i, j)
        L_env = np.einsum("wmn,ami,abwx,bnj->xij", L_env, A_conj, W, A)

    # After contracting all sites, L_env should have shape (1,1,1).
    energy = L_env[0, 0, 0]
    return energy.real


def main(image_folderpath: Path = Path(".")) -> None:
    # Number of lattice sites (1D with open boundary conditions)
    L = 10

    # Construct the MPO representation of the Heisenberg XXZ Hamiltonian.
    # Hamiltonian parameters: J = 1.0, Δ = 0.8, h = -0.1.
    mpoH = ptn.heisenberg_xxz_mpo(L, 1.0, 0.8, -0.1)
    mpoH.zero_qnumbers()

    # Initialize the wavefunction as an MPS with random entries.
    # Define virtual bond dimensions.
    D = [1, 2, 4, 8, 16, 28, 16, 8, 4, 2, 1]
    psi = ptn.MPS(mpoH.qd, [Di * [0] for Di in D], fill="random")

    # Clamp the virtual bond dimensions to Dinit.
    Dinit = 8
    for i in range(L):
        psi.A[i][:, Dinit:, :] = 0
        psi.A[i][:, :, Dinit:] = 0
    psi.orthonormalize(mode="left")

    # Define the time step (with both real and imaginary parts).
    dt = 0.01 - 0.05j
    numsteps = 20

    # List to store energy expectation values.
    energies = []

    # Evolve the state one step at a time.
    for step in range(numsteps):
        # Evolve the state by one time step.
        ptn.integrate_local_singlesite(mpoH, psi, dt, 1, numiter_lanczos=5)
        # Compute the energy expectation value.
        energy = compute_energy(psi, mpoH)
        energies.append(energy)
        print(f"Step {step+1}/{numsteps}, Energy = {energy:.6f}")

    # Plot the energy evolution over time.
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(numsteps), energies, marker="o", linestyle="-")
    plt.xlabel("Time Step")
    plt.ylabel("Energy Expectation Value")
    plt.title("Energy Evolution during TDVP Time Evolution")
    plt.grid(True)
    plt.savefig(image_folderpath.joinpath("example.png"), dpi=300)
    plt.clf()


if __name__ == "__main__":
    main()
