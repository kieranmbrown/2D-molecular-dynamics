import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import lobpcg

# params
x_min, x_max = -3, 3
y_min, y_max = -3, 3
dx = 0.01 # 0.02 - 0.1
dt = 0.001 
num_steps = 10000 
solve_every = 1 # decouples SE and nuclei dynamics by this factor
mass_proton = 1.0 
damping_factor = 0.9
max_lobpcg_iter = 1000 # higher eignestates especially struggle below like 500
num_eigenstates = 5
output_dir = "output_frames/try13/"


nucleus_positions = [(-1.0, 0.0), (-1/2, 0.87), (1/2, 0.87),  (1.0, 0.0), (1/2, -0.87), (-1/2, -0.87)]
velocities = [(0, 0)] * len(nucleus_positions)

def initialize_grid(x_min, x_max, y_min, y_max, dx):
    x = np.arange(x_min, x_max, dx)
    y = np.arange(y_min, y_max, dx)
    X, Y = np.meshgrid(x, y)

    return X, Y, x, y, dx

def laplacian_2D(Nx, Ny, dx):
    coeff = 1 / (dx ** 2)
    D = -4 * np.ones(Nx * Ny)
    U = np.ones(Nx * Ny - 1)
    U[np.arange(1, Nx * Ny) % Nx == 0] = 0
    L = np.ones(Nx * Ny - 1)
    VU = np.ones(Nx * Ny - Nx)
    VL = np.ones(Nx * Ny - Nx)
    Lap = diags([D, U, L, VU, VL], [0, 1, -1, Nx, -Nx]) * coeff

    return Lap

def potential_2D_multiple(X, Y, nucleus_positions):
    V = np.zeros_like(X)
    for (x0, y0) in nucleus_positions:
        r = np.sqrt((X - x0)**2 + (Y - y0)**2)
        V += -1 / (r + .00001)

    return V

def hamiltonian_2D_multiple(X, Y, dx, mass, nucleus_positions):
    Nx, Ny = X.shape
    Lap = laplacian_2D(Nx, Ny, dx)
    V = potential_2D_multiple(X, Y, nucleus_positions).flatten()
    H = - (1 / (2 * mass)) * Lap + diags(V)

    return H

def solve_schrodinger_lobpcg(H, num_states=1):
    H_csr = H.tocsr()
    X = np.random.rand(H.shape[0], num_states)
    eigenvalues, eigenvectors = lobpcg(H_csr, X, largest=False, maxiter=max_lobpcg_iter)

    return eigenvalues, eigenvectors

def compute_coulomb_forces(nucleus_positions, k=1, charge=4):

    forces = np.zeros((len(nucleus_positions), 2))

    for i, (x_i, y_i) in enumerate(nucleus_positions):
        force = np.array([0.0, 0.0])
        for j, (x_j, y_j) in enumerate(nucleus_positions):
            
            if i != j:
                r_vec = np.array([x_i - x_j, y_i - y_j])
                r_mag = np.linalg.norm(r_vec)
                force += k * charge * (r_vec / r_mag ** 2 + .00001)

        forces[i] = force
    return forces

def compute_electronics(X, Y, psi, nucleus_positions, cutoff=3e-2, alpha=1.0):
    P = np.abs(psi.reshape(X.shape))**2
    high_intensity_mask = P > cutoff

    forces = np.zeros((len(nucleus_positions), 2))

    for i, (x_i, y_i) in enumerate(nucleus_positions):
        force_jk = np.array([0.0, 0.0])

        for (j, k) in zip(*np.where(high_intensity_mask)):
            x_j, y_k = X[j, k], Y[j, k] 
            r_vec = np.array([x_j - x_i, y_k - y_i])
            r_mag = np.linalg.norm(r_vec) 

            force_jk += alpha * P[k, j] * r_vec / (r_mag**2 + .001)
            #force_y -= alpha * P[k, j] / r_mag**2

        forces[i] = force_jk
    #print(f"\nElectronic forces: {forces}\n")
    return forces

def update_nucleus_positions_damped(nucleus_positions, velocities, dt, mass, damping_factor, X, Y):
    forces = compute_coulomb_forces(nucleus_positions) + compute_electronics(X, Y, psi, nucleus_positions)
    new_velocities = damping_factor * (velocities + (forces / mass) * dt)
    new_positions = [(x + v_x * dt, y + v_y * dt) for (x, y), (v_x, v_y) in zip(nucleus_positions, new_velocities)]
    return new_positions, new_velocities

def normalize_wavefunction(psi, dx):
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx**2)

def plot_wavefunction_standard(X, Y, psi, eigenvalue, index, nucleus_positions, frame):

    probability_density = np.abs(psi.reshape(X.shape))**2

    dist = np.linalg.norm(np.array(nucleus_positions[0]) - np.array(nucleus_positions[1]))

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, probability_density, levels=50, cmap='inferno')
    plt.colorbar(label="Probability Density")

    for (x_p, y_p) in nucleus_positions:
        plt.scatter(x_p, y_p, color='white', edgecolor='black', marker='o', s=10)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"[{frame}] Eigenstate {index} (Energy = {eigenvalue:.3f}, Proton Distance: {dist:.2f}")
    plt.savefig(f"{output_dir}output_frame_{frame}.png")


X, Y, x, y, dx = initialize_grid(x_min, x_max, y_min, y_max, dx)


H_initial = hamiltonian_2D_multiple(X, Y, dx, mass_proton, nucleus_positions)
eigenvalues_initial, eigenvectors_initial = solve_schrodinger_lobpcg(H_initial, num_states=num_eigenstates)
psi = normalize_wavefunction(eigenvectors_initial[:, 3], dx)  # Use the 4th eigenstate

prev_nucleus_positions = np.array(nucleus_positions)
previous_psi_guess = eigenvectors_initial

for step in range(num_steps):
    nucleus_positions, velocities = update_nucleus_positions_damped(nucleus_positions, velocities, dt, mass_proton, damping_factor, X, Y)

    if step % solve_every == 0:
        H_time = hamiltonian_2D_multiple(X, Y, dx, mass_proton, nucleus_positions)
        eigenvalues_time, eigenvectors_time = lobpcg(H_time.tocsr(), previous_psi_guess, largest=False, maxiter=max_lobpcg_iter)
        lowest_index = np.argmin(eigenvalues_time)
        #psi = normalize_wavefunction(eigenvectors_time[:, lowest_index], dx)  # if dynamically choosing eigenstate
        psi = normalize_wavefunction(eigenvectors_time[:, num_eigenstates - 1], dx) #if hardcoding eigenstate
        previous_psi_guess = eigenvectors_time
        prev_nucleus_positions = np.array(nucleus_positions)

        #plot_wavefunction_standard(X, Y, psi, eigenvalues_time[lowest_index], lowest_index, nucleus_positions, frame=step)  # if dynamically choosing eigenstate
        plot_wavefunction_standard(X, Y, psi, eigenvalues_time[num_eigenstates - 1], num_eigenstates - 1, nucleus_positions, frame=step)  #if hardcoding eigenstate

        print(f"Step {step}: solve_every = {solve_every}")
