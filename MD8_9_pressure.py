import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random

# Parameters of the numerical computation
n_steps = 2000  # total number of steps to run
dt = 4         # time step (fs = 1e-15 s)

# Parameters of the confinement sphere
R = 10         # radius of confinement sphere (Å = 1e-10 m)
k = 1e-2       # harmonic force constant of confinement potential (u/fs²)

# Parameters of the Lennard-Jones interaction potential
LJ_depth = 1e-4  # interaction energy at LJ equilibrium distance (u.Å²/fs²)
LJ_dist = 3.0  # LJ equilibrium distance (Å)

# Parameters for electrostatic interactions
epsilon_r = 80.0      # relative dielectric permittivity
C = 1.39e-1 / epsilon_r  # scale of Coulomb interactions, u Å^3 fs^-2 e^-2
# Note: Coulomb's constant in SI units: 8.99e9 N m² C^−2

# System definition
q = 10 * [-1] + 10 * [1]   # particle charges (e = 1.6e-19 C)
m = 20         # particle mass - same for all particles (u = 1.66e-27 kg)

# Parameters for constant-temperature Langevin dynamics
T0 = 300.0     # target temperature (K)
gamma = 1e-2   # friction coefficient (fs^-1)
# Set gamma = 0 to go back to Newtonian dynamics and conserve energy

kB = 8.31e-7   # Boltzmann constant (u. in Å²/fs²/K)
sigma = np.sqrt(2.0 * kB * T0 * m * gamma / dt)  # standard deviation of stochastic force

# Name atoms automatically depending on the sign of their charge
names = []
for qi in q:
    if qi < 0:
        names.append('O')
    elif qi == 0:
        names.append('A')
    else:
        names.append('N')
N = len(q)     # total number of particles

# from particule to a single molecule
n_chain = 6  # number of particles linked with each other
k_bond = 10**(-2)   # force constant of harmonic spring force in u/fs²
d0 = 1.5        # equilibrium length in Å

# Draws N starting positions within the sphere, avoiding clashes


def starting_positions():
    positions = []

    for i in range(N):
        if i < n_chain:
            positions.append(np.array([((i - n_chain / 2) * d0), 0, 0]))

    # "excluded volume" around particles is set to half the sphere volume
    r_exclusion = R / np.power(2. * N, 1. / 3.)
    while len(positions) < N:
        trial_position = np.array([random.uniform(-R, R) for _ in range(3)])
        out_of_sphere = sum(trial_position**2) > R**2
        if out_of_sphere:
            continue  # give up this iteration and go back to while
        clash = False
        for xyz in positions:
            if sum((xyz - trial_position)**2) < r_exclusion**2:
                clash = True
                break   # found a clash, no need to test for more clashes with this particle
        if not clash:
            positions.append(trial_position)
    return positions


# Update list of forces F and return potential energy
def force_pot_energy(pos, v, F):

    # Reinit forces and potential energy
    for i in range(N):
        F[i] = np.array([0.0, 0.0, 0.0])
    Epot = 0.0
    Fwall = 0.0  # force exerted on boundary wall by colliding particles

    for i in range(N):
        # Forces exerted by the thermostat
        if gamma > 0:
            F[i] += -m * gamma * v[i] + np.array([random.normal(0.0, sigma) for _ in range(3)])

        # Confinement potential and force
        r = np.sqrt(sum(pos[i]**2))
        if r > R:
            F[i] += -k * (1.0 - R / r) * pos[i]
            Fwall += k * (r - R)  # sum of forces applied to the confinement wall
            Epot += 0.5 * k * (r - R)**2

        # Interactions between particules
        for j in range(i + 1, N):

            rij = pos[j] - pos[i]
            d2 = sum(rij**2)
            d = np.sqrt(d2)

            if j < n_chain and j == i + 1:
                # from particles to a simple molecule
                Epot += (k_bond * (d - d0) ** 2) / 2  # energy of the bond
                fij = (k_bond * (d - d0) * rij) / -d  # force of the bond

            else:
                # Lennard-Jones / Coulomb interactions
                ratio = LJ_dist / d
                ratio6 = ratio**6
                ratio12 = ratio**12
                fij = (LJ_depth * 12 * (ratio12 - ratio6) / d2 + q[i] * q[j] * C / (d**3)) * rij
                Epot += LJ_depth * (ratio12 - 2 * ratio6) + q[i] * q[j] * C / d

            F[i] -= fij
            F[j] += fij

    # Divide force by area to obtain pressure, convert to atm
    traj_P.append(Fwall / (4. * np.pi * R**2) * 1.64e8)
    return Epot


# Calculate and return kinetic energy
def kinetic_energy(v):
    Ekin = 0.0
    for vi in v:
        Ekin += 0.5 * m * sum(vi**2)
    return Ekin


# Save energies to lists for plotting, and positions to trajectory file
def output(t, pos, Ekin, Epot):
    T = 2 / 3 * Ekin / (N * kB)
    traj_t.append(t)
    traj_T.append(T)
    traj_E.append(Ekin + Epot)
    traj_Ekin.append(Ekin)
    traj_Epot.append(Epot)
    de = pos[n_chain - 1] - pos[0]
    traj_de.append(np.sqrt(sum(de**2)))

    # XYZ file readable by VMD
    traj_file.write('{}\n\n'.format(N))
    for i in range(0, N):
        traj_file.write('{} {} {} {}\n'.format(names[i], *pos[i]))


# Moves positions, velocities and forces forward in time: from t to (t + dt)
def step_forward(pos, v, F):
    # Leapfrog algorithm
    for i in range(N):
        v[i] += 0.5 * F[i] / m * dt
        pos[i] += v[i] * dt
    Epot = force_pot_energy(pos, v, F)
    for i in range(N):
        v[i] += 0.5 * F[i] / m * dt
    Ekin = kinetic_energy(v)
    return Ekin, Epot


# lists for plotting
traj_t = []     # time
traj_T = []     # temperature
traj_E = []     # total energy
traj_Ekin = []  # kinetic energy
traj_Epot = []  # potential energy
traj_P = []     # pressure
traj_de = []    # end-to-end distance between the first and the last polymer
traj_file = open('traj.xyz', 'w')

# Initial conditions
pos = starting_positions()
v = [np.array([random.normal(0.0, np.sqrt(kB * T0 / m)) for _ in range(3)]) for _ in range(N)]
F = [np.array([0.0, 0.0, 0.0]) for _ in range(N)]

# start by calculating force and energies at t = 0
Epot = force_pot_energy(pos, v, F)
Ekin = kinetic_energy(v)
output(0, pos, Ekin, Epot)


# Integration loop
for step in range(n_steps):
    Ekin, Epot = step_forward(pos, v, F)
    output((step + 1) * dt, pos, Ekin, Epot)
    if step % (n_steps / 20) == 0:
        print(f'Progress: {100*(step+1)/n_steps:.0f}%')


traj_file.close()

np.savetxt('pressure.txt', traj_P)
Pideal = N * kB * T0 / ((4. / 3) * np.pi * R**3) * 1.64e8
Pavg = np.array(traj_P).mean()
print(f'Pavg = {Pavg}, Pideal = {Pideal}')


plt.subplot(221)
plt.plot(traj_t, traj_T, label='T')
plt.plot([0, n_steps * dt], [T0, T0], label='T0')
plt.legend()
plt.title('Temperature')
plt.xlabel('t (fs)')
plt.ylabel('T (K)')

plt.subplot(222)
plt.plot(traj_t, traj_Ekin, label='Ekin')
plt.plot(traj_t, traj_Epot, label='Epot')
plt.plot(traj_t, traj_E, label='E')
plt.legend()
plt.title('Energies')
plt.xlabel('t (fs)')
plt.ylabel('E (u.Å^2.fs^-2)')
plt.tight_layout()

plt.subplot(223)
plt.plot(traj_t, traj_P, label='pressure')
plt.plot([0, n_steps * dt], [Pavg, Pavg], label='avg. P')
plt.plot([0, n_steps * dt], [Pideal, Pideal], label='ideal gas P')
plt.legend()
plt.title('Pressure')
plt.xlabel('t (fs)')
plt.ylabel('P (atm)')
plt.tight_layout()

plt.subplot(224)
plt.plot(traj_t, traj_de, label='chain length')
plt.plot(traj_t, [np.mean(traj_de)] * (n_steps + 1), label='average length')
plt.legend()
plt.title('Chain length')
plt.xlabel('t (fs)')
plt.ylabel('de (A)')
plt.tight_layout()
plt.show()
