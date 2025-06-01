import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Parameters ===
n_segments = 50
R = 10  # meters
theta = 1 * np.pi / n_segments  # total angle covered
mass = 10  # kg
g = 9.81  # gravity
k_drag = 0.1  # drag coefficient
pack_energy = (80 * 3600) * 12 # joules (battery energy)

# === Initial Conditions ===
V0 = 1.0  # initial velocity in m/s

# === Helper Functions ===
def segment_time(R, theta, V_in, V_out):
    return R * theta / ((V_in + V_out) / 2 + 1e-6)  # Avoid divide-by-zero

def objective(x):
    delta_Z = x[:n_segments]
    delta_E = x[n_segments:]

    V = np.zeros(n_segments + 1)
    V[0] = V0
    total_time = 0.0

    for i in range(n_segments):
        V_avg = V[i]
        drag = k_drag * V_avg**2 * R * theta
        delta_V2 = (-delta_E[i] - mass * g * delta_Z[i] - drag) * 2 / mass
        V[i+1] = np.sqrt(max(V[i]**2 + delta_V2, 1e-6))
        total_time += segment_time(R, theta, V[i], V[i+1])

    return total_time

# === Constraints ===
def total_altitude_change(x):
    return np.sum(x[:n_segments])  # Return to same altitude

def total_energy_used(x):
    return pack_energy - np.sum(x[n_segments:])  # Stay within energy limit

# === Initial Guess ===
x0 = np.zeros(2 * n_segments)

# === Bounds ===
bounds = [(-12, 12)] * n_segments + [(-pack_energy/n_segments, pack_energy/n_segments)] * n_segments

constraints = [
    {'type': 'eq', 'fun': total_altitude_change},
    {'type': 'ineq', 'fun': total_energy_used}
]

# === Run Optimization ===
result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
)

# === Plotting ===
if result.success:
    delta_Z = result.x[:n_segments]
    delta_E = result.x[n_segments:]
    altitude = np.cumsum(np.insert(delta_Z, 0, 0))
    V = [V0]
    segment_times = []

    for i in range(n_segments):
        V_avg = V[i]
        drag = k_drag * V_avg**2 * R * theta
        delta_V2 = (-delta_E[i] - mass * g * delta_Z[i] - drag) * 2 / mass
        V_next = np.sqrt(max(V[i]**2 + delta_V2, 1e-6))
        V.append(V_next)
        segment_times.append(segment_time(R, theta, V[i], V_next))

    total_time = result.fun
    print(f"Total time to complete the path: {total_time:.2f} seconds")

    # --- 2D Plots: Altitude and Velocity ---
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(altitude)
    plt.title("Altitude Profile")
    plt.xlabel("Segment")
    plt.ylabel("Altitude (m)")

    plt.subplot(1, 2, 2)
    plt.plot(V)
    plt.title("Velocity Profile")
    plt.xlabel("Segment")
    plt.ylabel("Velocity (m/s)")

    plt.tight_layout()
    plt.show()

    # --- 3D Path Plot ---
    angles = np.linspace(0, n_segments * theta, n_segments + 1)
    x = R * np.cos(angles)
    y = R * np.sin(angles)
    z = altitude

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, marker='o')
    ax.set_title("3D Optimized Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.view_init(elev=30, azim=120)  # Adjust view angle
    plt.tight_layout()
    plt.show()

else:
    print("Optimization failed:", result.message)

# --- Energy Usage per Segment ---
energy_used = delta_E  # already computed

plt.figure(figsize=(8, 4))
plt.plot(-energy_used, marker='o', linestyle='-')  # Positive = energy drawn
plt.ylabel("Energy Drawn (Joules)")
plt.title("Energy Expended per Segment")
plt.xlabel("Segment")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Energy Usage Check ===
total_energy_used = -np.sum(delta_E)  # Negative since usage is stored as negative in delta_E
energy_left = pack_energy - total_energy_used

print(f"Total battery energy: {pack_energy:.2f} J")
print(f"Total energy used   : {total_energy_used:.2f} J")
print(f"Energy remaining    : {energy_left:.2f} J")

if np.isclose(total_energy_used, pack_energy, rtol=1e-3):
    print("✅ Full battery energy was used (within tolerance).")
elif total_energy_used < pack_energy:
    print("⚠️  Not all battery energy was used — optimizer may be conserving energy.")
else:
    print("❌ Exceeded battery limit! (This shouldn't happen)")
