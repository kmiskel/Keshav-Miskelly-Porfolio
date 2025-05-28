import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Global Constants ===
mass = 10  # kg
g = 9.81  # m/s^2
k_drag = 0.1  # drag coefficient
pack_energy = 80 * 3600  # Joules
V0 = 1.0  # initial velocity (m/s)
n_per_section = 10  # resolution per segment
initial_radius = 10.0  # starting radius for turns

# === Path Structure ===
segment_types = ['straight', 'turn', 'straight', 'turn']
straight_length = 304.8
segment_lengths = [straight_length, np.pi * initial_radius, straight_length, np.pi * initial_radius]

# === Segment Metadata ===
segments = []
current_idx = 0
n_total = len(segment_types) * n_per_section

for i, (stype, L) in enumerate(zip(segment_types, segment_lengths)):
    n_seg = n_per_section if i < len(segment_types) - 1 else n_total - current_idx
    segments.append({
        'type': stype,
        'length': L,
        'n': n_seg,
        'start_idx': current_idx,
        'end_idx': current_idx + n_seg
    })
    current_idx += n_seg

# === Segment Lookup ===
def get_segment(i):
    for seg in segments:
        if seg['start_idx'] <= i < seg['end_idx']:
            return seg
    return segments[-1]

# === Segment Physics ===
def segment_time(L, V_in, V_out):
    return L / (0.5 * (V_in + V_out) + 1e-6)

# === Objective: Minimize Time ===
def objective(x):
    delta_Z = x[:n_total]
    delta_E = x[n_total:2 * n_total]
    radii = x[2 * n_total:]

    V = np.zeros(n_total + 1)
    V[0] = V0
    total_t = 0.0

    for i in range(n_total):
        seg = get_segment(i)
        if seg['type'] == 'straight':
            L = seg['length'] / seg['n']
        else:
            R = radii[i]
            L = np.pi * R / seg['n']

        drag = k_drag * V[i]**2 * L
        delta_V2 = (-delta_E[i] - mass * g * delta_Z[i] - drag) * 2 / mass
        V[i + 1] = np.sqrt(max(V[i]**2 + delta_V2, 1e-6))
        total_t += segment_time(L, V[i], V[i + 1])

    return total_t

# === Constraints ===
def total_altitude_zero(x):
    return np.sum(x[:n_total])

def total_energy_within_budget(x):
    return pack_energy - np.sum(x[n_total:2 * n_total])

def radius_above_min(x):
    return x[2 * n_total:] - 1.0

# === Initial Guess and Bounds ===
x0 = np.zeros(3 * n_total)
x0[2 * n_total:] = initial_radius  # radius guess

bounds = [(-10, 10)] * n_total + \
         [(-pack_energy / n_total, pack_energy / n_total)] * n_total + \
         [(1.0, 100.0)] * n_total

constraints = [
    {'type': 'eq', 'fun': total_altitude_zero},
    {'type': 'ineq', 'fun': total_energy_within_budget},
    
]

# === Optimization ===
result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'disp': True, 'maxiter': 1000, 'ftol': 1e-6}
)

# === Plotting if Successful ===
if result.success:
    delta_Z = result.x[:n_total]
    delta_E = result.x[n_total:2 * n_total]
    radii = result.x[2 * n_total:]
    altitude = np.cumsum(np.insert(delta_Z, 0, 0))

    # Velocity profile reconstruction
    V = [V0]
    for i in range(n_total):
        seg = get_segment(i)
        if seg['type'] == 'straight':
            L = seg['length'] / seg['n']
        else:
            R = radii[i]
            L = np.pi * R / seg['n']

        drag = k_drag * V[i]**2 * L
        delta_V2 = (-delta_E[i] - mass * g * delta_Z[i] - drag) * 2 / mass
        V.append(np.sqrt(max(V[i]**2 + delta_V2, 1e-6)))

    # 3D Trajectory
    pos = np.array([0.0, 0.0])
    angle = 0.0
    x_path, y_path, z_path = [pos[0]], [pos[1]], [altitude[0]]

    for i in range(n_total):
        seg = get_segment(i)
        if seg['type'] == 'straight':
            step = seg['length'] / seg['n']
            dx = np.cos(angle)
            dy = np.sin(angle)
            pos += step * np.array([dx, dy])
        else:
            R = radii[i]
            dtheta = np.pi / seg['n']
            angle += dtheta
            dx = np.cos(angle)
            dy = np.sin(angle)
            pos += R * dtheta * np.array([dx, dy])
        x_path.append(pos[0])
        y_path.append(pos[1])
        z_path.append(altitude[i + 1])

    # === Plots ===
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_path, y_path, z_path, label="Trajectory", color='blue')
    ax.set_title("Optimized 3D Path")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.legend()
    plt.show()

    plt.figure()
    plt.plot(altitude, label="Altitude")
    plt.title("Altitude vs Segment")
    plt.xlabel("Segment")
    plt.ylabel("Altitude (m)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(V, label="Velocity", color='green')
    plt.title("Velocity vs Segment")
    plt.xlabel("Segment")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(-delta_E, label="Energy Used per Segment", color='red')
    plt.title("Energy Consumption")
    plt.xlabel("Segment")
    plt.ylabel("Energy (J)")
    plt.grid(True)
    plt.show()

    print(f"✅ Optimization completed.")
    print(f"Total time: {result.fun:.2f} s")
    total_energy_used = -np.sum(delta_E)
    print(f"Energy used: {total_energy_used:.2f} J / {pack_energy:.2f} J")
else:
    print("❌ Optimization failed:", result.message)
