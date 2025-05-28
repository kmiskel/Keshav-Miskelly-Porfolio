import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Constants
DRAG_COEFFICIENT = 0.5  # kg/m - drag force = DRAG_COEFFICIENT * v^2
MASS = 70  # kg (vehicle + rider)
G = 9.81  # m/s^2

# Initial conditions
INITIAL_VELOCITY = 1.0  # m/s
INITIAL_RADIUS = 10.0  # m
PACK_ENERGY = 80 * 3600  # Joules
NUM_SEGMENTS = 5
DELTA_ALTITUDE = 0  # m (no altitude change)

def calculate_turn_parameters(v, R, segment_length):
    """
    Calculate energy expenditure and forces for a turn segment.
    
    Args:
        v: velocity (m/s)
        R: turn radius (m)
        segment_length: length of the segment (m)
        
    Returns:
        tuple: (time, centripetal_force, drag_force, energy_used)
    """
    # Time to complete segment
    t = segment_length / v if v != 0 else float('inf')
    
    # Centripetal force required
    F_centripetal = MASS * v**2 / R if R != 0 else float('inf')
    
    # Drag force
    F_drag = DRAG_COEFFICIENT * v**2
    
    # Total energy used (work against drag force)
    energy = F_drag * segment_length
    
    return t, F_centripetal, F_drag, energy

def simulate_turn(params):
    """
    Simulate the 180-degree turn with given parameters.
    
    Args:
        params: array of parameters (radius for each segment)
        
    Returns:
        tuple: (total_energy, success_flag, velocities, energies)
    """
    # Split the 180-degree turn into segments
    segment_angles = np.linspace(0, np.pi, NUM_SEGMENTS + 1)
    segment_angles_diff = np.diff(segment_angles)
    
    # Initialize variables
    v = INITIAL_VELOCITY
    total_energy = 0
    velocities = [v]
    energies_per_segment = []
    success = True
    
    # Simulate each segment
    for i in range(NUM_SEGMENTS):
        R = params[i]
        angle = segment_angles_diff[i]
        segment_length = R * angle  # Arc length
        
        # Calculate segment parameters
        t, F_cent, F_drag, energy = calculate_turn_parameters(v, R, segment_length)
        
        # Check if we have enough energy
        if total_energy + energy > PACK_ENERGY:
            success = False
            break
            
        # Update state
        total_energy += energy
        velocities.append(v)  # Assuming constant velocity per segment for now
        energies_per_segment.append(energy)
    
    return total_energy, success, velocities, energies_per_segment

def objective_function(params):
    """
    Objective function to minimize (total time through turn).
    """
    total_energy, success, velocities, _ = simulate_turn(params)
    
    if not success:
        return float('inf')  # Penalize solutions that exceed energy budget
    
    # Calculate total time (sum of segment times)
    segment_angles = np.linspace(0, np.pi, NUM_SEGMENTS + 1)
    segment_angles_diff = np.diff(segment_angles)
    total_time = 0
    
    for i in range(NUM_SEGMENTS):
        R = params[i]
        angle = segment_angles_diff[i]
        segment_length = R * angle
        v = velocities[i]
        total_time += segment_length / v
        
    return total_time

def optimize_turn():
    """
    Optimize the turn parameters to minimize time within constraints.
    """
    # Initial guess (constant radius)
    initial_params = [INITIAL_RADIUS] * NUM_SEGMENTS
    
    # Bounds (minimum radius could be physical limit, e.g., 1m)
    bounds = [(1, 20)] * NUM_SEGMENTS  # Example bounds
    
    # Constraints (none beyond bounds in this simple version)
    
    # Run optimization
    result = minimize(
        objective_function,
        initial_params,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 100}
    )
    
    return result

def visualize_turn(params):
    """
    Visualize the optimized turn path.
    """
    _, _, velocities, energies = simulate_turn(params)
    
    # Calculate turn segments
    segment_angles = np.linspace(0, np.pi, NUM_SEGMENTS + 1)
    
    # Plot the path
    plt.figure(figsize=(10, 5))
    
    # Path
    for i in range(NUM_SEGMENTS):
        R = params[i]
        start_angle = segment_angles[i]
        end_angle = segment_angles[i+1]
        theta = np.linspace(start_angle, end_angle, 100)
        x = R * np.cos(theta)
        y = R * np.sin(theta)
        plt.plot(x, y, label=f'Segment {i+1} (R={R:.1f}m)')
    
    plt.title('Optimized 180-Degree Turn Path')
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot velocities and energies
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(NUM_SEGMENTS), velocities[:-1], 'o-')
    plt.title('Velocity per Segment')
    plt.xlabel('Segment')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(NUM_SEGMENTS), energies)
    plt.title('Energy per Segment')
    plt.xlabel('Segment')
    plt.ylabel('Energy (J)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run optimization
if __name__ == "__main__":
    result = optimize_turn()
    
    if result.success:
        print("Optimization successful!")
        print(f"Optimal segment radii: {result.x}")
        print(f"Minimum time through turn: {result.fun:.2f} seconds")
        
        # Visualize the result
        visualize_turn(result.x)
    else:
        print("Optimization failed to converge.")
        print(result.message)