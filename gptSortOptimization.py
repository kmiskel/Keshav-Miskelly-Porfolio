#gpt sorted

import numpy as np
import itertools

# Constants
weight = 28  # lbs
rho = 0.002377  # slugs/ft^3
CL_max = 1.6
wing_area = 5.7  # ft^2
g = 32.2  # ft/s^2
maxLoadFactor = 8
vStall = ((2 * weight) / (rho * CL_max * wing_area)) ** 0.5
numSegments = 5
segmentAngleRad = np.deg2rad(180 / numSegments)

# Candidate grid
n_vals = np.linspace(1.01, maxLoadFactor, 40)
v_vals = np.linspace(vStall, 2 * vStall, 40)

# Store best candidates (n, v, time, radius)
candidates = []

for n in n_vals:
    for v in v_vals:
        R = v ** 2 / (g * np.sqrt(n ** 2 - 1))
        seg_distance = R * segmentAngleRad
        seg_time = seg_distance / v
        candidates.append((n, v, seg_time, R))

# Sort by segment time and take top K
K = 20  # adjustable
top_candidates = sorted(candidates, key=lambda x: x[2])[:K]

# Try all combinations of 5 segments
best_total_time = float('inf')
best_combination = None

for combo in itertools.product(top_candidates, repeat=numSegments):
    total_time = sum(seg[2] for seg in combo)
    if total_time < best_total_time:
        best_total_time = total_time
        best_combination = combo

# Output result
print(f"Best total time: {best_total_time:.2f} s")
for i, seg in enumerate(best_combination):
    n, v, seg_time, R = seg
    print(f"Segment {i+1}: n={n:.2f}, v={v:.2f} ft/s, R={R:.2f} ft, time={seg_time:.2f} s")
