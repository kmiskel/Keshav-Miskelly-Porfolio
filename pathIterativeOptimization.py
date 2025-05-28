import numpy as np

#GPT builds foundation --> build real structure here

#constraints

deltaAltitude = 0 
packEnergy = 80 * 3600 # joules
sumOfEnergy = 0
numSegments = 5
weight = 28 # lbs
rho = 0.002377 # slugs/ft^3
CL_max = 1.6
wing_area = 5.7 # ft^2
vStall = ((2 * weight) / (rho * CL_max * wing_area)) ** 0.5
maxLoadFactor = 8 # g's
g = 32.2 # ft/s^2
dragCoeffiecient = 0.05

# initial Conditions
v0 = vStall # m/s
R = ((v0) ** 2 ) / (g * (((maxLoadFactor) ** 2) - 1) ** 0.5) # ft
orientation = 0 # deg

def simulateTurn(numSegments, v_initial, R_initial):
    totalTime = 0
    # for first seg deltaV = 0
    for seg in range(numSegments):
        segTime = 0
        segDistance = R_initial * np.pi * 2 * ((180/5)/360)
        segTime = 1/v_initial * segDistance
        totalTime += segTime

    return totalTime

print(simulateTurn(numSegments, v0, R))

    

