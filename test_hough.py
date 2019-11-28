import os
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from hough import *
from conformalMap import *
#%matplotlib inline


event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('train_100_events', event_prefix))
averageNumberHits = particles['nhits'].mean()
#hits.head()
#cells.head()
particles.head()
#truth.head()
print("Average number of hits is:", averageNumberHits)


cond = (hits['volume_id'] == 8) | (hits['volume_id'] == 13) | (hits['volume_id'] == 17)
selected_indices = hits.index[cond].tolist()
selected_hits = hits.iloc[selected_indices]
selected_truth = truth.iloc[selected_indices]

percentageErrors = np.array([])
# Separate the tracks
allTracks = selected_truth.particle_id.unique()
for i in range(4500, 5000):    
    tracks = [allTracks[i]]
    if len(selected_truth[selected_truth.particle_id == tracks[0]]) < 3:
        continue
    allTrackPoints = np.empty((0, 2))
    for track in tracks:
        # A value of 0 means that the hit did not originate 
        # from a reconstructible particle, but e.g. from detector noise.
        if track == 0:
            continue
        t = selected_truth[selected_truth.particle_id == track]
        ctx, cty = conformalMapping(zip(t.tx, t.ty))
        ct = np.dstack([ctx, cty])[0]

        allTrackPoints = np.append(allTrackPoints, ct, axis=0)

    # Assign values to hough space
    from math import pi
    maxRho = max([np.linalg.norm(p) for p in allTrackPoints])
    minRho = -maxRho
    minTheta = 0
    maxTheta = pi - 10**-100
    # Best values seem to be 250 and 360 respectively 
    nrBinsRho = 1000 #360
    nrBinsTheta = 360 #360

    accumulator, maxRho, minRho, nrBinsRho, nrBinsTheta, thetas = createHoughSpace(allTrackPoints, minRho, maxRho, minTheta, maxTheta, nrBinsRho, nrBinsTheta)
    
    threshhold = averageNumberHits # Super dependent on this
    maxPoints = findMaxValues(accumulator, nrBinsRho, nrBinsTheta, threshhold)
    if len(maxPoints) == 0:
        maxIndex = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        maxPoints = [maxIndex]
        
    maxRhos, maxThetas = extractRhoAndTheta(maxPoints, thetas, minRho, maxRho, nrBinsRho)
    for i in range(len(maxRhos)):
        if maxRhos[i] == 0:
            maxRhos[i] = 10**-5
    
    lines = np.dstack((maxRhos, maxThetas))[0]
    circles = inverseConformalMapping(lines)
    print("Found", len(circles), "circles")

    momentum = np.array([])
    for circle in circles:
        R = circle[2] # radius of circle
        if R < 0:
            R *= -1
        q = 1 # charge of particle
        B = 2 # magnetic field strength
        p = 0.3 * B * R * 0.001 # 0.001 to get the 10th order correct ;)
        momentum = np.append(momentum, np.array([p]))
    trueMomentum = np.array([])
    
    for track in tracks:
        # A value of 0 means that the hit did not originate 
        # from a reconstructible particle, but e.g. from detector noise.
        if track == 0:
            continue
        t = selected_truth[selected_truth.particle_id == track]
        # Seems that whichever index I pick it has the same momentum (I belive it makes sense)
        px = t['tpx'].iloc[0]
        py = t['tpy'].iloc[0]
        trueMomentum = np.append(trueMomentum, np.linalg.norm(np.array([px, py])))
        
    minPercentage = float('inf')
    print(trueMomentum)
    for tm in trueMomentum:
        for m in momentum:
            actual = m
            ideal = tm
            percentageError = abs((actual - ideal) / ideal) * 100
            if percentageError < minPercentage:
                minPercentage = percentageError
    if len(trueMomentum) != 0 and len(momentum) != 0:
        percentageErrors = np.append(percentageErrors, minPercentage)
    print("mean percentage error:", np.mean(percentageErrors))

# mean percentage error for 500 tracks (4500 -> 5000) is 15.44