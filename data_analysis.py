import os

import numpy as np
import pandas as pd

from trackml.dataset import load_event
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

event_prefix = 'event000001000'
hits, cells, particles, truth = load_event(os.path.join('train_100_events', event_prefix))
cond = (hits['volume_id'] == 8) | (hits['volume_id'] == 13) | (hits['volume_id'] == 17)
selected_indices = hits.index[cond].tolist()
selected_hits = hits.iloc[selected_indices]
selected_truth = truth.iloc[selected_indices]
#assert len(selected_hits) == len(selected_truth)
#get every 100th particle

plt.figure(figsize=(10,10))
ax = plt.axes()
tracks = selected_truth.particle_id.unique()[:10]
for track in tracks:
    if track == 0:
        continue
    t = selected_truth[selected_truth.particle_id == track]
    ax.plot(t.tx, t.ty)

plt.show()

'''conf_x = []
conf_y = []
for i, obj in enumerate(zip(selected_hits.x, selected_hits.y)):
    x = obj[0]
    y = obj[1]
    denom = float(x*x) + float(y*y)
    cx = x / denom
    cy = y / denom
    conf_x.append(cx)
    conf_y.append(cy)


conf_x = []
conf_y = []
for i, obj in enumerate(zip(selected_truth.tx, selected_truth.ty)):
    x = obj[0]
    y = obj[1]
    denom = float(x*x) + float(y*y)
    cx = x / denom
    cy = y / denom
    conf_x.append(cx)
    conf_y.append(cy)

#radialview = sns.jointplot(selected_hits.x, selected_hits.y, size=10, s=1)
radialview = sns.jointplot(selected_truth.tx, selected_truth.ty, size=10, s=1)
radialview.set_axis_labels('x (mm)', 'y (mm)')
plt.show()'''