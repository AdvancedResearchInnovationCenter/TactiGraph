cases_dict = {
        0: [0, 0], # what to do with no contact case ??
        1: [0, 0],
        2: [0.01234134148078231, 0.01234134148078231],
        3: [0.02468268296156462, 0.02468268296156462],
        4: [0.03702402451305761, 0.03702402451305761],
        5: [0.053033008588991064, 0.053033008588991064],
        6: [0.06717514421272203, 0.06717514421272203],
        7: [0.08131727983645297, 0.08131727983645297],
        8: [0.09545941546018392, 0.09545941546018392],
        9: [0.10606601717798213, 0.10606601717798213],
        10: [1.0687059397353753e-18, 0.0174532925],
        11: [2.1374118794707506e-18, 0.034906585],
        12: [3.2061178253293598e-18, 0.0523598776],
        13: [4.592425496802574e-18, 0.075],
        14: [5.817072295949928e-18, 0.095],
        15: [7.04171909509728e-18, 0.115],
        16: [8.266365894244634e-18, 0.135],
        17: [9.184850993605149e-18, 0.15],
        18: [-0.012341341480782309, 0.01234134148078231],
        19: [-0.024682682961564617, 0.02468268296156462],
        20: [-0.037024024513057606, 0.03702402451305761],
        21: [-0.05303300858899106, 0.053033008588991064], 
        22: [-0.06717514421272201, 0.06717514421272203], 
        23: [-0.08131727983645295, 0.08131727983645297], 
        24: [-0.09545941546018391, 0.09545941546018392], 
        25: [-0.10606601717798211, 0.10606601717798213], 
        26: [-0.0174532925, 2.1374118794707506e-18], 
        27: [-0.034906585, 4.274823758941501e-18], 
        28: [-0.0523598776, 6.4122356506587196e-18], 
        29: [-0.075, 9.184850993605149e-18], 
        30: [-0.095, 1.1634144591899856e-17], 
        31: [-0.115, 1.408343819019456e-17], 
        32: [-0.135, 1.653273178848927e-17], 
        33: [-0.15, 1.8369701987210297e-17], 
        34: [-0.012341341480782312, -0.012341341480782309], 
        35: [-0.024682682961564624, -0.024682682961564617], 
        36: [-0.03702402451305762, -0.037024024513057606], 
        37: [-0.05303300858899108, -0.05303300858899106], 
        38: [-0.06717514421272203, -0.06717514421272201], 
        39: [-0.08131727983645298, -0.08131727983645295], 
        40: [-0.09545941546018394, -0.09545941546018391], 
        41: [-0.10606601717798216, -0.10606601717798211], 
        42: [-3.2061178192061255e-18, -0.0174532925],
        43: [-6.412235638412251e-18, -0.034906585], 
        44: [-9.618353475988079e-18, -0.0523598776],
        45: [-1.3777276490407722e-17, -0.075],
        46: [-1.745121688784978e-17, -0.095],
        47: [-2.1125157285291842e-17, -0.115], 
        48: [-2.4799097682733903e-17, -0.135],
        49: [-2.7554552980815445e-17, -0.15], 
        50: [0.012341341480782307, -0.012341341480782312], 
        51: [0.024682682961564614, -0.02468268296156462], 
        52: [0.0370240245130576, -0.03702402451305762], 
        53: [0.05303300858899105, -0.05303300858899108], 
        54: [0.067175144212722, -0.06717514421272203], 
        55: [0.08131727983645295, -0.08131727983645298], 
        56: [0.0954594154601839, -0.09545941546018394], 
        57: [0.1060660171779821, -0.10606601717798216], 
        58: [0.0174532925, -4.274823758941501e-18], 
        59: [0.034906585, -8.549647517883002e-18], 
        60: [0.0523598776, -1.2824471301317439e-17], 
        61: [0.075, -1.8369701987210297e-17], 
        62: [0.095, -2.3268289183799712e-17], 
        63: [0.115, -2.816687638038912e-17], 
        64: [0.135, -3.306546357697854e-17], 
        65: [0.15, -3.6739403974420595e-17]
    }



circle_center=(173, 130)
circle_rad=90
im_height=260
im_width=346

import pickle
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



with open('/home/hussain/tactile/data/raw/event_arrays.pkl', 'rb') as p:
    event_arrays=pickle.load(p)
    
with open('/home/hussain/tactile/data/raw/case.pkl', 'rb') as p:
    cases=pickle.load(p)


samples={}
for i, (case, event_array) in enumerate(zip(cases, event_arrays)):
    samples[f'sample_{i+1}'] = {
        'events': event_array.tolist(),
        'case': case
    }


ids = np.random.choice(list(samples.keys()), 1, replace=False)
dist_from_center = lambda x, y: np.sqrt((x - 173)**2 + (y - 130)**2)
id_ = ids[0]    
events = np.array(samples[id_]['events'])
in_circle = len(events)*[True]#dist_from_center(events[:, 0], events[:, 1]) < circle_rad 

events_ = events.copy().astype(float)
events_[:, 2] = (events[:, 2] - events[:, 2].min()) / (events[:, 2].max() - events[:, 2].min())
events_

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

events_ = events.copy().astype(float)
events_[:, 2] = (events[:, 2] - events[:, 2].min()).astype(float)/(events[:, 2].max() - events[:, 2].min())
plt.scatter(*np.flip(events_[:, :3]).T)
plt.show()
