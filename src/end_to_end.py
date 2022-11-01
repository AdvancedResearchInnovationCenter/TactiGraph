#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import models
import numpy as np
import h5py
import sys, os
import rospy
import rosbag
import math
from scipy.interpolate import interp1d
import h5py
import matplotlib.pylab as plt
import time 


# In[2]:


#parameters of processing
frequency = 30 #Hz
bag_file_name = '/data/dataset_ENVTACT_new2.bag'#'/home/hussain/me/projects/tactile/data/dataset_ENVTACT_new2.bag'
time_window_size = 6
examples_per_edge = 1

bag_file = rosbag.Bag(bag_file_name)


# In[3]:


events = []
contact_status = []
contact_status_ts = []
contact_case = [] #0:No contact 1: center, 2:remainder of contacts as in list_of_rotations
contact_case_ts = []
contact_angle = []

contact_case_updated = []
contact_case_updated_ts = []

event_packet_size = []
event_packet_size_ts = []
contact_angle = []


#generate labels
possible_angle = [0.0174532925, 0.034906585, 0.0523598776, 0.075, 0.095, 0.115, 0.135, 0.15]#
N_examples = 17
list_of_rotations = [[0, 0, 0]]

for i in range(1, N_examples):
    theta = i * 2 * math.pi/(N_examples - 1)
    for phi in possible_angle:
        rx = phi * math.cos(theta)
        ry = phi * math.sin(theta)
        rotvec = [rx, ry, 0]
        list_of_rotations.append(rotvec)

print(len(list_of_rotations))
print(list_of_rotations)


# In[4]:


from tqdm.auto import tqdm
for topic, msg, t in tqdm(bag_file.read_messages(topics=['/contact_status', '/dvs/events', '/contact_angle'])):
    if topic == '/dvs/events':
        for e in msg.events:
            event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
            events.append(event)
    elif topic == '/contact_status':
        contact_status.append(msg.data)
        contact_status_ts.append(t.to_nsec())
    elif topic == '/contact_angle':
        contact_angle.append([msg.x, msg.y, msg.z])
        
        # Updated contact status according to no. of events
        
            
#print(events)
bag_file.close()


# In[ ]:


case_span = 2.66e9
find_ts_idx = lambda ts: np.searchsorted(contact_status_ts, ts)

def look_ahead_big(ts, idx_ts):
    fin_ts = ts + case_span
    fin_idx = find_ts_idx(fin_ts)
    print(fin_idx)
    if contact_status[fin_idx]:
        #look further
        more = True
        fin_idx_ = fin_idx 
        while more:
            fin_idx_ += 1
            if fin_idx_ - fin_idx > 25:
                print('warning more than 25 idx away from init_ts + case_span')
            if contact_status[fin_idx_]:
                continue
            else:
                more = False
        print(f'was before case ended by {fin_idx_ - fin_idx} indexes')
        fin_idx = fin_idx_ - 1
    else:
        #look backwards
        more = True
        fin_idx_ = fin_idx 
        while more:
            fin_idx_ -= 1
            if not contact_status[fin_idx_]:
                continue
            else:
                more = False
        print(f'was ahead case ended by {fin_idx - fin_idx} indexes')
        fin_idx = fin_idx_ + 1
        
    return contact_status_ts[fin_idx]


def find_case(ts):
    idx = find_ts_idx(ts)
    best_rot_diff = 100
    best_rot_idx = 1
    i = 1
    x, y, z = contact_angle[idx]
    for rot in list_of_rotations:
        diff_vals = np.sqrt(np.power(rot[0] - x, 2) +  np.power(rot[1] - y, 2) + np.power(rot[2] - z, 2))
        if best_rot_diff > diff_vals:
            best_rot_diff = diff_vals
            best_rot_idx = i
        i = i + 1
    return best_rot_idx


i = 0 
cases_ts = []
cases_idx = []
cases = []
while i < len(contact_status):
    if contact_status[i]:
        init_ts = contact_status_ts[i]
        fin_ts = look_ahead_big(init_ts, i)
        fin_idx = find_ts_idx(fin_ts)
        case = find_case(np.mean([init_ts, fin_ts]))
        
        cases.append(case)
        cases_ts.append([init_ts, fin_ts])
        cases_idx.append([i, fin_idx])
        print(len(cases_ts), init_ts, fin_ts, (fin_ts - init_ts)*1e-9, i, fin_idx, case, '\n')
        i = fin_idx + 1
    else:
        i += 1


# In[ ]:


contact_case = np.zeros(len(contact_status))
for i, case in enumerate(cases):
    contact_case[cases_idx[i][0]:cases_idx[i][1] + 1] = case
    
plt.figure(figsize=(140, 40))
plt.plot(contact_status_ts, contact_case)


# In[ ]:


f_case = interp1d(contact_status_ts, contact_case, kind='previous')
contact_case_ts_int = np.arange(min(contact_status_ts), max(contact_status_ts), int(1e9/frequency))

contact_case_int = f_case(contact_case_ts_int)


# In[ ]:


def find_interp_idx(ts):
    return np.where(ts - contact_case_ts_int < 0)[0][0]


# In[ ]:


cases_int_idx = []

for i, case_ts in enumerate(cases_ts):
    case_int_idx = list(map(find_interp_idx, case_ts))
    cases_int_idx.append(case_int_idx)


# In[ ]:


events=  np.array(events)
ts = events[:, 2]


# In[ ]:


hist, bin_edges = np.histogram(ts, bins=int(1e5))


# In[ ]:


(ts[-1] - ts[0])*1e-4


# In[ ]:


3240219.741*1e-9


# In[ ]:


plt.figure(figsize=(200, 10))
plt.plot(contact_case_ts_int, contact_case_int > 0, color='black')
plt.plot(bin_edges[:-1], (hist - hist.min()) / (hist.max() - hist.min()), c='blue')


# In[ ]:


delta_t = 0.025e9
margin = -0.025e9
dist_from_center = lambda x, y: np.sqrt((x - 173)**2 + (y - 130)**2)
circle_rad=90

event_arrays = []
label_contact_case = []

for i, idx in enumerate(tqdm(cases_idx)):
    init_ts_idx = np.searchsorted(ts, cases_ts[i][0] + margin)
    fin_ts_idx = np.searchsorted(ts, cases_ts[i][0] + delta_t)
    if fin_ts_idx - init_ts_idx + 1 < 200:
        continue
    elif fin_ts_idx - init_ts_idx + 1 >= 2000:
        event_array = events[init_ts_idx:fin_ts_idx+1][-2001:]
        
    else:
        event_array = events[init_ts_idx:fin_ts_idx+1]
        
    in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < circle_rad  
    event_arrays.append(event_array[in_circle, :])
    label_contact_case.append(cases[i])


# In[ ]:


import seaborn as sns
def plot(ev):
    print(ev.shape)
    
    sns.scatterplot( x=ev[:, 0],y=ev[:, 1], hue=ev[:, 3])
    # sns.scatterplot(x=ev[:, 0], y=ev[:, 1], hue=ev[:, 3])
plot(event_arrays[52])


# In[ ]:


plt.figure(figsize=(10 , 10))
case = 1
all_ = cases_idx[5*case:5*case + 5]
init = all_[0][0]
final = all_[-1][1] 


# In[ ]:


def plot_case(case):
    plt.figure(figsize=(10 , 10))
    idx = np.where(np.array(cases) == case)[0]
    all_ = np.array(cases_int_idx)[idx]
    init = all_[0][0] -3 
    final = all_[-1][1] + 3
    
    plt.plot(contact_case_ts_int[init:final]-contact_case_ts_int[0], contact_case_int[init:final] / (case))
    
    
    init_events = np.searchsorted(bin_edges, np.array(cases_ts)[idx][0][0]) - 5
    final_events = np.searchsorted(bin_edges, np.array(cases_ts)[idx][-1][1])+ 5
    
    plt.plot(bin_edges[init_events:final_events] - bin_edges[0], hist[init_events:final_events] / hist.max())
    plt.scatter(bin_edges[init_events:final_events] - bin_edges[0], hist[init_events:final_events] / hist.max(), s=3)
    #print(hist.max())
    
    
    
plot_case(45)


# In[ ]:


n_events_case = np.array([len(a) for a in event_arrays])
len(n_events_case)


# In[ ]:


np.sort(n_events_case)[:10]


# In[ ]:


np.argsort(n_events_case)[:10]


# In[ ]:


for i in range(1, 129):
    idx= np.where(np.array(label_contact_case) == i)[0]
    sort = np.sort(n_events_case[idx])
    print(i,  len(idx), np.mean(sort), sort)


# In[ ]:


samples={}

for i, (case, event_array) in enumerate(zip(label_contact_case, event_arrays)):
    samples[f'sample_{i+1}'] = {
        'events': event_array.tolist(),
        'case': case
        }


# In[ ]:


from sklearn.model_selection import train_test_split
from pathlib import Path
import json

outdir = Path('/home/hussain/tactile/data/contact_extraction5')

sample_idx = list(samples.keys())

train_idx, val_test_idx = train_test_split(sample_idx, test_size=0.4, random_state=0) #fixed across extractions
val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=0) #fixed across extractions

params = {}


subsets = zip(['train', 'test', 'val'], [train_idx, val_idx, test_idx])

for sub_name, subset in subsets:
    if not (outdir / sub_name).exists():
        (outdir / sub_name / 'raw').mkdir(parents=True)
        (outdir / sub_name / 'processed').mkdir(parents=True)

        with open(outdir / sub_name / 'extraction_params.json', 'w') as f:
            json.dump(params, f, indent=4)

    with open(outdir / sub_name / 'raw' / 'contact_cases.json', 'w') as f:
        subset_samples = {}
        for i, subset_idx in enumerate(subset):
            sample = samples[subset_idx]
            sample['total_idx'] = subset_idx
            subset_samples[f'sample_{i+1}'] = sample
        json.dump(subset_samples, f, indent=4)


# In[ ]:


from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from torch_geometric.transforms import Distance, Cartesian
from imports.TrainModel import TrainModel
seed_everything(0)

from models.spline import nvs_no_skip as splinenet

model = splinenet().to('cuda')
model
#!rm ../data/contact_extraction2/{train,test,val}/processed/*

tm = TrainModel('/home/hussain/tactile/data/contact_extraction5/', model, n_epochs=150, transform=Cartesian(cat=False))


# In[ ]:


tm.train()

