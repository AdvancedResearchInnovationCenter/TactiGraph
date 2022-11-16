import numpy as np
import rosbag
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

possible_angle = [0.0174532925, 0.034906585, 0.0523598776, 0.075, 0.095, 0.115, 0.135, 0.15]#
N_examples = 17
list_of_rotations = [[0, 0, 0]]

for i in range(1, N_examples):
    theta = i * 2 * np.pi/(N_examples - 1)
    for phi in possible_angle:
        rx = phi * np.cos(theta)
        ry = phi * np.sin(theta)
        rotvec = [rx, ry, 0]
        list_of_rotations.append(rotvec)

cases_dict = {i+1: list_of_rotations[i][:2] for i in range(len(list_of_rotations))}
cases_dict[0] = [0, 0]

dist_from_center = lambda x, y: np.sqrt((x - 157)**2 + (y - 124)**2)
circle_rad=90

def rotate_case(ev_arr, label, angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    centered = ev_arr[:, :2] - np.array([157, 124])
    rot_ev = (R @ centered.T).T + np.array([157, 124])
    
    rot_v = np.array(cases_dict[label])
    new_rot_v = R @ rot_v
    #print(new_rot_v, cases_dict[label])

    best_rot_diff = 100
    best_rot_idx = 1
    i = 1
    
    for rot in list_of_rotations:
        diff_vals = np.sqrt( np.power(rot[0] - new_rot_v[0], 2) +  np.power(rot[1] - new_rot_v[1], 2))
        if best_rot_diff > diff_vals:
            best_rot_diff = diff_vals
            best_rot_idx = i
        i = i + 1
    
    return best_rot_idx, np.concatenate([rot_ev.astype(int), ev_arr[:, 2:]], -1)

class ExtractContactCases:

    def __init__(
        self,
        outdir,
        bag_file_name='/data/dataset_ENVTACT_new2.bag',     
        delta_t = 0.025e9,
        margin = -0.025e9,
        max_events_thresh = 2000,
        augment = False,
        train_prop = 0.6,
        _keep_raw = False
    ):
        self.outdir = Path(outdir)
        self.bag_file_name = bag_file_name

        self.parsed = False

        self.train_prop = train_prop 
        self.augment = augment
        self.delta_t = delta_t
        self.margin = margin
        self.max_events_thresh = max_events_thresh
        self._keep_raw = _keep_raw
        
        self.params = {
            'train_prop': train_prop,
            'augment': augment,
            'delta_t': delta_t,
            'margin': margin,
            'max_events_thresh': max_events_thresh
        }
        

    def parse_bag(self):
        bag_file = rosbag.Bag(self.bag_file_name)
        events = []
        contact_status = []
        contact_status_ts = []
        contact_angle = []
        contact_angle = []

        for topic, msg, t in tqdm(
            bag_file.read_messages(topics=['/contact_status', '/dvs/events', '/contact_angle']), desc='parsing bag'):
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

        self.case_span = 2.66e9
        self.find_ts_idx = lambda ts: np.searchsorted(contact_status_ts, ts)

        i = 0 
        cases_ts = []
        cases_idx = []
        cases = []
        while i < len(contact_status):
            if contact_status[i]:
                init_ts = contact_status_ts[i]
                fin_ts = self.look_ahead_big(init_ts, contact_status, contact_status_ts)
                fin_idx = self.find_ts_idx(fin_ts)
                case = self.find_case(np.mean([init_ts, fin_ts]), contact_angle)
                
                cases.append(case)
                cases_ts.append([init_ts, fin_ts])
                cases_idx.append([i, fin_idx])
                #print(len(cases_ts), init_ts, fin_ts, (fin_ts - init_ts)*1e-9, i, fin_idx, case, '\n')
                i = fin_idx + 1
            else:
                i += 1

        self.events = np.array(events)
        self.cases_idx = cases_idx
        self.cases_ts = cases_ts
        self.cases = cases


    def look_ahead_big(self, ts, contact_status, contact_status_ts):
        fin_ts = ts + self.case_span
        fin_idx = self.find_ts_idx(fin_ts)
        #print(fin_idx)
        if contact_status[fin_idx]:
            #look further
            more = True
            fin_idx_ = fin_idx 
            while more:
                fin_idx_ += 1
                if fin_idx_ - fin_idx > 25:
                    pass#print('warning more than 25 idx away from init_ts + case_span')
                if contact_status[fin_idx_]:
                    continue
                else:
                    more = False
            #print(f'was before case ended by {fin_idx_ - fin_idx} indexes')
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
            #print(f'was ahead case ended by {fin_idx - fin_idx} indexes')
            fin_idx = fin_idx_ + 1
            
        return contact_status_ts[fin_idx]


    def find_case(self, ts, contact_angle):
        idx = self.find_ts_idx(ts)
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

    def _save(self, samples):
        if self._keep_raw:
            self.samples = samples
        sample_idx = list(samples.keys())

        train_idx, val_test_idx = train_test_split(sample_idx, test_size=1-self.train_prop, random_state=0) #fixed across extractions
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=0) #fixed across extractions
        print(len(train_idx), len(val_idx), len(test_idx))
        subsets = zip(['train', 'test', 'val'], [train_idx, val_idx, test_idx])
        
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        with open(self.outdir / 'extraction_params.json', 'w') as f:
            json.dump(self.params, f, indent=4)
    
        for sub_name, subset in subsets:
            if not (self.outdir / sub_name).exists():
                (self.outdir / sub_name / 'raw').mkdir(parents=True)
                (self.outdir / sub_name / 'processed').mkdir(parents=True)
            with open(self.outdir / sub_name / 'raw' / 'contact_cases.json', 'w') as f:
                subset_samples = {}
                for i, subset_idx in enumerate(subset):
                    sample = samples[subset_idx]
                    sample['total_idx'] = subset_idx
                    subset_samples[f'sample_{i+1}'] = sample
                json.dump(subset_samples, f, indent=4)


    def sample_generator(self, label_contact_case, event_arrays):
        for case, event_array in zip(label_contact_case, event_arrays):
            if not self.augment:
               yield case, event_array 
            else:
                for angle in [0, 90, 180, 270]:
                    if angle == 0:
                        yield case, event_array, False
                    else:
                        yield rotate_case(event_array, case, angle), True
            
    def extract(self):
        if not self.parsed:
            self.parse_bag()

        dist_from_center = lambda x, y: np.sqrt((x - 173)**2 + (y - 130)**2)
        circle_rad=90

        event_arrays = []
        label_contact_case = []
        ts = np.array(self.events)[:, 2]
        for i, idx in enumerate(tqdm(self.cases_idx)):
            init_ts_idx = np.searchsorted(ts, self.cases_ts[i][0] + self.margin)
            fin_ts_idx = np.searchsorted(ts, self.cases_ts[i][0] + self.delta_t)
            if fin_ts_idx - init_ts_idx + 1 < 200:
                continue
            elif fin_ts_idx - init_ts_idx + 1 >= self.max_events_thresh:
                event_array = self.events[init_ts_idx:fin_ts_idx+1][-self.max_events_thresh - 1:]
                
            else:
                event_array = self.events[init_ts_idx:fin_ts_idx+1]
                
            in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < circle_rad  
            event_arrays.append(event_array[in_circle, :])
            label_contact_case.append(self.cases[i])

        samples={}
        
        gen = self.sample_generator(label_contact_case, event_arrays) 
        for i, (case, event_array) in enumerate(gen):
            samples[f'sample_{i+1}'] = {
                'events': event_array.tolist(),
                'case': case
                }

        print("saving")
        self.params['n'] = len(samples)
        self._save(samples)
        