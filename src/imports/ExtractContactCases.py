import numpy as np

try:
    import rosbag
except ImportError:
    print('ROS not found')

from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import math
from .TactileBag import TactileBag
import dask.array as da
EXTRACTIONS_DIR = Path('../data/extractions').resolve()

class ExtractContactCases:

    def __init__(
        self,
        tactile_bag_dir,     
        outdir,
        delta_t = 0.075e9,
        margin = -0.025e9,
        case_span = 2.66e9,
        min_n_events = 750,
        n_init_events = 5000,
        down_sample = 1,
        train_prop = 0.75,
        center = (180, 117),
        circle_rad=85,
        event_array_augmentations = [],
        event_array_filters = [],
        keep_interm = False
        ):
        self.outdir = Path(outdir).resolve()
        tactile_bag_dir = Path(tactile_bag_dir).resolve()
        self.tactile_bag = TactileBag(tactile_bag_dir)

        self.train_prop = train_prop 
        self.delta_t = delta_t
        self.margin = margin
        self.min_n_events = min_n_events
        self.n_init_events = n_init_events
        self.down_sample = down_sample
        self.center = center
        self.circle_rad = circle_rad
        self.dist_from_center = lambda x, y: np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        self.case_span = case_span
        self.keep_interm = keep_interm
        self.event_array_augmentations = event_array_augmentations if isinstance(event_array_augmentations, list) else [event_array_augmentations]
        self.event_array_filters = event_array_filters if isinstance(event_array_filters, list) else [event_array_filters]

        if not self.tactile_bag.is_parsed():
            self.tactile_bag.parse_exception()

        possible_angles = self.tactile_bag.params['possible_angles']
        N_examples = self.tactile_bag.params['N_examples']
        theta = self.tactile_bag.params['theta']
        list_of_rotations = [[0, 0, 0]]

        for i in range(1, N_examples):
            th = i * 2 * math.pi/(N_examples - 1) if theta=='full' else theta[i]#math.pi/2 #i * 2 * math.pi/(N_examples - 1)
            for phi in possible_angles:
                rx = phi * math.cos(th)
                ry = phi * math.sin(th)
                rotvec = [rx, ry, 0]
                list_of_rotations.append(rotvec)


        self.list_of_rotations = list_of_rotations

        self.params = {
            'bag': str(tactile_bag_dir),
            'train_prop': train_prop,
            'delta_t': delta_t,
            'margin': margin,
            'min_n_events': min_n_events,
            'n_init_events': n_init_events,
            'downsample_factor': down_sample,
            'center': center,
            'circle_rad': circle_rad,
            'case_span': case_span,
            'N_examples': N_examples,
            'possible_angles': possible_angles,
            'theta': theta,
            'augmentations': [str(aug) for aug in self.event_array_augmentations],
            'filters': [str(fil) for fil in self.event_array_filters]
        }
        

    def parse_bag(self):
        self.events = self.tactile_bag.events
        df = self.tactile_bag.parsed_bag

        self.find_ts_idx = lambda ts: np.searchsorted(df['ts'], ts)

        i = 0 
        cases_ts = []
        cases_idx = []
        cases = []
        pbar = tqdm(total=len(df['contact_status']), desc='extracting contact timestamps')
        while i < len(df['contact_status']):
            if df['contact_status'][i]:
                init_ts = df['ts'][i]
                fin_ts = self.look_ahead_big(init_ts, df['contact_status'], df['ts'])
                fin_idx = self.find_ts_idx(fin_ts)
                case = self.find_case(np.mean([init_ts, fin_ts]), df.values[:, -3:])
                
                cases.append(case)
                cases_ts.append([init_ts, fin_ts])
                cases_idx.append([i, fin_idx])
                #print(len(cases_ts), init_ts, fin_ts, (fin_ts - init_ts)*1e-9, i, fin_idx, case, '\n')
                pbar.update(fin_idx + 1 - i)
                i = fin_idx + 1
            else:
                i += 1
                pbar.update(1)
        pbar.close()

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
        for rot in self.list_of_rotations:
            diff_vals = np.sqrt(np.power(rot[0] - x, 2) +  np.power(rot[1] - y, 2) + np.power(rot[2] - z, 2))
            if best_rot_diff > diff_vals:
                best_rot_diff = diff_vals
                best_rot_idx = i
            i = i + 1
        return best_rot_idx
    
    def apply_augmentations(self, samples, train_idx):
        out = {f'sample_{i+1}': samples[idx] for i, idx in enumerate(train_idx)}
        if len(self.event_array_augmentations) == 0:
            return out
        out_idx = len(out)+1
        for aug_strat in self.event_array_augmentations:
            aug_strat.params = self.params
            if aug_strat.stackable: #apply to already augmented samples
                out_ = {}
                for idx, sample in out.items():
                    ev_arr_aug, label_aug = aug_strat.augment(sample['events'], sample['case'])
                    out_[f'sample_{out_idx}'] = {'events': ev_arr_aug, 'case': label_aug}
                    out_idx += 1
                out = dict(out, **out_)
            else:                   #apply to raw data only
                for s_idx in train_idx:
                    ev_arr_aug, label_aug = aug_strat.augment(samples[s_idx]['events'], samples[s_idx]['case'])
                    out[f'sample_{out_idx}'] = {'events': ev_arr_aug, 'case': label_aug}
                    out_idx += 1
        return out
    
    def assure(self):
        if self.tactile_bag.params['theta'] == 'full':
            desired_n_tot_cases = (1+(self.params['N_examples']-1)*len(self.params['possible_angles']))
        else:
            desired_n_tot_cases = (len(self.params['theta'])-1) * len(self.params['possible_angles'])

        desired_n_tot_cases *= self.tactile_bag.params['N_iters']
        print(desired_n_tot_cases, len(self.cases))
        assert desired_n_tot_cases == len(self.cases)

    def filter(self, samples):
        new_samples = {}
        i = 1
        for s_idx, sample in samples.items():
            vote = []
            for fil in self.event_array_filters:
                vote.append(fil.filter(sample['events'], sample['case']))

            if sum(vote) == len(vote):
                print(i)
                new_samples[f'sample_{i}'] = sample
                i += 1
            else: 
                continue
        return new_samples


    def _save(self, samples):
        self.assure()
        samples = self.filter(samples)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        with open(self.outdir / 'extraction_params.json', 'w') as f:
            json.dump(self.params, f, indent=4)
        
        with open(self.outdir / 'samples.json', 'w') as f:
            json.dump(samples, f, indent=4)

        sample_idx = list(samples.keys())
        cases = [str(samples[s_idx]['case']) for s_idx in sample_idx]

        train_idx, val_test_idx = train_test_split(sample_idx, test_size=1-self.train_prop, random_state=0) #fixed across extractions
        
        cases = [str(samples[s_idx]['case']) for s_idx in val_test_idx]
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=0) #fixed across extractions

        #print(len(train_idx), len(val_idx), len(test_idx))
        subsets = zip(['train', 'val', 'test'], [train_idx, val_idx, test_idx])
        
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        with open(self.outdir / 'extraction_params.json', 'w') as f:
            json.dump(self.params, f, indent=4)
    
        for sub_name, subset in subsets:
            if not (self.outdir / sub_name).exists():
                (self.outdir / sub_name / 'raw').mkdir(parents=True)
                (self.outdir / sub_name / 'processed').mkdir(parents=True)
            with open(self.outdir / sub_name / 'raw' / 'contact_cases.json', 'w') as f:
                if sub_name == 'train':
                    subset_samples = self.apply_augmentations(samples, train_idx)
                else:
                    subset_samples = {}
                    for i, subset_idx in enumerate(subset):
                        sample = samples[subset_idx]
                        sample['total_idx'] = subset_idx
                        subset_samples[f'sample_{i+1}'] = sample
                json.dump(subset_samples, f, indent=4)

    def extract(self):
        self.parse_bag()
        init_cases_ts = da.array(self.cases_ts)[:, 0]
        ts = self.events[:, 2]
        dist_from_center = lambda x, y: np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        init_ev_array_idx = da.searchsorted(ts, init_cases_ts + self.margin).compute()
        fin_ev_array_idx = da.searchsorted(ts, init_cases_ts + self.delta_t).compute()

        event_arrays = []
        label_contact_case = []

        for i, (ini, fin) in enumerate(zip(tqdm(init_ev_array_idx, desc='extracting event arrays'), fin_ev_array_idx)):
            ev_arr = self.events[ini:fin+1]
            in_circle = dist_from_center(ev_arr[:, 0], ev_arr[:, 1]) < self.circle_rad
            ev_arr = ev_arr[in_circle, :].compute()
            
            if len(ev_arr) < self.min_n_events:
                continue
            
            idx_downsample = np.random.choice([0, 1], ev_arr.shape[0], p=np.array([self.down_sample-1, 1]) / self.down_sample).astype(bool)
            event_arrays.append(ev_arr[idx_downsample])
            label_contact_case.append(self.cases[i])

        samples={}
        
        gen = zip(label_contact_case, event_arrays) 
        for i, (case, event_array) in enumerate(gen):
            samples[f'sample_{i+1}'] = {
                'events': event_array.tolist(),
                'case': case
                }

        print("saving")
        self.params['n'] = len(samples)
        self._save(samples)


    def ev_ts2idx(self, ts, guess_idx, init_stepsize=50000):
        stepsize = init_stepsize
        search_sorted = lambda st: da.searchsorted(self.events[guess_idx:guess_idx+st, 2], da.array([ts])).compute()[0]
        n_iters = 0
        
        out = search_sorted(stepsize)
        
        while out == stepsize:
            n_iters += 1
            stepsize = 10*stepsize
            
            out = search_sorted(stepsize)
        if n_iters > 1:
            print(ts, n_iters)
        return out + guess_idx
            

    def old_extract(self):
        self.parse_bag()
        dist_from_center = lambda x, y: np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
        event_arrays = []
        ev_idx_init = []
        label_contact_case = []

        guess_idx = 0
        dropped_cases = 0
        for i, idx in enumerate(tqdm(self.cases_idx, desc='extracting event arrays')):
            init_ts_idx = self.ev_ts2idx(self.cases_ts[i][0] + self.margin, guess_idx)
            ev_idx_init.append(init_ts_idx)
            guess_idx = init_ts_idx
            fin_ts_idx = self.ev_ts2idx(self.cases_ts[i][0] + self.delta_t, guess_idx)#da.searchsorted(events[:, 2], da.array([cases_ts[i][0] + delta_t]))
            guess_idx = fin_ts_idx
            if fin_ts_idx - init_ts_idx + 1 < self.min_n_events:
                dropped_cases += 1
                continue    
            else:
                event_array = self.events[init_ts_idx:fin_ts_idx+1].compute()
                in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < self.circle_rad  
                event_array = event_array[in_circle, :][:self.n_init_events]
                idx_downsample = np.random.choice([0, 1], event_array.shape[0], p=np.array([self.down_sample-1, 1]) / self.down_sample).astype(bool)
                event_arrays.append(event_array[idx_downsample])
                label_contact_case.append(self.cases[i])

        samples={}
        
        gen = zip(label_contact_case, event_arrays) 
        for i, (case, event_array) in enumerate(gen):
            samples[f'sample_{i+1}'] = {
                'events': event_array.tolist(),
                'case': case
                }

        print("saving")
        self.params['n'] = len(samples)
        self.params['dropped_cases'] = dropped_cases
        self._save(samples)

    def load(self):
        with open(self.outdir / 'samples.json', 'r') as f:
            return json.load(f)

    def load_train(self):
        with open(self.outdir / 'train' / 'raw' / 'contact_cases.json', 'r') as f:
            return json.load(f)

    @property
    def cases_dict(self):
        d = {i+1: self.list_of_rotations[i][:2] for i in range(len(self.list_of_rotations))}
        d[0] = [0, 0]
        return d