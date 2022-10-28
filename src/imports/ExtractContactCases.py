import numpy as np
import rosbag
from scipy.interpolate import interp1d
from tqdm import tqdm
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

dist_from_center = lambda x, y: np.sqrt((x - 173)**2 + (y - 130)**2)
circle_rad=90

class ExtractContactCases:

    def __init__(
        self,
        outdir,
        bag_file_name='/data/dataset_ENVTACT_new2.bag',
        frequency=30, 
        interpolate=True,
        time_frame=0.3e9,
        threshold=7500,
        _limit = False
    ):
        self.outdir = Path(outdir)
        self.bag_file_name = bag_file_name
        self.frequency = frequency
        self.interpolate = interpolate
        self.time_frame = time_frame
        self.threshold = threshold

        self.params = {
            'threshold': self.threshold,
            'time_frame': self.time_frame,
            'interpolate': self.interpolate,
            'frequency': self.frequency
        }

        self.parsed = False
        self._limit = _limit

    def parse_bag(self):
        bag_file = rosbag.Bag(self.bag_file_name)

        events = []
        contact_status = []
        contact_status_ts = []
        contact_case = []  # 0:No contact 1: center, 2:remainder of contacts as in list_of_rotations
        contact_case_ts = []
        k = 0
        xyz = []

        for topic, msg, t in tqdm(bag_file.read_messages(topics=['/contact_status', '/dvs/events', '/contact_angle'])):
            if self._limit:
                k += 1
            if k > 59999:
                break
            if topic == '/dvs/events':
                for e in msg.events:
                    event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
                    events.append(event)
                event_topic = True    
            elif topic == '/contact_status':
                contact_status.append(msg.data)
                contact_status_ts.append(t.to_nsec())
            elif topic == '/contact_angle':
                if (len(contact_status) > 1):
                    if (contact_status[-1] == True):
                        best_rot_diff = 100
                        best_rot_idx = 1
                        i = 1
                        for rot in list_of_rotations:
                            diff_vals = np.sqrt( np.power(rot[0] - msg.x, 2) +  np.power(rot[1] - msg.y, 2) + np.power(rot[2] - msg.z, 2) )
                            if best_rot_diff > diff_vals:
                                best_rot_diff = diff_vals
                                best_rot_idx = i
                            i = i + 1

                        contact_case.append(best_rot_idx)
                        contact_case_ts.append(t.to_nsec())
                    else:
                        contact_case.append(0)
                        contact_case_ts.append(t.to_nsec())
                else:
                    contact_case.append(0)
                    contact_case_ts.append(t.to_nsec())
        bag_file.close()
        self.parsed = True

        self.events = events
        self.contact_status = contact_status
        self.contact_status_ts = contact_status_ts
        # 0:No contact 1: center, 2:remainder of contacts as in
        # list_of_rotations
        self.contact_case = contact_case
        self.contact_case_ts = contact_case_ts

        self.event_time = np.array([events[i][2] for i in range(np.shape(events)[0])])

    def _parse_bag(self):
        bag_file = rosbag.Bag(self.bag_file_name)

        events = []
        contact_status = []
        contact_status_ts = []
        contact_case = []  # 0:No contact 1: center, 2:remainder of contacts as in list_of_rotations
        contact_case_ts = []
        k = 0
        xyz = []

        for topic, msg, t in tqdm(bag_file.read_messages(topics=['/contact_status', '/dvs/events', '/contact_angle']), desc='parsing rosbag'):
            if self._limit:
                k += 1
            if k > 59999:
                break
            if topic == '/dvs/events':
                for e in msg.events:
                    event = [e.x, e.y, e.ts.to_nsec(), e.polarity]
                    events.append(event)
                event_topic = True    
            elif topic == '/contact_status':
                contact_status.append(msg.data)
                contact_status_ts.append(t.to_nsec())
            elif topic == '/contact_angle':
                if (len(contact_status) > 1):
                    if (contact_status[-1] == True):
                        contact_case_ts.append(t.to_nsec())

                        xyz.append([msg.x, msg.y, msg.z])

                    else:
                        xyz.append([0, 0, 0])
                        contact_case_ts.append(t.to_nsec())
                else:
                    xyz.append([0, 0, 0])

                    contact_case_ts.append(t.to_nsec())
                        # Updated contact status according to no. of events

                # print(events)
        bag_file.close()
        self.parsed = True

        contact_angles = np.array(xyz)
        euc = []
        for rot in list_of_rotations:
            diff = contact_angles - rot
            euc.append(np.linalg.norm(diff, axis=1))
        contact_case = np.argmin(np.array(euc), axis=0) - 1

        self.events = events
        self.contact_status = contact_status
        self.contact_status_ts = contact_status_ts
        # 0:No contact 1: center, 2:remainder of contacts as in
        # list_of_rotations
        self.contact_case = contact_case
        self.contact_case_ts = contact_case_ts

        self.event_time = np.array([events[i][2] for i in range(np.shape(events)[0])])
        self.xyz = xyz




    def interpolation(self):
        f = interp1d(self.contact_case_ts, self.contact_case, kind='previous')
        contact_case_ts_int = range(min(self.contact_case_ts), max(
            self.contact_case_ts), int(1e9 / self.frequency))
        contact_case_int = f(contact_case_ts_int)
        return contact_case_ts_int, contact_case_int

    def filter_events_by_time(self, time_of_contact, time_frame = None):
        time_frame = self.time_frame if time_frame is None else time_frame
        event_in_time_idx = np.where((self.event_time > (
            time_of_contact - time_frame)) * (self.event_time < time_of_contact))[0]
        # print(len(event_in_time_idx))
        #time_of_contact - time_frame < ts < time_of_contact
        if len(event_in_time_idx) < self.threshold:
            return False, []
        else:
            # print(event_in_time_idx)
            output_events = np.array(self.events)[event_in_time_idx, :]
            return True, output_events

    def get_rise(self):
        if self.interpolate:
            self.case_ts, self.case = self.interpolation()
        else:
            self.case_ts, self.case = self.contact_case_ts, self.contact_case

        # out[i] = a[i+1] - a[i] if positive then at idx i the
        contact_case_diff = np.diff(self.case)
        contact_case_diff = np.insert(contact_case_diff, 0, 0)
        contact_rise_idx = np.where(contact_case_diff > 0.9)[0]

        contact_rise = len(self.case) * [0]
        for index in contact_rise_idx:
            contact_rise[index] = 70

        self.contact_rise = contact_rise
        self.contact_rise_idx = contact_rise_idx

    def _save(self, samples):
        sample_idx = list(samples.keys())

        train_idx, val_test_idx = train_test_split(sample_idx, test_size=0.4, random_state=0) #fixed across extractions
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, random_state=0) #fixed across extractions

        subsets = zip(['train', 'test', 'val'], [train_idx, val_idx, test_idx])

        for sub_name, subset in subsets:
            if not (self.outdir / sub_name).exists():
                (self.outdir / sub_name / 'raw').mkdir(parents=True)
                (self.outdir / sub_name / 'processed').mkdir(parents=True)

                with open(self.outdir / sub_name / 'extraction_params.json', 'w') as f:
                    json.dump(self.params, f, indent=4)
            
            with open(self.outdir / sub_name / 'raw' / 'contact_cases.json', 'w') as f:
                subset_samples = {}
                for i, subset_idx in enumerate(subset):
                    sample = samples[subset_idx]
                    sample['total_idx'] = subset_idx
                    subset_samples[f'sample_{i+1}'] = sample
                json.dump(subset_samples, f, indent=4)


    def sample_generator(self, time_frame = None):
        if not self.parsed:
            self.parse_bag()
        self.get_rise()

        i = 0

        for status_index in self.contact_rise_idx:
            for j in range(-7, 8):
                time_step = self.case_ts[status_index + j]
                detect, event_array = self.filter_events_by_time(time_step, time_frame = time_frame)
                # print(event_array)
                if detect:
                    i += 1
                    in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < circle_rad 
                    yield {f'sample_{i}': {'events': event_array[in_circle, :], 'case': np.array(self.case)[status_index + 1]}}
                    break
            
    
    def extract(self):
        if not self.parsed:
            self.parse_bag()
        self.get_rise()

        self.label_contact_case = []
        i = 0
        self.event_arrays = []

        for status_index in tqdm(self.contact_rise_idx):
            for j in range(-7, 8):
                time_step = self.case_ts[status_index + j]
                detect, event_array = self.filter_events_by_time(time_step)
                # print(event_array)
                if detect:
                    in_circle = dist_from_center(event_array[:, 0], event_array[:, 1]) < circle_rad 
                    self.event_arrays.append(event_array[in_circle, :])
                    self.label_contact_case.append(
                        np.array(self.case)[status_index + 1])
                    break

        samples={}

        for i, (case, event_array) in enumerate(zip(self.label_contact_case, self.event_arrays)):
            samples[f'sample_{i+1}'] = {
                'events': event_array.tolist(),
                'case': case
                }

        print("saving")
        self._save(samples)
        

class nothresh(ExtractContactCases):
    def __init__(self, outdir, bag_file_name='/data/dataset_ENVTACT_new2.bag', frequency=30, interpolate=True, time_frame=200000000, threshold=7500, training_size=0.8):
        super().__init__(outdir, bag_file_name, time_frame=time_frame)

    def filter_events_by_time(self, time_of_contact, time_frame = None):
        time_frame = self.time_frame if time_frame is None else time_frame
        event_in_time_idx = np.where((self.event_time > (
            time_of_contact - time_frame)) * (self.event_time < time_of_contact))[0]
        #time_of_contact - time_frame < ts < time_of_contact
      
        output_events = np.array(self.events)[event_in_time_idx, :]
        return True, output_events    