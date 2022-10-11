from random import sample
import numpy as np
import rosbag
from scipy.interpolate import interp1d
from tqdm.auto import tqdm
import json
from pathlib import Path



class ExtractContactCases:

    cases_dict = {
        0: [0, 0],  # what to do with no contact case ??
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

    def __init__(
        self,
        outdir,
        bag_file_name='/data/dataset_ENVTACT_new2.bag',
        frequency=30, 
        interpolate=True,
        time_frame=0.3e9,
        threshold=7500
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

    def parse_bag(self):
        bag_file = rosbag.Bag(self.bag_file_name)

        events = []
        contact_status = []
        contact_status_ts = []
        contact_case = []  # 0:No contact 1: center, 2:remainder of contacts as in list_of_rotations
        contact_case_ts = []

        for topic, msg, t in tqdm(bag_file.read_messages(
                topics=['/contact_status', '/dvs/events', '/contact_angle']), desc='parsing ros bag'):
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
                    if (contact_status[-1]):
                        best_rot_diff = 100
                        best_rot_idx = 1
                        i = 1
                        for case in self.cases_dict.keys():
                            rot = self.cases_dict[case]
                            rot.append(0)
                            diff_vals = np.sqrt(np.power(rot[0] - msg.x, 2) + np.power(rot[1] - msg.y, 2) + np.power(rot[2] - msg.z, 2))
                            if best_rot_diff > diff_vals:
                                best_rot_diff = diff_vals
                                best_rot_idx = i
                            i = i + 1
                            #print(best_rot_diff, best_rot_idx)

                        contact_case.append(best_rot_idx)
                        contact_case_ts.append(t.to_nsec())
                    else:
                        contact_case.append(0)
                        contact_case_ts.append(t.to_nsec())
                else:
                    contact_case.append(0)
                    contact_case_ts.append(t.to_nsec())

                # Updated contact status according to no. of events

        # print(events)
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


    def interpolation(self):
        f = interp1d(self.contact_case_ts, self.contact_case, kind='previous')
        contact_case_ts_int = range(min(self.contact_case_ts), max(
            self.contact_case_ts), int(1e9 / self.frequency))
        contact_case_int = f(contact_case_ts_int)
        return contact_case_ts_int, contact_case_int

    def filter_events_by_time(self, time_of_contact):
        event_in_time_idx = np.where((self.event_time > (
            time_of_contact - self.time_frame)) * (self.event_time < time_of_contact))[0]
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
        contact_rise_idx = np.where(contact_case_diff > 0.4)[0]

        contact_rise = len(self.case) * [0]
        for index in contact_rise_idx:
            contact_rise[index] = 70

        self.contact_rise = contact_rise
        self.contact_rise_idx = contact_rise_idx

    def _save(self, samples):
        if not self.outdir.exists():
            (self.outdir / 'raw').mkdir(parents=True)
            (self.outdir / 'processed').mkdir(parents=True)

            with open(self.outdir / 'extraction_params.json', 'w') as f:
                json.dumps(self.params, f, indent=4)
        
        with open(self.outdir / 'raw' / 'contact_cases.json', 'w') as f:
            json.dump(samples, f, indent=4)


    def sample_generator(self):
        if not self.parsed:
            self.parse_bag()
        self.get_rise()

        i = 0

        for status_index in self.contact_rise_idx:
            for j in range(-7, 8):
                time_step = self.case_ts[status_index + j]
                detect, event_array = self.filter_events_by_time(time_step)
                # print(event_array)
                if detect:
                    i += 1
                    yield {f'sample_{i}': {'events': event_array, 'case': np.array(self.case)[status_index + 1]}}
                    break
            
    
    def extract(self):
        if not self.parsed:
            self.parse_bag()
        self.get_rise()

        label_contact_case = []
        i = 0
        event_arrays = []

        for status_index in tqdm(self.contact_rise_idx):
            for j in range(-7, 8):
                time_step = self.case_ts[status_index + j]
                detect, event_array = self.filter_events_by_time(time_step)
                # print(event_array)
                if detect:
                    event_arrays.append(event_array)
                    label_contact_case.append(
                        np.array(self.case)[status_index + 1])
                    break

        samples={}

        for i, (case, event_array) in enumerate(zip(label_contact_case, event_arrays)):
            samples[f'sample_{i+1}'] = {
                'events': event_array.tolist(),
                'case': case
                }

        self._save(samples)
        

    
