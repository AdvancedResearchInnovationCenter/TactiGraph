import numpy as np

class EventArrayAugmentation:

    def __init__(self, stackable) -> None:
        self.stackable = stackable
        self.params = None

    def augment(self, ev_arr, label):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


class JitterEvents(EventArrayAugmentation):

    def __init__(self, max_delta_x=3, stackable=True) -> None:
        self.max_delta_x = max_delta_x
        super().__init__(stackable=stackable)

    def augment(self, ev_arr, label):
        ev_arr = np.array(ev_arr)
        delta_x = np.random.randint(-self.max_delta_x, self.max_delta_x+1, size=(len(ev_arr), 2))
        aug = ev_arr[:, :2] + delta_x
        out_ev_arr = np.c_[aug, ev_arr[:, 2:]]
        return out_ev_arr.tolist(), label
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_delta_x={self.max_delta_x})"

class JitterTemporal(EventArrayAugmentation):
    
    def __init__(self, dt = 1e6, stackable=False) -> None:
        self.dt = dt
        super().__init__(stackable)

    def augment(self, ev_arr, label):
        ev_arr = np.array(ev_arr)
        jitter = self.dt * np.random.randn(len(ev_arr))
        ev_arr[:, 2] = ev_arr[:, 2] + jitter
        return ev_arr.tolist(), label
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dt={self.dt})"


class RotateEvents(EventArrayAugmentation):

    def __init__(self, angle, stackable=False) -> None:
        super().__init__(stackable=stackable)
        self.angle = angle


    def augment(self, ev_arr, case):

        self.list_of_rotations = [[0, 0, 0]]

        for i in range(1, self.params['N_examples']):
            th = i * 2 * np.pi/(self.params['N_examples'] - 1) if self.params['theta']=='full' else self.params['theta'][i]#math.pi/2 #i * 2 * math.pi/(N_examples - 1)
            for phi in self.params['possible_angles']:
                rx = phi * np.cos(th)
                ry = phi * np.sin(th)
                rotvec = [rx, ry, 0]
                self.list_of_rotations.append(rotvec)        
                
        self.cases_dict = {i+1: self.list_of_rotations[i][:2] for i in range(len(self.list_of_rotations))}
        self.cases_dict[0] = [0, 0]
        self.center = self.params['center']

        ev_arr = np.array(ev_arr)
        theta = np.radians(self.angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        centered = ev_arr[:, :2] - np.array(self.center)
        centered[:, 1] = - centered[:, 1]
        rot_ev = (R @ centered.T).T 
        rot_ev[:, 1] = - rot_ev[:, 1]
        rot_ev += np.array(self.center)

        
        rot_v = np.array(self.cases_dict[case])
        new_rot_v = R @ rot_v

        best_rot_diff = 100
        best_rot_idx = 1
        i = 1
        
        for rot in self.list_of_rotations:
            diff_vals = np.sqrt( np.power(rot[0] - new_rot_v[0], 2) +  np.power(rot[1] - new_rot_v[1], 2))
            if best_rot_diff > diff_vals:
                best_rot_diff = diff_vals
                best_rot_idx = i
            i = i + 1

        return np.concatenate([rot_ev.astype(int), ev_arr[:, 2:]], -1).tolist(), best_rot_idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angle={self.angle})"