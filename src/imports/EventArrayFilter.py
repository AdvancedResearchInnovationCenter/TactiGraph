import numpy as np
class EventArrayFilter():

    def __init__(self) -> None:
        self.params = None
        pass

    def filter(self, sample, case):
        return True

class RemovePhiBiggerThan(EventArrayFilter):

    def __init__(self, max_phi=np.radians(9)) -> None:
        self.max_phi = max_phi
        super().__init__()

    def filter(self, sample, case):
        if np.linalg.norm(case) <= self.max_phi:
            return True
        else:
            return False 

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_phi={self.max_phi})"

class RemoveNEventsLessThan(EventArrayFilter):

    def __init__(self, min_n_events=3500) -> None:
        self.min_n_events = min_n_events
        super().__init__()

    def filter(self, sample, case):
        if len(sample) <= self.min_n_events:
            return False
        else:
            return True