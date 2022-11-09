class Augment():
    
    def generator(self, event_array):
        raise NotImplementedError

class Rotate_aug(Augment):

    def generator(self):
        return super().generator()