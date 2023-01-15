import sys

from tactile.src.imports.ExtractContactCases import ExtractContactCases
from tactile.src.imports.EventArrayAugmention import *
import math
import numpy as np

ex = ExtractContactCases(
    outdir='./extractions/contact_extraction4', 
    possible_angles=[math.radians(i) for i in range(1,11)], 
    N_examples=20,
    theta='full',
    bag_file_name='/home/hussain/catkin_ws/small_example4_heavy_one_phi_4thetas.bag',
    center=(180, 117),
    circle_rad=85,
    n_init_events=10000,
    delta_t=0.075e9,
    margin=-0.025e9,
    event_array_augmentations=[
        RotateEvents(90),
        RotateEvents(180)
        JitterEvents(2)
    ]
)