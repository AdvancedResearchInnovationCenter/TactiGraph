from imports.TactileBag import TactileBag
from imports.EventArrayAugmention import JitterEvents, RotateEvents, JitterTemporal
from torch_geometric import seed_everything
from torch_geometric.transforms import Distance, RandomJitter, Compose, Cartesian
seed_everything(0)

from tuner import model2
print()
# from imports.ExtractContactCases import ExtractContactCases

# ex = ExtractContactCases(
#     '../data/bags/data2',
#     '../data/extractions/morethan3500ev_lessthan_9deg/',
#     n_init_events = -1,
#     margin=-0.05e9,
#     delta_t=0.15e9,
#     min_n_events=3500,
#     down_sample=2,
#     event_array_augmentations=[
#         JitterTemporal(dt=5e6, stackable=False)
#     ]
# )
#ex.extract()

from imports.TrainModel import TrainModel


#from models.modules import GraphRes

#rm -rf ../data/temp_0_lcc_1_2hop_0/{test,train,val}/processed/*.pt
#model = model3().cuda()
import torch
transform = Compose([RandomJitter((0.003, 0.003, 0)), Cartesian(cat=False, norm=True)])
model = model2().cuda()
#model.load_state_dict(torch.load('/home/hussain/tactile/results/hpc_ymak_filtered/state_dict', map_location='cuda'))
tm = TrainModel(
    '../data/bags/data2/extractions/online', 
    model, 
    lr=0.001, 
    features='pol', 
    batch=4, 
    n_epochs=2000, 
    experiment_name='tuner st', 
    desc='temporal shuffle augment',
    merge_test_val=False,
    transform=transform
)

tm.train()

