from imports.TactileBag import TactileBag
from imports.EventArrayAugmention import JitterEvents, RotateEvents, JitterTemporal
from torch_geometric import seed_everything
seed_everything(0)
tbag = TactileBag('../data/bags/data2')
print(tbag.events)
print(tbag.parsed_bag)

from imports.ExtractContactCases import ExtractContactCases

ex = ExtractContactCases(
    '../data/bags/data2',
    '../data/extractions/temp_0_lcc_1_2hop_0_jitter_50microseconds/',
    n_init_events = -1,
    margin=-0.05e9,
    delta_t=0.15e9,
    min_n_events=3500,
    down_sample=2,
    event_array_augmentations=[
        JitterTemporal(dt=5e6, stackable=False)
    ]
)
ex.extract()

from imports.TrainModel import TrainModel
from models.modules import model2


#from models.modules import GraphRes

#rm -rf ../data/temp_0_lcc_1_2hop_0/{test,train,val}/processed/*.pt
model = model2().cuda()
#import torch
#model.load_state_dict(torch.load('/home/hussain/tactile/results/temp_0_lcc_1_2hop_0_model2/ckpt_150'))
tm = TrainModel(
    '../data/extractions/temp_0_lcc_1_2hop_0_jitter_50microseconds/', 
    model, 
    lr=0.01, 
    features='pol', 
    batch=6, 
    n_epochs=300, 
    experiment_name='temp_0_lcc_1_2hop_0_jitter_50microseconds', 
    desc='temp_0_lcc_1_2hop_0_jitter_1microseconds',
    merge_test_val=False,
    
)

tm.train()

