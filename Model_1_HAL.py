
from __future__ import division
import os.path as osp
import collections
import os.path as osp
import os
import errno
import numpy as np
import glob
import scipy.io as sio
import torch
import torch_geometric.transforms as T
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch.utils.data
from torch_geometric.data import Data, DataLoader, Dataset

import numpy as np
import math
import shutil
import os

import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn 
from torch.nn import Sequential, Linear, ReLU
#from torch import autograd1
from torch.autograd import Variable
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, GCNConv,  global_mean_pool
#from torch_geometric.nn.conv import SplineConv
#from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import InMemoryDataset, download_url
from collections.abc import Sequence
from torch_geometric.nn import global_mean_pool
import time

import os.path as osp

import torch
from torch_geometric.data import Dataset
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph
def files_exist(files):
    return all([osp.exists(f) for f in files])


number_of_epoch=200


FOLDERTOSAVE = 'models_and_results/' +"Model1_Epoch" +str(number_of_epoch)
if not os.path.isdir(FOLDERTOSAVE):
    os.makedirs(FOLDERTOSAVE)

class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self._indices: Optional[Sequence] = None
        
    def indices(self) -> Sequence:
        return range(self.len()) if self._indices is None else self._indices
    @property
    def raw_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir, '*.npz'))
        print(filenames)
        file = [f.split('/')[-1] for f in filenames]
        #print(file)
        return file

    @property
#     def processed_file_names(self):
#         filenames = glob.glob(os.path.join(self.raw_dir, '*.npz'))
#         file = [f.split('/')[-1] for f in filenames]
#         saved_file = [f.replace('.npz','.pt') for f in file]
#         return saved_file
    def processed_file_names(self):
        filenames = glob.glob(os.path.join(self.raw_dir,'../processed/', '*_.pt'))
#         print('rawdire', self.raw_dir)
        file = [f.split('/')[-1] for f in filenames]
#         print('file', file)
        saved_file = [f.replace('.pt','.pt') for f in file]
        return saved_file

    def __len__(self):
        return len(self.processed_file_names)
    def indices(self) -> Sequence:
        return range(self.__len__()) if self._indices is None else self._indices
    
    def download(self):
        if files_exist(self.raw_paths):
            return
        print('No found data!!!!!!!')


    def process(self):
        for raw_path in self.raw_paths:
             # Read data from `raw_path`.
            content =np.load(raw_path)#np.load('data_Hal_in_out.npz')# sio.loadmat(raw_path)
#             print("kkk", len(content['X']), len(content['Y']))
            
            for sample1 in range(0, len(content['X'])):
 
            
                feature = torch.tensor(content['X'][sample1,:,0:2])
#                 print("feature",feature)
                c1 =  torch.tensor(content['X'][sample1,:,0:1]).double()#.float()
                c2 = torch.tensor(content['X'][sample1,:,1:2]).double()#.float()
    #             pos=(torch.tensor(content['X'][0:1][0])).view(-1,2)[:,0:1].double()#.float()

                pos2 = torch.stack((c1/346,  c2 /260),1)#.view(-1,3)
#                 print("pos2",pos2)
                # plt.plot(c1/346, c2/260, '.')
                # plt.show()
#                 feature = torch.tensor(content['X'][sample1][0:1])
    #             print("pos", torch.stack(feature))
                edge_index = radius_graph(feature, r=300, max_num_neighbors=10,flow='source_to_target',num_workers= 1)
                pos = torch.tensor(content['X'][sample1])

                label_idx = torch.tensor(content['Y'][sample1], dtype=torch.long)

                data = Data(x=pos2, edge_index=edge_index, pos=pos2, y=label_idx.squeeze(0))

                if self.pre_filter is not None and not self.pre_filter(data):
                     continue

                if self.pre_transform is not None:
                     data = self.pre_transform(data)

                saved_name = raw_path.split('/')[-1].replace('.npz','.pt')
                #torch.save(data, osp.join(self.processed_dir,"_sample_"+ str(sample1)+saved_name  ))
                torch.save(data, osp.join(self.processed_dir, "sample_"+ str(sample1)+"_"+saved_name  ))

                print("GRAPH DATA ARE SAVED!!!!!!!!!!!!!!!!!!!!!! in ", self.processed_dir)

    def get(self, idx):
        # print("I'm in get ", self.processed_dir)

        data = torch.load(osp.join(self.processed_paths[idx]))
        return data
pi = 3.14159265359

def predict_angletrain( model, testloader):
#print(y_pred)
    angle_error = []
    for i, data in enumerate(testloader):
        model.to("cpu")
        y_pred = model(data.to("cpu"))
        # print(y_pred[0])
        # print(y_pred[0].detach().numpy()[0])
        # print(data.y[0])
        # print(data.y[0].detach().numpy()[0])

        diff_x = np.absolute(y_pred[0].detach().numpy()[0] - data.y[0])#.detach().numpy()[0])
        diff_y = np.absolute(y_pred[0].detach().numpy()[1] - data.y[1])#.detach().numpy()[1])
        angle_diff = np.sqrt(np.square(diff_x) + np.square(diff_y))
        angle_error.append(angle_diff)

    max_error = max(angle_error)*180/pi
    print ('maximum error in deg: {:.4f}'.format(max_error))

    mean_error = np.mean(angle_error)*180/pi
    print ('mean error in deg: {:.4f}'.format(mean_error))
    print(angle_error)

    predictions = open(FOLDERTOSAVE+"/TRAINING"+'models_and_predictions.txt',"a+")

    predictions.write("\n #### Max error: {} #### Mean error: {} \n".format( max_error, mean_error))

    predictions.close()
def predict_angletest( model, testloader):
#print(y_pred)
    angle_error = []
    for i, data in enumerate(testloader):
        model.to("cpu")
        y_pred = model(data.to("cpu"))
        # print(y_pred[0])
        # print(y_pred[0].detach().numpy()[0])
        # print(data.y[0])
        # print(data.y[0].detach().numpy()[0])

        diff_x = np.absolute(y_pred[0].detach().numpy()[0] - data.y[0])#.detach().numpy()[0])
        diff_y = np.absolute(y_pred[0].detach().numpy()[1] - data.y[1])#.detach().numpy()[1])
        angle_diff = np.sqrt(np.square(diff_x) + np.square(diff_y))
        angle_error.append(angle_diff)

    max_error = max(angle_error)*180/pi
    print ('maximum error in deg: {:.4f}'.format(max_error))

    mean_error = np.mean(angle_error)*180/pi
    print ('mean error in deg: {:.4f}'.format(mean_error))
    print(angle_error)

    predictions = open(FOLDERTOSAVE+ "/TESTING"+'models_and_predictions.txt',"a+")

    predictions.write("\n #### Max error: {} #### Mean error: {} \n".format( max_error, mean_error))

    predictions.close()
def predict_anglevalid( model, testloader):

#print(y_pred)
    angle_error = []
    for i, data in enumerate(testloader):
        model.to("cpu")
        y_pred = model(data.to("cpu"))
        # print(y_pred[0])
        # print(y_pred[0].detach().numpy()[0])
        # print(data.y[0])
        # print(data.y[0].detach().numpy()[0])

        diff_x = np.absolute(y_pred[0].detach().numpy()[0] - data.y[0])#.detach().numpy()[0])
        diff_y = np.absolute(y_pred[0].detach().numpy()[1] - data.y[1])#.detach().numpy()[1])
        angle_diff = np.sqrt(np.square(diff_x) + np.square(diff_y))
        angle_error.append(angle_diff)

    max_error = max(angle_error)*180/pi
    print ('maximum error in deg: {:.4f}'.format(max_error))

    mean_error = np.mean(angle_error)*180/pi
    print ('mean error in deg: {:.4f}'.format(mean_error))
    print(angle_error)

    predictions = open(FOLDERTOSAVE+ "/Validation"+'models_and_predictions.txt',"a+")

    predictions.write("\n #### Max error: {} #### Mean error: {} \n".format( max_error, mean_error))

    predictions.close()




train_path = osp.join('/home/kucarst3-dlws/Desktop/HALWANY/halwanyfiles/dataset_hal/train')
train_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.95,0.999]) ])
train_dataset = MyOwnDataset(train_path, transform=train_data_aug)      #### transform=T.Cartesian()


valid_path = osp.join('/home/kucarst3-dlws/Desktop/HALWANY/halwanyfiles/dataset_hal/validate')
valid_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.95,0.999]) ])
valid_dataset = MyOwnDataset(valid_path, transform=valid_data_aug)      #### transform=T.Cartesian()


test_path = osp.join('/home/kucarst3-dlws/Desktop/HALWANY/halwanyfiles/dataset_hal/test')
test_data_aug = T.Compose([T.Cartesian(cat=False), T.RandomFlip(axis=0, p=0.3), T.RandomScale([0.95,0.999]) ])
test_dataset = MyOwnDataset(test_path, transform=test_data_aug)      #### transform=T.Cartesian()






import random
import numpy
seed_val = int(1)
print("Random Seed ID is: ", seed_val)
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)
os.environ['PYTHONHASHSEED'] = str(seed_val)



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = GCNConv(2, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        
        self.conv3 = GCNConv(128, 512)
        self.bn3 = torch.nn.BatchNorm1d(512) 
        
        self.conv4 = GCNConv(512, 256)
        self.bn4 = torch.nn.BatchNorm1d(256)  
        
        self.conv5 = GCNConv(256, 64)
        self.bn5 = torch.nn.BatchNorm1d(64)
        
        self.conv6 = GCNConv(64, 32)
        self.bn6 = torch.nn.BatchNorm1d(32)

        self.conv7 = GCNConv(32, 16)
        self.bn7 = torch.nn.BatchNorm1d(16)        

        self.conv8 = GCNConv(16, 2)
        self.bn8 = torch.nn.BatchNorm1d(16)    
        

        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 2)
        


    def forward(self, data):
        #data.x = F.elu(self.conv1(data.x , data.edge_index , data.edge_attr ))
        # print("index", data.x.view(-1,2).shape)
        data.x = F.sigmoid(self.conv1(torch.tensor(data.x.view(-1,2), dtype=torch.float32), data.edge_index) )#, data.edge_attr ))

        # data.x = F.sigmoid(self.conv1(torch.tensor(data.x.clone().detach(), dtype=torch.float32), data.edge_index))
        data.x = self.bn1(data.x)
        part1= data.x


        data.x = F.sigmoid(self.conv2(data.x, data.edge_index))
        data.x = self.bn2(data.x)
        part2= data.x

        
        data.x=torch.cat((part1,part2), dim=1) #64+64
        data.x = F.sigmoid(self.conv3(data.x, data.edge_index))
        data.x = self.bn3(data.x)
        
        data.x = F.sigmoid(self.conv4(data.x, data.edge_index))
        data.x = self.bn4(data.x)
        
        data.x = F.sigmoid(self.conv5(data.x, data.edge_index))
        data.x = self.bn5(data.x)

        data.x = F.sigmoid(self.conv6(data.x, data.edge_index))
        data.x = self.bn6(data.x)
        
        data.x = F.sigmoid(self.conv7(data.x, data.edge_index))
        data.x = self.bn7(data.x)

        # data.x = F.sigmoid(self.conv8(data.x, data.edge_index))
                # 2. Readout layer
        x = global_mean_pool(data.x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.sigmoid(self.fc1(x))
        #print("abdulrahman2 done",x.size())

        out = F.sigmoid(self.fc2(x))
        
        # print("out", out)
#         out=F.softmax(data.x, dim=1)
        return out


model=Net()

print(len(train_dataset))
print(len(valid_dataset))

print(len(test_dataset))



train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device", device)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
i=0
iv=0
epoch_losses = []
epoch_lossesv = []

print ("Training will start now")
for epoch in range(number_of_epoch):
    print("Epoch", epoch)
    epoch_loss = 0
    acc=0
    start=time.time()    

    for i, data in enumerate(train_loader):
        # print(data)
    #         with autograd.detect_anomaly():
        data = data.to(device)
        #print(data.y)
        optimizer.zero_grad()
        end_point = model(data)
        #print("bbbbb end_point",(end_point))    
        # print("bbbbb data.y",(data.y.view(-1,2) ))  
        # print("bbbbb end_point",(end_point.view(-1,2)[0] ))  

        loss = nn.L1Loss()(end_point.view(-1,2)[0], data.y.view(-1,2)[0])
        pred = end_point.max(1)[1]
        acc += (pred.eq(data.y).sum().item())/len(data.y)

        loss.backward()
        optimizer.step()
        # print("loss",loss)
        # if i % 10 == 0:
        #     print({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
        
        epoch_loss += loss.detach().item()
        i=i+1
    epoch_loss /= (i + 1)
    acc /= (i + 1)

    epoch_lossv = 0


    for iv, datav in enumerate(test_loader):
# print(data)
#         with autograd.detect_anomaly():
        datav = datav.to(device)
        #print(data.y)
        end_pointv = model(datav)
        #print("bbbbb end_point",(end_point))    
        # print("bbbbb data.y",(data.y.view(-1,2) ))  
        # print("bbbbb end_point",(end_point.view(-1,2)[0] ))  

        lossv = nn.L1Loss()(end_pointv.view(-1,2)[0], datav.y.view(-1,2)[0])
        

        # print("loss",loss)
        # if i % 10 == 0:
        #     print({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
        
        epoch_lossv += lossv.detach().item()
        iv=iv+1
    epoch_lossv /= (i + 1)



    end = time.time()
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), ' Elapsed time: ', end-start, 'validation loss', epoch_lossv)
    epoch_losses.append(epoch_loss)
    epoch_lossesv.append(epoch_lossv)

torch.save(model.state_dict(),FOLDERTOSAVE+'/model_weights.pth')
torch.save(model, FOLDERTOSAVE+ '/model.pkl')
plt.plot(np.stack(epoch_losses), label='train')
plt.plot(np.stack(epoch_lossesv), label='valid')
plt.legend()
plt.savefig(FOLDERTOSAVE+"/"+ str(number_of_epoch)+'epochs.png',dpi=300, bbox_inches='tight')
    # plt.savefig( str(number_of_epoch)+'epochs.pdf', format='pdf', dpi=1200)



print("training")
predict_angletrain(model, train_loader)

print("validation")
predict_anglevalid(model, valid_loader)

print("testing")
predict_angletest(model, test_loader)
