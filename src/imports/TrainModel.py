import json
import torch
import torch_geometric as pyg
from tqdm.auto import tqdm, trange  
from pathlib import Path
from numpy import pi
from pandas import DataFrame
from .TactileDataset import TactileDataset
import numpy as np

class TrainModel():

    def __init__(
        self, 
        extraction_case_dir, 
        model,
        n_epochs = 150,
        optimizer = 'adam',
        lr = 0.001,
        loss_func = torch.nn.L1Loss(),
        transform = None,
        features = 'all',
        weight_decay=0,
        patience=10,
        batch = 1,
        augment=False,
        seed=0
        ):

        self.extraction_case_dir = Path(extraction_case_dir)
        self.transform = transform

        self.train_data = TactileDataset(self.extraction_case_dir / 'train', transform=transform, features=features, augment=augment)
        self.val_data = TactileDataset(self.extraction_case_dir / 'val', features=features)
        self.test_data = TactileDataset(self.extraction_case_dir / 'test', features=features)

        self.train_loader = pyg.loader.DataLoader(self.train_data, shuffle=True, batch_size=batch)
        self.val_loader = pyg.loader.DataLoader(self.val_data)
        self.test_loader = pyg.loader.DataLoader(self.test_data)

        self.model = model
        self.n_epochs = n_epochs


        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError('use tm.optimizer = torch.optim.<optimizer>')
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-5, patience=patience)

        self.loss_func = loss_func

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(0)
        np.random.seed(0)
        

    def train(self):
        self.train_losses = []
        self.val_losses = []
        self.lr = []
        self.max_losses = []

        name = str(type(self.model)).split('.')[-1][:-2]
        path = Path('results') / name

        for epoch in trange(self.n_epochs, desc='training', unit='epoch'):
            #bunny(epoch)
            epoch_loss = 0
            
            if (epoch == 25):
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.001
    
            if epoch == 110:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.0001

            if epoch == 200:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = 0.00001
                    
            lr = self.optimizer.param_groups[0]['lr']
            self.lr.append(lr)
            val_loss = torch.inf
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    with torch.autograd.detect_anomaly():
                        data = data.to(self.device)
                        self.optimizer.zero_grad()
                        end_point = self.model(data)
                        loss = self.loss_func(end_point, data.y)
                        loss.backward()
                        self.optimizer.step()
                        lr = self.optimizer.param_groups[0]['lr']

                        epoch_loss += loss.detach().item()
                    
                        tepoch.set_postfix({
                            'train_loss': epoch_loss / (i + 1), 
                            'train_loss_degrees': epoch_loss / (i + 1) * 180/pi, 
                            'val_loss': self.val_losses[epoch - 1] if epoch > 0 else 'na',
                            'val_loss_degrees': self.val_losses[epoch - 1] * 180/pi if epoch > 0 else 'na',
                            'max_loss_degree': self.max_losses[epoch - 1] * 180/pi if epoch >0 else 'na',
                            'lr': lr
                            })

                #self.scheduler.step(val_loss)
                epoch_loss /= i + 1
                val_loss, max_loss = self.validate()
                self.max_losses.append(max_loss)
                tepoch.set_postfix({'train_loss': epoch_loss, 'val_loss': val_loss})
                self.train_losses.append(epoch_loss)
                self.val_losses.append(val_loss)
            if (epoch + 1) % 1 == 0:
                self.log(current_epoch=epoch)
        torch.save(self.model, path / 'model.pt')

    def validate(self):
        loss = 0
        losses = []
        for i, data in enumerate(self.val_loader):      
            data = data.to(self.device)
            end_point = self.model(data)
            l = self.loss_func(end_point, data.y).detach().item()
            loss += l
            losses.append(l)
        loss /= len(self.val_data)
        return loss, max(losses)
    
    def test(self):
        loss = 0
        for i, data in enumerate(self.test_loader):      
            data = data.to(self.device)
            end_point = self.model(data)

            loss += self.loss_func(end_point, data.y).detach().item()
        loss /= len(self.test_data)
        return loss

    def augment(self, batch):
        pass

    def log(self, current_epoch):
        #find model name
        print('logging')
        name = str(type(self.model)).split('.')[-1][:-2]
        path = Path('results') / name
        if not path.exists():
            path.mkdir(parents=True)

        with open(path / 'training_params.json', 'w') as f:
            params = {
                'model': name,
                'extraction_used': str(self.extraction_case_dir),
                'n_epochs': self.n_epochs,
                'final_val_loss_degrees': self.val_losses[-1] * 180 / pi,
            }
            json.dump(params, f, indent=4)

        train_log = { 
            'epoch': [i for i in range(1, current_epoch+2)],
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'max_loss': self.max_losses,
            'lr': self.lr
        }
        DataFrame(train_log).to_csv(path / 'train_log.csv', index=False)

