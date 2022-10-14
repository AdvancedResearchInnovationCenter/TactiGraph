import torch
import torch_geometric as pyg
from tqdm.auto import tqdm
from .TactileDataset import TactileDataset
from pathlib import Path

class TrainModel():

    def __init__(
        self, 
        extraction_case_dir, 
        model,
        n_epochs = 150,
        optimizer = 'adam',
        lr = 0.001,
        loss_func = torch.nn.MSELoss()
        ):

        self.extraction_case_dir = Path(extraction_case_dir)

        self.train_data = TactileDataset(self.extraction_case_dir / 'train')
        self.val_data = TactileDataset(self.extraction_case_dir / 'val')
        self.test_data = TactileDataset(self.extraction_case_dir / 'test')

        self.train_loader = pyg.loader.DataLoader(self.train_data)
        self.val_loader = pyg.loader.DataLoader(self.val_data)
        self.test_loader = pyg.loader  .DataLoader(self.test_data)

        self.model = model
        self.n_epochs = n_epochs

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError('use tm.optimizer = torch.optim.<optimizer>')
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.loss_func = loss_func

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        


    def train(self):
        train_losses = []
        val_losses = []
        for epoch in range(self.n_epochs):
            epoch_loss = 0
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    with torch.autograd.detect_anomaly():
                        data = data.to(self.device)
                        self.optimizer.zero_grad()
                        end_point = self.model(data)

                        loss = self.loss_func(end_point[0], data.y[0])
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.detach().item()
                        tepoch.set_postfix({'train_loss': epoch_loss / (i + 1)})

                epoch_loss /= len(self.train_data)
                val_loss = self.validate()
                self.scheduler.step(val_loss)

                train_losses.append(epoch_loss)
                val_losses.append(val_loss)

    def validate(self):
        loss = 0
        for i, data in enumerate(self.val_loader):
            data = data.to(self.device)
            end_point = self.model(data)

            loss += self.loss_func(end_point[0], data.y[0]).detach().item()
        loss /= len(self.val_data)

        return loss
        







