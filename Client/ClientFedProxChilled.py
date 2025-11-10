
import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
from sklearn.cluster import KMeans
import torch.optim as optim
import torch.nn.functional as F
from utils import Accuracy,soft_predict
from Client.ClientBase import Client
from models import TempNet

class ClientFedProxChilled(Client):
    """
    This class is for train the local model with input global model(copied) and output the updated weight
    args: argument 
    Loader_train,Loader_val,Loaders_test: input for training and inference
    user: the index of local model
    idxs: the index for data of this local model
    logger: log the loss and the process
    """
    def __init__(self, args, model, Loader_train,loader_test,idx, code_length, num_classes, device):
        super().__init__(args, model, Loader_train,loader_test,idx, code_length, num_classes, device)

        self.tempnet = TempNet(feature_dim=512).to(device)

    
    def update_weights_Prox(self,global_round, lam):
        self.model.cuda()
        self.model.train()
        global_model = copy.deepcopy(self.model)
        global_model.eval()
        global_weight_collector = list(global_model.parameters())
        epoch_loss = []
        optimizer = optim.SGD(self.model.parameters(),lr=self.args.lr)
        temp_optimizer = optim.SGD(self.tempnet.parameters(), lr=self.args.lr)

        for iterr in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (X, y) in enumerate(self.trainloader):
                X = X.to(self.device)
                y = y.to(self.device).long()
                optimizer.zero_grad()
                feats,p = self.model(X)
                tau = self.tempnet(feats.detach())
                scaled_p = p / tau
                y_pred = scaled_p.argmax(1)
                loss1 = self.ce(scaled_p,y)
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.model.parameters()):
                    fed_prox_reg += ((lam / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss = loss1 + lam*fed_prox_reg
                loss.backward()
                if self.args.clip_grad != None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = self.args.clip_grad)
                optimizer.step()
                temp_optimizer.step()
                if batch_idx % 10 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t prox_loss: {:.6f}'.format(
                        global_round, iterr, batch_idx * len(X),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(),fed_prox_reg.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        with torch.no_grad():

            sample_data, _ = next(iter(self.trainloader))
            sample_data = sample_data.to(self.device)
            f, _ = self.model(sample_data)
            tau_val = self.tempnet(f).item()

        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), tau_val