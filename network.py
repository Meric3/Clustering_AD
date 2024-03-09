import pdb
import sys
import os 
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

import util.dataset as dataset
import util.logger as logger
import model.model as model

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import time
from datetime import datetime 
from loss import *


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    
class EmbeddingNet_(nn.Module):
    def __init__(self, args):
        super(EmbeddingNet, self).__init__()
        self.args = args
        """ 
        self.convnet = nn.Sequential(nn.Conv1d(6, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv1d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))
        """ 
        self.convnet = nn.Sequential(nn.Conv1d(12,128 , 2, padding = 1), nn.PReLU(),
                                     nn.Conv1d(128, 256, 2, padding = 1), nn.PReLU(),
                                    ) 

        self.fc = nn.Sequential(nn.Linear(2560, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
#         print(x.shape)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)









class Solver():
    def __init__(self, args):
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        np.random.seed(777)
        
        self.args = args
       
        self.dataset = dataset.triplet_dataset(data_path = self.args.data_path, \
                                   window_length = self.args.window_length, \
                                   nrow = self.args.nrow, train = self.args.train)
        self.train_loader = DataLoader(dataset = self.dataset, batch_size = 32, shuffle = True, num_workers = 2, drop_last = True)
        self.args.input_size = self.dataset.train_data.shape[1]
        
        
        # device = torch.device('cuda' if torch.cuda.is_available else "cpu")
        self.device = 'cuda'

        self.embedding = EmbeddingNet(args)        
        self.embedding.to(self.device)
        
        self.model = TripletNet(self.embedding)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)
        self.loss_fn = TripletLoss(10)  
        
#         self.decoder = model.Decoder(args)
#         self.decoder.to(self.device)
#         self.optim_enc   = torch.optim.Adam(self.encoder.parameters(), self.args.lr)
#         self.optim_dec   = torch.optim.Adam(self.decoder.parameters(), self.args.lr)
#         self.loss_fn = nn.MSELoss()
        self.logger = logger.Logger(self.args.tensorboard_log_path + time.strftime('%Y-%m-%d',\
                time.localtime(time.time())))
        
        self.check_tb_size = int(len(self.train_loader.dataset) / 100)
        
        
    def fit(self, load = False):
        print("fit")

        self.model.train()
        
        start_time = time.time()
        
        log_time = 0
        cumul_loss= 0
        
        for epoch in range(self.args.epoch):
            for i, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                
                output = self.model.forward(data[0].type(torch.FloatTensor).to(self.device), \
                                   data[1].type(torch.FloatTensor).to(self.device), \
                                   data[2].type(torch.FloatTensor).to(self.device))
                loss = self.loss_fn(output[1], output[1], output[2])

                loss.backward()

                self.optimizer.step()
    
                if i % self.check_tb_size == 0 and self.args.show_tensorboard == True:

                    # 1. Log scalar values (scalar summary)
                    info = { 'loss': loss.item()}

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, log_time)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in self.model.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), log_time)
                        
                    log_time += 1
                cumul_loss += loss.item()
    
    
    
            print('culmul loss {}'.format(cumul_loss))
            cumul_loss = 0

        
        
        
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="enc_dec")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--clip", type=int, default=0.1)

    parser.add_argument("--data_path", type=str, default= './wadi_data/')
    
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--nrow", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=30)

    parser.add_argument("--cell_type", type=str, default="LSTM")
    parser.add_argument("--window_length", type=int, default=12)
    
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument('--selected_dim', nargs='+', type=int, default=[36, 38, 28, 40])

    parser.add_argument("--teaching_force", default=True, action="store_true")
    parser.add_argument("--train", default=True, action="store_true")
    parser.add_argument("--show_tensorboard", default=False, action="store_true")
    parser.add_argument("--tensorboard_log_path", type=str, default="./tf_logs/")
    args = parser.parse_args()

    args.show_tensorboard = True
    args.tensorboard_log_path = "./tf_logs/test2/"
    solver = Solver(args = args)
    solver.fit()
