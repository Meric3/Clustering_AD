"""
sdfasdkf
i

Code for ..

"""

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

from torch.autograd import Variable

import time
from datetime import datetime 

class Solver():
    def __init__(self, args):
        torch.manual_seed(777)
        torch.cuda.manual_seed_all(777)
        np.random.seed(777)
        
        self.args = args
        
        self.dataset = dataset.base_dataset(data_path = self.args.data_path, \
                                   window_length = self.args.window_length, \
                                   nrow = self.args.nrow, train = self.args.train)
        
        self.train_loader = DataLoader(dataset = self.dataset, batch_size = 32, shuffle = True, num_workers = 2, drop_last = True)
   
    
        self.args.input_size = self.dataset.data.shape[1]
        
        
        # device = torch.device('cuda' if torch.cuda.is_available else "cpu")
        self.device = 'cuda'

        self.encoder = model.Encoder(args)
        self.encoder.to(self.device)

        self.decoder = model.Decoder(args)
        self.decoder.to(self.device)
        
        self.optim_enc   = torch.optim.Adam(self.encoder.parameters(), self.args.lr)
        self.optim_dec   = torch.optim.Adam(self.decoder.parameters(), self.args.lr)
        
        self.loss_fn = nn.MSELoss()
        
        self.logger = logger.Logger(self.args.tensorboard_log_path)
        
        self.check_tb_size = int(len(self.train_loader.dataset) / 100)
        
        
    def fit(self, load = False):
        print("fit")
        
        start_time = time.time()
        
        log_time = 0
        encoder_loss = 0
        decoder_loss = 0
        
        for epoch in range(self.args.epoch):
            print("-"*99)
            print("epoch ", epoch)
            print("-"*99)
            hidden_enc = self.encoder.init_hidden(self.args.batch_size)
            for i, data in enumerate(self.train_loader):
                
                
#                 pdb.set_trace()
#                 hidden_enc = self.encoder.init_hidden(self.args.batch_size)
                self.optim_enc.zero_grad()
                self.optim_dec.zero_grad()
                
                inputs, labels = data[0].type(torch.FloatTensor).to(self.device),\
                data[1].type(torch.FloatTensor).to(self.device)

#                 print(hidden_enc[0].shape)
                hidden_enc = self.encoder.repackage_hidden(hidden_enc)
#                 print(hidden_enc[0].shape)
#                 print(inputs.shape)

                outputseq_enc, hidden_enc = self.encoder.forward(inputs, hidden_enc, return_hiddens = True)
#                 print(hidden_enc[0].shape)
                
#                 loss_enc = self.loss_fn(outputseq_enc[:,-1,:].view(self.args.batch_size, -1), inputs[:,-1,:].contiguous().view(self.args.batch_size, -1))
                loss_enc = self.loss_fn(outputseq_enc[:,-1,:], inputs[:,-1,:].contiguous())
                loss_enc.backward(retain_graph= True)      
                
                encoder_norm = sum(p.grad.data.abs().sum() for p in self.encoder.parameters())
                # print('encoder param {}'.format(encoder_norm))
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.args.clip)
                # print('encoder param {}'.format(encoder_norm))
#                 pdb.set_trace()
#                 print('encoder loss {}'.format(encoder_norm))
                self.optim_enc.step()  


                if self.args.teaching_force == True :

                    decoder_input = Variable(torch.zeros(outputseq_enc.size())).cuda()
                    decoder_input[:,0,:] = outputseq_enc[:,-1,:] 
                    decoder_input[:,1:,:] = inputs.flip(1)[:, 1:, :]
                    
                    output_dec, hidden_enc = self.decoder.forward(decoder_input, hidden_enc, return_hiddens=True)
   

                    loss_dec = self.loss_fn(output_dec.view(self.args.batch_size, -1), inputs.flip(1).contiguous().view(self.args.batch_size, -1))   
                    loss_dec.backward()
                    decoder_norm = sum(p.grad.data.abs().sum() for p in self.decoder.parameters())
                
#                     print('decoder loss {}'.format(decoder_norm))
                    self.optim_dec.step()
    
                if i % self.check_tb_size == 0 and self.args.show_tensorboard == True:

                    # 1. Log scalar values (scalar summary)
                    info = { 'enc_loss': loss_enc.item(), 'dec_loss' : loss_dec.item() }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, log_time)

                    # 2. Log values and gradients of the parameters (histogram summary)
                    for tag, value in self.encoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), log_time)
                        
                    for tag, value in self.decoder.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, value.data.cpu().numpy(), log_time)
                    log_time += 1
                encoder_loss += loss_enc.item()
    
    
    
            print('encoder loss {}'.format(encoder_loss))
            encoder_loss = 0
#             print('decoder loss {}'.format(decoder_norm))

        
        
        
        


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
    parser.add_argument("--window_length", type=int, default=6)
    
    parser.add_argument("--seq_length", type=int, default=2)
    parser.add_argument('--selected_dim', nargs='+', type=int, default=[36, 38, 28, 40])

    parser.add_argument("--teaching_force", default=True, action="store_true")
    parser.add_argument("--train", default=True, action="store_true")
    parser.add_argument("--show_tensorboard", default=False, action="store_true")
    parser.add_argument("--tensorboard_log_path", type=str, default="./tf_logs/")
    args = parser.parse_args()

    args.show_tensorboard = True
    args.tensorboard_log_path = "./tf_logs/test/"
    solver = Solver(args = args)
    solver.fit()



#     dataset = dataset.base_dataset(data_path = args.data_path, \
#                                    window_length = args.window_length, \
#                                    nrow = args.nrow, train = args.train)
    
#     args.input_size = dataset.data.shape[1]




#     train_loader = DataLoader(dataset = dataset, batch_size = 32, shuffle = True, num_workers = 2)
    
#     # device = torch.device('cuda' if torch.cuda.is_available else "cpu")
#     device = 'cuda'
    
#     encoder = model.Encoder(args)
#     encoder.to(device)

#     decoder = model.Decoder(args)
#     decoder.to(device)

#     optim_enc   = torch.optim.Adam(encoder.parameters(), args.lr)
#     optim_dec   = torch.optim.Adam(decoder.parameters(), args.lr)

#     loss_fn = nn.MSELoss()   

#     for epoch in range(2):
        
#         hidden_enc = encoder.init_hidden(args.batch_size)
#         for i, data in enumerate(train_loader):
#             inputs, labels = data[0].type(torch.FloatTensor).to(device),\
#             data[1].type(torch.FloatTensor).to(device)
          
            
#             hidden_enc = encoder.repackage_hidden(hidden_enc)
            
#             outputseq_enc, hidden_enc = encoder.forward(inputs[:,0:int(args.window_length/2),:], hidden_enc, return_hiddens = True)
            
# #             pdb.set_trace()
            
#             if args.teaching_force == True:
                
#                 deccoder_input = Variable(torch.zeros(outputseq_enc.size())).cuda()
#                 deccoder_input[:,0,:] = outputseq_enc[:,-1,:] 
#                 deccoder_input[:,1:,:] = inputs[:,int(args.window_length/2) + 1:-1,:]
            
            
#                 pdb.set_trace()
            
            
            
#     pdb.set_trace()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
