#CUDA_VISIBLE_DEVICES=1
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import numpy as np
import json
import os.path
import subprocess
import random
from operator import itemgetter
import sklearn.metrics as metrics 
import sys

from benchmark_tools import *

np.set_printoptions(linewidth=1000000000)
torch.cuda.manual_seed(1)

verbose = False#True
file_name = sys.argv[1]
results_path = sys.argv[2]
if len(sys.argv)>3:
  verbose = True

training = dict()
testing = dict()

training['features'], training['sequences'] = read_seq_pssm("pssm_dimers_450_list.tsv","dimers_450")
training['contacts'] = read_contacts("rri_dimers_450_list.tsv","dimers_450_rri",training['features'])
training['scop'] = read_scop("450_dimers_list.tsv")

testing['features'], testing['sequences'] = read_seq_pssm("pssm_dimers_bipspi_list.tsv","dimers_bipspi")
testing['contacts'] = read_contacts("rri_dimers_bipspi_list.tsv","dimers_bipspi_rri",testing['features'])

class BiLSTM(nn.Module):

    def __init__( self, input_dim=22, lstm_hidden_dim=256, hidden_1_dim=1024, hidden_2_dim=512, hidden_3_dim=1024, rri_size=2 ):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_1_dim = hidden_1_dim
        self.hidden_2_dim = hidden_2_dim
        self.hidden_3_dim = hidden_3_dim
        self.rri_size = rri_size

        self.lstm_h0 = None
        self.lstm_c0 = None
        self.update_lstm_hidden()

        self.LSTM = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

        self.drop_hidden_1 = nn.Dropout(p=0.5)
        self.lstm2hidden_1 = nn.Linear(2*lstm_hidden_dim, hidden_1_dim)
        
        self.drop_hidden_2 = nn.Dropout(p=0.5)
        self.hidden2hidden_2 = nn.Linear(hidden_1_dim, hidden_2_dim)

        self.drop_hidden_3 = nn.Dropout(p=0.5)
        self.hidden2hidden_3 = nn.Linear(2*hidden_2_dim, hidden_3_dim)

        self.hidden2out = nn.Linear(hidden_3_dim, rri_size)

    def update_lstm_hidden(self):
        self.lstm_h0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        self.lstm_c0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        

    def prepare_data(self, pdb, sequence, features):
        list_pssm = []
        for aa in sequence:
          v = list(features[pdb][aa]["pssm"])
          list_pssm.append(v)
        return autograd.Variable( torch.unsqueeze(torch.FloatTensor(list_pssm),dim=1) ).cuda() 

    def forward( self, pdb, sequence_r, sequence_l, features, contacts, ch_r=None, ch_l=None, Flag=True ):

        v_r = self.prepare_data( pdb, sequence_r, features )
        v_l = self.prepare_data( pdb, sequence_l, features )

        self.update_lstm_hidden()
        out_LSTM_r, (hidden_LSTM_r, content_LSTM_r) = self.LSTM( v_r, (self.lstm_h0, self.lstm_c0))

        self.update_lstm_hidden()
        out_LSTM_l, (hidden_LSTM_l, content_LSTM_l) = self.LSTM( v_l, (self.lstm_h0, self.lstm_c0))

        hidden_r_1 = self.lstm2hidden_1( out_LSTM_r.view(len(sequence_r), -1) )
        hidden_r_1 = self.drop_hidden_1(hidden_r_1)
        out_hidden_r_1 = F.relu(hidden_r_1)

        hidden_r_2 = self.hidden2hidden_2( out_hidden_r_1 )
        hidden_r_2 = self.drop_hidden_2(hidden_r_2)
        out_hidden_r_2 = F.relu(hidden_r_2)

        hidden_l_1 = self.lstm2hidden_1( out_LSTM_l.view(len(sequence_l), -1) )
        hidden_l_1 = self.drop_hidden_1(hidden_l_1)
        out_hidden_l_1 = F.relu(hidden_l_1)

        hidden_l_2 = self.hidden2hidden_2( out_hidden_l_1 )
        hidden_l_2 = self.drop_hidden_2(hidden_l_2)
        out_hidden_l_2 = F.relu(hidden_l_2)

        rl = 0
        N_r = len(sequence_r)
        N_l = len(sequence_l)
        native_rri = []

        if Flag:
          res_rri = list(contacts['rri_ch_ch'][pdb][ch_r+":"+ch_l].keys())
          N_rri = len(res_rri)
          v_in = autograd.Variable( torch.FloatTensor(N_rri+len(res_rri),2*self.hidden_2_dim) ).cuda()
          v_in_t = autograd.Variable( torch.FloatTensor(N_rri+len(res_rri),2*self.hidden_2_dim) ).cuda()

          for rr in res_rri:
            R = rr.split(":")
            try:
              i_r = sequence_r.index(R[0])
            except ValueError as err:
              raise err
            try:
              j_l = sequence_l.index(R[1])
            except ValueError as err:
              raise err

            w_rc = out_hidden_r_2[i_r,:]
            w_lc = out_hidden_l_2[j_l,:]

            v_in[rl,:] = torch.cat( [w_rc, w_lc], dim=0 )
            v_in_t[rl,:] = torch.cat( [w_lc, w_rc], dim=0 )
            rl += 1
            native_rri.append(1)

          while N_rri>0:
            i_r = random.randint(0,N_r-1)
            j_l = random.randint(0,N_l-1)
            if( not sequence_r[i_r]+":"+sequence_l[j_l] in contacts['rri'][pdb]):
              w_rc = out_hidden_r_2[i_r,:]
              w_lc = out_hidden_l_2[j_l,:]

              v_in[rl,:] = torch.cat( [w_rc,w_lc], dim=0 )
              v_in_t[rl,:] = torch.cat( [w_lc,w_rc], dim=0 )

              rl += 1
              N_rri -= 1
              native_rri.append(0)

          hidden_3 = self.hidden2hidden_3(v_in)
          hidden_3 = self.drop_hidden_3(hidden_3)

          hidden_3_t = self.hidden2hidden_3(v_in_t)
          hidden_3_t = self.drop_hidden_3(hidden_3_t)

          out_hidden_3 = F.relu(0.5*(hidden_3+hidden_3_t))

          rri_out = self.hidden2out( out_hidden_3 )
          rri_out = F.log_softmax( rri_out, dim=1 )

          native_rri = autograd.Variable(torch.LongTensor(native_rri)).cuda()

          return rri_out, native_rri

        else:
          v_in = autograd.Variable( torch.FloatTensor(N_r*N_l,2*self.hidden_2_dim) ).cuda()
          v_in_t = autograd.Variable( torch.FloatTensor(N_r*N_l,2*self.hidden_2_dim) ).cuda()
          for i_r in range( N_r ):
            for j_l in range( N_l ): 
              if sequence_r[i_r]+":"+sequence_l[j_l] in contacts['rri'][pdb]:
                native_rri.append(1)
              else:
                native_rri.append(0)

              w_rc = out_hidden_r_2[i_r,:]
              w_lc = out_hidden_l_2[j_l,:]

              v_in[rl,:] = torch.cat( [w_rc,w_lc], dim=0 )
              v_in_t[rl,:] = torch.cat( [w_lc,w_rc], dim=0 )
              rl += 1

          hidden_3 = self.hidden2hidden_3(v_in)
          hidden_3 = self.drop_hidden_3(hidden_3)

          hidden_3_t = self.hidden2hidden_3(v_in_t)
          hidden_3_t = self.drop_hidden_3(hidden_3_t)

          out_hidden_3 = F.relu(0.5*(hidden_3+hidden_3_t))

          rri_out = self.hidden2out( out_hidden_3 )
          rri_out = F.log_softmax( rri_out, dim=1 )

          native_rri = autograd.Variable(torch.LongTensor(native_rri)).cuda()

          return rri_out, native_rri
          
input_dim=22
lstm_hidden_dim=256
hidden_1_dim=512
hidden_2_dim=256
hidden_3_dim=512

#input_dim=22
#lstm_hidden_dim=16
#hidden_1_dim=32
#hidden_2_dim=16
#hidden_3_dim=64

model = BiLSTM( input_dim=input_dim, lstm_hidden_dim=lstm_hidden_dim, hidden_1_dim=hidden_1_dim, hidden_2_dim=hidden_2_dim, hidden_3_dim=hidden_3_dim, rri_size=2 )
model.cuda()
if(verbose): print(model)
loss_function = nn.NLLLoss()

if(verbose): print("Neural networking ...")

TRAGETS = list(map(str.strip, open(file_name,"r").readlines()))

for target_ in TRAGETS:
  
  r = target_.split("\t")
  target = r[0]
  scop_r = r[3]
  scop_l = r[4]

  if os.path.isfile(results_path+"/"+target+".tsv"):
    continue 

  lr  = 0.1
  model = BiLSTM( input_dim=input_dim, lstm_hidden_dim=lstm_hidden_dim, hidden_1_dim=hidden_1_dim, hidden_2_dim=hidden_2_dim, hidden_3_dim=hidden_3_dim, rri_size=2 )
  model.cuda()

  for epoch in range(50):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr *= 0.99
    N_current = training['contacts']['N_cci']-1
    cci_ = list(training['contacts']['cci'].keys())
    random.shuffle(cci_)

    for pdb in cci_:
      for ch_ch in training['contacts']['cci'][pdb]:
        R = ch_ch.split(":")
        ch_r = R[0]
        ch_l = R[1]
        if training['scop'][pdb][ch_ch] == scop_r+":::"+scop_l or training['scop'][pdb][ch_ch] == scop_l+":::"+scop_r:
          N_current -= 1
          continue

        if(verbose): print("%d - %s %s:%s         \r"  %(N_current, pdb,ch_r,ch_l),end="")
        N_current -= 1

        local_sequence_r = training['sequences'][pdb][ch_r]
        local_sequence_l = training['sequences'][pdb][ch_l]

        model.zero_grad()
        optimizer.zero_grad()

        predicted_rri, native_rri = model( pdb, local_sequence_r, local_sequence_l, training['features'], training['contacts'], ch_r=ch_r, ch_l=ch_l, Flag=True )
        loss = loss_function( predicted_rri, native_rri )
        loss.backward()
        optimizer.step()

    model.train(mode=False)
    ##TRAINING AUC SCORE FOR EACH EPOCH
    AUC = []
    for pdb in cci_:
      for ch_ch in training['contacts']['cci'][pdb]:
        R = ch_ch.split(":")
        ch_r = R[0]
        ch_l = R[1]

        local_sequence_r = training['sequences'][pdb][ch_r]
        local_sequence_l = training['sequences'][pdb][ch_l]

        predicted_rri, native_rri = model( pdb, local_sequence_r, local_sequence_l, training['features'], training['contacts'], ch_r=ch_r, ch_l=ch_l, Flag=True )
        np_class = native_rri.data.cpu().numpy() 
        np_prediction = predicted_rri.data.cpu()[:,1].numpy()
  
        fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
        new_auc = metrics.auc(fpr, tpr)
        AUC.append(new_auc)
    training_auc =  np.mean(AUC)

    ##TESTING FOR EACH EPOCH
    for ch_ch in testing['contacts']['cci'][target]:
      R = ch_ch.split(":")
      ch_r = R[0]
      ch_l = R[1]

      local_sequence_r = testing['sequences'][target][ch_r]
      local_sequence_l = testing['sequences'][target][ch_l]

      N_r = len(local_sequence_r)
      N_l = len(local_sequence_l)

      model.zero_grad()
      optimizer.zero_grad()

      predicted_rri, native_rri = model( target, local_sequence_r, local_sequence_l, testing['features'], testing['contacts'], ch_r=ch_r, ch_l=ch_l, Flag=False )

      np_class = native_rri.data.cpu().numpy() 
      np_prediction = predicted_rri.data.cpu()[:,1].numpy()

      np_all = np.stack((np_prediction,np_class),  axis=-1)
      np_all = np.insert(np_all, 0, np.array((N_r,N_l)), 0) 
      np.savetxt(results_path+"/predictions/"+target+"."+ch_r+":"+ch_l+"."+str(epoch)+".tsv",np_all)

      fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
      testing_auc = metrics.auc(fpr, tpr)

      if(verbose): print( "%d - %s %s:%s - AUC=%0.4f - TRAINING_AUC=%0.4f" % (epoch, target,ch_r,ch_l,testing_auc,training_auc) )
      open(results_path+"/"+target+".tsv", "a").write("%d - %s %s:%s - AUC=%0.4f - TRAINING_AUC=%0.4f\n" % (epoch, target,ch_r,ch_l,testing_auc,training_auc) )
    model.train(mode=True)

