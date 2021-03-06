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

training = dict()
testing = dict()

#training['features'], training['sequences'] = read_seq_pssm("pssm_dimers_280_list.tsv","dimers_280")
#training['contacts'] = read_contacts("rri_dimers_280_list.tsv","dimers_280_rri",training['features'])
#training['scop'] = read_scop("280_dimers_list.tsv")
training['features'], training['sequences'] = read_seq_pssm("pssm_dimers_450_list.tsv","dimers_450")
training['contacts'] = read_contacts("rri_dimers_450_list.tsv","dimers_450_rri",training['features'])
training['scop'] = read_scop("450_dimers_list.tsv")

testing['features'], testing['sequences'] = read_seq_pssm("pssm_dimers_bipspi_list.tsv","dimers_bipspi")
testing['contacts'] = read_contacts("rri_dimers_bipspi_list.tsv","dimers_bipspi_rri",testing['features'])


class BiLSTM(nn.Module):

    def __init__( self, input_dim=22, rri_size=2 ):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.conv2d_channels_added = 16
        self.n_conv2d = 2
        self.k_size = 11
        self.pad_size = int(self.k_size/2)

        self.rri_size = rri_size
        
        self.conv2d = []
        out_channels = -1

        in_channels = 2*self.input_dim
        out_channels = 32#in_channels 
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, stride=1, padding=self.pad_size, dilation=1, groups=1, bias=True)

        in_channels = out_channels
        out_channels = in_channels + self.conv2d_channels_added
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, stride=1, padding=self.pad_size, dilation=1, groups=1, bias=True)

        in_channels = out_channels
        out_channels = in_channels + self.conv2d_channels_added
        self.conv2d_2 = nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, stride=1, padding=self.pad_size, dilation=1, groups=1, bias=True)

        in_channels = out_channels
        out_channels = in_channels + self.conv2d_channels_added
        self.conv2d_3 = nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, stride=1, padding=self.pad_size, dilation=1, groups=1, bias=True)

        in_channels = out_channels
        out_channels = in_channels + self.conv2d_channels_added
        self.conv2d_4 = nn.Conv2d(in_channels, out_channels, kernel_size=self.k_size, stride=1, padding=self.pad_size, dilation=1, groups=1, bias=True)

        self.conv2d_dim = out_channels
        self.hidden2out = nn.Linear(out_channels, self.rri_size)

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

        rl = 0
        N_r = len(sequence_r)
        N_l = len(sequence_l)
        native_rri = []

        conv_matrix = autograd.Variable( torch.FloatTensor(2*self.input_dim,N_r,N_l) ).cuda()

        if(verbose == "full"):print("\tBUILDING CONV MATRIX %sx%s"%(N_r,N_l))

        dim_2 = self.input_dim
        dim_4 = 2*dim_2

        v_r = v_r.squeeze(1)
        v_l = v_l.squeeze(1)


        _out_LSTM_r = torch.cat((v_r, autograd.Variable( torch.zeros(N_r,dim_2)).cuda() ),dim=1)
        _out_LSTM_l = torch.cat((autograd.Variable( torch.zeros(N_l,dim_2)).cuda(), v_l),dim=1)

        pre_conv = _out_LSTM_r.repeat(N_l,1,1).permute(1,0,2) + _out_LSTM_l.repeat(N_r,1,1)

        _out_LSTM_r = torch.cat((autograd.Variable( torch.zeros(N_r,dim_2)).cuda(),v_r ),dim=1)
        _out_LSTM_l = torch.cat((v_l,autograd.Variable( torch.zeros(N_l,dim_2)).cuda()),dim=1)

        pre_conv_t = _out_LSTM_r.repeat(N_l,1,1).permute(1,0,2) + _out_LSTM_l.repeat(N_r,1,1)

        pre_conv = pre_conv.permute(2,0,1).unsqueeze(0)
        pre_conv_t = pre_conv_t.permute(2,0,1).unsqueeze(0)

        conv_matrix_0 = F.relu( 0.5*(self.initial_conv(pre_conv)+self.initial_conv(pre_conv_t)) )

        if(verbose == "full"):print("\tCONVOLVING")
        conv_matrix_1 = F.relu( self.conv2d_1(conv_matrix_0) )
        conv_matrix_2 = F.relu( self.conv2d_2(conv_matrix_1) )
        conv_matrix_3 = F.relu( self.conv2d_3(conv_matrix_2) )
        conv_matrix_4 = F.relu( self.conv2d_4(conv_matrix_3) )

        conv_matrix = conv_matrix_4
        if Flag:
          res_rri = list(contacts['rri_ch_ch'][pdb][ch_r+":"+ch_l].keys())
          P_rri = int(len(res_rri)*0.25)
          if(P_rri == 0): P_rri=1
          N_rri = P_rri
          v_in = autograd.Variable( torch.FloatTensor(N_rri+P_rri,self.conv2d_dim) ).cuda()

          if(verbose == "full"):print("\tCOLLECTING POSITIVES %s" % P_rri)
          while P_rri>0:
            rr = res_rri[ random.randint(0,len(res_rri))-1 ]
            R = rr.split(":")
            try:
              i_r = sequence_r.index(R[0])
            except ValueError as err:
              raise err
            try:
              j_l = sequence_l.index(R[1])
            except ValueError as err:
              raise err

            v_in[rl,:] = conv_matrix[0,:,i_r,j_l]
            rl += 1
            native_rri.append(1)
            P_rri -= 1

          if(verbose == "full"):print("\tCOLLECTING NEGATIVES %s" % N_rri)
          while N_rri>0:
            i_r = random.randint(0,N_r-1)
            j_l = random.randint(0,N_l-1)
            if( not sequence_r[i_r]+":"+sequence_l[j_l] in contacts['rri'][pdb]):

              v_in[rl,:] = conv_matrix[0,:,i_r,j_l]

              rl += 1
              N_rri -= 1
              native_rri.append(0)

          if(verbose == "full"):print("\tLOG_SOFTMAX")
          rri_out = self.hidden2out( v_in )
          rri_out = F.log_softmax( rri_out, dim=1 )

          native_rri = autograd.Variable(torch.LongTensor(native_rri)).cuda()

          return rri_out, native_rri

        else:
          v_in = autograd.Variable( torch.FloatTensor(N_r*N_l,self.conv2d_dim) ).cuda()

          for i_r in range( N_r ):
            for j_l in range( N_l ): 
              if sequence_r[i_r]+":"+sequence_l[j_l] in contacts['rri'][pdb]:
                native_rri.append(1)
              else:
                native_rri.append(0)

              v_in[rl,:] = conv_matrix[0,:,i_r,j_l]
              rl += 1

          rri_out = self.hidden2out( v_in )
          rri_out = F.log_softmax( rri_out, dim=1 )

          native_rri = autograd.Variable(torch.LongTensor(native_rri)).cuda()

          return rri_out, native_rri
          
input_dim=22
lstm_hidden_dim=64

model = BiLSTM( input_dim=input_dim, rri_size=2 )
model.cuda()
if(verbose): 
  print(model)
  for c2d in model.conv2d:
    print(c2d)

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

  lr  = 0.01
  model = BiLSTM( input_dim=input_dim, rri_size=2 )
  model.cuda()
  optimizer = optim.SGD(model.parameters(), lr=lr)
  #optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
  for epoch in range(100):


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

        if(verbose): print("%d - %s %s:%s                            \r"  %(N_current, pdb,ch_r,ch_l),end="")
        N_current -= 1

        local_sequence_r = training['sequences'][pdb][ch_r]
        local_sequence_l = training['sequences'][pdb][ch_l]

        model.zero_grad()
        optimizer.zero_grad()

        if(verbose == "full"):print("\nPREDICTING")
        predicted_rri, native_rri = model( pdb, local_sequence_r, local_sequence_l, training['features'], training['contacts'], ch_r=ch_r, ch_l=ch_l, Flag=True )
        if(verbose == "full"):print("LOSSING")
        loss = loss_function( predicted_rri, native_rri )
        if(verbose == "full"):print("OPTIMISING")
        loss.backward()
        optimizer.step()
        if(verbose == "full"):print("END")
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

