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

np.set_printoptions(linewidth=1000000000)
torch.cuda.manual_seed(1)

training_data = []
testing_data = []

I = open("pssm_list.tsv","r").readlines()
pssm_data = list(map(str.strip, I))

pdb_features = dict()
all_sequence = dict()
for i in pssm_data:
  I = iter(list(map(str.strip,open("PSSM/"+i,"r").readlines())))
  r = i.split("_")
  pdb = r[0]+"_"+r[1]
  ch  = r[2]

  if not pdb in pdb_features:
    pdb_features[pdb] = dict()
  if not pdb in all_sequence:
    all_sequence[pdb] = dict()
  next(I)
  for j in I:
    r = j.split(" ")
    res_id = r[1]
    pdb_features[pdb][res_id+ch] = dict()
    if not ch in all_sequence[pdb]:
      all_sequence[pdb][ch] = list()
    all_sequence[pdb][ch].append(res_id+ch)
    pdb_features[pdb][res_id+ch]['pssm'] = list(map(float,r[3:23]))

rri = dict()
cci = dict()
N_cci = 0
PDB = list(map(str.strip, open("rri_list.tsv","r").readlines()))
for i in PDB:
  if not i in rri:
    rri[i] = dict()
    cci[i] = dict()
  J = iter(list(map(str.strip,open("pairPred_contactMap/"+i+".int","r").readlines())))
  for j in J:
    r =  j.split("\t")
    ch_r = r[0][-1]
    ch_l = r[1][-1]
    rri[i][r[0]+":"+r[1]]=True
    if not ch_r+":"+ch_l in cci[i]:
      N_cci += 1
      cci[i][ch_r+":"+ch_l] = True


class BiLSTM(nn.Module):

    def __init__( self, input_dim=20, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, hidden_3_dim=512, rri_size=2 ):
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
        

    def prepare_data(self, pdb, sequence):
        list_pssm = []
        for aa in sequence:
          list_pssm.append(list(pdb_features[pdb][aa]["pssm"]))
        return autograd.Variable( torch.unsqueeze(torch.FloatTensor(list_pssm),dim=1) ).cuda() 

    def forward(self, pdb, sequence_r, sequence_l ):

        v_r = self.prepare_data( pdb+"_r", sequence_r )
        v_l = self.prepare_data( pdb+"_l", sequence_l )

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

        rri_out = autograd.Variable( torch.FloatTensor(len(sequence_r)*len(sequence_l),self.rri_size) ).cuda()

        rl = 0
        N_r = len(sequence_r)
        N_l = len(sequence_l)

        v_in = autograd.Variable( torch.FloatTensor(N_r*N_l,2*self.hidden_2_dim) ).cuda()
        for r in range( N_r ):
          for l in range( N_l ): 
            v_in[rl,:] = torch.cat([out_hidden_r_2[r,:],out_hidden_l_2[l,:]],dim=0)
            rl += 1

        hidden_3 = self.hidden2hidden_3(v_in)
        hidden_3 = self.drop_hidden_3(hidden_3)
        out_hidden_3 = F.relu(hidden_3)

        rri_out = self.hidden2out( out_hidden_3 )
        rri_out = F.log_softmax( rri_out )

        return rri_out

def get_native_rri( pdb, sequence_r, sequence_l):

  N_r = len(sequence_r)
  N_l = len(sequence_l)

  RRI = []
  for r in range( N_r ):
    for l in range( N_l ): 
      if sequence_r[r]+":"+sequence_l[l] in rri[pdb]:
        RRI.append(1)
      else:
        RRI.append(0)

  return autograd.Variable(torch.LongTensor(RRI)).cuda()


model = BiLSTM(input_dim=20, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, hidden_3_dim=512, rri_size=2 )
model.cuda()
print(model)
loss_function = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)

N = len(training_data)

print("Neural networking ...")

for target in cci:
  lr  = 0.1
  for epoch in range(2000):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr *= 0.99
    N_current = N_cci
    for pdb in cci:
      if pdb == target:
        continue
      for ch_ch in cci[pdb]:
        R = ch_ch.split(":")
        ch_r = R[0]
        ch_l = R[1]

        print("%d - %s %s:%s         \n"  %(N_current, pdb,ch_r,ch_l),end="")
        N_current -= 1

        local_sequence_r = all_sequence[pdb+"_r"][ch_r]
        local_sequence_l = all_sequence[pdb+"_l"][ch_l]

        model.zero_grad()
        optimizer.zero_grad()

        predicted_rri = model( pdb, local_sequence_r, local_sequence_l )
        #print("\tprediction done")
        native_rri =  get_native_rri( pdb, local_sequence_r, local_sequence_l )
        #print("\tnative compiled")
        loss = loss_function( predicted_rri, native_rri )
        #print("\tloss computed")
        loss.backward()
        #print("\toptimized step")
        optimizer.step()
    
        np_class = native_rri.data.cpu().numpy() 
        np_prediction = predicted_rri.data.cpu()[:,1].numpy()
  
        fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
        new_auc = metrics.auc(fpr, tpr)
          
        print("\tAUC=%0.4f"%(new_auc))

    ##TESTING FOR EACH EPOCH
    model.train(mode=False)
    for ch_ch in cci[target]:
      R = ch_ch.split(":")
      ch_r = R[0]
      ch_l = R[1]

      print("EVALUATION %s %s:%s         \n"  %(target,ch_r,ch_l),end="")

      local_sequence_r = all_sequence[target+"_r"][ch_r]
      local_sequence_l = all_sequence[target+"_l"][ch_l]

      model.zero_grad()
      optimizer.zero_grad()

      predicted_rri = model( target, local_sequence_r, local_sequence_l )
      native_rri =  get_native_rri( target, local_sequence_r, local_sequence_l )

      np_class = native_rri.data.cpu().numpy() 
      np_prediction = predicted_rri.data.cpu()[:,1].numpy()

      fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
      new_auc = metrics.auc(fpr, tpr)
        
      print("AUC=%0.4f"%(new_auc))
    model.train(mode=True)

  exit()

