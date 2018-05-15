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

I = open("rri_list.tsv","r").readlines()
pdb_list = list(map(str.strip, I))

pdb_bs = dict()
chain_list = dict()
N_chains = 0
for i in pdb_list:

  pdb_bs[ i+"_l" ] = dict()
  chain_list[i] = { "r":{}, "l":{} }
  I = iter(list(map(str.strip,open("bestResults/struct_2/"+i+".res.tab.lig","r").readlines())))
  next(I)
  next(I)
  for j in I:
    R = j.split(" ")
    if int(R[2]) > 0:
      pdb_bs[ i+"_l" ][ R[1]+R[0] ]= True
    
    if not R[0] in chain_list[i]["l"]:
      N_chains += 1
    if R[1]+R[0] in pdb_features[ i+"_l" ]:
      pdb_features[ i+"_l" ][ R[1]+R[0] ]['score'] = float(R[3])
      chain_list[i]["l"][R[0]] = True

  pdb_bs[ i+"_r" ] = dict()
  I = iter(list(map(str.strip,open("bestResults/struct_2/"+i+".res.tab.rec","r").readlines())))
  next(I)
  next(I)
  for j in I:
    R = j.split(" ")
    if int(R[2]) > 0:
      pdb_bs[ i+"_r" ][ R[1]+R[0] ]= True
    if not R[0] in chain_list[i]["r"]:
      N_chains += 1
    if R[1]+R[0] in pdb_features[ i+"_r" ]:
      pdb_features[ i+"_r" ][ R[1]+R[0] ]['score'] = float(R[3])
      chain_list[i]["r"][R[0]] = True

def get_native_bs( pdb, ch):
  BS = []
  for aa in all_sequence[pdb][ch]:
    if aa in pdb_bs[pdb]:
      BS.append(1)
    else:
      BS.append(0)
  return autograd.Variable(torch.LongTensor(BS)).cuda()

class BiLSTM(nn.Module):

    def __init__( self, input_dim=21, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, bs_size=2 ):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_1_dim = hidden_1_dim
        self.hidden_2_dim = hidden_2_dim
        self.bs_size = bs_size

        self.lstm_h0 = None
        self.lstm_c0 = None
        self.update_lstm_hidden()

        self.LSTM = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

        self.drop_hidden_1 = nn.Dropout(p=0.5)
        self.lstm2hidden_1 = nn.Linear(2*lstm_hidden_dim, hidden_1_dim)
        
        self.drop_hidden_2 = nn.Dropout(p=0.5)
        self.hidden2hidden_2 = nn.Linear(hidden_1_dim, hidden_2_dim)

        self.hidden2out = nn.Linear(hidden_2_dim, bs_size)

    def update_lstm_hidden(self):
        self.lstm_h0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        self.lstm_c0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        

    def prepare_data(self, pdb, sequence):
        list_pssm = []
        list_initial_scores = []
        for aa in sequence:
          v = list(pdb_features[pdb][aa]["pssm"])
          if "score" in pdb_features[pdb][aa]: 
            v.append(pdb_features[pdb][aa]["score"])
            list_initial_scores.append( pdb_features[pdb][aa]["score"] )
          else:########SCORE  WAS  NOT FOUND !!!!!!  WHY ????
            v.append(0)
            list_initial_scores.append(0)
          list_pssm.append( v )

        return autograd.Variable( torch.unsqueeze(torch.FloatTensor(list_pssm),dim=1) ).cuda(), torch.FloatTensor(list_initial_scores) 

    def forward(self, pdb, sequence ):

        v_in, init_scores = self.prepare_data( pdb, sequence )

        out_LSTM, (hidden_LSTM, content_LSTM) = self.LSTM( v_in, (self.lstm_h0, self.lstm_c0))

        hidden_1 = self.lstm2hidden_1( out_LSTM.view(len(sequence), -1) )
        hidden_1 = self.drop_hidden_1(hidden_1)
        out_hidden_1 = F.relu(hidden_1)

        hidden_2 = self.hidden2hidden_2( out_hidden_1 )
        hidden_2 = self.drop_hidden_2(hidden_2)
        out_hidden_2 = F.relu(hidden_2)

        bs_out = self.hidden2out( out_hidden_2 )
        bs_out = F.log_softmax( bs_out )

        return bs_out, init_scores


model = BiLSTM(input_dim=21, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, bs_size=2)
model.cuda()
print(model)
loss_function = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)

N = len(training_data)
current_n = 1

print("Neural networking ...")

for target in chain_list:
  lr  = 0.1
  for epoch in range(1000):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr *= 0.99
    current_n = N_chains
    for pdb in chain_list:
      if pdb == target:
        continue

      for rl in ["r","l"]:
        for ch in chain_list[pdb][rl]:
          print("%d %s_%s - %s         \r"  %(current_n, pdb,rl,ch),end="")
          current_n -= 1

          local_sequence = all_sequence[pdb+"_"+rl][ch]

          model.update_lstm_hidden()
          model.zero_grad()
          optimizer.zero_grad()

          predicted_bs, init_scores = model( pdb+"_"+rl, local_sequence )
          native_bs = get_native_bs( pdb+"_"+rl, ch )

          loss = loss_function( predicted_bs, native_bs )
          loss.backward()
          optimizer.step()
    
          #np_prediction = predicted_bs.data.cpu()[:,1].numpy()
          #np_class = native_bs.data.cpu().numpy() 
          #np_init = init_scores.numpy()

    ##TESTING FOR EACH EPOCH
    model.train(mode=False)
    for rl in ["r","l"]:
      for ch in chain_list[target][rl]:
        print("%s : %s : %s : %d"%(target,rl,ch,epoch))

        local_sequence = all_sequence[target+"_"+rl][ch]

        model.update_lstm_hidden()
        model.zero_grad()
        optimizer.zero_grad()

        predicted_bs, init_scores = model( target+"_"+rl, local_sequence )
        native_bs = get_native_bs( target+"_"+rl, ch )

        np_class = native_bs.data.cpu().numpy() 
        np_init = init_scores.numpy()
        np_prediction = predicted_bs.data.cpu()[:,1].numpy()

        fpr, tpr, thresholds = metrics.roc_curve(np_class, np_init, pos_label=1)
        init_auc = metrics.auc(fpr, tpr)

        fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
        new_auc = metrics.auc(fpr, tpr)
          
        print("INIT AUC=%0.4f - NEW AUC=%0.4f"%(init_auc, new_auc))
    model.train(mode=True)

  exit()

