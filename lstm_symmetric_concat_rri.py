import sys
list_file_name = sys.argv[1]
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

I = open("blank_list.tsv","r").readlines()
pssm_data = list(map(str.strip, I))

AA = {
  'A' :[-1.56 ,-1.67 ,-0.97 ,-0.27 ,-0.93 ,-0.78 ,-0.20 ,-0.08 ,0.21 ,-0.48 ],
  'R' :[0.22 ,1.27 ,1.37 ,1.87 ,-1.70 ,0.46 ,0.92 ,-0.39 ,0.23 ,0.93 ],
  'N' :[1.14 ,-0.07 ,-0.12 ,0.81 ,0.18 ,0.37 ,-0.09 ,1.23 ,1.10 ,-1.73 ],
  'D' :[0.58 ,-0.22 ,-1.58 ,0.81 ,-0.92 ,0.15 ,-1.52 ,0.47 ,0.76 ,0.70 ],
  'C' :[0.12 ,-0.89 ,0.45 ,-1.05 ,-0.71 ,2.41 ,1.52 ,-0.69 ,1.13 ,1.10 ],
  'Q' :[-0.47 ,0.24 ,0.07 ,1.10 ,1.10 ,0.59 ,0.84 ,-0.71 ,-0.03 ,-2.33 ],
  'E' :[-1.45 ,0.19 ,-1.61 ,1.17 ,-1.31 ,0.40 ,0.04 ,0.38 ,-0.35 ,-0.12 ],
  'G' :[1.46 ,-1.96 ,-0.23 ,-0.16 ,0.10 ,-0.11 ,1.32 ,2.36 ,-1.66 ,0.46 ],
  'H' :[-0.41 ,0.52 ,-0.28 ,0.28 ,1.61 ,1.01 ,-1.85 ,0.47 ,1.13 ,1.63 ],
  'I' :[-0.73 ,-0.16 ,1.79 ,-0.77 ,-0.54 ,0.03 ,-0.83 ,0.51 ,0.66 ,-1.78 ],
  'L' :[-1.04 ,0.00 ,-0.24 ,-1.10 ,-0.55 ,-2.05 ,0.96 ,-0.76 ,0.45 ,0.93 ],
  'K' :[-0.34 ,0.82 ,-0.23 ,1.70 ,1.54 ,-1.62 ,1.15 ,-0.08 ,-0.48 ,0.60 ],
  'M' :[-1.40 ,0.18 ,-0.42 ,-0.73 ,2.00 ,1.52 ,0.26 ,0.11 ,-1.27 ,0.27 ],
  'F' :[-0.21 ,0.98 ,-0.36 ,-1.43 ,0.22 ,-0.81 ,0.67 ,1.10 ,1.71 ,-0.44 ],
  'P' :[2.06 ,-0.33 ,-1.15 ,-0.75 ,0.88 ,-0.45 ,0.30 ,-2.30 ,0.74 ,-0.28 ],
  'S' :[0.81 ,-1.08 ,0.16 ,0.42 ,-0.21 ,-0.43 ,-1.89 ,-1.15 ,-0.97 ,-0.23 ],
  'T' :[0.26 ,-0.70 ,1.21 ,0.63 ,-0.10 ,0.21 ,0.24 ,-1.15 ,-0.56 ,0.19 ],
  'W' :[0.30 ,2.10 ,-0.72 ,-1.57 ,-1.16 ,0.57 ,-0.48 ,-0.40 ,-2.30 ,-0.60 ],
  'Y' :[1.38 ,1.48 ,0.80 ,-0.56 ,-0.00 ,-0.68 ,-0.31 ,1.03 ,-0.05 ,0.53 ],
  'V' :[-0.74 ,-0.71 ,2.04 ,-0.40 ,0.50 ,-0.81 ,-1.07 ,0.06 ,-0.46 ,0.65 ]
}

pdb_features = dict()
all_sequence = dict()
aa_to_ix = {}
for i in pssm_data:
  I = iter(list(map(str.strip,open("PSSM/"+i+".pssm","r").readlines())))
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
    aa = AA[r[2]]
    if not r[2] in aa_to_ix:
      aa_to_ix[r[2]] = len(aa_to_ix)
    pdb_features[pdb][res_id+ch] = dict()
    if not ch in all_sequence[pdb]:
      all_sequence[pdb][ch] = list()
    all_sequence[pdb][ch].append(res_id+ch)
    pdb_features[pdb][res_id+ch]['pssm'] = list(map(float,r[3:23]))
    pdb_features[pdb][res_id+ch]['aa'] = aa
    pdb_features[pdb][res_id+ch]['type'] = r[2]

  I = iter(list(map(str.strip,open("SPIDER2/"+i+".spider2","r").readlines())))
  next(I);next(I)

  for j in I:
    r = j.split(" ")
    res_id = r[1]
    pdb_features[pdb][res_id+ch]['spider'] = list(map(float,[ r[i] for i in [4,9,10,11]]))
  
cci = dict()
N_cci = 0

I = open("rri_list.tsv","r").readlines()
pdb_list = list(map(str.strip, I))

for i in pdb_list:
  if not i in cci:
    cci[i] = dict()
  J = iter(list(map(str.strip,open("pairPred_contactMap/"+i+".int","r").readlines())))
  for j in J:
    r =  j.split("\t")
    ch_r = r[0][-1]
    ch_l = r[1][-1]
    if not ch_r+":"+ch_l in cci[i]:
      N_cci += 1
      cci[i][ch_r+":"+ch_l] = True

pdb_bs = dict()

for i in pdb_list:

  pdb_bs[ i+"_l" ] = dict()
  I = iter(list(map(str.strip,open("bestResults/struct_2/"+i+".res.tab.lig","r").readlines())))
  next(I)
  next(I)
  for j in I:
    R = j.split(" ")
    if int(R[2]) > 0:
      pdb_bs[ i+"_l" ][ R[1]+R[0] ]= True
    
  pdb_bs[ i+"_r" ] = dict()
  I = iter(list(map(str.strip,open("bestResults/struct_2/"+i+".res.tab.rec","r").readlines())))
  next(I)
  next(I)
  for j in I:
    R = j.split(" ")
    if int(R[2]) > 0:
      pdb_bs[ i+"_r" ][ R[1]+R[0] ]= True

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

        #self.aa_embedding = nn.Embedding(20, 10)

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
        #embed = []
        for aa in sequence:
          v = list(pdb_features[pdb][aa]["pssm"])
          v.extend( list(pdb_features[pdb][aa]["aa"]) )
          v.extend( list(pdb_features[pdb][aa]["spider"]) )
          list_pssm.append(v)
          #embed.append( aa_to_ix[pdb_features[pdb][aa]["type"]] )
        return autograd.Variable( torch.unsqueeze(torch.FloatTensor(list_pssm),dim=1) ).cuda(), 1#autograd.Variable( torch.LongTensor(embed) ).cuda() 

    def forward(self, pdb, sequence_r, sequence_l ):

        N_r = len(sequence_r) 
        N_l = len(sequence_l)

        v_r, embed_r = self.prepare_data( pdb+"_r", sequence_r )
        v_l, embed_l = self.prepare_data( pdb+"_l", sequence_l )
        zeros = autograd.Variable( torch.zeros(1,1,self.input_dim) ).cuda()

        #embed_r = self.aa_embedding(embed_r)
        #embed_r  = torch.unsqueeze(embed_r,dim=1)
        #v_r = torch.cat([v_r,embed_r],dim=2)

        #embed_l = self.aa_embedding(embed_l)
        #embed_l  = torch.unsqueeze(embed_l,dim=1)
        #v_l = torch.cat([v_l,embed_l],dim=2)      
        
        #RL
        v_in =  torch.cat([v_r,zeros,v_l],dim=0)

        self.update_lstm_hidden()
        out_LSTM, (hidden_LSTM, content_LSTM) = self.LSTM( v_in, (self.lstm_h0, self.lstm_c0))

        hidden_1 = self.lstm2hidden_1( out_LSTM.view(len(sequence_r)+len(sequence_l)+1, -1) )
        hidden_1 = self.drop_hidden_1(hidden_1)
        out_hidden_1 = F.relu(hidden_1)

        hidden_2 = self.hidden2hidden_2( out_hidden_1 )
        hidden_2 = self.drop_hidden_2(hidden_2)
        out_hidden_2 = F.relu(hidden_2)

        #hidden_r = out_hidden_2
        bs_out_r = self.hidden2out( out_hidden_2 )

        #LR
        v_in =  torch.cat([v_l,zeros,v_r],dim=0)

        self.update_lstm_hidden()
        out_LSTM, (hidden_LSTM, content_LSTM) = self.LSTM( v_in, (self.lstm_h0, self.lstm_c0))

        hidden_1 = self.lstm2hidden_1( out_LSTM.view(len(sequence_r)+len(sequence_l)+1, -1) )
        hidden_1 = self.drop_hidden_1(hidden_1)
        out_hidden_1 = F.relu(hidden_1)

        hidden_2 = self.hidden2hidden_2( out_hidden_1 )
        hidden_2 = self.drop_hidden_2(hidden_2)
        out_hidden_2 = F.relu(hidden_2)

        #hidden_l = out_hidden_2
        #hidden_l = torch.cat([ hidden_l[N_l+1:], torch.unsqueeze(hidden_l[N_l],dim=0), hidden_l[0:N_l]], dim=0)
        bs_out_l = self.hidden2out( out_hidden_2 )
        bs_out_l = torch.cat([ bs_out_l[N_l+1:], torch.unsqueeze(bs_out_l[N_l],dim=0), bs_out_l[0:N_l]], dim=0)

        #bs_out = self.hidden2out( 0.5*(hidden_r+hidden_l) )
        bs_out = 0.5*(bs_out_r+bs_out_l)

        bs_out = F.log_softmax( bs_out )

        return bs_out

def get_native_rri( pdb, ch_r, ch_l ):

  BS = []
  for aa in all_sequence[pdb+"_r"][ch_r]:
    if aa in pdb_bs[pdb+"_r"]:
      BS.append(1)
    else:
      BS.append(0)
  BS.append(0)
  for aa in all_sequence[pdb+"_l"][ch_l]:
    if aa in pdb_bs[pdb+"_l"]:
      BS.append(1)
    else:
      BS.append(0)
  return autograd.Variable(torch.LongTensor(BS)).cuda()


model = BiLSTM( input_dim=34, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, bs_size=2 )
model.cuda()
print(model)
loss_function = nn.NLLLoss()
#optimizer = optim.Adam(model.parameters(), lr=0.01)

N = len(training_data)

print("Neural networking ...")

partial_list = list(map(str.strip, open(list_file_name,"r").readlines()))

for target in partial_list:
  model = BiLSTM( input_dim=34, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, bs_size=2 )
  model.cuda()
  lr  = 0.1
  for epoch in range(200):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr *= 0.99
    N_current = N_cci
    cci_ = list(cci.keys())
    random.shuffle(cci_)
    for pdb in cci_:
      if pdb == target:
        continue
      for ch_ch in cci[pdb]:
        R = ch_ch.split(":")
        ch_r = R[0]
        ch_l = R[1]

        print("%d - %s %s:%s         \r"  %(N_current, pdb,ch_r,ch_l),end="")
        N_current -= 1

        local_sequence_r = all_sequence[pdb+"_r"][ch_r]
        local_sequence_l = all_sequence[pdb+"_l"][ch_l]

        model.zero_grad()
        optimizer.zero_grad()

        predicted_rri = model( pdb, local_sequence_r, local_sequence_l )
        native_rri =  get_native_rri( pdb, ch_r, ch_l )
        loss = loss_function( predicted_rri, native_rri )
        loss.backward()
        optimizer.step()
    
    model.train(mode=False)
    ##TRAINING AUC SCORE FOR EACH EPOCH
    AUC = []
    for pdb in cci_:
      if pdb == target:
        continue
      for ch_ch in cci[pdb]:
        R = ch_ch.split(":")
        ch_r = R[0]
        ch_l = R[1]

        local_sequence_r = all_sequence[pdb+"_r"][ch_r]
        local_sequence_l = all_sequence[pdb+"_l"][ch_l]

        predicted_rri = model( pdb, local_sequence_r, local_sequence_l )
        native_rri =  get_native_rri( pdb, ch_r, ch_l )   
        np_class = native_rri.data.cpu().numpy() 
        np_prediction = predicted_rri.data.cpu()[:,1].numpy()
  
        fpr, tpr, thresholds = metrics.roc_curve(np_class, np_prediction, pos_label=1)
        new_auc = metrics.auc(fpr, tpr)
        AUC.append(new_auc)
          
    training_auc =  np.mean(AUC)

    ##TESTING FOR EACH EPOCH

    for ch_ch in cci[target]:
      R = ch_ch.split(":")
      ch_r = R[0]
      ch_l = R[1]


      local_sequence_r = all_sequence[target+"_r"][ch_r]
      local_sequence_l = all_sequence[target+"_l"][ch_l]

      N_r = len(local_sequence_r)
      N_l = len(local_sequence_l)

      model.zero_grad()
      optimizer.zero_grad()

      predicted_rri = model( target, local_sequence_r, local_sequence_l )
      native_rri =  get_native_rri( target, ch_r, ch_l )

      np_class = native_rri.data.cpu().numpy() 
      np_prediction = predicted_rri.data.cpu()[:,1].numpy()
      np_all = np.stack((np_prediction,np_class),  axis=-1)
      np_all = np.insert(np_all, 0, np.array((N_r,N_l)), 0) 
      np.savetxt("results/predictions/"+target+"."+ch_r+":"+ch_l+"."+str(epoch)+".tsv",np_all)

      fpr, tpr, thresholds = metrics.roc_curve(np_class[0:N_r], np_prediction[0:N_r], pos_label=1)
      auc_r = metrics.auc(fpr, tpr)

      fpr, tpr, thresholds = metrics.roc_curve(np_class[N_r+1:], np_prediction[N_r+1:], pos_label=1)
      auc_l = metrics.auc(fpr, tpr)
        
      print( "%d - %s %s:%s - AUC=%0.4f:%0.4f - TRAINING_AUC=%0.4f" % (epoch, target,ch_r,ch_l,auc_r,auc_l,training_auc) )
      open("results/"+target+".tsv", "a").write("%d - %s %s:%s - AUC=%0.4f:%0.4f - TRAINING_AUC=%0.4f\n" % (epoch, target,ch_r,ch_l,auc_r,auc_l,training_auc) )
    print("")
    model.train(mode=True)

