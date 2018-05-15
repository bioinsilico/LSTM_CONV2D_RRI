import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import os.path
import subprocess

np.set_printoptions(linewidth=1000000000)
torch.cuda.manual_seed(1)

training_data = []
testing_data = []

I = open("psipred_data/train1.lst","r").readlines()
training_data = list(map(str.strip, I))

I = open("psipred_data/test1.lst","r").readlines()
testing_data = list(map(str.strip, I))

ss_values = {"G":"H", "H":"H", "I":"H", "B":"E", "E":"E", "S":"C", "T":"C", "P":"C", "A":"C", "C":"C"}
ss_to_ix = {"C": 0, "E": 1, "H": 2}

def get_data(pdb):
  sequence = []
  A = subprocess.check_output(['grep -v "#" psipred_data/'+pdb+'.tdb | cut -c6-9,184-'],shell=True ).decode('utf-8').split("\n")
  for i in A:
    if len(i)==0:
      continue
    v = i[0:4].split(" ")
    aa = v[0]
    if aa == "-":
      aa = "X"
    ss = "C"
    if v[1]:
      ss=ss_values[v[1]]
    pssm = list(map(float, i[4:].split("  ")))
    if len(pssm)!=20:
      exit(pdb+"!!!!")
    sequence.append({'aa':aa,'ss':ss,'pssm':pssm})
  return sequence

def get_native_ss(data,tag_to_ix):
  out = []
  for i in data:
    out.append(tag_to_ix[ i['ss'] ])
  return autograd.Variable(torch.LongTensor(out)).cuda()

class BiLSTM(nn.Module):

    def __init__( self, input_dim=20, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, ss_size=3 ):
        super(BiLSTM, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.hidden_1_dim = hidden_1_dim
        self.hidden_2_dim = hidden_2_dim
        self.ss_size = ss_size

        self.lstm_h0 = None
        self.lstm_c0 = None
        self.update_lstm_hidden()

        self.LSTM = nn.LSTM(input_dim, lstm_hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)

        self.drop_hidden_1 = nn.Dropout(p=0.5)
        self.lstm2hidden_1 = nn.Linear(2*lstm_hidden_dim, hidden_1_dim)
        
        self.drop_hidden_2 = nn.Dropout(p=0.5)
        self.hidden2hidden_2 = nn.Linear(hidden_1_dim, hidden_2_dim)

        self.hidden2out = nn.Linear(hidden_2_dim, ss_size)

    def update_lstm_hidden(self):
        self.lstm_h0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        self.lstm_c0 = autograd.Variable(torch.zeros(4, 1, self.lstm_hidden_dim)).cuda()
        

    def prepare_data(self, sequence):
        list_pssm = []
        for aa in sequence:
          list_pssm.append( list(aa["pssm"]) )

        return autograd.Variable( torch.unsqueeze(torch.FloatTensor(list_pssm),dim=1) ).cuda()

    def forward(self, sequence ):

        v_in = self.prepare_data( sequence )

        out_LSTM, (hidden_LSTM, content_LSTM) = self.LSTM( v_in, (self.lstm_h0, self.lstm_c0))

        hidden_1 = self.lstm2hidden_1( out_LSTM.view(len(sequence), -1) )
        hidden_1 = self.drop_hidden_1(hidden_1)
        out_hidden_1 = F.relu(hidden_1)

        hidden_2 = self.hidden2hidden_2( out_hidden_1 )
        hidden_2 = self.drop_hidden_2(hidden_2)
        out_hidden_2 = F.relu(hidden_2)

        ss_out = self.hidden2out( out_hidden_2 )
        ss_out = F.log_softmax( ss_out )

        return ss_out


print("Collecting data ...")
all_sequences = {}
set_aa = {}
for pdb in training_data:
  data = get_data(pdb)
  all_sequences[pdb] = {}
  all_sequences[pdb]['data'] = data

for pdb in testing_data:
  data = get_data(pdb)
  all_sequences[pdb] = {}
  all_sequences[pdb]['data'] = data

model = BiLSTM(input_dim=20, lstm_hidden_dim=250, hidden_1_dim=1024, hidden_2_dim=512, ss_size=3)
model.cuda()
print(model)
loss_function = nn.NLLLoss()

#optimizer = optim.Adam(model.parameters(), lr=0.01)

N = len(training_data)
current_n = 1

lr = 0.1
print("Neural networking ...")
for epoch in range(20):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    lr *= 0.9
    for pdb in training_data:
        sequence = all_sequences[pdb]['data']

        model.update_lstm_hidden()
        model.zero_grad()
        optimizer.zero_grad()

        predicted_ss = model( sequence )
        native_ss = get_native_ss( sequence, ss_to_ix )

        loss = loss_function( predicted_ss, native_ss )
        loss.backward()
        optimizer.step()

        current_n += 1
        frac = 100*current_n/N
        print("%0.2f%%\r" %(frac), end="")
        if current_n == N:
          model.train(mode=False)
          print("")
          print("Testing ...")
          Q3 = []
          for pdb in testing_data:
            sequence = all_sequences[pdb]['data']
            predicted_ss = model( sequence )
            all_sequences[pdb]['predcitions'] = predicted_ss
            
            native_ss = get_native_ss( sequence, ss_to_ix )
         
            value, index = torch.max(predicted_ss,1)
            eq = torch.eq(native_ss.data.cpu(), index.data.cpu())
            q3 = torch.sum(eq)/eq.size()[0]
            Q3.append(q3)

            current_n = 1
          Q3 = torch.Tensor(Q3)
          print( torch.mean(Q3) )
          model.train(mode=True)

