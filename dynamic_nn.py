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
for i in pssm_data:
  I = iter(list(map(str.strip,open("PSSM/"+i,"r").readlines())))
  r = i.split("_")
  pdb = r[0]+"_"+r[1]
  ch  = r[2]

  if not pdb in pdb_features:
    pdb_features[pdb] = dict()
  next(I)
  for j in I:
    r = j.split(" ")
    res_id = r[1]
    pdb_features[pdb][res_id+ch] = dict()
    pdb_features[pdb][res_id+ch]['pssm'] = list(map(float,r[3:23]))

for pdb in pdb_features.keys():
  I = iter(list(map(str.strip,open("NEIGHBOURS/"+pdb+"_u.vd","r").readlines())))
  for j in I:
     r = iter(j.split("\t"))
     res_ch =  next(r)
     pdb_features[pdb][res_ch]['vd'] = list(r)

rri = dict()
PDB = list(map(str.strip, open("rri_list.tsv","r").readlines()))
for i in PDB:
  if not i in rri:
    rri[i] = dict()
  J = iter(list(map(str.strip,open("pairPred_contactMap/"+i+".int","r").readlines())))
  for j in J:
    r =  j.split("\t")
    rri[i][r[0]+":"+r[1]]=True

def get_native_rri( pdb,res_i,res_j ):
  if res_i+":"+res_j in rri[pdb]:
    return autograd.Variable(torch.LongTensor([1])).cuda()
  return autograd.Variable(torch.LongTensor([0])).cuda()

def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Linear') != -1:
    nn.init.xavier_normal(m.weight)
    m.bias.data.fill_(0)
    #nn.init.xavier_normal(m.bias)

class DyNet(nn.Module):

    def __init__( self, input_dim=20, direct_pair_dim=512, neighbour_pair_dim=512, neighbour_pair_out_dim=256, out_size=2 ):
        super(DyNet, self).__init__()

        self.input_dim = input_dim
        self.direct_pair_dim = direct_pair_dim
        self.neighbour_pair_dim = neighbour_pair_dim
        self.neighbour_pair_out_dim = neighbour_pair_out_dim
        self.out_size = out_size

        #NN DIRECT PAIR
        self.drop_direct_pair = nn.Dropout(p=0.5)
        self.direct_pair = nn.Linear(2*input_dim+2*neighbour_pair_out_dim, direct_pair_dim)
        self.direct_pair_out = nn.Linear(direct_pair_dim, out_size) 
        
        #NN NEIGHBOURS
        self.drop_neighbour_pair = nn.Dropout(p=0.5)
        self.neighbour_pair = nn.Linear(2*input_dim, neighbour_pair_dim)
        self.neighbour_pair_out = nn.Linear(neighbour_pair_dim, neighbour_pair_out_dim)

    def prepare_data(self, pdb_i, res_i, pdb_j, res_j):
        a_i = torch.unsqueeze( torch.FloatTensor(pdb_features[pdb_i][res_i]['pssm']),dim=0 )
        b_j = torch.unsqueeze( torch.FloatTensor(pdb_features[pdb_j][res_j]['pssm']),dim=0 )

        flag = True 
        for i in pdb_features[pdb_i][res_i]['vd']:

          v = list(pdb_features[pdb_j][res_j]['pssm'])
          v.extend( pdb_features[pdb_i][i]['pssm'] )
          v = torch.unsqueeze(torch.FloatTensor(v),dim=0)
          if not flag:
            A_i = torch.cat( (A_i, v) ,dim=0 )
          else:
            flag = False
            A_i = v 

        flag = True
        for j in pdb_features[pdb_j][res_j]['vd']:

          v = list(pdb_features[pdb_i][res_i]['pssm'])
          v.extend(pdb_features[pdb_j][j]['pssm'])
          v = torch.unsqueeze(torch.FloatTensor(v),dim=0)
          if not flag:
            B_j = torch.cat( (B_j, v),dim=0 )
          else:
            flag = False
            B_j = v

        return autograd.Variable(a_i).cuda(), autograd.Variable(A_i).cuda(), autograd.Variable(b_j).cuda(), autograd.Variable(B_j).cuda()

    def forward(self, pdb_i, res_i, pdb_j, res_j ):

        a_i, A_i, b_j, B_j = self.prepare_data( pdb_i, res_i, pdb_j, res_j )

        N_i = self.neighbour_pair(A_i)
        N_i = self.drop_neighbour_pair(N_i)
        N_i = F.relu(N_i)
        N_i = self.neighbour_pair_out(N_i)
        N_i = F.relu( N_i )
        N_i = torch.mean(N_i,0)

        N_j = self.neighbour_pair(B_j)
        N_j = self.drop_neighbour_pair(N_j)
        N_j = F.relu(N_j)
        N_j = self.neighbour_pair_out(N_j)
        N_j = F.relu( N_j )
        N_j = torch.mean(N_j,0)

        v_in = torch.cat([a_i,b_j,N_i,N_j],dim=1)

        out = self.direct_pair(v_in)
        out = self.drop_direct_pair(out)
        out = F.relu(out)
        out = self.direct_pair_out(out)
        out = F.log_softmax( out )

        return out


input_dim=20
direct_pair_dim=512
neighbour_pair_dim=512
neighbour_pair_out_dim=1
out_size=2

model = DyNet(input_dim=input_dim, direct_pair_dim=direct_pair_dim, neighbour_pair_dim=neighbour_pair_dim, neighbour_pair_out_dim=neighbour_pair_out_dim, out_size=out_size)
model.cuda()

print(model)
loss_function = nn.NLLLoss()

#optimizer = optim.Adam(model.parameters(), lr=0.01)

N = len(training_data)
current_n = 1

print("Neural networking ...")

for target in PDB:
  lr = 0.1
  model = model = DyNet(input_dim=input_dim, direct_pair_dim=direct_pair_dim, neighbour_pair_dim=neighbour_pair_dim, neighbour_pair_out_dim=neighbour_pair_out_dim, out_size=out_size)
  model.cuda()

  for epoch in range(100):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    #lr *= 0.9
    curr_n = len(PDB)
    results = list()
    for pdb in PDB:
      curr_n -=1
      if pdb == target:
        continue
      print("%d:%d - %s                 \r"%(epoch,curr_n,pdb),end="")

      for R in list(rri[pdb].keys()):
        [res_i,res_j] = R.split(":")
    
        model.zero_grad()
        optimizer.zero_grad()

        predicted = model( pdb+"_r", res_i, pdb+"_l", res_j )
        native = get_native_rri( pdb,res_i,res_j )

        loss = loss_function( predicted, native )
        loss.backward()
        optimizer.step()

        results.append( [1,predicted.data.cpu()[0,1]] )

      neg = len( list(rri[pdb].keys()) )
      while(neg>0):
        res_i = random.choice( list(pdb_features[pdb+"_r"].keys()) )
        res_j = random.choice( list(pdb_features[pdb+"_l"].keys()) )

        if res_i+":"+res_j in rri:
          continue
        if not "vd" in pdb_features[pdb+"_r"][res_i]:
          continue
        if not "vd" in pdb_features[pdb+"_l"][res_j]:
          continue

        neg -= 1

        model.zero_grad()
        optimizer.zero_grad()
    
        predicted = model( pdb+"_r", res_i, pdb+"_l", res_j )
        native = get_native_rri( pdb,res_i,res_j )

        loss = loss_function( predicted, native )
        loss.backward()
        optimizer.step()

        results.append( [0,predicted.data.cpu()[0,1]] )

    soreted_res = np.array(sorted(results, key=itemgetter(1),reverse=True))
    fpr, tpr, thresholds = metrics.roc_curve(soreted_res[:,0], soreted_res[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("Training %s:%d  AUC=%0.4f\n"%(target,epoch,auc),end="")

    if epoch % 10 == 0:
      print("Evaluating %s:%d\n"%(target,epoch),end="")
      model.train(mode=False)
      results = list()

      for res_i in list(pdb_features[target+"_r"].keys()):
        for res_j in list(pdb_features[target+"_l"].keys()):

          if res_i+":"+res_j in rri:
            continue
          if not "vd" in pdb_features[target+"_r"][res_i]:
            continue
          if not "vd" in pdb_features[target+"_l"][res_j]:
            continue

          predicted = model( target+"_r", res_i, target+"_l", res_j )
          results.append( [0,predicted.data.cpu()[0,1]] )
      
      for R in list(rri[target].keys()):
      
        [res_i,res_j] = R.split(":")
      
        predicted = model( target+"_r", res_i, target+"_l", res_j )
        results.append( [1,predicted.data.cpu()[0,1]] )

      soreted_res = np.array(sorted(results, key=itemgetter(1),reverse=True))

      fpr, tpr, thresholds = metrics.roc_curve(soreted_res[:,0], soreted_res[:,1], pos_label=1)
      auc = metrics.auc(fpr, tpr)

      p_10 = np.sum(soreted_res[0:10,0])/10
      p_100 = np.sum(soreted_res[0:100,0])/100
      p_500 = np.sum(soreted_res[0:500,0])/500

      print( "%s - AUC=%0.4f - P10=%0.4f - P100=%0.4f - P500=%0.4f" % (target, auc, p_10, p_100, p_500) )
    
      model.train(mode=True)



