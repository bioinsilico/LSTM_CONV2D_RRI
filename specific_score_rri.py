import sys

import numpy as np
import json
import os.path
import subprocess
import random
from operator import itemgetter
import sklearn.metrics as metrics 
import matplotlib.pyplot as plt

from os import listdir
import os.path

for i in range(50,200):
  thr = -1
  
  F = list(map(str.strip,subprocess.check_output(['ls results/predictions/*.'+str(i)+'.tsv'],shell=True ).decode('utf-8').split("\n")))
  del F[-1]
  
  M = []
  PRE = []
  REC = []
  
  chains = {}
  
  for f in F:
    pdb = f.split("/")[2].split(".")[0]
  
    if not pdb in chains:
      #chains[pdb] = { 'r':{'M':[], 'PRE':[], 'REC':[]}, 'l':{'M':[], 'PRE':[], 'REC':[]} }
      chains[pdb] = { 'r':{'y_pred':[], 'y_true':[] }, 'l':{'y_pred':[], 'y_true':[]} }
  
    if os.path.isfile(f):
      data = np.genfromtxt(f)
  
      N_i = int(data[0,0])
      N_j = int(data[0,1])
      
      I = data[1:N_i+1]
      J = data[N_i+2:N_i+2+N_j+1]
  
      I_pred = (1.0*(I[:,0]>=thr))
      chains[pdb]['r']['y_pred'].append(I_pred)
      I_true = I[:,1]
      chains[pdb]['r']['y_true'].append(I_true)
  
      J_pred = (1.0*(J[:,0]>=thr))
      chains[pdb]['l']['y_pred'].append(J_pred)
      J_true = J[:,1]
      chains[pdb]['l']['y_true'].append(J_true)
  
      I_mcc = metrics.matthews_corrcoef(I_true, I_pred)
      I_pre = metrics.precision_score(I_true, I_pred)
      I_rec = metrics.recall_score(I_true, I_pred)
  
      J_mcc = metrics.matthews_corrcoef(J_true, J_pred)
      J_pre = metrics.precision_score(J_true, J_pred)
      J_rec = metrics.recall_score(J_true, J_pred)
      
      M.append(np.sum(I_pred))
      M.append(np.sum(J_pred))
      
  
      if np.sum(I_pred) >= 0:
        PRE.append(I_pre)
        REC.append(I_rec)
  
      if np.sum(J_pred) >= 0:
        PRE.append(J_pre)
        REC.append(J_rec)
  
  M= np.array(M)
  
  print("Iter %d" % i)
  print( "\tF_CH %f0.4" % (np.sum(  (1.0*(M>0)) )/M.size) )
  print( "\tREC %f0.4"% (np.mean(PRE)) )
  print( "\tPRE %f0.f"% (np.mean(REC)) )

sys.exit(0)

M = []
PRE = []
REC = []

for pdb in chains:
  I_pred = np.concatenate( chains[pdb]['r']['y_pred'], axis=0 )
  I_true = np.concatenate( chains[pdb]['r']['y_true'], axis=0 )

  J_pred = np.concatenate( chains[pdb]['l']['y_pred'], axis=0 )
  J_true = np.concatenate( chains[pdb]['l']['y_true'], axis=0 )

  I_mcc = metrics.matthews_corrcoef(I_true, I_pred)
  I_pre = metrics.precision_score(I_true, I_pred)
  I_rec = metrics.recall_score(I_true, I_pred)

  J_mcc = metrics.matthews_corrcoef(J_true, J_pred)
  J_pre = metrics.precision_score(J_true, J_pred)
  J_rec = metrics.recall_score(J_true, J_pred)
  
  M.append(np.sum(I_pred))
  M.append(np.sum(J_pred))
  

  if np.sum(I_pred) > 0:
    PRE.append(I_pre)
    REC.append(I_rec)

  if np.sum(J_pred) > 0:
    PRE.append(J_pre)
    REC.append(J_rec)

M= np.array(M)

#print( np.sum(  (1.0*(M>0)) )/M.size )
#print( np.mean(PRE) )
#print( np.mean(REC) )
