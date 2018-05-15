import sys

import numpy as np
import json
import os.path
import subprocess
import random
from operator import itemgetter
import sklearn.metrics as metrics 

from os import listdir

file_list = listdir('results/predictions/')

for f in file_list:
  print(f)
  V = f.split(".")
  W = V[1].split(":")
  pdb = V[0]
  ch_i = W[0]
  ch_j = W[0]
  n_iter = V[2]

  data = np.genfromtxt('results/predictions/'+f)
  
  N_i = int(data[0,0])
  N_j = int(data[0,1])
  
  I = data[1:N_i+1]
  
  J = data[N_i+2:N_i+2+N_j+1]
  
  file_name ="results/evaluations/"+pdb+".r."+ch_i+"."+n_iter+".tsv" 
  fh = open(file_name, "w")

  s = -37
  while s<=0:
    y_pred = (1.0*(I[:,0]>=s))
    y_true = I[:,1]
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    #print("%0.4f\t%0.4f\t%0.4f\t\t(%0.4f)"%(mcc,pre,rec,s))
    fh.write("%0.4f\t%0.4f\t%0.4f\t\t(%0.4f)\n"%(mcc,pre,rec,s))
    s += 0.05
  fh.close()

  file_name ="results/evaluations/"+pdb+".l."+ch_j+"."+n_iter+".tsv" 
  fh = open(file_name, "w")

  s = -37
  while s<=0:
    y_pred = (1.0*(J[:,0]>=s))
    y_true = J[:,1]
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    #print("%0.4f\t%0.4f\t%0.4f\t\t(%0.4f)"%(mcc,pre,rec,s))
    fh.write("%0.4f\t%0.4f\t%0.4f\t\t(%0.4f)\n"%(mcc,pre,rec,s))
    s += 0.05
  fh.close()

