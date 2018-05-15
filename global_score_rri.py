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

for i in range(1,200):
  J = iter(list(map(str.strip,subprocess.check_output(['ls results/predictions/*.'+str(i)+'.tsv'],shell=True ).decode('utf-8').split("\n"))))

  j = next(J)
  M = np.genfromtxt(j)[1:,:]
  for j in J:
    if os.path.isfile(j):
      X = np.genfromtxt(j)[1:,:]
      M = np.concatenate((M,X),axis=0)
  
  y_true = M[:,1].astype(int)
  y_pred = M[:,0]
  #precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
  #thr = np.concatenate((thresholds,[0]))
  #plt.plot(recall,precision)
  ##plt.plot(thr,precision)
  #plt.title('Iter '+str(i))
  #plt.show()

  #for s in thresholds:
  #  if s>-5:
  MCC = []
  PRE = []
  REC = []
  S = []
  s = -2
  while s<=0:
    y_pred = (1.0*(M[:,0]>=s))
    y_true = M[:,1]
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    MCC.append(mcc)
    PRE.append(pre)
    REC.append(rec)
    S.append(s)
    #print("%f0.4 %f0.4"  %(s,mcc))
    s += 0.05
  argmax_mcc = np.argmax(MCC)
  if np.isscalar(argmax_mcc):
    k = argmax_mcc
    print( "%d %0.4f %0.4f %0.4f %0.4f" % (i,S[k], MCC[k], PRE[k], REC[k]) )
  else:
    for k in argmax_mcc:
      print( "%d %0.4f %0.4f %0.4f %0.4f" % (i,S[k], MCC[k], PRE[k], REC[k]) )
