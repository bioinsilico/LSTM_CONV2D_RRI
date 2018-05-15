import sys

import numpy as np
import json
import os.path
import subprocess
import random
from operator import itemgetter
import sklearn.metrics as metrics 

from os import listdir
import os.path

for i in range(1,200):
  J = iter(list(map(str.strip,subprocess.check_output(['ls results/evaluations/*.'+str(i)+'.tsv'],shell=True ).decode('utf-8').split("\n"))))

  j = next(J)
  M = np.genfromtxt(j)[:,0]

  j = next(J)
  M = np.stack([M,np.genfromtxt(j)[:,0]],axis=1)

  for j in J:
    if os.path.isfile(j):
      X = np.genfromtxt(j)[:,0]
      X = X.reshape(X.size,1)
      M = np.concatenate((M,X),axis=1)
  
  avg = M.sum(1)/M.shape[1]
  print("%d - %f0.4"%(i,avg.max()))
    





