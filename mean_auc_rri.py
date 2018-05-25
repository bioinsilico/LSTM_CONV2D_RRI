import sys

import json
import os.path
import subprocess
import random

from os import listdir
import os.path

import matplotlib.pyplot as plt

resultls = sys.argv[1]
n_epoch = sys.argv[2] 

fN = int(n_epoch)
A=[]
for i in range(1,fN):
  J = iter(list(map(str.strip,subprocess.check_output(['ls '+resultls+'/*.tsv'],shell=True ).decode('utf-8').split("\n"))))

  auc = 0
  buc = 0
  n_auc = 0

  for j in J:
    if not os.path.isfile(j):
      continue
    K = list( map(str.strip,open(j,"r").readlines()) )
    if len(K)<fN:
      continue
    auc += float(K[i].split("AUC=")[1].split(" -")[0])
    buc += float(K[i].split("TRAINING_AUC=")[1])
    n_auc += 1
  try:
    A.append( (auc/n_auc,buc/n_auc) )
  except:
    print("ERROR")
    print(j)
    sys.exit(1)
  print("%d %0.4f %0.4f"%(i,(auc/n_auc),(buc/n_auc)))

X,Y = zip(*A)
plt.plot( range(fN-1),X,'ro' )
plt.plot( range(fN-1),Y ,'bo')
plt.show()
