import sys

import json
import os.path
import subprocess
import random

from os import listdir
import os.path

import matplotlib.pyplot as plt

resultls = sys.argv[1]
resultls_2 = sys.argv[2]
n_epoch = sys.argv[3] 

fN = int(n_epoch)

for i in range(1,fN):
  J = list(map(str.strip,subprocess.check_output(['ls '+resultls+'/ | grep tsv'],shell=True ).decode('utf-8').split("\n")))
  JJ = list(map(str.strip,subprocess.check_output(['ls '+resultls_2+'/ | grep tsv'],shell=True ).decode('utf-8').split("\n")))

  C = [value for value in J if value in JJ and len(value)>1] 

  auc_1 = 0
  buc_1 = 0
  n_auc_1 = 0

  auc_2 = 0
  buc_2 = 0
  n_auc_2 = 0

  for j in C:
    K1 = list( map(str.strip,open(resultls+'/'+j,"r").readlines()) )
    K2 = list( map(str.strip,open(resultls_2+'/'+j,"r").readlines()) )


    if len(K1)<fN or len(K2)<fN:
      continue

    auc_1 += float(K1[i].split("AUC=")[1].split(" -")[0])
    buc_1 += float(K1[i].split("TRAINING_AUC=")[1])
    n_auc_1 += 1

    auc_2 += float(K2[i].split("AUC=")[1].split(" -")[0])
    buc_2 += float(K2[i].split("TRAINING_AUC=")[1])
    n_auc_2 += 1

  if n_auc_1 == 0 or n_auc_2 == 0:
    continue

  print("%d\t%0.4f %0.4f\t%0.4f %0.4f"%(i,(auc_1/n_auc_1),(auc_2/n_auc_2),(buc_1/n_auc_1),(buc_2/n_auc_2)))

