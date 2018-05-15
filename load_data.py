import subprocess
import numpy as np
from aa_to_ch import *
import random


r_features = dict()
l_features = dict()

def create_data(r,l):
  R = [r]
  L = [l]
  dR = dict()
  dL = dict()

  dR[r] = 0
  for i in range( 0,len(r_features[r]['nn']) ):
    dR[ r_features[r]['nn'][i] ] = r_features[r]['nn_dist'][i] 

  dL[l] = 0
  for i in range( 0,len(l_features[l]['nn']) ):
    dL[ l_features[l]['nn'][i] ] = l_features[l]['nn_dist'][i] 

  R.extend( r_features[r]['nn'] )
  L.extend( l_features[l]['nn'] )
  
  matrix = []
  for i in R:
    row = []
    for j in L:
      data = []
      data.extend( r_features[ i ][ 'aa' ]  )
      data.extend( r_features[ i ][ 'features' ]  )
      data.append( dR[i]  )
      data.extend( l_features[ j ][ 'aa' ]  )
      data.extend( l_features[ j ][ 'features' ]  )
      data.append( dL[j]  )
      row.append(data)
    matrix.append(row)
  return matrix

def  collect_complex(r_features_file,l_features_file,rri_file,pdb,n_features=6,n_neigh=8):
  fh = open(r_features_file,'r')
  for i in fh:
    r = i.strip()
    R=r.split("\t")
    r_features[ R[0] ] = dict()
    r_features[ R[0] ][ 'aa_' ] = R[1] 
    r_features[ R[0] ][ 'aa' ] = [ float(x) for x in AA[ R[1] ]]
    r_features[ R[0] ][ 'features' ] = [ float(x) for x in R[2:(n_features+2)] ]
    r_features[ R[0] ][ 'nn' ] = R[(n_features+2):(n_features+n_neigh+2)]
    r_features[ R[0] ][ 'nn_dist' ] = [ float(x) for x in R[(n_features+n_neigh+2):(n_features+n_neigh+2+n_neigh*2)] ]
  fh.close()
  
  fh = open(l_features_file,'r')
  for i in fh:
    r = i.strip()
    R=r.split("\t")
    l_features[ R[0] ] = dict()
    l_features[ R[0] ][ 'aa_' ] = R[1]
    l_features[ R[0] ][ 'aa' ] = [ float(x) for x in AA[ R[1] ]]
    l_features[ R[0] ][ 'features' ] = [ float(x) for x in R[2:(n_features+2)]]
    l_features[ R[0] ][ 'nn' ] = R[(n_features+2):(n_features+n_neigh+2)]
    l_features[ R[0] ][ 'nn_dist' ] = [ float(x) for x in R[(n_features+n_neigh+2):(n_features+n_neigh+2+n_neigh*2)] ]
  fh.close()
  
  rri = dict()
  fh = open(rri_file,'r')
  for i in fh:
    r = i.strip()
    R=r.split("\t")
    rri[ R[0]+":"+R[1] ] = 1
  fh.close()

  np = 0
  collection  = []
  labels  = []
  negatives = []

  #print("\tbatching positives")
  for rr in rri:
    R = rr.split(":")
    r = R[0]
    l = R[1]
    np += 1
    __data  = create_data(r,l)
    collection.append( create_data(r,l) )
    labels.append([1.0,0.0])

  #print("\tbatching negatives")
  R = list(r_features.keys())
  L = list(l_features.keys())
  while np>0:
    r = random.choice(R)
    l = random.choice(L)
    if not r+":"+l in rri:
      __data = create_data(r,l)
      collection.append( __data )
      labels.append([0.0,1.0])
      np-=1
  return [collection, labels]
  

def __batch(pdb,n_features=6,n_neigh=8):
  rri_file = "/home/joan/tools/RRI/DEEP_LEARNING/pairPred_contactMap/"+pdb+".int"
  r_features_file = "/home/joan/tools/RRI/DEEP_LEARNING/features/"+pdb+"_r_u.nn.tsv"
  l_features_file = "/home/joan/tools/RRI/DEEP_LEARNING/features/"+pdb+"_l_u.nn.tsv"
  return collect_complex(r_features_file,l_features_file,rri_file,pdb,n_features=6,n_neigh=8)

def random_batch_excluding(out):
  batch_x = []
  labels_y = []
  fh=open('/home/joan/tools/RRI/DEEP_LEARNING/pdb_list.tsv','r')
  for i in fh:
    pdb = i.strip()
    if i != out:
      #print(pdb)
      [batch,labels] = __batch(pdb)
      batch_x.extend(batch)
      labels_y.extend(labels_y)
  fh.close()
  return [np.array(batch_x),np.array(labels_y)]

def random_batch_of(pdb):
  batch_x = []
  labels_y = []
  [batch,labels] = __batch(pdb)
  batch_x.extend(batch)
  labels_y.extend(labels_y)
  return [np.array(batch_x),np.array(labels_y)]

