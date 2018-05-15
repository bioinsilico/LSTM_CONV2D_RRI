import random


kd = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'X':0, 'Z': 0}


def read_seq_pssm(pssm_file,path,hydro=False):
  pdb_features = dict()
  all_sequence = dict()
  
  I = open(pssm_file,"r").readlines()
  pssm_data = list(map(str.strip, I))
  for i in pssm_data:
    I = iter(list(map(str.strip,open(path+"/"+i,"r").readlines())))
    r = i.split("_")
    pdb = r[0]
    ch  = r[1]
  
    if not pdb in pdb_features:
      pdb_features[pdb] = dict()
    if not pdb in all_sequence:
      all_sequence[pdb] = dict()
    next(I)
    for j in I:
      r = j.split(" ")
      res_id = r[2]
      aa_type = kd[r[3]]
      pdb_features[pdb][res_id+ch] = dict()
      if not ch in all_sequence[pdb]:
        all_sequence[pdb][ch] = list()
      all_sequence[pdb][ch].append(res_id+ch)
      pdb_features[pdb][res_id+ch]['pssm'] = list(map(float,r[4:24]))
      pdb_features[pdb][res_id+ch]['pssm'].extend(list(map(float,r[44:46])))
      if hydro:
        pdb_features[pdb][res_id+ch]['pssm'].append( aa_type )

  return pdb_features, all_sequence

def read_ignore(ignore_file,path):
  ignored_residues = dict()
  
  I = open(ignore_file,"r").readlines()
  ignore_data = list(map(str.strip, I))
  for i in ignore_data:
    r = i.split(".")
    r = r[0].split("_")
    pdb = r[0]
    ch  = r[1]
    if not pdb in ignored_residues:
      ignored_residues[pdb]={}
    ignored_residues[pdb][ch] = {}
    I = iter(list(map(str.strip,open(path+"/"+i,"r").readlines())))
    for j in I:
      #print("%s %s %s"%(pdb,ch,j))
      ignored_residues[pdb][ch][j]=True

  return ignored_residues
      


def read_contacts(rri_file,path,features,verbose=False):
  rri = dict()
  rri_ch_ch = dict()
  cci = dict()
  N_cci = 0

  PDB = list(map(str.strip, open(rri_file,"r").readlines()))
  for i in PDB:
    if not i in features:
      print(features.keys())
      raise Exception("PDB "+i+" features not found")
    if not i in rri:
      rri[i] = dict()
      rri_ch_ch[i] = dict()
      cci[i] = dict()
    J = iter(list(map(str.strip,open(path+"/"+i+".int","r").readlines())))
    for j in J:
      r =  j.split("\t")
      if not (r[0] in features[i] and r[1] in features[i]):
        if(verbose): print("IGNORING %s %s %s"%(r[0],r[1],i))
        continue
      ch_r = r[0][-1]
      ch_l = r[1][-1]
      if ch_r > ch_l:
        aux = ch_r
        ch_r = ch_l
        ch_l = aux
        aux = r[0]
        r[0] = r[1]
        r[1] = aux
      if not ch_r+":"+ch_l in rri_ch_ch[i]:
        rri_ch_ch[i][ch_r+":"+ch_l] = {}
  
      rri_ch_ch[i][ch_r+":"+ch_l][r[0]+":"+r[1]] = True
      rri[i][r[0]+":"+r[1]]=True
      if not ch_r+":"+ch_l in cci[i]:
        N_cci += 1
        cci[i][ch_r+":"+ch_l] = True
  return {'rri':rri, 'rri_ch_ch':rri_ch_ch, 'cci':rri_ch_ch, 'N_cci':N_cci}


def read_scop(scop_file):
  scop = dict()
  I = iter(list(map(str.strip,open(scop_file,"r").readlines())))
  for i in I:
    r =  i.split("\t")
    pdb = r[0]
    ch = r[1]+":"+r[2]
    scop_ = r[3]+":::"+r[4]
    if r[1] > r[2]:
      ch = r[2]+":"+r[1]
      scop_ = r[4]+":::"+r[3]
    if not pdb in scop:
      scop[pdb]=dict()
    if not ch in scop[pdb]:
      scop[pdb][ch]=scop_
  return scop

def build_nr_training_set(training):
  PDBs = training['contacts']['cci'].keys()
  scop_pairs = {}
  for pdb in PDBs:
    for ch_ch in training['contacts']['cci'][pdb]:
      SCOP = training['scop'][pdb][ch_ch].split(":::")
      scop_1 = SCOP[0]
      scop_2 = SCOP[1]
      if not scop_1 in scop_pairs:
        scop_pairs[scop_1] = {}
      scop_pairs[scop_1][scop_2] = [pdb,ch_ch]

      if not scop_2 in scop_pairs:
        scop_pairs[scop_2] = {}
      scop_pairs[scop_2][scop_1] = [pdb,ch_ch]

  out = {}
  keys = list(scop_pairs.keys())
  n_cci = 0
  while len(keys) > 0:
    n = random.randint(0,len(keys)-1)
    scop = keys[n]
    keys_ = list(scop_pairs[scop].keys())
    n_ = random.randint(0,len(keys_)-1)
    scop_ = keys_[n_]

    if not scop_pairs[scop][scop_][0] in out:
      out[ scop_pairs[scop][scop_][0] ] = {}

    out[ scop_pairs[scop][scop_][0] ][ scop_pairs[scop][scop_][1] ] = True
    n_cci +=1

    if scop in scop_pairs:
      del scop_pairs[scop]

    for i in keys_:
      if i in scop_pairs:
        del scop_pairs[i]

    for i in scop_pairs:
      if scop in scop_pairs[i]:
        del scop_pairs[i][scop]
      for j in keys_:
        if j in scop_pairs[i]:
          del scop_pairs[i][j]

    keys = list(scop_pairs.keys())
    for i in keys:
      if len(scop_pairs[i])==0:
        del scop_pairs[i]

    keys = list(scop_pairs.keys())
    
  return out, n_cci
