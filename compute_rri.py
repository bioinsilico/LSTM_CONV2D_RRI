from Bio.PDB.PDBParser import PDBParser 
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.mmtf import MMTFParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Polypeptide import CaPPBuilder
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import Selection
from Bio.PDB import NeighborSearch
from Bio.PDB.DSSP import DSSP
from Bio.PDB.NACCESS import *
from Bio.PDB import PDBIO


thr = 6
pdb_list = open('../437_dimers_list.merge.tsv')
for pdb_chi_chj in pdb_list:
  print(pdb_chi_chj)
  x = pdb_chi_chj.rstrip().split("\t")
  pdb = x[0]
  chi = x[1]
  chj = x[2]
  p = MMTFParser()
  structure = MMTFParser.get_structure_from_url(pdb)
  
  s = structure[0]
  
  atom_list = [ atom for atom in s[chi].get_atoms() if atom.name != 'H' ]
  atom_list.extend( [ atom for atom in s[chj].get_atoms() if atom.name != 'H' ] )
  RRI = NeighborSearch( atom_list ).search_all( thr ,'A' )
  MAP = {}
  for rri in RRI:
    if(rri[0].get_parent().get_id()[0][0:2]=="H_" or rri[0].get_parent().get_id()[0] == 'W'):
      #print(rri[0].get_parent().get_id()[0][0:2])
      continue
    if(rri[1].get_parent().get_id()[0][0:2]=="H_" or rri[1].get_parent().get_id()[0] == 'W'):
      #print(rri[1].get_parent().get_id()[0][0:2])
      continue
    ch_i = rri[0].get_full_id()[2]
    res_i = rri[0].get_full_id()[3][1]
    if rri[0].get_full_id()[3][2] != " ":
      res_i = str(res_i)+rri[0].get_full_id()[3][2]
    ch_j = rri[1].get_full_id()[2]
    res_j = rri[1].get_full_id()[3][1]
    if rri[1].get_full_id()[3][2] != " ":
      res_j = str(res_j)+rri[1].get_full_id()[3][2]
    if ch_i == ch_j:
      continue

    if ch_i == chi:
      if not res_i in MAP:
        MAP[res_i] = {}
      MAP[res_i][res_j] = True
    else:
      if not res_j in MAP:
        MAP[res_j] = {}
      MAP[res_j][res_i] = True
  fh = open("./"+pdb+".int","a")
  for i in MAP:
    for j in MAP[i]:
      fh.write(str(i)+chi+"\t"+str(j)+chj+"\n")
  fh.close()
