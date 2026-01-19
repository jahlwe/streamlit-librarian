# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 23:22:43 2025

@author: Jakob
"""

import utils.genericUtilities as gu
import utils.compilerUtilities as cu
import re
import itertools
import IsoSpecPy as iso
import numpy as np
from numpy import array, dot
from numpy.linalg import norm
from scipy.stats import norm as scipy_norm
from rdkit import Chem
from rdkit.Chem import AllChem, RDKFingerprint, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import BRICS 
from rdkit.Chem.Recap import RecapDecompose
from itertools import islice, combinations
import pandas as pd
import copy
from utils.MacFrag import MacFrag
import csv

E_MASS = 0.00054858
MASSES = { # maybe we can get masses from a package instead.
    'C': 12.00000,
    'H': 1.00783,
    'N': 14.00307,
    'O': 15.99491,
    'P': 30.97376,
    'S': 31.97207,
    # halogens
    'F': 18.99840,
    'Cl': 34.96885,
    'Br': 78.91834,
    'I': 126.90447,
    # others
    'Na': 22.98977,
    'K': 38.96371,
    'Hg': 201.97064,
    'As': 74.92159,
    'Au': 196.96657
}

# adduct atoms --- use to update atom_counts
# maybe do something more sophisticated later where we just read the
# ion type/adduct string and take it from there
ADDUCTS = {
    # pos
    '[M]+': {},
    '[M]2+': {},
    '[M+H]+': {'H': 1},
    '[M+NH4]+': {'N': 1, 'H': 4},
    '[M+Na]+': {'Na': 1},
    '[M+K]+': {'K': 1},
    '[M+2H]2+': {'H': 2},
    '[M+H-H2O]+': {'H': 1, 'O': -1},
    # neg
    '[M-H]-': {'H': -1},
    '[M+Cl]-': {'Cl': 1},
    '[M+F]-': {'F': 1},
    '[M-2H]2-': {'H': -2},
    '[M-H-H2O]-': {'H': -1, 'O': -1}
}

atom_names = sorted(MASSES.keys(), key=lambda x: -len(x))
atom_pattern = r'(' + '|'.join(atom_names) + r')(\d*)'

def parse_formula(formula):
    """
    Helper, reads a molecular formula and returns a dictionary with atom counts.
    Atoms are stored as keys, counts as values.
    """
    
    atom_counts = {}
    for (atom, count) in re.findall(atom_pattern, formula, re.IGNORECASE):
        atom_counts[atom] = atom_counts.get(atom, 0) + int(count or 1)
    return atom_counts

def regenerate_formula_hill(atom_counts):
    """
    Recreates a molecular formula string from an atom count dictionary.
    """
    
    elements = list(atom_counts.keys())
    formula = ''
    if 'C' in atom_counts and atom_counts['C'] > 0:
        # if C is present: C first, then H, rest alphabetical
        count = atom_counts['C']
        formula += 'C' + (str(count) if count > 1 else '')
        # H second (if present)
        if 'H' in atom_counts and atom_counts['H'] > 0:
            count = atom_counts['H']
            formula += 'H' + (str(count) if count > 1 else '')
        for elem in sorted(e for e in elements if e not in ('C', 'H')):
            count = atom_counts[elem]
            if count > 0:
                formula += elem + (str(count) if count > 1 else '')
    else:
        # if no C (unlikely for us but, anyway): all alphabetical
        for elem in sorted(elements):
            count = atom_counts[elem]
            if count > 0:
                formula += elem + (str(count) if count > 1 else '')
    return formula

def apply_adduct(atom_counts, adduct):
    """
    Helper, adds adduct atoms to an atom count dictionary.
    """
    
    for atom, increment in ADDUCTS[adduct].items():
        atom_counts[atom] = atom_counts.get(atom, 0) + increment
    return atom_counts

def get_charge(adduct):
    """
    Helper, returns an integer charge value based on adduct / ion type.
    """
    
    # defines groups to match in a string 
    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        number = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * number
    return 0 # if no charge 

# for annotation.
# if we ever take this on tour, remember MB wants [formula]2+
def get_charge_annotation(adduct):
    """
    Helper, returns string with charge information for annotation.
    """
    
    # supply the adduct again and do this
    if adduct:
        match = re.search(r'(\d*)([+-])$', adduct)
        if match:
            number = match.group(1)
            sign = match.group(2)
            charge = sign + number if number else sign
            return charge
    else:
        return ''

def get_mz_noCharge(atom_counts):
    """
    Helper, sums and returns the total mass of atoms in an atom count dictionary.
    """
    
    mz = (sum(n * MASSES[a] for a, n in atom_counts.items()))
    return mz

def get_molecular_ion_mz(atom_counts, adduct):
    """
    Helper, calculates theoretical m/z of an ion given an atom count dictionary and adduct / ion type.
    """
    
    charge = get_charge(adduct)
    mol_ion_mz = (sum(n * MASSES[a] for a, n in atom_counts.items()) - (charge * E_MASS))/ abs(charge) # need abs for neg to work
    return mol_ion_mz

def mass_IsoSpecPy(formula, adduct=None):
    """
    Generates an isotopic distribution from an input molecular formula.
    
    Parameters & args:
        formula (string): A molecular formula
        adduct (string): Adduct / ion type
    Returns:
        massDist (dict): Dictionary with isotopic distribution (peak m/z as keys, probabilities as values)
    """
    
    massDist = {}
    # use isospecpy to generate an isotopic distribution
    isoDist = iso.IsoTotalProb(formula=formula, prob_to_cover=0.999)
    if adduct: # adjust neutral masses from ISP according to charge
        charge = get_charge(adduct)
        if charge != 0:
            massDist = {((mass - (charge * E_MASS))/ abs(charge)): prob for mass, prob in isoDist}
        else:
            massDist = {mass: prob for mass, prob in isoDist}
    else:
        massDist = {mass: prob for mass, prob in isoDist}
    return massDist

def calculate_ppm_dev(exp_mz, ref_mz):
    """
    Helper, calculates parts-per-million deviation given two m/z values.
    """
    
    return ((ref_mz-exp_mz) / ref_mz) * 1e6

def normalize_atom(atom, masses_dict):
    """
    Helper, checks whether atom key in atom counts dictionary is valid.
    """
    
    # ma 2014 formulas being read as lower case for some unknown reason
    for key in masses_dict:
        if key.lower() == atom.lower():
            return key
    raise KeyError(f"can't find {atom} in MASSES list")

# ---- MACFRAG & FINGERPRINTING ----
MACCS_KEYS = { # MACCS keys, from source code. index = bit, content = SMARTS.
  1: ('?', 0),  # ISOTOPE
  #2:('[#104,#105,#106,#107,#106,#109,#110,#111,#112]',0),  # atomic num >103 Not complete
  2: ('[#104]', 0),  # limit the above def'n since the RDKit only accepts up to #104
  3: ('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 0),  # Group IVa,Va,VIa Rows 4-6 
  4: ('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 0),  # actinide
  5: ('[Sc,Ti,Y,Zr,Hf]', 0),  # Group IIIB,IVB (Sc...)  
  6: ('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0),  # Lanthanide
  7: ('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0),  # Group VB,VIB,VIIB
  8: ('[!#6;!#1]1~*~*~*~1', 0),  # QAAA@1
  9: ('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0),  # Group VIII (Fe...)
  10: ('[Be,Mg,Ca,Sr,Ba,Ra]', 0),  # Group IIa (Alkaline earth)
  11: ('*1~*~*~*~1', 0),  # 4M Ring
  12: ('[Cu,Zn,Ag,Cd,Au,Hg]', 0),  # Group IB,IIB (Cu..)
  13: ('[#8]~[#7](~[#6])~[#6]', 0),  # ON(C)C
  14: ('[#16]-[#16]', 0),  # S-S
  15: ('[#8]~[#6](~[#8])~[#8]', 0),  # OC(O)O
  16: ('[!#6;!#1]1~*~*~1', 0),  # QAA@1
  17: ('[#6]#[#6]', 0),  #CTC
  18: ('[#5,#13,#31,#49,#81]', 0),  # Group IIIA (B...) 
  19: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
  20: ('[#14]', 0),  #Si
  21: ('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),  # C=C(Q)Q
  22: ('*1~*~*~1', 0),  # 3M Ring
  23: ('[#7]~[#6](~[#8])~[#8]', 0),  # NC(O)O
  24: ('[#7]-[#8]', 0),  # N-O
  25: ('[#7]~[#6](~[#7])~[#7]', 0),  # NC(N)N
  26: ('[#6]=;@[#6](@*)@*', 0),  # C$=C($A)$A
  27: ('[I]', 0),  # I
  28: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 0),  # QCH2Q
  29: ('[#15]', 0),  # P
  30: ('[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),  # CQ(C)(C)A
  31: ('[!#6;!#1]~[F,Cl,Br,I]', 0),  # QX
  32: ('[#6]~[#16]~[#7]', 0),  # CSN
  33: ('[#7]~[#16]', 0),  # NS
  34: ('[CH2]=*', 0),  # CH2=A
  35: ('[Li,Na,K,Rb,Cs,Fr]', 0),  # Group IA (Alkali Metal)
  36: ('[#16R]', 0),  # S Heterocycle
  37: ('[#7]~[#6](~[#8])~[#7]', 0),  # NC(O)N
  38: ('[#7]~[#6](~[#6])~[#7]', 0),  # NC(C)N
  39: ('[#8]~[#16](~[#8])~[#8]', 0),  # OS(O)O
  40: ('[#16]-[#8]', 0),  # S-O
  41: ('[#6]#[#7]', 0),  # CTN
  42: ('F', 0),  # F
  43: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 0),  # QHAQH
  44: ('[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]', 0),  # OTHER
  45: ('[#6]=[#6]~[#7]', 0),  # C=CN
  46: ('Br', 0),  # BR
  47: ('[#16]~*~[#7]', 0),  # SAN
  48: ('[#8]~[!#6;!#1](~[#8])(~[#8])', 0),  # OQ(O)O
  49: ('[!+0]', 0),  # CHARGE  
  50: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
  51: ('[#6]~[#16]~[#8]', 0),  # CSO
  52: ('[#7]~[#7]', 0),  # NN
  53: ('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),  # QHAAAQH
  54: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),  # QHAAQH
  55: ('[#8]~[#16]~[#8]', 0),  #OSO
  56: ('[#8]~[#7](~[#8])~[#6]', 0),  # ON(O)C
  57: ('[#8R]', 0),  # O Heterocycle
  58: ('[!#6;!#1]~[#16]~[!#6;!#1]', 0),  # QSQ
  59: ('[#16]!:*:*', 0),  # Snot%A%A
  60: ('[#16]=[#8]', 0),  # S=O
  61: ('*~[#16](~*)~*', 0),  # AS(A)A
  62: ('*@*!@*@*', 0),  # A$!A$A
  63: ('[#7]=[#8]', 0),  # N=O
  64: ('*@*!@[#16]', 0),  # A$A!S
  65: ('c:n', 0),  # C%N
  66: ('[#6]~[#6](~[#6])(~[#6])~*', 0),  # CC(C)(C)A
  67: ('[!#6;!#1]~[#16]', 0),  # QS
  68: ('[!#6;!#1;!H0]~[!#6;!#1;!H0]', 0),  # QHQH (&...) SPEC Incomplete
  69: ('[!#6;!#1]~[!#6;!#1;!H0]', 0),  # QQH
  70: ('[!#6;!#1]~[#7]~[!#6;!#1]', 0),  # QNQ
  71: ('[#7]~[#8]', 0),  # NO
  72: ('[#8]~*~*~[#8]', 0),  # OAAO
  73: ('[#16]=*', 0),  # S=A
  74: ('[CH3]~*~[CH3]', 0),  # CH3ACH3
  75: ('*!@[#7]@*', 0),  # A!N$A
  76: ('[#6]=[#6](~*)~*', 0),  # C=C(A)A
  77: ('[#7]~*~[#7]', 0),  # NAN
  78: ('[#6]=[#7]', 0),  # C=N
  79: ('[#7]~*~*~[#7]', 0),  # NAAN
  80: ('[#7]~*~*~*~[#7]', 0),  # NAAAN
  81: ('[#16]~*(~*)~*', 0),  # SA(A)A
  82: ('*~[CH2]~[!#6;!#1;!H0]', 0),  # ACH2QH
  83: ('[!#6;!#1]1~*~*~*~*~1', 0),  # QAAAA@1
  84: ('[NH2]', 0),  #NH2
  85: ('[#6]~[#7](~[#6])~[#6]', 0),  # CN(C)C
  86: ('[C;H2,H3][!#6;!#1][C;H2,H3]', 0),  # CH2QCH2
  87: ('[F,Cl,Br,I]!@*@*', 0),  # X!A$A
  88: ('[#16]', 0),  # S
  89: ('[#8]~*~*~*~[#8]', 0),  # OAAAO
  90:
  ('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',
   0),  # QHAACH2A
  91:
  ('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',
   0),  # QHAAACH2A
  92: ('[#8]~[#6](~[#7])~[#6]', 0),  # OC(N)C
  93: ('[!#6;!#1]~[CH3]', 0),  # QCH3
  94: ('[!#6;!#1]~[#7]', 0),  # QN
  95: ('[#7]~*~*~[#8]', 0),  # NAAO
  96: ('*1~*~*~*~*~1', 0),  # 5 M ring
  97: ('[#7]~*~*~*~[#8]', 0),  # NAAAO
  98: ('[!#6;!#1]1~*~*~*~*~*~1', 0),  # QAAAAA@1
  99: ('[#6]=[#6]', 0),  # C=C
  100: ('*~[CH2]~[#7]', 0),  # ACH2N
  101:
  ('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',
   0),  # 8M Ring or larger. This only handles up to ring sizes of 14
  102: ('[!#6;!#1]~[#8]', 0),  # QO
  103: ('Cl', 0),  # CL
  104: ('[!#6;!#1;!H0]~*~[CH2]~*', 0),  # QHACH2A
  105: ('*@*(@*)@*', 0),  # A$A($A)$A
  106: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),  # QA(Q)Q
  107: ('[F,Cl,Br,I]~*(~*)~*', 0),  # XA(A)A
  108: ('[CH3]~*~*~*~[CH2]~*', 0),  # CH3AAACH2A
  109: ('*~[CH2]~[#8]', 0),  # ACH2O
  110: ('[#7]~[#6]~[#8]', 0),  # NCO
  111: ('[#7]~*~[CH2]~*', 0),  # NACH2A
  112: ('*~*(~*)(~*)~*', 0),  # AA(A)(A)A
  113: ('[#8]!:*:*', 0),  # Onot%A%A
  114: ('[CH3]~[CH2]~*', 0),  # CH3CH2A
  115: ('[CH3]~*~[CH2]~*', 0),  # CH3ACH2A
  116: ('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]', 0),  # CH3AACH2A
  117: ('[#7]~*~[#8]', 0),  # NAO
  118: ('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),  # ACH2CH2A > 1
  119: ('[#7]=*', 0),  # N=A
  120: ('[!#6;R]', 1),  # Heterocyclic atom > 1 (&...) Spec Incomplete
  121: ('[#7;R]', 0),  # N Heterocycle
  122: ('*~[#7](~*)~*', 0),  # AN(A)A
  123: ('[#8]~[#6]~[#8]', 0),  # OCO
  124: ('[!#6;!#1]~[!#6;!#1]', 0),  # QQ
  125: ('?', 0),  # Aromatic Ring > 1
  126: ('*!@[#8]!@*', 0),  # A!O!A
  127: ('*@*!@[#8]', 1),  # A$A!O > 1 (&...) Spec Incomplete
  128:
  ('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',
   0),  # ACH2AAACH2A
  129: ('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',
        0),  # ACH2AACH2A
  130: ('[!#6;!#1]~[!#6;!#1]', 1),  # QQ > 1 (&...)  Spec Incomplete
  131: ('[!#6;!#1;!H0]', 1),  # QH > 1
  132: ('[#8]~*~[CH2]~*', 0),  # OACH2A
  133: ('*@*!@[#7]', 0),  # A$A!N
  134: ('[F,Cl,Br,I]', 0),  # X (HALOGEN)
  135: ('[#7]!:*:*', 0),  # Nnot%A%A
  136: ('[#8]=*', 1),  # O=A>1 
  137: ('[!C;!c;R]', 0),  # Heterocycle
  138: ('[!#6;!#1]~[CH2]~*', 1),  # QCH2A>1 (&...) Spec Incomplete
  139: ('[O;!H0]', 0),  # OH
  140: ('[#8]', 3),  # O > 3 (&...) Spec Incomplete
  141: ('[CH3]', 2),  # CH3 > 2  (&...) Spec Incomplete
  142: ('[#7]', 1),  # N > 1
  143: ('*@*!@[#8]', 0),  # A$A!O
  144: ('*!:*:*!:*', 0),  # Anot%A%Anot%A
  145: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
  146: ('[#8]', 2),  # O > 2
  147: ('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),  # ACH2CH2A
  148: ('*~[!#6;!#1](~*)~*', 0),  # AQ(A)A
  149: ('[C;H3,H4]', 1),  # CH3 > 1
  150: ('*!@*@*!@*', 0),  # A!A$A!A
  151: ('[#7;!H0]', 0),  # NH
  152: ('[#8]~[#6](~[#6])~[#6]', 0),  # OC(C)C
  153: ('[!#6;!#1]~[CH2]~*', 0),  # QCH2A
  154: ('[#6]=[#8]', 0),  # C=O
  155: ('*!@[CH2]!@*', 0),  # A!CH2!A
  156: ('[#7]~*(~*)~*', 0),  # NA(A)A
  157: ('[#6]-[#8]', 0),  # C-O
  158: ('[#6]-[#7]', 0),  # C-N
  159: ('[#8]', 1),  # O>1
  160: ('[C;H3,H4]', 0),  #CH3
  161: ('[#7]', 0),  # N
  162: ('a', 0),  # Aromatic
  163: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
  164: ('[#8]', 0),  # O
  165: ('[R]', 0),  # Ring
  166: ('?', 0),  # Fragments  FIX: this can't be done in SMARTS
}

def clean_dummy_from_formula(formula):
    """
    Helper, removes dummy elements from formulas generated via fingerprinting etc.
    """
    
    # dummy artifacts are * or *(number)
    return re.sub(r'\*\d*', '', formula)

def atom_count_from_mol(mol):
    """
    Helper, gets atom count dictionary for an RDKit mol object.
    """
    
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return sum(parse_formula(formula).values())

def smiles_to_mol_to_formula_to_count(smiles):
    """
    Helper, gets atom count starting from a SMILES extracted from an RDKit mol object.
    """
    
    # ...
    temp_mol = Chem.MolFromSmiles(smiles)
    atom_count = atom_count_from_mol(temp_mol)
    return atom_count

def get_MacFrag(parent_mol):
    """
    Generates MacFrag fragments. Uses external code, see MacFrag.py.
    Also, see Diao et al., 2023 (https://doi.org/10.1093/bioinformatics/btad012)
    
    Parameters & args:
        parent_mol: RDKit mol object
    
    Returns:
        (dict): Dictionary with MacFrag fragments described as SMILES as keys, atom count dictionaries as values
    """
    
    # this returns a list of SMILES.
    MacFrags = MacFrag(parent_mol)
    # no need for position cleaning, they do this themselves
    if len(MacFrags) > 1:
        return {smiles:smiles_to_mol_to_formula_to_count(smiles) for smiles in MacFrags}
    else:
        return {}
    
def extract_fragments_from_bitinfo(mol, bitInfo):
    """
    Extracts Morgan/ECFP substructures from bitInfo.
    Reconstructs fragments in atom environments and converts to SMILES,
    which are returned as keys in a dictionary with atom counts as values.
    
    Parameters & args:
        mol: RDKit mol object
        bitInfo (dict): Dictionary with Morgan bitInfo
        
    Returns:
        cleaned_dict (dict): Dictionary with unique fragments described as SMILES
    """
    
    fp_dict = {}
    for bit, atom_radius_list in bitInfo.items():
        for atom_idx, radius in atom_radius_list:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            atoms = set([atom_idx])
            for bond_idx in env:
                bond = mol.GetBondWithIdx(bond_idx)
                atoms.add(bond.GetBeginAtomIdx())
                atoms.add(bond.GetEndAtomIdx())
            submol = Chem.PathToSubmol(mol, env)
            if submol:
                smiles = Chem.MolToSmiles(submol)
                fp_dict[smiles] = submol.GetNumAtoms()
    # empty fragments sometimes --- do this
    cleaned_dict = {entity:n_atoms for entity, n_atoms in fp_dict.items() if n_atoms > 0}
    return cleaned_dict

def extract_fragments_from_bitpaths(mol, bitPaths):
    """
    Extracts path-based fingerprint substructures from bitPaths.
    Reconstructs fragments directly from bond index paths and converts to smiles,
    which are returned as keys in a dictionary with atom counts as values.
    
    Parameters & args:
        mol: RDKit Mol object
        bitPaths: Path-based fingerprint dict {bitID: [bond_indices_list, ...]}
    
    Returns:
        cleaned_dict (dict): Dictionary with unique fragments described as SMILES
    """
    
    fp_dict = {}
    for bit, paths in bitPaths.items():
        for bond_indices in paths:
            submol = Chem.PathToSubmol(mol, bond_indices) # ... PathToSubmol needs ATOM indices?
            if submol.GetNumAtoms() > 0:
                smiles = Chem.MolToSmiles(submol)
                fp_dict[smiles] = submol.GetNumAtoms()
    # in case we have empty fragments here too...
    cleaned_dict = {entity:n_atoms for entity, n_atoms in fp_dict.items() if n_atoms > 0}
    return cleaned_dict

def get_rdkit_fingerprints(mol):
    """
    Generates RDKit Daylight-like path-based fingerprints.
    Uses the extract_fragments_from_bitpaths fn to return fragments.
    
    Parameters & args:
        mol: RDKit Mol object
    
    Returns:
        dict: Dictionary with unique fragments described as SMILES
    """
    
    # better for linear structures, chains? 'daylight-like'
    fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitPaths()
    fp = fp_gen.GetFingerprint(mol, additionalOutput=ao)
    bitPaths = ao.GetBitPaths()
    return extract_fragments_from_bitpaths(mol, bitPaths)

def get_morgan_fingerprints(mol, radius=2, nBits=2048):
    """
    Generates Morgan/ECFP circular fingerprints.
    Uses the extract_fragments_from_bitinfo fn to return fragments.
    
    Parameters & args:
        mol: RDKit Mol object
        radius (int): Morgan radius
        nBits (int): Fingerprint size
    
    Returns:
        dict: Dictionary with unique fragments described as SMILES
    """
    
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = fp_gen.GetFingerprint(mol, additionalOutput=ao)
    bitInfo = ao.GetBitInfoMap()
    return extract_fragments_from_bitinfo(mol, bitInfo)

def get_maccs_fingerprints(mol, maccs_dict):
    """
    Generates MACCS structural keys fingerprints.
    Matches SMARTS substructures for each on-bit and extracts connected bond components.
    
    Parameters & args:
        mol: RDKit Mol object
        maccs_dict: (dict) Dictionary mapping MACCS bit IDs to (SMARTS, count) tuples
    
    Returns:
        cleaned_dict (dict): Dictionary with unique fragments described as SMILES
    """

    # now working like the others, except we need to supply
    # the maccs_dict reference (we could use it global-style, but whatever)
    fp_dict = {}
    fp = MACCSkeys.GenMACCSKeys(mol)
    maccs_bits = fp.GetOnBits()
    for bit in maccs_bits:
        smarts = maccs_dict.get(bit, ('?', 0))[0]
        if smarts == '?' or not smarts:
            continue
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        # this returns sets of atoms in our parent mol that match a given
        # substructure query (from our smarts, which are very generalized 
        # representations of chemical motifs)
        matches = mol.GetSubstructMatches(query)
        for match in matches:
            if len(match) < 2:
                continue
            # find all bonds where both atoms are in the match
            atom_set = set(match)
            bond_indices = []
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in atom_set and a2 in atom_set:
                    bond_indices.append(bond.GetIdx())
            if not bond_indices:
                continue  # no bonds between these atoms --- skip
            submol = Chem.PathToSubmol(mol, bond_indices)
            if submol.GetNumAtoms() > 0:
                # check for disconnected fragments
                # if we generate stuff, but it is disjointed...
                if len(Chem.GetMolFrags(submol)) > 1:
                    continue
                smiles = Chem.MolToSmiles(submol)
                if smiles not in fp_dict.keys():
                    fp_dict[smiles] = submol.GetNumAtoms()
    cleaned_dict = {entity:n_atoms for entity, n_atoms in fp_dict.items() if n_atoms > 0}
    return cleaned_dict

# ---- READ THE NEUTRAL LOSS LIST ---
def read_loss_ref():
    """
    Reads the curated list of neutral losses and fragments from Ma, 2014.
    DOI: https://doi.org/10.1021/ac502818e
    """
    data = {}
    with open('utils/ma2014.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, line in enumerate(reader):
            if i == 0:
                continue
            if len(line) < 7: continue
            ref_formula = line[3]
            atom_counts = parse_formula(ref_formula)
            # Normalize atom names to match MASSES keys
            normalized_counts = {}
            for atom, count in atom_counts.items():
                normalized_atom = normalize_atom(atom, MASSES)
                normalized_counts[normalized_atom] = count
            mass = get_mz_noCharge(normalized_counts)
            key = regenerate_formula_hill(normalized_counts)
            pos = (line[5] == '+')
            neg = (line[6] == '+')
            data[key] = {
                'mass': round(mass, 5), 'atom_count': normalized_counts,
                'pos': pos, 'neg': neg
            }
    return data

# we seem to be missing ~3 of these, but whatever
NEUTRAL_LOSS_REF = read_loss_ref()

# also add our own stuff.
common_fragments = [
    'C7H7'
]

def add_common_fragments(fragments, loss_ref):
    """
    Adds common fragments to the global NEUTRAL_LOSS_REF dictionary.
    More fragments can be added to the list common_fragments later...
    """
    
    for formula in fragments:
        atom_counts = parse_formula(formula)
        mass = get_mz_noCharge(atom_counts)
        if formula not in loss_ref.keys():
            loss_ref[formula] = {
                'mass': round(mass, 5), 'atom_count': atom_counts,
                'pos': True, 'neg': True}
    return loss_ref

NEUTRAL_LOSS_REF = add_common_fragments(common_fragments, NEUTRAL_LOSS_REF)

def identify_parent(theo_parent_mz, ms2_data, ppm_threshold=10):
    """
    Helper that identifies which peak in an MS2 spectrum (if any) 
    corresponds to the parent ion.
    """
    
    best_match = (None, None)  # (index, ppm_dev)
    for idx, (mz, abs_int, rel_int) in enumerate(ms2_data):
        ppm_dev = calculate_ppm_dev(mz, theo_parent_mz)
        if abs(ppm_dev) < ppm_threshold:
            if best_match[1] is None or abs(ppm_dev) < abs(best_match[1]):
                best_match = (idx, ppm_dev)
    return best_match

def find_compatible_losses(atom_count, mode=None):
    """
    Helper that identifies losses from the dictionary of common fragments that 
    are compatible with the atom constitution of a compound.    
    """
    
    # don't do the mode thing.
    loss_reference = copy.deepcopy(NEUTRAL_LOSS_REF)
    compatible_losses = {}
    for key, data in loss_reference.items():
        loss_count = data['atom_count']
        compatible = True
        for atom, count in loss_count.items():
            if atom not in atom_count or count > atom_count[atom]:
                compatible = False
                break
        if compatible:
            compatible_losses[key] = data
    return compatible_losses

def generate_ref_fragments(
    data, fragment_and_loss=True
):
    """
    Collects candidate fragment formulas for a compound using the collection of 
    a-priori known (currently, mostly Ma et al., 2014) reference losses and fragments.
    
    Both the individual candidate fragment formulas and the resulting 
    parent-minus-fragment formulas are collected.
    
    Parameters & args:
        data (dict): Compound data
    
    Returns:
        fragment_dict (dict): Collected formulas 
    """
    
    # store all the precursor-loss types
    fragment_dict = {}
    
    base_formula = data.get('molecularFormula')
    adduct = data.get('ion_type')
    ms2_data = data.get('ms2_norm')
    if not (base_formula and adduct and ms2_data):
        return {}
    # first, we find out if we have a precursor m/z to use from our ms2 data
    parent_atom_count = apply_adduct(parse_formula(base_formula), adduct)
    theo_parent_mz = get_molecular_ion_mz(parent_atom_count, adduct)
    best_match = identify_parent(theo_parent_mz, ms2_data, 10)
    annot_charge = get_charge_annotation(adduct)
    
    # we will need charge state for multiply charged compounds
    # we just need the absolute value to divide our losses with it
    charge = abs(get_charge(adduct))
    
    # get parent mz like this --- if we can't find exp, use theo
    parent_mz_list = []
    parent_mz_list.append(ms2_data[best_match[0]][0] if best_match[0] is not None else theo_parent_mz)
    # for heavily chlorinated e.g. we have to do this
    if 'Cl' or 'Br' in parent_atom_count and not best_match[0]:
        iso_envelope = mass_IsoSpecPy(regenerate_formula_hill(parent_atom_count), adduct)
        parent_mz_list.append(max(iso_envelope, key=lambda mass: iso_envelope[mass]))
    
    # now, we find out which [precursor-loss] entities to calculate
    compatible_losses = find_compatible_losses(parent_atom_count)

    # then we calculate them.
    # they should all be valid (no atoms < 0) because we check loss compatibility
    for formula, data in compatible_losses.items():
        loss_count = parse_formula(formula)
        # precursor_charge = get_charge(adduct) # maybe think about 2x-charge states later
        fragment_count = {a: parent_atom_count[a] - loss_count.get(a, 0) for a in parent_atom_count}
        fragment_formula = regenerate_formula_hill(fragment_count)
        # we probably should calculate the mass of the fragment with reference
        # to the observed mz (in cases where we have it)
        # first we need the mz of the loss
        loss_mz = (get_mz_noCharge(loss_count)) / charge # now also considering charge 
        fragment_mz = round(parent_mz_list[0] - loss_mz, 5) # keep thinking about this.
        fragment_dict[fragment_formula] = {
            'mz':fragment_mz, 'loss':formula, 'source':'ref_fragment'
        }
        # create matchable fragment for loss if mz > 50
        # assuming neutral losses, specifically, to be able to create charged fragments
        # is probably physicochemically iffy but it sounds harmless to try tbh
        if data.get('mass') > 50 and fragment_and_loss:
            # maybe this is redundant --- but i want to calibrate our mass
            # to our observed parent mz. maybe don't do this.
            # if we do this, we need to first get the theo mz of the fragment above.

            # trying this now. sure, go for theo mz --- but think about electrons. 
            loss_fragment_mz = get_molecular_ion_mz(loss_count, adduct)
            # and we swap places for the formulas from above
            fragment_dict[formula] = {
                'mz':loss_fragment_mz, 'loss':fragment_formula, 'source':'ref_loss'
            }
    # now we should have all the losses we need for a given compound
    # reformulated as fragments of the parent --- not as the loss itself!
    # ONE MORE THING! include the molecular ion.
    for i, possible_parent_mz in enumerate(parent_mz_list):
        # we have to do this --- add a parenthesis to enumerate identical formulas
        # will it be a problem downstream?
        fragment_dict[regenerate_formula_hill(parent_atom_count) + f'({i})'] = {
            'mz':round(possible_parent_mz, 5), 'loss':None, 'source':'parent'
        }
    return fragment_dict

def generate_more_fragments(
    data, fragment_dict, fragment_and_loss=True
):
    """
    Collects candidate fragment formulas for a compound using MacFrag and
    RDKit fingerprinting approaches.
    
    Both the individual candidate fragment formulas and the resulting 
    parent-minus-fragment formulas are collected.
    
    Parameters & args:
        data (dict): Compound data
        fragment_dict (dict): Prior set of collected formulas
    
    Returns:
        fragment_dict (dict): Updated set of collected formulas 
    """
    
    # --- REQUIRED FIELDS ---
    parent_smiles = data.get('smiles')
    base_formula = data.get('molecularFormula')
    adduct = data.get('ion_type')
    ms2_data = data.get('ms2_norm')
    if not (base_formula and adduct and ms2_data and parent_smiles):
        print('missing data, cannot generate more loss fragments')
        return fragment_dict
    # same thing as for the other function...
    parent_atom_count = apply_adduct(parse_formula(base_formula), adduct)
    theo_parent_mz = get_molecular_ion_mz(parent_atom_count, adduct)
    best_match = identify_parent(theo_parent_mz, ms2_data, 10)
    charge = abs(get_charge(adduct))
    annot_charge = get_charge_annotation(adduct)
    
    # get parent mz like this --- if we can't find exp, use theo
    # but we don't use it here, do we.
    parent_mz = ms2_data[best_match[0]][0] if best_match[0] is not None else data.get('precursor_mz')
    
    # --- INITIALIZE ---
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    more_losses = {}
    
    # we NEED to track sources. for diagnostics.
    sources = {
        'macfrag': 'macfrag',
        'rdkit': 'rdkit_fp',
        'morgan': 'morgan_fp',
        'maccs': 'maccs_fp'
    }
    
    # --- MACFRAG BLOCK ---
    temp_dict = get_MacFrag(parent_mol)
    if len(temp_dict) > 0:
        more_losses.update((smiles, (loss, sources['macfrag'])) for smiles, loss in temp_dict.items())
    
    # --- FINGERPRINTING BLOCK ---
    temp_dict = get_rdkit_fingerprints(parent_mol)
    if len(temp_dict) > 0:
        more_losses.update((smiles, (loss, sources['rdkit'])) for smiles, loss in temp_dict.items())
    temp_dict = get_morgan_fingerprints(parent_mol)
    if len(temp_dict) > 0:
        more_losses.update((smiles, (loss, sources['morgan'])) for smiles, loss in temp_dict.items())
    temp_dict = get_maccs_fingerprints(parent_mol, MACCS_KEYS)
    if len(temp_dict) > 0:
       more_losses.update((smiles, (loss, sources['maccs'])) for smiles, loss in temp_dict.items())
        
    # now, turn losses into new fragments
    more_fragments = {}
    for smiles, (loss, source) in more_losses.items():
        loss_mol = Chem.MolFromSmiles(smiles)
        if not loss_mol:
            continue
        loss_formula = clean_dummy_from_formula(
            Chem.rdMolDescriptors.CalcMolFormula(loss_mol)
        )
        loss_count = parse_formula(loss_formula)
        loss_mz = (get_mz_noCharge(loss_count)) / charge
        
        # --- CALCULATE FOR NEW FRAGMENT ---
        fragment_count = {a: parent_atom_count[a] - loss_count.get(a, 0) for a in parent_atom_count}
        fragment_formula = regenerate_formula_hill(fragment_count)
        # fragment_mz = round(parent_mz - loss_mz, 5) # keep thinking about this.
        # it seems getting a fresh mass may actually be better.
        # AT LEAST when masses become smaller. maybe balance this.
        fragment_mz = round(get_molecular_ion_mz(fragment_count, adduct), 5)
        more_fragments[fragment_formula] = {
            'mz':fragment_mz, 'loss':loss_formula, 'source': source + '_fragment'
        }
        
        # ALSO! charged version. if it is valid after adduct atom update.
        
        # actually --- to cover our butts, it seems we may need to both add and subtract H
        # we don't always know where the hydrogen will end up and sometimes we miss stuff
        # because of it
    
        for h_modifier in ('[M+H]+', '[M-H]-'): # create plus and minus one H for fragment
            fragment_count_charged = apply_adduct(copy.deepcopy(fragment_count), h_modifier)
            if not any(n < 0 for atom, n in fragment_count_charged.items()):
                fragment_formula_charged = regenerate_formula_hill(fragment_count_charged)
                # do NOT supply the h_modifier to generate the mz, it will be messed up for 
                # the -H fragment if we are in pos, and vice versa for neg
                fragment_mz_charged = round(get_molecular_ion_mz(fragment_count_charged, adduct), 5)
                more_fragments[fragment_formula_charged] = {
                    'mz':fragment_mz_charged, 'loss':loss_formula, 'source': source + '_fragment'
                }
                
        if loss_mz > 50 and fragment_and_loss:
            loss_fragment_mz = get_molecular_ion_mz(loss_count, adduct)
            more_fragments[loss_formula] = {
                'mz':loss_fragment_mz, 'loss':fragment_formula, 'source': source + '_loss'
            }
            
            # and again --- we create TWO "charged" versions, depending on H behavior
            
            for h_modifier in ('[M+H]+', '[M-H]-'): # create plus and minus one H for fragment
                loss_count_charged = apply_adduct(copy.deepcopy(loss_count), h_modifier)
                if not any(n < 0 for atom, n in loss_count_charged.items()):
                    loss_formula_charged = regenerate_formula_hill(loss_count_charged)
                    loss_fragment_mz_charged = round(get_molecular_ion_mz(loss_count_charged, adduct), 5)
                    more_fragments[loss_formula_charged] = {
                        'mz':loss_fragment_mz_charged, 'loss':fragment_formula, 'source': source + '_loss'
                    }

        #charge_placeholder_adduct = '[M+H]+' if get_charge(adduct) > 0 else '[M-H]-'
        #fragment_count_charged = apply_adduct(fragment_count, charge_placeholder_adduct)
        #if not any(n < 0 for atom, n in fragment_count_charged.items()): # valid, or not?
        #    fragment_formula_charged = regenerate_formula_hill(fragment_count_charged)
        #    fragment_mz_charged = (fragment_mz + 1.00783 - E_MASS) / charge if get_charge(adduct) > 0 else (fragment_mz - 1.00783 + E_MASS) / charge
        #    more_fragments[fragment_formula_charged] = {
        #        'mz':fragment_mz_charged, 'loss':loss_formula, 'source': source + '_fragment'
        #    }
        
        #if loss_mz > 50 and fragment_and_loss:
        #    loss_fragment_mz = get_molecular_ion_mz(loss_count, adduct)
        #    # and we swap places for the formulas from above
        #    more_fragments[loss_formula] = {
        #        'mz':loss_fragment_mz, 'loss':fragment_formula, 'source': source + '_loss'
        #    }
        #    # also! create a charged version --- charge may be iffy.
        #    loss_count_charged = apply_adduct(loss_count, charge_placeholder_adduct)
        #    if not any(n < 0 for atom, n in loss_count_charged.items()):
        #        loss_formula_charged = regenerate_formula_hill(loss_count_charged)
        #        loss_fragment_mz_charged = (get_mz_noCharge(loss_count_charged) - E_MASS) / charge if get_charge(adduct) > 0 else (get_mz_noCharge(loss_count_charged) + E_MASS) / charge
        #        more_fragments[loss_formula_charged] = {
        #            'mz':loss_fragment_mz_charged, 'loss':fragment_formula, 'source': source + '_loss'
        #        }
    merged_fragments = {**fragment_dict, **more_fragments}
    # we could mass filter here but we don't really need to. (sometimes H is in the dict, etc...)
    return merged_fragments

# this one here is a weapon indeed. 
def find_possible_fragment_formulas(
    data, target_mz, min_mass=50, ppm_threshold=5 # more stringent ppm
):
    """
    For MS2 fragments without any identified candidate formulas, this function
    generates hypothetical fragments limited by the atom count of the compound.
    The m/z of these hypothetical fragments are matched against the m/z of the 
    unannotated fragment.
    
    Parameters & args:
        data (dict): Compound data
        target_mz (float): MS2 fragment m/z to match
        min_mass (float): Minimum fragment mass
        ppm_threshold (float): ppm deviation threshold for matching
    
    Returns:
        best_match_formula (string), best_match_mz (float),
        best_match_ppm (float), best_match_source (string)
    """
    
    # adapt our brute-forcer from before for this purpose?
    base_formula = data.get('molecularFormula')
    adduct = data.get('ion_type')
    frag_annot = get_charge_annotation(adduct)
    
    # --- INITIALIZE RETURN ---
    best_match_formula = None
    best_match_mz = None
    best_match_ppm = float('inf')
    best_match_source = None
    
    # ENSURE NON-NA
    if not base_formula or not adduct:
        print('missing data, cannot use the swiss army knife')
        return best_match_formula, best_match_mz, best_match_ppm, best_match_source
    
    # parent sets the limits, of course
    atom_counts = apply_adduct(parse_formula(base_formula), adduct)
    atoms = list(atom_counts.keys())
    max_counts = [atom_counts[atom] for atom in atoms]
    
    # it would be more efficient if we could match all remaining masses to each 
    # generated fragment as they are being created...?
    for counts in itertools.product(*(range(max_c + 1) for max_c in max_counts)):
        if sum(counts) == 0: # skip null fragment
            continue
        fragment_dict = {atom: count for atom, count in zip(atoms, counts) if count > 0}
        fragment_formula = regenerate_formula_hill(fragment_dict)
        fragment_mz = get_molecular_ion_mz(fragment_dict, adduct)
        if fragment_mz < min_mass:
            continue
        ppm_dev = abs(calculate_ppm_dev(target_mz, fragment_mz))
        raw_ppm_dev = calculate_ppm_dev(target_mz, fragment_mz)
        if abs(ppm_dev) < ppm_threshold and ppm_dev < best_match_ppm:
            best_match_formula = fragment_formula + frag_annot if frag_annot not in fragment_formula else fragment_formula
            best_match_mz = round(fragment_mz, 5)
            best_match_ppm = round(raw_ppm_dev, 2)
            best_match_source = 'swiss_army_knife'
    return best_match_formula, best_match_mz, best_match_ppm, best_match_source

def isotopic_envelope_scoring(match_formula, match_idx, adduct, ms2_data, ppm_threshold=10):
    """
    Identifies isotopic peaks in MS2 spectra following candidate fragment matching.
    Generates theoretical isotopic envelopes from the candidate formula,
    and identifies isotopic peaks (M+1, etc) present in the raw data.
    Identification of isotopic peaks as such is done by cosine similarity scoring.
    Identifying an experimental peak as part of an isotopic pattern takes
    precedence over other annotation, and "claims" it as such.
    
    Parameters & args:
        match_formula (string):  Matched candidate formula to generate isotopic dist for
        match_idx (int): Index of peak in ms2_data to which the candidate formula is matched
        adduct (string): Ion type / adduct
        ms2_data (list): List of normalized MS2 data
        ppm_threshold (float: ppm deviation threshold for matching peaks
    
    Returns:
        cosine_sim (float), peaks_to_claim (tuple)
    """
    
    # ok. first we generate the iso envelope given our current peak match.
    cleaned_match_formula = re.sub(r'(\(\d+\))?[\+-][0-9A-Za-z]*$|(\(\d+\))$', '', match_formula)
    iso_envelope = mass_IsoSpecPy(cleaned_match_formula, adduct)
    # no probability filtering for now...
    iso_envelope = [(mz, prob) for mz, prob in iso_envelope.items() if prob > 0]
    
    # exp peaks to match against. only extract those with sub-max-envelope masses.
    # maybe change that to a ppm thing later to add some leeway at the top end.
    max_env_mz = max(mz for mz, _ in iso_envelope)
    max_env_mz_leeway = max_env_mz + ((max_env_mz / 1e6) * 10)
    exp_peaks = [(*data, j) for j, data in enumerate(ms2_data) if j >= match_idx and data[0] <= max_env_mz_leeway]
    
    # then we find matches
    mz_matched_peaks = []
    claimed_exp_peaks = [] # keep track of this
    
    for iso_idx, (iso_mz, prob) in enumerate(iso_envelope):
        best_match_mz = None
        best_match_ppm = float('inf')
        best_match_idx = None
        for mz_exp, ai_exp, ri_exp, exp_idx in exp_peaks:
            if mz_exp in claimed_exp_peaks:
                continue
            ppm_dev = calculate_ppm_dev(mz_exp, iso_mz)
            if abs(ppm_dev) < best_match_ppm and abs(ppm_dev) < ppm_threshold:
                best_match_mz = mz_exp
                best_match_idx = exp_idx
                best_match_ppm = ppm_dev
        if best_match_mz is not None:
            # create a reconciled mz for matches
            avg_mz = (iso_mz + best_match_mz) / 2 # actually... we don't need this?
            mz_matched_peaks.append((best_match_mz, best_match_ppm, iso_idx, best_match_idx))
            claimed_exp_peaks.append(best_match_mz)
            
    if len(mz_matched_peaks) == 0:
        return 0.0, []
    
    # align and score
    num_iso_peaks = len(iso_envelope)
    theo_intensities = np.zeros(num_iso_peaks)
    exp_intensities = np.zeros(num_iso_peaks)
    
    # lookup dict for exp (rel) intensities
    # key = isotopic peak index. value = relative int for experimental peak that was matched to that isotopic peak.
    matched_exp_intensity = {iso_idx: ms2_data[exp_idx][2] for _, ppm, iso_idx, exp_idx in mz_matched_peaks}
    
    # aligned intensity vectors...
    for i, (mz, intensity) in enumerate(iso_envelope):
        theo_intensities[i] = intensity
        exp_intensities[i] = matched_exp_intensity.get(i, 0)
    
    # avoid division by zero
    if np.sum(theo_intensities) == 0 or np.sum(exp_intensities) == 0:
        return 0.0, []
        
    theo_norm = theo_intensities / norm(theo_intensities)    
    exp_norm = exp_intensities / norm(exp_intensities)
    cosine_sim = np.dot(theo_norm, exp_norm)
    
    # send back which exp peaks to claim as isotopic
    peaks_to_claim = []
    if cosine_sim > 0.8:
        # we need to return
        # exp mz, formula, iso_mz, ppm dev, exp ri, 'isotopic'
        # this is a little insane...
        for mz_exp, ppm, iso_idx, matched_idx in mz_matched_peaks:
            peaks_to_claim.append(
                (ms2_data[matched_idx][0], match_formula, round(iso_envelope[iso_idx][0], 5), round(ppm, 2), ms2_data[matched_idx][2], 'isotopic', matched_idx)
            )
        return cosine_sim, peaks_to_claim
    else:
        return cosine_sim, []

def match_loss_fragments(
    data, loss_dict, ppm_threshold=10, swiss_army_knife=True
):
    """
    Coordinates fragment matching and annotation through use of functions above.
    
    Parameters & args:
        data (dict): Compound data
        loss_dict (dict): Dictionary with candidate fragment formulas
        ppm_threshold (float): ppm deviation threshold for matching
        swiss_army_knife (bool): Controls whether to execute find_possible_fragment_formulas
    
    Returns:
        List of matches.
    """

    base_formula = data.get('molecularFormula')
    if not base_formula:
        return []
    ms2_data = data.get('ms2_norm')    
    adduct = data.get('ion_type')
    matched = []
    # for keeping track of isotopic envelope-claimed peaks
    claimed = [] # maybe change at some point to use indices instead
    frag_annot = get_charge_annotation(adduct)
    
    total_atoms = sum(v for a, v in (parse_formula(base_formula)).items())
    for idx, (mz, ai, ri) in enumerate(ms2_data):
        if mz not in claimed:
            best_match_formula = None
            best_match_mz = None
            best_match_ppm = float('inf')
            best_match_source = None
            for formula, frag_data in loss_dict.items():
                fragment_mz = frag_data['mz']
                ppm_dev = abs(calculate_ppm_dev(mz, fragment_mz))
                raw_ppm_dev = calculate_ppm_dev(mz, fragment_mz)
                if ppm_dev < ppm_threshold and ppm_dev < best_match_ppm:
                    best_match_formula = formula + frag_annot if frag_annot not in formula else formula
                    best_match_mz = round(frag_data.get('mz'), 5)
                    best_match_ppm = round(raw_ppm_dev, 2)
                    best_match_source = frag_data.get('source')
            # the special sauce
            scaled_ri_limit = total_atoms * 2.5 # scale its use like this?
            if not best_match_formula and ri >= scaled_ri_limit and swiss_army_knife and total_atoms < 100:
                print('using swiss army knife.')
                print(f'target mz {mz}')
                best_match_formula, best_match_mz, best_match_ppm, best_match_source = find_possible_fragment_formulas(data, mz)
            # from this we store
            # original peak mz, best formula match, mz of best match, ppm of best match, original peak index
            match_result = (
                mz, best_match_formula, best_match_mz, best_match_ppm, ri, best_match_source, idx
            )
            matched.append(match_result)
            if best_match_formula:
                claimed.append(mz) # first, add matched peak to claimed list
                # iso envelope --- now with cosine scoring in a separate helper
                cos_score, peaks_to_claim = isotopic_envelope_scoring(best_match_formula, idx, adduct, ms2_data)
                if len(peaks_to_claim) > 0:
                    for mz_j, formula_j, mz_match_j, ppm_j, ri_j, source_j, idx_j in peaks_to_claim:
                        if mz_j not in claimed:
                            claimed.append(mz_j)
                            match_result = (
                                mz_j, re.sub(r'\(\d+\)', '', formula_j), mz_match_j, ppm_j, ri_j, source_j, idx_j
                            )
                            matched.append(match_result)
    # we need to reformat (or recalculate ppm, rather) a little for the parent entry
    for i, (mz, formula, match_mz, match_ppm, ri, source, idx) in enumerate(matched):
        if source == 'parent':
            parent_atom_count = apply_adduct(parse_formula(base_formula), adduct)
            theo_parent_mz = get_molecular_ion_mz(parent_atom_count, adduct)
            actual_ppm = round(calculate_ppm_dev(mz, theo_parent_mz), 2) if '(0)' in formula else round(calculate_ppm_dev(mz, match_mz), 2)
            # also do this to take care of parent-heses
            matched[i] = (mz, re.sub(r'\(\d+\)', '', formula), match_mz, actual_ppm, ri, source, idx)
    return sorted(matched, key=lambda x: x[6])

def format_annotation(data, match_list):
    """
    Reformats fragment annotation data for visualization in the web app.
    """
    
    formatted = []
    for (mz, formula, match_mz, match_ppm, ri, source, idx) in match_list:
        theo_mz = match_mz
        fragment_formula = cu.reformat_charged_formula(formula) if formula else None
        formula_count = sum(1 for entry in match_list if entry[1] == formula)
        exp_mz = mz
        ppm = match_ppm
        formatted.append((theo_mz, fragment_formula, formula_count, exp_mz, ppm))
    return formatted

#stuff = format_annotation(dictionary['Clonidine'], match_list)

#annot_lookup = {}
#for (theo_mz, formula, count, exp_mz, ppm) in stuff:
#    if not theo_mz:
#        continue
#    key = round(exp_mz, 4)  # round to 4 decimals to avoid floating point issues
#    annot_lookup[key] = {
#        'theo_mz': theo_mz,
#        'formula': formula,
#        'count': count,
#        'ppm': ppm
#    }
    
# reconcile with full peak list
#ms2_display = []
# run through each MS2 peak, w/ or w/out annotation
#for mz, abs_int, norm_int in dictionary['Clonidine']['ms2_norm']:
#    key = round(mz, 4)
    # add data from annotated lookup dict
#    if annot_lookup:
#        annotation = annot_lookup.get(key, {})
#        ms2_display.append({
#            'exp_mz': mz,
#            'abs_int': abs_int,
#            'norm_int': norm_int,
#            'theo_mz': annotation.get('theo_mz'),
#            'formula': annotation.get('formula'),
#            'count': annotation.get('count'),
#            'ppm': annotation.get('ppm')
#        })
#    else: # if there is no annotations at all, we have to do this
#        ms2_display.append({
#            'exp_mz': mz,
#            'abs_int': abs_int,
#            'norm_int': norm_int
#        })
#
#    print(ms2_display)