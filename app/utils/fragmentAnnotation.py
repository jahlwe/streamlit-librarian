# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:00:22 2025

@author: Jakob
"""
import utils.genericUtilities as gu
import utils.MacFrag as MacFrag
import re
import itertools
import IsoSpecPy as iso
from numpy import array, dot
from numpy.linalg import norm
from scipy.stats import norm as scipy_norm
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import BRICS 
from rdkit.Chem import AllChem, RDKFingerprint, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Recap import RecapDecompose
from itertools import islice, combinations
import pandas as pd
import copy

e_mass = 0.00054858
masses = { # maybe we can get masses from a package instead.
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
adducts = {
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

atom_names = sorted(masses.keys(), key=lambda x: -len(x))
atom_pattern = r'(' + '|'.join(atom_names) + r')(\d*)'

def parse_formula(formula):
    atom_counts = {}
    for (atom, count) in re.findall(atom_pattern, formula, re.IGNORECASE):
        atom_counts[atom] = atom_counts.get(atom, 0) + int(count or 1)
    return atom_counts

def regenerate_formula_hill(atom_counts):
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
    for atom, increment in adducts[adduct].items():
        atom_counts[atom] = atom_counts.get(atom, 0) + increment
    return atom_counts

def get_charge(adduct):
    # defines groups to match in a string 
    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        number = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * number
    return 0 # if no charge 

def get_molecular_ion_mz(atom_counts, adduct):
    charge = get_charge(adduct)
    mol_ion_mz = (sum(n * masses[a] for a, n in atom_counts.items()) - (charge * e_mass))/charge
    return mol_ion_mz

def mass_IsoSpecPy(formula, adduct=None):
    massDist = {}
    # use isospecpy to generate an isotopic distribution
    isoDist = iso.IsoTotalProb(formula=formula, prob_to_cover=0.999)
    if adduct: # adjust neutral masses from ISP according to charge
        charge = get_charge(adduct)
        if charge != 0:
            massDist = {((mass - (charge * e_mass))/charge): prob for mass, prob in isoDist}
        else:
            massDist = {mass: prob for mass, prob in isoDist}
    else:
        massDist = {mass: prob for mass, prob in isoDist}
    return massDist

def generate_possible_fragments(atom_counts, adduct=None, min_mass=50, iso_cutoff=0.01):
    atoms = list(atom_counts.keys())
    max_counts = [atom_counts[atom] for atom in atoms]
    possible_fragments = {}
    # generate all combinations of atom counts
    for counts in itertools.product(*(range(max_c + 1) for max_c in max_counts)):
        if sum(counts) == 0:  # skip null fragment
            continue
        frag_dict = {atom: count for atom, count in zip(atoms, counts) if count > 0}
        frag_formula = regenerate_formula_hill(frag_dict)
        frag_massDist = mass_IsoSpecPy(frag_formula, adduct)
        filtered_dist = {m: p for m, p in frag_massDist.items() if p >= iso_cutoff}
        # mass of most prevalent fragment decides whether to filter
        if filtered_dist and max(filtered_dist, key=filtered_dist.get) >= min_mass:
            possible_fragments[frag_formula] = filtered_dist
    return possible_fragments

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

def calculate_ppm_dev(exp_mz, ref_mz):
    return abs((ref_mz-exp_mz) / ref_mz) * 1e6

def read_spectrum(path):
    spectrum = []
    reading_peaks = False
    n_peaks = 0
    n_peaks_read = 0
    types = (float, int, int)
    with open(path, 'r') as s:
        for line in s:
            line = line.rstrip()
            if pattern := re.search(r'(PK\$NUM_PEAK: )(\d+)$', line):
                n_peaks = int(pattern.group(2))
            if 'PK$PEAK:' in line:
                reading_peaks = True
                continue
            if reading_peaks and n_peaks_read < n_peaks:
                peak_data = line.split()
                if len(peak_data) >= 3:
                    spectrum.append(tuple(t(val) for t, val in zip(types, peak_data[:3])))
                n_peaks_read += 1
                if n_peaks_read >= n_peaks:
                    reading_peaks = False
    return spectrum

# use isotopic envelopes for scoring rather than single fragments
def score_envelope_match(exp_data, envelope, mass_tol=0.005, ppm_tol=5):
    theo_mz = sorted(envelope.keys())
    theo_probs = array([envelope[m] for m in theo_mz])
    theo_probs /= theo_probs.sum()
    
    if len(exp_data) == 0:
        return 0.0, []
    if isinstance(exp_data[0], tuple) and len(exp_data[0]) == 2 and isinstance(exp_data[0][1], tuple):
        # called from greedy_envelope_assignment, exp_data is [(idx, (mz, abs_int, rel_int)), ...]
        orig_indices, peak_data = zip(*exp_data)
        exp_mz, abs_int, rel_int = zip(*peak_data)
    else:
        # called directly, exp_data is [(mz, abs_int, rel_int), ...]
        orig_indices = list(range(len(exp_data)))
        exp_mz, abs_int, rel_int = zip(*exp_data)
    
    matched_rel_ints = []
    matched_indices = []
    for mz in theo_mz:
        closest = None
        min_diff = float('inf')
        closest_idx = None
        for idx, (obs_mz, obs_abs_int, obs_rel_int) in enumerate(zip(exp_mz, abs_int, rel_int)):
            diff = abs(obs_mz - mz)
            ppm_dev = calculate_ppm_dev(obs_mz, mz)
            if diff < mass_tol and ppm_dev < ppm_tol and diff < min_diff:
                closest = obs_rel_int
                min_diff = diff
                closest_idx = orig_indices[idx]
        matched_rel_ints.append(closest if closest is not None else 0.0)
        matched_indices.append(closest_idx if closest_idx is not None else None)

    matched_rel_ints = array(matched_rel_ints, dtype=float)
    if matched_rel_ints.sum() == 0:
        return 0.0, []
    matched_rel_ints /= matched_rel_ints.sum()

    return cosine_similarity(theo_probs, matched_rel_ints), matched_indices

def greedy_envelope_assignment(
        exp_data, possible_fragments, mass_tol=0.005, ppm_tol=5, min_score=0.9):
    # start with original indices
    remaining_peaks = list(enumerate(exp_data))
    assignments = []
    while True:
        best_match = None
        best_indices = None
        best_score = 0
        for formula, envelope in possible_fragments.items():
            score, matched_orig_indices = score_envelope_match(remaining_peaks, envelope, mass_tol, ppm_tol)
            if score > best_score and score >= min_score and any(idx is not None for idx in matched_orig_indices):
                best_score = score
                best_match = {
                    'formula': formula,
                    'score': score,
                    'matched_indices': matched_orig_indices
                }
                best_indices = matched_orig_indices
        if best_match is None:
            break
        assignments.append(best_match)
        # remove assigned peaks by original index
        used_indices = set(idx for idx in best_indices if idx is not None)
        remaining_peaks = [peak for peak in remaining_peaks if peak[0] not in used_indices]
    return assignments

#### hybrid scoring? ####

# basically we need to boost scoring for molecular ion peaks
# these should be high-confidence given just a sufficient ppm match
# just envelope matching is a problem especially for rich envelopes (Br, Cl) 

def get_key_peaks(envelope, prob_threshold=0.1, max_peaks=3):
    sorted_peaks = sorted(envelope.items(), key=lambda x: -x[1])
    return [(mz, prob) for mz, prob in sorted_peaks 
            if prob >= prob_threshold][:max_peaks]

def peak_match_quality(exp_mz, theo_mz, ppm_tol=5, sigma_ppm=3):
    ppm_dev = calculate_ppm_dev(exp_mz, theo_mz)
    if abs(ppm_dev) > ppm_tol:
        return 0.0
    return scipy_norm.pdf(ppm_dev, loc=0, scale=sigma_ppm) / scipy_norm.pdf(0, loc=0, scale=sigma_ppm)

def hybrid_scoring(
        exp_data, envelope, mol_ion_mz=None, 
        mass_tol=0.005, ppm_tol=5, prob_threshold=0.3):
    # we only use peak scoring for envelopes that contain the molecular ion peak
    cos_score, matched_indices = score_envelope_match(exp_data, envelope, mass_tol, ppm_tol=ppm_tol)
    best_score = cos_score  # always default to cos_score
    
    # when called by the greedy function, we enumerate the peaks
    # so we will have an idx (mz, abs_int, rel_int) format
    if isinstance(exp_data[0], tuple) and len(exp_data[0]) == 2 and isinstance(exp_data[0][1], tuple):
        # [(idx, (mz, abs_int, rel_int)), ...]
        orig_indices, peak_data = zip(*exp_data)
        exp_mzs = [p[0] for p in peak_data]
    else:
        # [(mz, abs_int, rel_int), ...]
        exp_mzs = [p[0] for p in exp_data]

    if mol_ion_mz is not None:
        # check first whether the mol_ion we provide is present in the envelope
        # only then do we want to start looking for peak matches 
        envelope_has_mol_ion = any(abs(calculate_ppm_dev(mol_ion_mz, theo_mz)) < ppm_tol for theo_mz in envelope.keys())
        if envelope_has_mol_ion:
            possible_mol_ions = [k for k, v in envelope.items() if v > prob_threshold]
            for mol_ion_theo_mz in possible_mol_ions:
                if abs(calculate_ppm_dev(mol_ion_mz, mol_ion_theo_mz)) < ppm_tol:
                    for idx, exp_mz in enumerate(exp_mzs):
                        if abs(calculate_ppm_dev(exp_mz, mol_ion_theo_mz)) < ppm_tol:
                            ion_score = peak_match_quality(exp_mz, mol_ion_theo_mz, ppm_tol)
                            best_score = max(cos_score, ion_score)
                            break
    return best_score, matched_indices

def greedy_envelope_hybrid(
        exp_data, possible_fragments, mol_ion_mz,
        mass_tol=0.005, ppm_tol=5, min_score=0.9):
    # start with original indices
    remaining_peaks = list(enumerate(exp_data))
    assignments = []
    while True:
        best_match = None
        best_indices = None
        best_score = 0
        for formula, envelope in possible_fragments.items():
            # scoring is exactly the same as envelope matching
            # except if we encounter envelope that contains molecular ion 
            # if we find the molecular ion in exp data, that is used for scoring
            # we fallback to cos scoring if that still gives a better score
            score, matched_orig_indices = hybrid_scoring(remaining_peaks, envelope, mol_ion_mz, ppm_tol)
            if score > best_score and score >= min_score and any(idx is not None for idx in matched_orig_indices):
                best_score = score
                best_match = {
                    'formula': formula,
                    'score': score,
                    'matched_indices': matched_orig_indices
                }
                best_indices = matched_orig_indices
        if best_match is None:
            break
        assignments.append(best_match)
        # remove assigned peaks by original index
        used_indices = set(idx for idx in best_indices if idx is not None)
        remaining_peaks = [peak for peak in remaining_peaks if peak[0] not in used_indices]
    return assignments

def get_charge_annotation(adduct):
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

def annotate_from_assignments(exp_data, possible_fragments, assignments, adduct=None):
    annotation = []
    charge_annotation = get_charge_annotation(adduct)
    for match in assignments:
        envelope = possible_fragments[match['formula']]
        theo_mz_list = sorted(envelope.keys())
        for i, idx in enumerate(match['matched_indices']):
            if idx is not None:
                annotation.append({
                    'peak_mz': exp_data[idx][0],
                    'formula': match['formula'] + charge_annotation,
                    'theoretical_mz': theo_mz_list[i],
                    'ppm_dev': round(calculate_ppm_dev(theo_mz_list[i], exp_data[idx][0]), 5)
                })
    return sorted(annotation, key=lambda x: x['peak_mz'])

# need this if we annotate from postComp --- but we probably shouldn't do that
#def rewrap_ms2(ms2_peaks):
#    spectrum = []
#    lines = ms2_peaks.strip().split('\n')
#    for line in lines[1:]:
#        parts = line.strip().split()
#        if len(parts) == 3:
#            mz = float(parts[0])
#            intensity = int(parts[1])
#            rel_intensity = int(parts[2])
#            spectrum.append((mz, intensity, rel_intensity))
#    return spectrum

# ---- NEW CANDIDATE FORMULA GENERATION ----

# we need alternatives to brute force - it is too insane (for large cpds espec)
# can we...
# generate all possible connected atom subsets (subgraphs)
# create the corresponding molecule (substructure)

# We can't do this either. It's too rough as the molecule grows large.
# Try BRICS.

def get_connected_subgraphs_old(mol, min_size=2, max_size=None):
    n_atoms = mol.GetNumAtoms() # get n atoms
    if max_size is None:
        max_size = n_atoms
    seen = set() # keep track of unique fragments
    fragments = [] # store submol objects
    size_range = list(range(min_size, max_size))
    # iterate over fragment sizes
    # for each possible, form 2 up to max_size (won't include full molecule)
    for size in size_range:
        # for each combination of atom indices in the mol of the current size
        for idx, atom_indices in enumerate(combinations(range(n_atoms), size)):
            # build a list of all bonds that exist between atoms in current combo
            # check if the subgraph is connected
            bonds = []
            for i in atom_indices:
                for j in atom_indices:
                    if i < j:
                        bond = mol.GetBondBetweenAtoms(i, j)
                        if bond is not None:
                            bonds.append(bond.GetIdx())
            # if there are no bonds in the current combo, move on
            if not bonds:
                continue
            # if there are bonds, this RDKit function extracts the submol
            # defined by those bonds
            submol = Chem.PathToSubmol(mol, bonds)
            if submol.GetNumAtoms() == size:
                smiles = Chem.MolToSmiles(submol)
                if smiles not in seen:
                    seen.add(smiles)
                    fragments.append(submol)
    formulas = [Chem.rdMolDescriptors.CalcMolFormula(frag) for frag in fragments]
    # we also do want the full molecule for candidate formulas list...
    formulas.append(Chem.rdMolDescriptors.CalcMolFormula(mol))
    return formulas

def get_connected_subgraphs(mol, min_size=2, max_size=None, max_total_frags=500,
                            max_samples_per_size=250):
    n_atoms = mol.GetNumAtoms() # get n atoms
    if max_size is None:
        max_size = n_atoms
    seen = set() # keep track of unique fragments
    fragments = [] # store submol objects
    # ---- CAN WE LIMIT SUBGRAPHING ----
    size_range = list(range(min_size, max_size))
    max_per_size = max_total_frags // len(size_range)
    # iterate over fragment sizes
    # for each possible, form 2 up to max_size (won't include full molecule)
    for size in size_range:
        # trying to implement limitations on subgraphing...
        size_count = 0
        # for each combination of atom indices in the mol of the current size
        for atom_indices in islice(combinations(range(n_atoms), size), max_samples_per_size):
            # build a list of all bonds that exist between atoms in current combo
            # check if the subgraph is connected
            bonds = []
            for i in atom_indices:
                for j in atom_indices:
                    if i < j:
                        bond = mol.GetBondBetweenAtoms(i, j)
                        if bond is not None:
                            bonds.append(bond.GetIdx())
            # if there are no bonds in the current combo, move on
            if not bonds:
                continue
            # if there are bonds, this RDKit function extracts the submol
            # defined by those bonds
            submol = Chem.PathToSubmol(mol, bonds)
            if submol.GetNumAtoms() == size:
                smiles = Chem.MolToSmiles(submol)
                if smiles not in seen:
                    seen.add(smiles)
                    fragments.append(submol)
                    size_count += 1
                    if size_count >= max_per_size:
                        print(f'reached max n fragments for {size}, moving on')
                        break
    formulas = [Chem.rdMolDescriptors.CalcMolFormula(frag) for frag in fragments]
    # we also do want the full molecule for candidate formulas list...
    formulas.append(Chem.rdMolDescriptors.CalcMolFormula(mol))
    return formulas

def formulas_to_counts(formulas, adduct=None, min_mass=50, iso_cutoff=0.01):
    formula_dict = {}
    for formula in formulas:
        mode = get_charge(adduct)
        # we do this --- if it's not +H, -H, we apply both the adduct AND
        # the baseline +H or -H so we cover all possibilities.
        all_adducts = (adduct, '[M+H]+') if mode == 1 else (adduct, '[M-H]-')
        for a in set(all_adducts):
            # count-ify the formula and apply adduct
            # BUT ONLY IF THE FORMULA IS ALREADY UNCHARGED! right?
            if ('+' not in formula) and ('-' not in formula):
                current_counts = apply_adduct(parse_formula(formula), a)
            else:
                current_counts = parse_formula(formula)
            if min(current_counts.values()) < 0:
                pass # if adduct addition brings us below 0 of any count, skip
            current_formula = regenerate_formula_hill(current_counts)
            current_massDist = mass_IsoSpecPy(current_formula, adduct)
            filtered_dist = {m: p for m, p in current_massDist.items() if p >= iso_cutoff}
            # mass of most prevalent fragment decides whether to filter
            if filtered_dist and max(filtered_dist, key=filtered_dist.get) >= min_mass and current_formula not in formula_dict:
                formula_dict[current_formula] = filtered_dist
    return formula_dict

# ---- INCORPORATING BRICS ----

def clean_dummy_from_formula(formula):
    # dummy artifacts are * or *(number)
    return re.sub(r'\*\d*', '', formula)

def atom_count_from_mol(mol):
    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return sum(parse_formula(formula).values())

def get_BRICS(parent_mol):
    break_mol = BRICS.BreakBRICSBonds(parent_mol)
    frags = Chem.GetMolFrags(break_mol, asMols=True)
    if len(frags) > 1:
        for frag in frags:
            for atom in frag.GetAtoms():
                atom.SetAtomMapNum(0)
        return {Chem.MolToSmiles(frag, True):atom_count_from_mol(frag) for frag in frags}
    else:
        return {}
    
def collect_fragment_formulas_BRICS(data):
    formulas = []
    if data.get('smiles'):
        # get parent mol and store smiles & atom count
        parent_mol = Chem.MolFromSmiles(data['smiles'])
        smiles_dict = {data['smiles']:atom_count_from_mol(parent_mol)}
        # get brics fragments
        BRICS_dict = get_BRICS(parent_mol)
        if len(BRICS_dict) > 0:
            smiles_dict.update(BRICS_dict)
        # update the list of formulas with what we have so far
        for smiles in smiles_dict:
            temp_mol = Chem.MolFromSmiles(smiles)
            formulas.append(Chem.rdMolDescriptors.CalcMolFormula(temp_mol))
        # now find subgraph fragments for parent + BRICS that are of OK size
        for smiles, atom_count in smiles_dict.items():
            if atom_count > 6 and atom_count < 20:
                # print(f'subgraph calculation for entity of length {atom_count}')
                temp_mol = Chem.MolFromSmiles(smiles)
                # we should adjust the minimum size of the subgraphs
                min_submol_size = int(atom_count * 0.66)
                returned_formulas = get_connected_subgraphs(temp_mol, min_submol_size)
                formulas.extend(returned_formulas)
        # clean formulas of dummy artifacts
        formulas = [clean_dummy_from_formula(formula) for formula in formulas]
        # keep uniques
        unique_formulas = sorted(set(formulas))
        return [len(unique_formulas), list(unique_formulas)]
    else:
        return [0, []]

# ---- FINGERPRINTING ----

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

# helpers for MACCS...
def maccs_smarts_from_bits(bits, maccs_dict):
    return [maccs_dict.get(bit, ('?', 0))[0] for bit in bits]

def smarts_to_smiles(smarts):
    mol = Chem.MolFromSmarts(smarts)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    else:
        return None

# extraction functions
def extract_fragments_from_bitinfo(mol, bitInfo):
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
    fp_dict = {}
    for bit, paths in bitPaths.items():
        for bond_indices in paths:
            submol = Chem.PathToSubmol(mol, bond_indices)
            if submol.GetNumAtoms() > 0:
                smiles = Chem.MolToSmiles(submol)
                fp_dict[smiles] = submol.GetNumAtoms()
    # in case we have empty fragments here too...
    cleaned_dict = {entity:n_atoms for entity, n_atoms in fp_dict.items() if n_atoms > 0}
    return cleaned_dict

# fingerprinters  
def get_rdkit_fingerprints(mol):
    # better for linear structures, chains? 'daylight-like'
    fp_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitPaths()
    fp = fp_gen.GetFingerprint(mol, additionalOutput=ao)
    bitPaths = ao.GetBitPaths()
    return extract_fragments_from_bitpaths(mol, bitPaths)

def get_morgan_fingerprints(mol, radius=2, nBits=2048):
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = fp_gen.GetFingerprint(mol, additionalOutput=ao)
    bitInfo = ao.GetBitInfoMap()
    return extract_fragments_from_bitinfo(mol, bitInfo)

def get_maccs_fingerprints(mol, maccs_dict):
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

# BRICS-equivalent for MacFrag (used further down...)
def get_MacFrag(parent_mol):
    # this returns a list of SMILES.
    MacFrags = MacFrag.MacFrag(parent_mol)
    # no need for position cleaning, they do this themselves
    if len(MacFrags) > 1:
        # an annoying thing is that we need atom counts
        # and for that we need mols
        def smiles_to_mol_to_formula_to_count(smiles):
            temp_mol = Chem.MolFromSmiles(smiles)
            atom_count = atom_count_from_mol(temp_mol)
            return atom_count
        return {smiles:smiles_to_mol_to_formula_to_count(smiles) for smiles in MacFrags}
    else:
        return {}

# AN IDEA FOR LATER
# Is to subtract our smaller fingerprint fragment atom counts
# from our larger structures (full molecule and larger MacFrags)
# Subtract and keep new structures that don't result in negative atom counts
# also! add up smaller fragments up to maybe... 80% of total count?

def smiles_to_mol_to_formula_to_count(smiles):
    # ...
    temp_mol = Chem.MolFromSmiles(smiles)
    atom_count = atom_count_from_mol(temp_mol)
    return atom_count

def smiles_to_mol_to_formula_to_count_dict(smiles):
    # ...
    temp_mol = Chem.MolFromSmiles(smiles)
    if temp_mol:
        temp_fm = Chem.rdMolDescriptors.CalcMolFormula(temp_mol)
        atom_counts = parse_formula(temp_fm)
        return atom_counts
    else:
        return None
    
def smiles_to_mol_to_formula(smiles):
    temp_mol = Chem.MolFromSmiles(smiles)
    if temp_mol:
        temp_fm = Chem.rdMolDescriptors.CalcMolFormula(temp_mol)
        return temp_fm if temp_fm else None
    else:
        return None

# ---- MACFRAG ----

def get_MacFrag(parent_mol):
    # this returns a list of SMILES.
    MacFrags = MacFrag.MacFrag(parent_mol)
    # no need for position cleaning, they do this themselves
    if len(MacFrags) > 1:
        return {smiles:smiles_to_mol_to_formula_to_count(smiles) for smiles in MacFrags}
    else:
        return {}
    
# ---- GET! MORE! FRAGMENTS! ----
def add_charge_annotation(formula, charge):
    # sigh.
    if charge == 0:
        return formula
    elif charge == 1:
        return formula + '+'
    elif charge == -1:
        return formula + '-'
    elif charge > 1:
        return formula + f'+{charge}'
    elif charge < -1:
        return formula + f'{-charge}-'
    else:
        return formula
    
def get_more_fragments(smiles_dict, fp_dict, formulas):
    # what we do is --- feed in the MacFrags, and our fingerprints, and
    # our current set of formulas from thus derived
    
    # actually --- first, we just use our full molecule. more madness later.
    # convert smiles to formulas to counts, and subtract all fp_dict counts
    # from our parent molecule count.
    # keep those that result in no less-than-0 atom counts
    
    # get all parent info
    parent_smiles = next(iter(smiles_dict)) # parent always first
    parent_atom_count = smiles_to_mol_to_formula_to_count_dict(parent_smiles)
    parent_charge = Chem.GetFormalCharge(Chem.MolFromSmiles(parent_smiles))      
    for smiles in fp_dict.keys():
        # get all fragment info
        # and make a mol first to check that it's a valid smiles
        fragment_mol = Chem.MolFromSmiles(smiles)
        if not fragment_mol:
            continue
        fragment_atom_count = smiles_to_mol_to_formula_to_count_dict(smiles)
        fragment_charge = Chem.GetFormalCharge(fragment_mol)
        # --- SUBTRACTION BLOCK... ---
        if fragment_atom_count:
            parent_minus_fragment = copy.deepcopy(parent_atom_count)
            valid = True
            for atom, count in fragment_atom_count.items():
                if atom in parent_minus_fragment.keys():
                    parent_minus_fragment[atom] -= count
                    if parent_minus_fragment[atom] < 0:
                        valid = False
                        break
            if not valid:
                continue
            # --- FINALIZATION ---
            minus_fragment_formula = regenerate_formula_hill(parent_minus_fragment)
            comp_charge = 0
            if parent_charge != 0 and fragment_charge == 0:
                # we only care about cases with charged parents and neutral fragments
                # allow the others to be weird/ad hoccy...
                comp_charge = parent_charge
            # Annotate formula with charge
            annotated_formula = add_charge_annotation(minus_fragment_formula, comp_charge)
            if annotated_formula not in formulas:
                formulas.append(annotated_formula)
    return formulas
    
def collect_fragment_formulas_MacFrag(
    data, fingerprint=True, more_fragments=True, subgraph=False, subgraph_range=(8, 32)
):
    formulas = []
    if data.get('smiles'):
        # get parent mol and store smiles & atom count
        parent_mol = Chem.MolFromSmiles(data.get('smiles'))
        parent_atom_count = atom_count_from_mol(parent_mol)
        smiles_dict = {data['smiles']:atom_count_from_mol(parent_mol)}
        
        # fingerprint storage
        fp_dict = {} # store all fingerprints
        
        # --- MACFRAG BLOCK ---
        # get MacFrag (!) fragments
        MacFrag_dict = get_MacFrag(parent_mol)
        if len(MacFrag_dict) > 0:
            smiles_dict.update(MacFrag_dict)
            
        # --- FINGERPRINTING BLOCK ---
        # fingerprinting DOES work for most compounds and can add a decent 
        # chunk of new candidates --- NB though, for some, all fragments are 
        # invalid and the mols generated will be None-objects
        if fingerprint:
            # do all of them for now
            rdkit_fp = get_rdkit_fingerprints(parent_mol)
            if len(rdkit_fp) > 0:
                fp_dict.update(rdkit_fp)
            morgan_fp = get_morgan_fingerprints(parent_mol)
            if len(morgan_fp) > 0:
                fp_dict.update(morgan_fp)
            maccs_fp = get_maccs_fingerprints(parent_mol, MACCS_KEYS)
            if len(maccs_fp) > 0:
                fp_dict.update(maccs_fp)
                
        # update the list of formulas with what we have so far
        for smiles in smiles_dict.keys():
            temp_mol = Chem.MolFromSmiles(smiles)
            if temp_mol is not None:
                formulas.append(Chem.rdMolDescriptors.CalcMolFormula(temp_mol))
        for smiles in fp_dict.keys():
            temp_mol = Chem.MolFromSmiles(smiles)
            if temp_mol is not None:
                formulas.append(Chem.rdMolDescriptors.CalcMolFormula(temp_mol))
        # now find subgraph fragments for parent + MacFrags that are of OK size
        if subgraph: # subgraphing is a problem
            # how about this. we do subgraphing for our parent, all the way from the.
            # we allow a maximum of 1000 fragments, which will be split up across
            # the different size steps
            returned_formulas = get_connected_subgraphs(
                parent_mol, 8, parent_mol.GetNumAtoms(), 200
            )
            formulas.extend(returned_formulas)
            #for smiles, atom_count in smiles_dict.items():
            #    if atom_count > subgraph_range[0] and atom_count < subgraph_range[1]:
            #        # print(f'subgraph calculation for entity of length {atom_count}')
            #        temp_mol = Chem.MolFromSmiles(smiles)
            #        # we should adjust the minimum size of the subgraphs
            #        min_size, max_size = int(atom_count * 0.3), int(atom_count * 0.7)
            #        returned_formulas = get_connected_subgraphs(temp_mol, min_size, max_size)
            #        formulas.extend(returned_formulas)
        # clean formulas of dummy artifacts
        formulas = [clean_dummy_from_formula(formula) for formula in formulas]
        # keep uniques
        unique_formulas = list(sorted(set(formulas)))
        if more_fragments:
            unique_formulas = get_more_fragments(smiles_dict, fp_dict, unique_formulas)
        return unique_formulas
        #return smiles_dict, fp_dict, list(unique_formulas)
    else:
        return []
            
# ---- ANNOTATION FUNCTION ----
def annotate_spectrum(compound, data, fragments='macfrag'):
    spectrum = data.get('ms2_norm', None)
    formula = data.get('molecularFormula', None).strip()
    adduct = data.get('ion_type', None)
    smiles = data.get('smiles', None)
    mode = '+' if get_charge(adduct) == 1 else '-'
    anti_mode = '-' if get_charge(adduct) == 1 else '+'
    # ready check...
    if not (spectrum and formula and adduct and smiles):
        print(f'missing data for {compound} --- cannot annotate ms2')
        return None
    atom_counts = parse_formula(formula)
    atom_counts = apply_adduct(atom_counts, adduct)
    molecular_ion_mz = get_molecular_ion_mz(atom_counts, adduct)
    # brute force --- obsoleted, i think!
    if fragments == 'brute':
        possible_fragments = generate_possible_fragments(atom_counts, adduct)
        assignments = greedy_envelope_hybrid(spectrum, possible_fragments, molecular_ion_mz)
    elif fragments == 'graph':
        mol = Chem.MolFromSmiles(smiles)
        print("Generating fragments...")
        formulas = get_connected_subgraphs(mol)
        print(f"Number of fragments: {len(formulas)}")
        print("Converting formulas to counts...")
        formulas = formulas_to_counts(formulas, adduct) # SUPPLY ADDUCT!
        print("Scoring fragments...")
        assignments = greedy_envelope_hybrid(spectrum, formulas, molecular_ion_mz)
    elif fragments == 'brics':
        pass
    elif fragments == 'macfrag':
        formulas = collect_fragment_formulas_MacFrag(
            data, fingerprint=True, more_fragments=True, subgraph=False)
        
        possible_fragments = formulas_to_counts(formulas, adduct)
        assignments = greedy_envelope_hybrid(spectrum, possible_fragments, molecular_ion_mz)
    
    # to be placed under PK$ANNOTATION:
    # nb - i've noticed that apparently greater mz peaks have been annotated
    # as being BEFORE lower mz peaks. i think? keep a lookout...
    # actually the order was just scrambled. i think. still keep a lookout.
    annotation = []
    if len(assignments) > 0:
        for entry in assignments:
            # first - theoretical mass & formula
            theo_mz_dict = possible_fragments[entry['formula']]
            #theo_mz_dict = formulas[entry['formula']]
            # extract from envelope...
            theo_mz = max(theo_mz_dict.items(), key=lambda x: x[1])[0]
            fragment_formula = str(entry['formula'] + mode)
            
            # then - formula counts and exp mz
            matched_exp_indices = entry['matched_indices']
            matched_exp_indices = [i for i in matched_exp_indices if i is not None]
            annot_fragments = [spectrum[i] for i in matched_exp_indices]
            formula_count = len(annot_fragments)
            exp_mz = annot_fragments[0][0] # always give monoisotopic peak
            
            # ppm.
            ppm = round(calculate_ppm_dev(exp_mz, theo_mz), 2)
            
            # bunch it all up
            annotation.append((theo_mz, fragment_formula, 
                               formula_count, exp_mz, ppm))
    return sorted(annotation, key=lambda x: x[3])

#sd, fp, fms = collect_fragment_formulas_MacFrag(dictionary['Clonidine'], fingerprint=True)

#get_more_fragments(sd, fp, fms)

#collect_fragment_formulas_MacFrag(
#    dictionary['Thiamine'], fingerprint=True, more_fragments=True, subgraph=False)

#### ALRIGHT
# Worked example for fragment generation pipeline including BRICS
#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv')
#mol = Chem.MolFromSmiles(dictionary['Theobromine']['smiles'])
#collect_fragment_formulas_BRICS(dictionary['Sarafloxacin'])
#result = collect_fragment_formulas_MacFrag(dictionary['Deflazacort'], subgraph=True)

#mol = Chem.MolFromSmiles(dictionary['Theobromine']['smiles'])

#dictionary['Theobromine']['frags']

#mol = Chem.MolFromSmiles(dictionary['Deoxycorticosterone']['smiles'])
#smiles_collection = get_morgan_fingerprints(mol)
#for smiles in smiles_collection.keys():
#    temp_mol = Chem.MolFromSmiles(smiles)
#    if temp_mol is not None:
#        print(Chem.rdMolDescriptors.CalcMolFormula(temp_mol))

#len(collect_fragment_formulas_MacFrag(dictionary['Deflazacort'], fingerprint=True)[1])
#len(collect_fragment_formulas_MacFrag(dictionary['Deflazacort'], fingerprint=False)[1])

#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv')
#for i, (compound, data) in enumerate(dictionary.items()):
#    print(f'getting fragments for {compound}, number {i} of {len(dictionary)}')
#    # print(collect_fragment_formulas_BRICS(compound, data))
#    if data.get('smiles'):
#        data['frags'] = collect_fragment_formulas_MacFrag(
#            data, fingerprint=True, more_fragments=True, subgraph=False)
#        data['atom_count'] = sum(parse_formula(data['molecularFormula']).values())

#data_rows = []
#for compound, data in dictionary.items():
#    row = {
#        'name': compound,
#        'monoisotopicMass': data.get('monoisotopicMass'),
#        'molecularFormula': data.get('molecularFormula'),
#        'atom_count': data.get('atom_count'),
#        'frags': data.get('frags')
#    }
#    data_rows.append(row)
    
#df = pd.DataFrame(data_rows)
#df.to_excel('MacFrag_and_fingerprintingMORGAN.xlsx')
    




# IDEA --- använd BRICS först, och se hur många fragment som genereras.
# Sedan, om tillräckligt många cpds genererar åtminstone två fragment,
# som inte är för stora, kan vi applicera grafteoretisk approach på dessa

# USE THIS TO SEE HOW BRICS IS GONNA WORK FOR OUR SET.
#def apply_BRICS(dictionary):
#    for compound, data in dictionary.items():
#        if data.get('smiles'):
#            mol = Chem.MolFromSmiles(data['smiles'])
#            broken_mol = BRICS.BreakBRICSBonds(mol)
#            frags = Chem.GetMolFrags(broken_mol, asMols=True)
#            frag_smiles = [Chem.MolToSmiles(frag, True) for frag in frags]
#            frag_formulas = [Chem.rdMolDescriptors.CalcMolFormula(frag) for frag in frags]
#            # we clean dummies from FORMULAS instead.
#            frag_formulas = [clean_dummy_from_formula(formula) for formula in frag_formulas]
#           data['BRICS_frag_n'] = len(frag_formulas)
#            data['BRICS_frags'] = frag_smiles
#            data['BRICS_formulas'] = frag_formulas
#            data['BRICS_frag_sizes'] = [
#                sum(parse_formula(formula).values()) for formula in frag_formulas
#            ]
#        else:
#            print(f'no SMILES available for {compound}')
#    return dictionary

#dictionary = apply_BRICS(dictionary)

# ONCE WE HAVE OUR FRAGMENTS
# WE SHOULD ALSO REMOVE ATOMS FROM THE PARENT COMPOUND BY ALL THE FRAGMENTS
# AND COVER ANY (PARENT - FRAGMENT) ENTITIES WE ALREADY DO NOT HAVE IN OUR LIST 

#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv', 'internalName')
#mol = Chem.MolFromSmiles(dictionary['Theobromine']['smiles'])  
#formulas = get_connected_subgraphs(mol)
#formulas = formulas_to_counts(formulas, '[M+H]+')
#formulas['C7H9N4O2']
#current_counts = apply_adduct(parse_formula('N2O2'), '[M+H]+')
#current_counts.values()
#dictionary['Theobromine']
#annotate_spectrum('Theobromine', dictionary['Theobromine'])

# caffeine
#formula = 'C8H10N4O2'
#atom_counts = parse_formula(formula)
#atom_counts = apply_adduct(atom_counts, '[M+H]+')
#possible_fragments = generate_possible_fragments(atom_counts, '[M+H]+')
#possible_fragments['C8H11N4O2']
#exp_data = read_spectrum('output/compiler/2025-06-06/pos/MSBNK-ACES_SU-AS000914.txt')
#score_envelope_match(exp_data, possible_fragments['C8H11N4O2'])
#assignments = greedy_envelope_assignment(exp_data, possible_fragments)
#annotate_from_assignments(exp_data, possible_fragments, assignments, '[M+H]+')

#formula = 'C14H11Cl2NO2'
#atom_counts = parse_formula(formula)
#atom_counts = apply_adduct(atom_counts, '[M+H]+')
#possible_fragments = generate_possible_fragments(atom_counts, '[M+H]+')
#exp_data = read_spectrum('output/compiler/2025-06-06/pos/MSBNK-ACES_SU-AS000461.txt')
#score_envelope_match(exp_data, possible_fragments['C14H12Cl2NO2'])
#possible_fragments['C14H12Cl2NO2']
#assignments = greedy_envelope_assignment(exp_data, possible_fragments)
#annotate_from_assignments(exp_data, possible_fragments, assignments, '[M+H]+')

#formula = 'C9H13N2O2+'
#atom_counts = parse_formula(formula)
#atom_counts = apply_adduct(atom_counts, '[M]+')
#possible_fragments = generate_possible_fragments(atom_counts)
#exp_data = read_spectrum('output/compiler/pos/MSBNK-ACES_SU-AS0001133.txt')
#score_envelope_match(exp_data, possible_fragments['C9H13N2O2'])
#assignments = greedy_envelope_assignment(exp_data, possible_fragments)
#annotate_from_assignments(exp_data, possible_fragments, assignments, '[M+H]+')

# TO DO
# chlorinated compounds with missing significant envelope peaks scoring low
# e.g. diclofenac molecular ion peak
# double charge support