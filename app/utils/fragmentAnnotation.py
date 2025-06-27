# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 18:00:22 2025

@author: Jakob
"""
import utils.genericUtilities as gu
import utils.Macfrag as MacFrag
import re
import itertools
import IsoSpecPy as iso
from numpy import array, dot
from numpy.linalg import norm
from scipy.stats import norm as scipy_norm
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import BRICS # TRY BRICS TOMORROW.
from itertools import combinations

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

def get_connected_subgraphs(mol, min_size=2, max_size=None):
    n_atoms = mol.GetNumAtoms() # get n atoms
    print(n_atoms)
    if max_size is None:
        max_size = n_atoms
    seen = set() # keep track of unique fragments
    fragments = [] # store submol objects
    # iterate over fragment sizes
    # for each possible, form 2 up to max_size (won't include full molecule)
    for size in range(min_size, max_size):
        print(f'  Fragment size: {size}')
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
                    print(smiles)
                    seen.add(smiles)
                    fragments.append(submol)
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
            current_counts = apply_adduct(parse_formula(formula), a)
            if min(current_counts.values()) < 0:
                pass # if adduct addition brings us below 0 of any count, skip
            current_formula = regenerate_formula_hill(current_counts)
            current_massDist = mass_IsoSpecPy(current_formula, adduct)
            filtered_dist = {m: p for m, p in current_massDist.items() if p >= iso_cutoff}
            # mass of most prevalent fragment decides whether to filter
            if filtered_dist and max(filtered_dist, key=filtered_dist.get) >= min_mass and current_formula not in formula_dict:
                formula_dict[current_formula] = filtered_dist
    return formula_dict

# ---- BRIGS & MACFRAG ----

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
    
def collect_fragment_formulas_MacFrag(data, subgraph=True):
    formulas = []
    if data.get('smiles'):
        # get parent mol and store smiles & atom count
        parent_mol = Chem.MolFromSmiles(data.get('smiles'))
        parent_atom_count = atom_count_from_mol(parent_mol)
        smiles_dict = {data['smiles']:atom_count_from_mol(parent_mol)}
        # get MacFrag (!) fragments
        MacFrag_dict = get_MacFrag(parent_mol)
        if len(MacFrag_dict) > 0:
            smiles_dict.update(MacFrag_dict)
        # update the list of formulas with what we have so far
        for smiles in smiles_dict:
            temp_mol = Chem.MolFromSmiles(smiles)
            formulas.append(Chem.rdMolDescriptors.CalcMolFormula(temp_mol))
        # now find subgraph fragments for parent + MacFrags that are of OK size
        if subgraph:
            pass
        # clean formulas of dummy artifacts
        formulas = [clean_dummy_from_formula(formula) for formula in formulas]
        # keep uniques
        unique_formulas = sorted(set(formulas))
        print(unique_formulas)
        return [len(unique_formulas), list(unique_formulas)]
    else:
        return [0, []]

# ---- ANNOTATION FUNCTION ----

def annotate_spectrum(compound, data, fragments='brute'):
    spectrum = data.get('ms2_norm', None)
    formula = data.get('molecularFormula', None).strip()
    adduct = data.get('ion_type', None)
    smiles = data.get('smiles', None)
    mode = '+' if get_charge(adduct) == 1 else '-'
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
        pass
    
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

#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv', 'internalName')
#mol = Chem.MolFromSmiles(dictionary['Theobromine']['smiles'])  
#formulas = get_connected_subgraphs(mol)
#formulas = formulas_to_counts(formulas, '[M+H]+')
#formulas['C7H9N4O2']
#current_counts = apply_adduct(parse_formula('N2O2'), '[M+H]+')
#current_counts.values()
#dictionary['Theobromine']
#annotate_spectrum('Acetopromazine', dictionary['Acetopromazine'])

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