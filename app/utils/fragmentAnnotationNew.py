# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 23:22:43 2025

@author: Jakob
"""

import utils.genericUtilities as gu
# import utils.compilerUtilities as cu --- imports this file, which can give errors
# we dont need it anymore though
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
import time
from functools import partial

ELECTRON_MASS = 0.000548579909070
PROTON_MASS = 1.007276466621
ELEMENTS = {
    'C': 12.000000000,
    'H': 1.00782503224,
    'N': 14.00307400443,
    'O': 15.99491461957,
    'P': 30.97376199842,
    'S': 31.9720711744,
    # halogens
    'F': 18.99840316273,
    'Cl': 34.968852682,
    'Br': 78.9183376,
    'I': 126.9044684,
    # others
    'Na': 22.9897692820,
    'K': 38.9637064864,
    'Hg': 201.970652,
    'As': 74.921594,
    'Au': 196.966570,
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

ATOM_NAMES = sorted(ELEMENTS.keys(), key=lambda x: -len(x))
ATOM_PATTERN = r'(' + '|'.join(ATOM_NAMES) + r')(\d*)'

# REVISION 260210
# Recursive formula generation limited by parent --- simple.

def parse_formula(formula):
    """
    Helper, reads a molecular formula and returns a dictionary with atom counts.
    Atoms are stored as keys, counts as values.
    """
    atom_counts = {}
    for (atom, count) in re.findall(ATOM_PATTERN, formula, re.IGNORECASE):
        atom_counts[atom] = atom_counts.get(atom, 0) + int(count or 1)
    return atom_counts

def apply_adduct(atom_counts, adduct):
    for atom, increment in ADDUCTS[adduct].items():
        atom_counts[atom] = atom_counts.get(atom, 0) + increment
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

def get_charge(adduct):
    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        value = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * value
    return 0 # if no charge    

def get_neutral_mass(atom_counts):
    return sum(atom_counts.get(elem, 0) * mass for elem, mass in ELEMENTS.items())

def get_charged_mass(atom_counts, charge):
    return (sum((atom_counts.get(elem, 0) * mass) for elem, mass in ELEMENTS.items()) - (charge * ELECTRON_MASS)) / abs(charge)

def is_subformula(parent_counts, fragment_counts):
    return all(fragment_counts.get(elem, 0) <= parent_counts.get(elem, 0) for elem in fragment_counts)

def get_ppm_dev(exp_mz, ref_mz):
    return round(((ref_mz-exp_mz) / exp_mz) * 1e6, 2)

def get_hc_ratio(atom_counts):
    # Expect H/C ratios between 0.2-3.1
    # At least within 0.1-6
    # as per Fiehn and Kind
    # With appropriate caveats for very small fragments
    h_count, c_count = atom_counts.get('H', 0), atom_counts.get('C', 0)
    return h_count / c_count if h_count and c_count else float('Inf')

def generate_iso_pattern(formula, charge):
    # This should be enough...
    massDist = {}
    isoDist = iso.IsoTotalProb(formula=formula, prob_to_cover=0.999)
    if charge != 0:
        # This has been tested and generates doubly charged isotope patterns fine
        massDist = {((mass - (charge * ELECTRON_MASS))/ abs(charge)): prob for mass, prob in isoDist}
    else:
        massDist = {mass: prob for mass, prob in isoDist}
    return massDist
    
def generate_subformulas(data, ppm_tol=10):
    # ----- SETUP -----
    parent_formula = data.get('molecularFormula', None)
    adduct = data.get('ion_type', None)
    is_natively_charged = '[M]' in adduct
    peak_list = [peak for (peak, _) in data.get('ms2_data', None)]
    if not parent_formula or not adduct or not peak_list:
        return None
    charge = get_charge(adduct)
    is_singly_charged = abs(charge) == 1 # Double charge implementation...?
    parent_atoms = apply_adduct(parse_formula(parent_formula), adduct)
    molecular_ion_formula = regenerate_formula_hill(parent_atoms)
    neutral_parent_atoms = parse_formula(parent_formula)
    neutral_parent_formula = regenerate_formula_hill(neutral_parent_atoms)
    # parent_atoms = {elem: 2 * parent_atoms.get(elem) for elem in parent_atoms} ...? Mah.
    # parent_atoms['F'] = 2
    
    # Need this for now to not get stuck on super-large compounds
    start_time = time.time()
    deadline = start_time + 120
    
    # Do it once for all peaks, sounds more efficient...
    mass_windows = {target_mz: target_mz * ppm_tol * 1e-6 for target_mz in peak_list}
    peak_candidates = {mz: {} for mz in peak_list}
    
    #candidates = []  # (atoms, matched_mz)

    # Ggenerate subformulas within parent atom limits
    def recurse(current, remaining_elements):
        # Runtime check
        if time.time() > deadline:
            return None
        
        # We will have atom compositions here that include adduct atoms
        # So, we just have to add or remove an electron mass according to mode
        
        # Now with a double charge implementation
        current_mass = get_charged_mass(current, charge / abs(charge))
        current_mass_dbl_charge = get_charged_mass(current, charge) if not is_singly_charged else None
        current_formula = regenerate_formula_hill(current)
        is_parent = current_formula == molecular_ion_formula
        
        # Do this...? To avoid some obvious boo-boos.
        def plausible_loss(current, parent_atoms):
            diff = {}
            all_elems = set(parent_atoms) | set(current)
            for elem in all_elems:
                dp = parent_atoms.get(elem, 0)
                dc = current.get(elem, 0)
                diff[elem] = dp - dc
                
            # Find what elements differ
            nonzero_diff_elems = [e for e, d in diff.items() if d != 0]

            # _Only_ a H loss of 2 or less should not be happening
            if nonzero_diff_elems == ['H'] and abs(diff['H']) <= 2:
                return False
            # Add more later maybe...
            return True
        
        is_plausible_loss = plausible_loss(current, parent_atoms)
        if not is_parent:
            if not is_plausible_loss:
                return
        
        if current_mass > max(peak_list) * 1.02 and is_singly_charged:  # 2% above highest peak
            return # Only if singly charged though...!
        
        # Check if current matches ANY peak within its window
        for peak_idx, target_mz in enumerate(peak_list):
            # If current mass falls within the window...
            if abs(current_mass - target_mz) <= mass_windows[target_mz]:
                # We store it as a candidate --- but first!
                # Check if it has isotopic pattern peak matches that could be claimed
                # On second thought --- get candidates for everything first, do isotopic stuff later
                current_formula = regenerate_formula_hill(current)
                # It happens sometimes that a co-isolation with near-proton difference
                # will be interpreted as being the "neutral" parent formula
                # Which should not be possible of course
                if (current_formula == neutral_parent_formula and not is_natively_charged):
                    continue
                # Deduplicate on formula
                peak_candidates[target_mz][current_formula] = {
                    'peak_idx': peak_idx,
                    'mass': current_mass,
                    'error_ppm': get_ppm_dev(target_mz, current_mass),
                    'atom_counts': current.copy(),
                    'iso_pattern': generate_iso_pattern(current_formula, charge),
                    # Evaluate stuff. Initiate variables here...
                    'iso_score': 0.0,
                    'iso_peaks': [],
                    'hc_ratio': get_hc_ratio(current.copy()),
                    # Store flag for molecular ion
                    'is_parent': True if current_formula == molecular_ion_formula else False,
                    # Store this if we start dabbling with non-single charges
                    'charge': charge / abs(charge)
                }
                break # Move to new subformula after match
                
            # Double charge implementation...?
            if current_mass_dbl_charge and not is_singly_charged:
                n = current.get('N', 0)
                s = current.get('S', 0)
                p = current.get('P', 0)
                # Only consider (at least two) quaternary ammonium groups, or sulfonium/phosphonium
                atom_req = (n + s + p) >= 2
                if abs(current_mass_dbl_charge - target_mz) <= mass_windows[target_mz] and atom_req:
                    # print('considering doubly charged fragment')
                    # Do this again
                    current_formula = regenerate_formula_hill(current)
                    # print(current_formula)
                    # And this although maybe irrelevant
                    if (current_formula == neutral_parent_formula and not is_natively_charged):
                        continue
                    # Without thinking too much about it there should be no clashes possible here
                    peak_candidates[target_mz][current_formula] = {
                        'peak_idx': peak_idx,
                        'mass': current_mass_dbl_charge,
                        'error_ppm': get_ppm_dev(target_mz, current_mass_dbl_charge),
                        'atom_counts': current.copy(),
                        'iso_pattern': generate_iso_pattern(current_formula, charge),
                        # Evaluate stuff. Initiate variables here...
                        'iso_score': 0.0,
                        'iso_peaks': [],
                        'hc_ratio': get_hc_ratio(current.copy()),
                        # Store flag for molecular ion
                        'is_parent': True if current_formula == molecular_ion_formula else False,
                        # Store this if we start dabbling with non-single charges
                        'charge': charge
                    }
                    break # Move to new subformula after match
        
        if not remaining_elements:
            return
        
        # Branch on next element
        elem = remaining_elements[0]
        max_count = parent_atoms.get(elem, 0)
        for count in range(max_count + 1):
            was_present = elem in current
            prev_value = current.get(elem, None)
            current[elem] = count
            recurse(current, remaining_elements[1:])
            if not was_present:
                del current[elem]
            else:
                current[elem] = prev_value

    recurse({}, list(parent_atoms.keys()))
    
    return peak_candidates

def match_iso_patterns(data, peak_candidates, ppm_leeway=10, min_iso_prob=0.01):
    peak_data = data.get('ms2_data', None)
    if not peak_data:
        return None
    
    tot_intensity = sum(intensity for _, intensity in peak_data) # Use % intensity to go with the isotopic probabilities
    sorted_peaks = sorted(mz for mz, _ in peak_data)
    
    # Convert peak_data to dict for fast lookup: {mz: (intensity, available)}
    peak_lookup = {mz: {'rel_intensity': intensity / tot_intensity, 'claimed': False} 
                   for mz, intensity in peak_data}
    
    for target_mz in sorted_peaks:
        if target_mz not in peak_candidates:
            continue
            
        candidates = peak_candidates[target_mz]
        
        for formula, candidate in candidates.items():
            # We have to come to a decision which formula candidate is
            # the best one, if we have multiple
            # But maybe we can do that at a later point
            iso_pattern = candidate.get('iso_pattern', {})
            charge = candidate.get('charge', 0)
            # Index of "base" fragment peak is stored in here, use it
            peak_idx = candidate.get('peak_idx')
            if not iso_pattern:
                continue
                
            # Set up the theoretical isotopic envelope
            # We already have it, but prune it to sufficiently abundant peaks onöy
            theo_peaks = []
            # Only include above-X%-abundant isotopic peaks
            # Using 0.01 for now, 1%
            # This will include the "M" fragment peak as well, which we want
            for theo_mz, prob in iso_pattern.items():
                if prob >= min_iso_prob:
                    theo_peaks.append((theo_mz, prob))
            
            # First, find experimental peaks that correspond to iso peaks
            # Maybe we could enter the "M" fragment peak here already
            iso_assignments = []
            
            for theo_idx, (theo_mz, theo_prob) in enumerate(theo_peaks):
                # Select peaks from the experimental data within the window of the current isotopic peak
                # Include the "base" fragment peak in this, for cosine scoring below
                candidate_peaks = [p_mz for p_mz in sorted_peaks[peak_idx:] if abs(p_mz - theo_mz) <= (theo_mz * ppm_leeway * 1e-6)]
                
                best_match = None
                best_score = float('inf')
                
                for exp_mz in candidate_peaks:
                    # Calculate ppm error between current iso peak and experimental
                    ppm_error = abs((theo_mz - exp_mz) / exp_mz * 1e6)
                    # The theo - exp abundance should be as close to 0 as possible
                    # E.g., theo prob 3%, exp abundance 1.5%, diff = 1.5%
                    # Or, theo prob 1.5%, exp abundance 3%, diff = -1.5%
                    score = abs(theo_prob - peak_lookup[exp_mz]['rel_intensity'])
                    
                    if score < best_score:
                        best_score = score
                        best_match = (exp_mz, ppm_error)
                
                if best_match:
                    exp_mz, ppm_error = best_match
                    # Store exp_mz, rel_intensity, corresponding theo_idx --- and also, theo_mz and ppm dev is nice
                    # ALSO MUST ADD EXP_IDX!
                    iso_assignments.append((exp_mz, peak_lookup[exp_mz]['rel_intensity'], sorted_peaks.index(exp_mz),
                                            theo_idx, theo_mz, ppm_error, charge))
                    
                else:
                    # If no best match, we have to store an empty slot for the corresponding theo_idx peak
                    iso_assignments.append((None, 0, None, theo_idx, None, float('Inf'), 0))
            
            # Then, align and score
            # Check if we have any actual matches in the iso assignments
            # Don't include the "M" peak in that check, as that should always not be None
            if any(entry[0] is not None for entry in iso_assignments[1:]): 
                # Create aligned intensity vectors
                # They should be aligned by default since we add empty entries for no-matches to iso_assignments                  
                theo_int = np.array([theo_prob for _, theo_prob in theo_peaks], dtype=float)
                exp_int = np.array([exp_int for _, exp_int, _, _, _, _, _ in iso_assignments], dtype=float)
                
                #print(f"{target_mz}: {formula}")
                #print(theo_int)
                #print(exp_int)
                
                if theo_int.size != exp_int.size:
                    cosine_sim = 0.0
                else:
                    # Avoid division by zero
                    if theo_int.sum() == 0 or exp_int.sum() == 0:
                        cosine_sim = 0.0
                    else:
                        theo_norm = theo_int / norm(theo_int)    
                        exp_norm = exp_int / norm(exp_int)
                        cosine_sim = float(np.dot(theo_norm, exp_norm))
                
                candidate['iso_score'] = cosine_sim
                # This should work? iso_assignment peaks are exp peaks, if they are not none
                # And we are working with the "base" formula match index, which if we add the 
                # isotopic index to it ("M" is idx 0, etc), should be the corresponding exp peak index
                candidate['iso_peaks'] = [
                    (exp_mz, exp_idx, theo_mz, round(ppm_error, 2)) for exp_mz, _, exp_idx, _, theo_mz, ppm_error, _ in iso_assignments if exp_mz is not None and exp_idx > peak_idx
                    ]
            else:
                candidate['iso_score'] = 0.0
                candidate['iso_peaks'] = []
    
    # Should now be updated with iso_score and peaks for each formula-fragment match
    return peak_candidates          

def finalize_annotation(data, peak_candidates, molecular_ion_formula=None):
    # ----- SETUP -----
    parent_formula = data.get('molecularFormula', None)
    adduct = data.get('ion_type', None)
    peak_data = data.get('ms2_data', None)
    
    if not parent_formula or not adduct or not peak_data:
        return None
    
    # We need to find if this formula is present in the peak candidates,
    # and if it is, we need to use the peak_idx stored in there
    molecular_ion_formula = regenerate_formula_hill(apply_adduct(parse_formula(parent_formula), adduct))
    # We need to keep track of the parent ion index so it is not claimed as an isotopic peak
    parent_idx = None
    for target_mz, candidates in peak_candidates.items():
        if molecular_ion_formula in candidates:
            parent_idx = candidates[molecular_ion_formula]['peak_idx']
            break
    
    # We store [(mz, matched_formula, matched_mz, ppm_error, is_isotopic, is_parent, charge)]
    # MAYBE ADD CHARGE AS WELL? Yes.
    result = [(mz, None, None, None, False, False, 0) for mz, _ in peak_data]
    claimed_isotopic = set() # Keep track of peaks claimed as isotopic

    peak_indices = sorted(range(len(peak_data)), key=lambda i: peak_data[i][0])
    
    for peak_idx in peak_indices:
        exp_mz = peak_data[peak_idx][0]
        
        # Skip if already claimed as isotopic
        if peak_idx in claimed_isotopic or result[peak_idx][4]:
            continue
            
        # We already store our peak candidates by exp_mz
        candidates = peak_candidates.get(exp_mz, {})
        
        if not candidates:
            continue
            
        # Single candidate
        if len(candidates) == 1:
            formula, candidate = next(iter(candidates.items()))
            result[peak_idx] = (
                exp_mz, formula, round(candidate['mass'], 5), candidate['error_ppm'], 
                False, formula == molecular_ion_formula, candidate['charge']
            )
        else:
            # Multiple candidates, gotta score somehow...
            scored_candidates = []
            for formula, candidate in candidates.items():
                iso_score = candidate.get('iso_score', 0.0)
                ppm_error = abs(candidate.get('error_ppm', 0.0))
                # This needs some work
                total_score = iso_score * 0.9 + (1.0 - ppm_error/50) * 0.1
                scored_candidates.append((total_score, formula, candidate))
            
            if scored_candidates:
                _, best_formula, best_candidate = max(scored_candidates)
                result[peak_idx] = (
                    exp_mz, best_formula, round(best_candidate['mass'], 5), 
                    best_candidate['error_ppm'], False, 
                    best_formula == molecular_ion_formula,
                    best_candidate['charge']
                )
        
        # Claim isotopic peaks
        assigned_formula = result[peak_idx][1]
        # The parent can have isotopic peaks being assigned beyond its mz
        # What CANNOT be allowed is that the actual parent ion peak is
        # claimed by a prior peak as part of that peaks isotopic pattern
        if assigned_formula: 
            matched_iso_envelope = candidates[assigned_formula].get('iso_peaks', [])
            charge = candidates[assigned_formula].get('charge', 0)
            #print(matched_iso_envelope)
            #print(candidates[assigned_formula].get('iso_score', []))
            if (matched_iso_envelope and candidates[assigned_formula].get('iso_score', 0.0) > 0.7):
                for iso_mz, exp_idx, theo_mz, iso_ppm_error in matched_iso_envelope:
                    print(f"{iso_mz} - {exp_idx} - {theo_mz}")
                    if (exp_idx not in claimed_isotopic and not result[exp_idx][4] and exp_idx != parent_idx):               
                        result[exp_idx] = (
                            iso_mz, assigned_formula, round(theo_mz, 5), iso_ppm_error, True, False, charge
                        )
                        claimed_isotopic.add(exp_idx)
    
    return result

def format_formula(data, formula, charge):
    # Now supporting double charges...
    def format_charge(charge):
        n = abs(charge)
        if n == 1:
            return '+' if charge > 0 else '-'
        else:
            return f'+{n}' if abs(charge) > 0 else f'-{n}'
        
    if formula and charge:
        return f'[{formula}]{format_charge(charge)}'
    
    # Should be obsolete now...
    adduct = data.get('adduct', '')
    if adduct:
        sign = re.search(r'(\d*)([+-])$', adduct)
        return f'[{formula}]{sign[0]}'
    else:
        mode = data.get('ion_mode', '')
        if not mode:
            print(f'Missing adduct and/or ion mode data, cannot reformat formula {formula}')
            return None
        sign = '+' if 'pos' in mode.lower() else '-' if 'neg' in mode.lower() else ''
        if not sign:
            return None
        return f'[{formula}]{sign[0]}'

def format_annotation(data, result):
    """
    Reformats fragment annotation data for visualization in the web app.
    """
    # Small changes to go with the new recursive pipeline...
    formatted = []
    # mz, formula, theo_mz, ppm_error, is_isotopic, is_parent, CHARGE (now relevant)
    for (mz, formula, theo_mz, ppm_error, _, _, charge) in result:
        formatted_formula = format_formula(data, formula, charge) if formula else None
        formula_count = sum(1 for row in result if row[1] == formula)
        # Before we were appending theo_mz first, and mz later.
        # This should be the proper MassBank order...
        # Charge is dropped of course and embedded in the formatted formula
        formatted.append((mz, formatted_formula, formula_count, theo_mz, ppm_error))
    return formatted
    
#def annotate(compound, data):
#    cands = generate_subformulas(data[compound], ppm_tol=10)
    # peaks = data[compound]['ms2_data']
#    cands = match_iso_patterns(data[compound], cands)
#    final = finalize_annotation(data[compound], cands)
#    return format_annotation(data[compound], final)

#neg_data = gu.sheet_to_dict('preAssembly_neg.csv')
#pos_data = gu.sheet_to_dict('preAssembly_pos.csv')
#annotate('Theobromine', pos_data)

# TEST DOUBLE CHARGE PERFORMANCE
#pos_data = gu.sheet_to_dict('PA_panalc.csv')
#pos_data['Alcuronium']['ms2_data']
#generate_subformulas(pos_data['Alcuronium'], ppm_tol=10)

#annotate('Pancuronium', pos_data) # Seems alright

#annotate('Diclofenac', pos_data)
#final = annotate('Hydrocortisone acetate', pos_data)
#annotate('Benzethonium', pos_data)

#data = {'Thiostrepton': {'keyColumn': 'library_id', 'file_name': None, 'short_accession': None, 'accession': None, 'title': None, 'date': None, 'authors': 'ACESx, National Facility for Exposomics', 'license': 'CC BY', 'copyright': 'Stockholm University, ACESx, National Facility for Exposomics', 'comment_1': 'CONFIDENCE Standard Compound (Level 1)', 'comment_2': None, 'library_id': 'Thiostrepton', 'iupacName': 'N-[3-[(3-amino-3-oxoprop-1-en-2-yl)amino]-3-oxoprop-1-en-2-yl]-2-[(11E)-37-butan-2-yl-18-(2,3-dihydroxybutan-2-yl)-11-ethylidene-59-hydroxy-8,31-bis(1-hydroxyethyl)-26,40,46-trimethyl-43-methylidene-6,9,16,23,28,38,41,44,47-nonaoxo-27-oxa-3,13,20,56-tetrathia-7,10,17,24,36,39,42,45,48,52,58,61,62,63,64-pentadecazanonacyclo[23.23.9.329,35.12,5.112,15.119,22.154,57.01,53.032,60]tetrahexaconta-2(64),4,12(63),19(62),21,29(61),30,32(60),33,51,54,57-dodecaen-51-yl]-1,3-thiazole-4-carboxamide', 'class': None, 'molecularFormula': 'C72H85N19O18S5', 'monoisotopicMass': 1663.49235, 'smiles': 'CCC(C)C1C(=O)NC(C(=O)NC(=C)C(=O)NC(C(=O)NC23CCC(=NC2C4=CSC(=N4)C(C(OC(=O)C5=NC6=C(C=CC(C6O)N1)C(=C5)C(C)O)C)NC(=O)C7=CSC(=N7)C(NC(=O)C8CSC(=N8)/C(=C\\C)/NC(=O)C(NC(=O)C9=CSC3=N9)C(C)O)C(C)(C(C)O)O)C1=NC(=CS1)C(=O)NC(=C)C(=O)NC(=C)C(=O)N)C)C', 'inchi': 'InChI=1S/C72H85N19O18S5/c1-14-26(3)47-63(105)78-30(7)57(99)75-28(5)56(98)76-31(8)58(100)91-72-19-18-40(66-85-43(22-111-66)59(101)77-29(6)55(97)74-27(4)54(73)96)81-52(72)42-21-112-67(83-42)49(34(11)109-69(107)41-20-37(32(9)92)36-16-17-39(79-47)51(95)50(36)80-41)89-60(102)44-24-113-68(86-44)53(71(13,108)35(12)94)90-62(104)45-23-110-65(84-45)38(15-2)82-64(106)48(33(10)93)88-61(103)46-25-114-70(72)87-46/h15-17,20-22,24-26,30-35,39,45,47-49,51-53,79,92-95,108H,4-6,14,18-19,23H2,1-3,7-13H3,(H2,73,96)(H,74,97)(H,75,99)(H,76,98)(H,77,101)(H,78,105)(H,82,106)(H,88,103)(H,89,102)(H,90,104)(H,91,100)/b38-15+', 'cas': None, 'pubchemCID': 16129666, 'inchikey': 'NSFFHOGKXHRQEW-DVRIZHICSA-N', 'comptoxURL': None, 'instrument': 'Exploris 480 Orbitrap (Thermo Scientific)', 'instrument_type': 'LC-ESI-QFT', 'ms_type': 'MS2', 'ion_mode': 'Positive', 'ionization': 'ESI', 'fragmentation_mode': 'HCD', 'collision_energy': 'Ramp 20%-70% (nominal)', 'resolution': '30000', 'column_name': 'Waters; Acquity UPLC BEH C18, 3.0 x 100 mm, 1.7 um, Waters', 'flow_gradient': '95/5 at 0.7 min, 0/100 at 12 min, 0/100 at 15.5 min, 95/5 at 15.6 min, 95/5 at 20 min', 'flow_rate': '0.4 mL/min', 'retention_time': 12.6, 'rti': None, 'solvent_a': '1mM ammonium fluoride in water', 'solvent_b': 'MeOH', 'chromatography_comment_1': 'Column oven at 50°C', 'base_peak': 86.09634, 'precursor_mz': 832.7526, 'ion_type': '[M+2H]2+', 'data_processing': 'WHOLE MS-DIAL', 'splash': None, 'num_peak': 76, 'ms2_peaks': None, 'ms2_annot': None, 'ms2_data': [(52.17369, 43038), (63.15478, 38536), (64.72392, 37071), (65.35354, 40518), (67.71963, 38017), (69.07003, 77563), (76.80721, 35914), (83.58787, 41084), (86.09634, 624598), (87.92103, 36516), (111.05509, 50563), (111.98528, 98066), (113.07068, 108870), (115.05418, 71410), (124.02164, 108005), (137.3911, 38551), (138.19003, 36498), (138.90945, 42219), (151.03203, 75693), (157.13301, 101027), (159.02129, 44457), (160.07602, 36618), (160.08453, 39963), (167.02757, 49256), (185.12825, 129123), (188.0701, 254545), (195.02185, 56676), (199.73651, 37154), (206.08134, 209665), (225.81429, 38703), (230.32471, 44608), (230.3373, 49825), (231.05865, 40010), (234.07678, 423797), (257.16541, 54497), (278.00513, 43643), (298.07657, 39195), (301.15497, 170872), (319.16397, 282015), (320.16888, 48577), (418.19656, 245515), (424.20407, 44389), (581.12762, 172756), (582.14038, 77633), (591.69818, 47605), (670.19226, 47020), (697.66724, 72894), (714.21436, 166346), (714.71869, 125022), (731.70166, 44117), (740.20947, 166149), (740.71655, 242077), (750.23358, 42862), (750.97412, 43772), (754.71826, 66747), (755.22876, 50938), (763.73322, 66385), (764.22748, 79719), (767.71375, 49463), (771.72546, 57917), (774.71979, 173454), (775.22437, 85408), (775.71686, 54427), (776.23584, 66491), (780.72034, 163097), (781.21814, 60914), (783.23944, 316692), (783.74127, 197369), (784.24109, 79695), (789.72583, 353866), (790.23383, 330108), (790.72681, 154486), (815.23724, 47507), (824.2384, 132244), (824.74994, 74528), (832.75439, 138998)], 'ms2_norm': [(52.17369, 43038, 68), (63.15478, 38536, 61), (64.72392, 37071, 59), (65.35354, 40518, 64), (67.71963, 38017, 60), (69.07003, 77563, 124), (76.80721, 35914, 57), (83.58787, 41084, 65), (86.09634, 624598, 999), (87.92103, 36516, 58), (111.05509, 50563, 80), (111.98528, 98066, 156), (113.07068, 108870, 174), (115.05418, 71410, 114), (124.02164, 108005, 172), (137.3911, 38551, 61), (138.19003, 36498, 58), (138.90945, 42219, 67), (151.03203, 75693, 121), (157.13301, 101027, 161), (159.02129, 44457, 71), (160.07602, 36618, 58), (160.08453, 39963, 63), (167.02757, 49256, 78), (185.12825, 129123, 206), (188.0701, 254545, 407), (195.02185, 56676, 90), (199.73651, 37154, 59), (206.08134, 209665, 335), (225.81429, 38703, 61), (230.32471, 44608, 71), (230.3373, 49825, 79), (231.05865, 40010, 63), (234.07678, 423797, 677), (257.16541, 54497, 87), (278.00513, 43643, 69), (298.07657, 39195, 62), (301.15497, 170872, 273), (319.16397, 282015, 451), (320.16888, 48577, 77), (418.19656, 245515, 392), (424.20407, 44389, 70), (581.12762, 172756, 276), (582.14038, 77633, 124), (591.69818, 47605, 76), (670.19226, 47020, 75), (697.66724, 72894, 116), (714.21436, 166346, 266), (714.71869, 125022, 199), (731.70166, 44117, 70), (740.20947, 166149, 265), (740.71655, 242077, 387), (750.23358, 42862, 68), (750.97412, 43772, 70), (754.71826, 66747, 106), (755.22876, 50938, 81), (763.73322, 66385, 106), (764.22748, 79719, 127), (767.71375, 49463, 79), (771.72546, 57917, 92), (774.71979, 173454, 277), (775.22437, 85408, 136), (775.71686, 54427, 87), (776.23584, 66491, 106), (780.72034, 163097, 260), (781.21814, 60914, 97), (783.23944, 316692, 506), (783.74127, 197369, 315), (784.24109, 79695, 127), (789.72583, 353866, 565), (790.23383, 330108, 527), (790.72681, 154486, 247), (815.23724, 47507, 75), (824.2384, 132244, 211), (824.74994, 74528, 119), (832.75439, 138998, 222)], 'frag_annot': None, 'submitted_to_MBEU': None}}

#generate_subformulas(data['Thiostrepton'], ppm_tol=10)
#annotate('Thiostrepton', data) # Seems alright
