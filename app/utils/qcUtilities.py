# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:25:20 2025

@author: Jakob
"""

import utils.surveyUtilities as su
import utils.genericUtilities as gu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import os
from collections import defaultdict

def normalize_peaks(peak_list):
    '''
    Takes a list of (mz, raw intensity) pairs and returns
    a list with (mz, raw, normalized intensity) values
    
    Normalized intensities are scaled to the base peak intensity
    '''
    if not peak_list:
        return []
    base_peak_int = max(intensity for mz, intensity in peak_list)
    return [(mz, intensity, int((intensity / base_peak_int)*999)) for mz, intensity in peak_list]

# maybe we want to do something like this, maybe not.
def filter_ms2(peak_list_w_norm, norm_cutoff=10):
    filtered_ms2 = [(mz, abs_int, norm_int) for mz, abs_int, norm_int in peak_list_w_norm if norm_int >= norm_cutoff]
    return filtered_ms2

# need this when getting MS2 data from postComp-sheets
def extract_ms2(ms2_in_postcomp_format):
    parts = ms2_in_postcomp_format.split()
    values = parts[3:] # skip string headers
    ms2_peaks = []
    for i in range(0, len(values), 3):
        try: # skip abs int, because .msp only has normalized
            mz = float(values[i])
            rel_int = int(values[i+2])
            ms2_peaks.append((mz, rel_int))
        except (IndexError, ValueError):
            # Handle possible malformed input gracefully
            continue
    return ms2_peaks

def parse_msp(msp_path):
    records = {}
    current_accession = None
    current_name = None
    current_synonyms = None
    with open(msp_path, 'r') as msp_records:
        for line in msp_records:
            line = line.strip()
            if not line:
                continue
            if not line[0].isdigit():
                key, _, value = line.partition(':')
                key, value = key.strip(), value.strip()
                if key == 'Name':
                    current_name = value
                elif key == 'Synon':
                    current_synonyms = value
                elif key == 'DB#':
                    if value not in records.keys():
                        current_accession = value
                        records[current_accession] = {
                        'Name': current_name,
                        'synonyms': current_synonyms,
                        'ms2_peaks': []
                    }
                else:
                    if current_accession:
                        records[current_accession][key] = value
            else:
                parts = line.split()
                if len(parts) >= 2 and current_accession:
                    mz, intensity = map(float, parts[:2])
                    records[current_accession]['ms2_peaks'].append((mz, intensity))

    return records

# this function seems to have created too liberal bins, 
# causing faulty matching for borderline-similar spectra
def align_peaks_composite(query, reference, tolerance_ppm=10):
    # create a reconciled, composite set of mz without duplicates 
    composite_mz = set()
    
    # add all mz from original spectra
    for mz, _ in query:
        composite_mz.add(mz)
    for mz, _ in reference:
        composite_mz.add(mz)
    
    # find matches within tolerance
    matched_pairs = set()
    
    for q_mz, q_int in query: 
        # for each query mz, we loop through each ref mz
        for r_mz, r_int in reference:
            ppm_diff = abs(q_mz - r_mz)/min(q_mz, r_mz)*1e6  # symmetric ppm calculation
            if ppm_diff <= tolerance_ppm:
                # store match as (query_mz, reference_mz)
                matched_pairs.add((q_mz, r_mz))
    
    # for matches, create a new reconciled mz and remove respective match mz
    for q_mz, r_mz in matched_pairs:
        avg_mz = (q_mz + r_mz)/2
        composite_mz.add(avg_mz)
        composite_mz.discard(q_mz)  # remove original if matched
        composite_mz.discard(r_mz)
    
    # nb - unmatched peaks are still retained in the composite set
    # this means they will still be accounted for
    composite_mz = sorted(composite_mz)
    
    # build intensity vectors with strict assignment
    def build_vector(spectrum, composite_mz):
        vector = []
        used_peaks = set() # ensures no peaks are assigned more than once
        
        # run through the composite vector and assign...
        for target_mz in composite_mz:
            best_intensity = 0
            best_ppm = float('inf')
            best_mz = None
            
            for mz, intensity in spectrum:
                if mz in used_peaks:
                    continue
                
                ppm_diff = abs(mz - target_mz)/target_mz*1e6
                if ppm_diff <= tolerance_ppm and ppm_diff < best_ppm:
                    best_intensity = intensity
                    best_ppm = ppm_diff
                    best_mz = mz
                    
            # do this, to avoid adding too many peaks to used peaks 
            # before, this was in the if loop above.
            if best_mz is not None: 
                used_peaks.add(best_mz)
            
            # this returns both the mz in the composite set and
            # the intensity. for scoring, we only need intensity,
            # remember that. but the mz is needed for calculating
            # weights for massbank-style scoring.
            vector.append((target_mz, best_intensity))
        return vector
    
    query_vector = build_vector(query, composite_mz)
    reference_vector = build_vector(reference, composite_mz)
    
    return query_vector, reference_vector, composite_mz

def align_peaks_strict(query, reference, tolerance_ppm=10):
    # List to hold matched pairs
    matched_query = set()
    matched_ref = set()
    composite_mz = []

    # For each query peak, find the closest reference peak within tolerance
    for i, (q_mz, q_int) in enumerate(query):
        best_j = None
        best_ppm = tolerance_ppm + 1
        for j, (r_mz, r_int) in enumerate(reference):
            if j in matched_ref:
                continue
            ppm_diff = abs(q_mz - r_mz) / ((q_mz + r_mz)/2) * 1e6
            if ppm_diff < best_ppm and ppm_diff <= tolerance_ppm:
                best_ppm = ppm_diff
                best_j = j
        if best_j is not None:
            # Matched pair
            matched_query.add(i)
            matched_ref.add(best_j)
            composite_mz.append((q_mz + reference[best_j][0]) / 2)
    
    # Add unmatched query peaks
    for i, (q_mz, _) in enumerate(query):
        if i not in matched_query:
            composite_mz.append(q_mz)
    # Add unmatched reference peaks
    for j, (r_mz, _) in enumerate(reference):
        if j not in matched_ref:
            composite_mz.append(r_mz)

    composite_mz = sorted(composite_mz)

    # Now build vectors as before, but only assign a peak once
    def build_vector(spectrum, composite_mz):
        vector = []
        used_peaks = set()
        for target_mz in composite_mz:
            best_intensity = 0
            best_ppm = float('inf')
            best_mz = None
            for i, (mz, intensity) in enumerate(spectrum):
                if i in used_peaks:
                    continue
                ppm_diff = abs(mz - target_mz)/target_mz*1e6
                if ppm_diff <= tolerance_ppm and ppm_diff < best_ppm:
                    best_intensity = intensity
                    best_ppm = ppm_diff
                    best_mz = i
            if best_mz is not None:
                used_peaks.add(best_mz)
            vector.append((target_mz, best_intensity))
        return vector

    query_vector = build_vector(query, composite_mz)
    reference_vector = build_vector(reference, composite_mz)
    return query_vector, reference_vector, composite_mz

# Now we can calculate cosine similarities...
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    
    return dot_product / norm_product if norm_product > 0 else 0

def weighted_cosine(query_peaks, ref_peaks, composite_mz):
    # MassBank-style weighted scoring
    def build_weighted_vector(spectrum, composite_mz):
        vector = []
        used_peaks = set()
        
        for target_mz in composite_mz:
            best_weighted = 0
            best_ppm = float('inf')
            best_mz = None
            
            for mz, intensity in spectrum:
                if mz in used_peaks:
                    continue
                
                ppm_diff = abs(mz - target_mz)/target_mz*1e6
                if ppm_diff <= 10 and ppm_diff < best_ppm:
                    # this system gives higher-mass peaks more
                    # influence in the similarity scoring.
                    # (mz**2) and sqrt(intensity)
                    weight = (mz**2) * np.sqrt(intensity)
                    best_weighted = weight
                    best_ppm = ppm_diff
                    best_mz = mz
                
            if best_mz is not None:
                used_peaks.add(mz)
            vector.append(best_weighted)
        return np.array(vector)
    
    q_vec = build_weighted_vector(query_peaks, composite_mz)
    r_vec = build_weighted_vector(ref_peaks, composite_mz)
    
    return cosine_similarity(q_vec, r_vec)

# OK, now lets do the actual matching thing. MassBank first.
def massbank_msp_indexes(records):
    '''
    Helper like the one we used in surveyUtilities.
    Create indexes (from the .msp) that can be used for fast lookups.
    '''
    inchikey_index = defaultdict(list)
    smiles_index = defaultdict(list)
    formula_index = defaultdict(list)
    mass_index = defaultdict(list)
    for accession, data in records.items():
        if 'InChIKey' in data and data['InChIKey']:
            inchikey_index[data['InChIKey']].append((accession, data))
        if 'SMILES' in data and data['SMILES']:
            smiles_index[data['SMILES']].append((accession, data))
        if 'Formula' in data and data['Formula']:
            formula_index[data['Formula']].append((accession, data))
        if 'ExactMass' in data and data['ExactMass']:
            try:
                mass = round(float(data['ExactMass']), 5)
                mass_index[mass].append((accession, data))
            except ValueError:
                continue
    return inchikey_index, smiles_index, formula_index, mass_index

def subset_massbank_records(compound_data, indexes):
    inchikey_index, smiles_index, formula_index, mass_index = indexes

    current_mode = compound_data.get('AC$MASS_SPECTROMETRY: ION_MODE', '').lower()[:3]
    subset = []

    current_mass = compound_data.get('CH$EXACT_MASS:', None)
    current_inchikey = compound_data.get('CH$LINK: INCHIKEY', None)
    current_smiles = compound_data.get('CH$SMILES:', None)
    current_formula = compound_data.get('CH$FORMULA:', None)

    if current_mass:
        try:
            mass = round(float(current_mass), 5)
            subset += mass_index.get(mass, [])
        except ValueError:
            pass
    if current_inchikey:
        subset += inchikey_index.get(current_inchikey, [])
    if current_smiles:
        subset += smiles_index.get(current_smiles, [])
    if current_formula:
        subset += formula_index.get(current_formula, [])

    # remove duplicates
    subset_dict = {acc: data for acc, data in subset}

    # filter by mode
    filtered_subset = {
        acc: data for acc, data in subset_dict.items()
        if 'Ion_mode' in data and current_mode in data['Ion_mode'].lower()
    }

    return filtered_subset

def find_massbank_matches(
    dictionary,
    massbank_msp_path, 
    ppm_tolerance=5, 
    score_cutoff=0.7,
    
):
    # Parse MSP and build indexes ONCE
    records = parse_msp(massbank_msp_path)
    indexes = massbank_msp_indexes(records)

    for compound, data in dictionary.items():
        # Ensure output fields exist
        data.setdefault('match_n_massbank', 0)
        data.setdefault('match_id&score_massbank', [])
        data.setdefault('match_ms2_massbank', [])

        # Prepare query MS2 (assumes ms2_norm is [(mz, abs_int, norm_int), ...])
        query_ms2 = extract_ms2(data['PK$PEAK:'])

        # Subset MassBank records using fast indexes
        record_subset = subset_massbank_records(data, indexes)

        n_matches = 0
        match_id_score = []
        match_ms2 = []

        # For each candidate in the subset, compute similarity
        for ref_accession, ref_data in record_subset.items():
            ref_ms2 = ref_data['ms2_peaks']
            q_vec, r_vec, composite_mz = align_peaks_strict(query_ms2, ref_ms2, ppm_tolerance)
            score = weighted_cosine(q_vec, r_vec, composite_mz)
            if score > score_cutoff:
                n_matches += 1
                match_id_score.append((ref_accession, score))
                match_ms2.append(ref_ms2)

        # Store results
        data['match_n_massbank'] = n_matches
        data['match_id&score_massbank'] = match_id_score
        data['match_ms2_massbank'] = match_ms2

    return dictionary

def export_massbank_report(
    dictionary,
    mode,
    report_name='qcReport', 
    report_path='output/compiler/'
):
    fieldnames = [
        'CH$NAME:',
        'CH$FORMULA:',
        'CH$EXACT_MASS:',
        'CH$SMILES:',
        'MS_TYPE',
        'AC$MASS_SPECTROMETRY: ION_MODE',
        'MS$FOCUSED_ION: PRECURSOR_M/Z',
        'MS$FOCUSED_ION: ION_TYPE',
        'PK$NUM_PEAK:',
        'PK$PEAK:',
        'match_n_massbank',
        'match_id&score_massbank',
        'match_ms2_massbank'
    ]
    
    # set the output file path
    report_name_full = f'{report_path}{report_name}_{mode}.csv'
    output_csv_path = os.path.join(report_name_full)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for compound, data in dictionary.items():
            row = {
                'CH$NAME:': compound,
                'CH$FORMULA:': data.get('CH$FORMULA:', ''),
                'CH$EXACT_MASS:': data.get('CH$EXACT_MASS:', ''),
                'CH$SMILES:': data.get('CH$SMILES:', ''),
                'MS_TYPE': data.get('MS_TYPE', ''),
                'AC$MASS_SPECTROMETRY: ION_MODE': data.get('AC$MASS_SPECTROMETRY: ION_MODE', ''),
                'MS$FOCUSED_ION: PRECURSOR_M/Z': data.get('MS$FOCUSED_ION: PRECURSOR_M/Z', ''),
                'MS$FOCUSED_ION: ION_TYPE': data.get('MS$FOCUSED_ION: ION_TYPE', ''),
                'PK$NUM_PEAK:': data.get('PK$NUM_PEAK:', ''),
                'PK$PEAK:': data.get('PK$PEAK:', ''),
                'match_n_massbank': data.get('match_n_massbank', 0),
                # Convert lists to strings for CSV output
                'match_id&score_massbank': str(data.get('match_id&score_massbank', [])),
                'match_ms2_massbank': str(data.get('match_ms2_massbank', [])),
            }
            writer.writerow(row)

#q_vec, r_vec, composite_mz = align_peaks_composite(test_peaks, ref_peaks, 5)
#msp_records = parse_msp('files/survey/MassBank_NIST.msp')
#example_ref = msp_records['MSBNK-MPI_for_Chemical_Ecology-CE000334']
#ref_peaks = example_ref['ms2_peaks']
#query_peaks = extract_ms2(dict_neg['Idazoxan']['PK$PEAK:'])

#ref_peaks
#query_peaks
#alignment = align_peaks_strict(query_peaks, ref_peaks)


#dictionary = gu.sheet_to_dict('output/compiler/postCompilationSheet_pos.xlsx', 'RECORD_TITLE:')
#test_peaks = [(mz, norm_int) for mz, abs_int, norm_int in dictionary['Diclofenac']['ms2_norm']]
#ref_peaks = msp_records['MSBNK-Eawag-EA020103']['ms2_peaks']
#dictionary['Diclofenac']['PK$PEAK:']
#weighted_cosine(q_vec, r_vec, composite_mz)
#compound_data = dictionary['Diclofenac']
#query_ms2 = extract_ms2(compound_data['PK$PEAK:'])

#dictionary = gu.sheet_to_dict('output/compiler/postComp_pos.csv', 'CH$NAME:')
#dictionary = find_massbank_matches(dictionary, 'files/survey/MassBank_NIST.msp')
#sum(1 for k, v in dictionary.items() if v['match_n_massbank'] == 0)
#dictionary['Caffeine']
#export_massbank_report(dictionary, 'pos', 'qcReport')

#dictionary = gu.sheet_to_dict('output/compiler/postComp_neg.csv', 'CH$NAME:')
#dictionary = find_massbank_matches(dictionary, 'files/survey/MassBank_NIST.msp')
#sum(1 for k, v in dictionary.items() if v['match_n_massbank'] == 0)
#dictionary['Caffeine']
#export_massbank_report(dictionary, 'neg', 'qcReport')

# Okay, next database...
def load_gnps_csv(csv_path):
    gnps_records = pd.read_csv(csv_path)
    gnps_records = gnps_records.set_index('spectrum_id').to_dict(orient='index')
    return gnps_records

# Actually, the .csv doesn't contain MS2 data. So...
#gnps_records = load_gnps_csv('files/survey/GNPS_neg.csv')

def find_gnps_matches():
    pass