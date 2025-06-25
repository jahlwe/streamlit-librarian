# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 13:48:29 2025

@author: martingroup
"""

import json
import re
import pandas as pd
import utils.genericUtilities as gu
from collections import defaultdict
import math

# MassBank (.json-format) functions
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8',
              errors='ignore') as f:
        json_records = json.load(f)
    return json_records

def extract_instrument(mb_entry): 
    if 'name' in mb_entry: # check in name field
        instrument_match = re.search(r';\s*([^;]+);\s*MS', mb_entry['name'])
        if instrument_match:
            return instrument_match.group(1).strip()

    if 'measurementTechnique' in mb_entry: # check in measurementTechnique
        if isinstance(mb_entry['measurementTechnique'], list):
            for technique in mb_entry['measurementTechnique']:
                if isinstance(technique, dict) and 'name' in technique:
                    return technique['name']
        elif isinstance(mb_entry['measurementTechnique'], dict) and 'name' in mb_entry['measurementTechnique']:
            return mb_entry['measurementTechnique']['name']

    if 'description' in mb_entry: # check in description
        instrument_match = re.search(
            r'(LC-[^;]+|GC-[^;]+|[A-Z]+-[A-Z]+)', mb_entry['description'])
        if instrument_match:
            return instrument_match.group(1)

    return None

def extract_ionization_mode(mb_name):
    try:
        parts = mb_name.split(';')
        for part in parts:
            current_part = part.strip().lower()
            if current_part == 'pos' or current_part == 'positive' or current_part[-1] == '+':
                return 'pos'
            elif current_part == 'neg' or current_part == 'negative' or current_part[-1] == '-':
                return 'neg'
    except:
        return None

def extract_fields(source, fields):
    '''
    Helper for parse_json.
    '''
    return {field: source.get(field, '') or '' for field in fields}

def is_valid_name(name):
    '''
    Deals with missing synonyms for making name_list
    in survey functions.
    '''
    if name is None:
        return False
    if isinstance(name, float) and math.isnan(name):
        return False
    if not isinstance(name, str):
        return False
    if name.lower() == 'nan':
        return False
    return True

def parse_json(json_records, exclude_by_type=True):
    '''
    Creates a dictionary from .json MassBank records.
    '''
    records = {}
    for entry in json_records:
        accession = entry.get('identifier')
        if not accession:
            continue
        # creates a reference to records[accession]
        rec = records.setdefault(accession, {})
        entry_type = entry.get('@type')        
        if entry_type == 'Dataset': # first part of entry
            instrument = extract_instrument(entry)
            rec['instrument'] = instrument.strip() if instrument else ''
            mode = extract_ionization_mode(entry.get('name'))
            rec['mode'] = mode if mode else ''
        elif entry_type == 'ChemicalSubstance':
            if 'hasBioChemEntityPart' in entry and isinstance(entry['hasBioChemEntityPart'], list):
                for part in entry['hasBioChemEntityPart']:
                    if isinstance(part, dict):
                        field_map = {
                            # record category: storage name
                            'name': 'name',
                            'inChI': 'inchi',
                            'inChIKey': 'inchikey',
                            'smiles': 'smiles',
                            'monoisotopicMolecularWeight': 'monoisotopic',
                            'molecularFormula': 'formula'
                        }
                        for src_field, out_field in field_map.items():
                            rec[out_field] = part.get(src_field, '') or ''
    if records and exclude_by_type: # we should exclude these...
        exclude_prefixes = ('EI', 'GC', 'CI', 'FAB', 'FD', 'FI', 'MALDI', 'SI')
        def keep_record(v):
            instr = v.get('instrument', '').upper()
            return not any(instr.startswith(prefix) for prefix in exclude_prefixes)
        records = {k: v for k, v in records.items() if keep_record(v)}
    return records

def massbank_record_indexes(records):
    '''
    Helper for survey_massbank, builds indexes 
    that can be used for fast lookups.
    (ALSO USING IT FOR GNPS .MGF)
    '''
    inchikey_index = {}
    inchi_index = {}
    smiles_index = {}
    for accession, data in records.items():
        # stores record data accessible via identifiers
        if 'inchikey' in data and data['inchikey']:
            inchikey_index[data['inchikey']] = (accession, data)
        if 'inchi' in data and data['inchi']:
            inchi_index[data['inchi']] = (accession, data)
        if 'smiles' in data and data['smiles']:
            smiles_index[data['smiles']] = (accession, data)
    return inchikey_index, inchi_index, smiles_index

def survey_massbank(dictionary, records):
    '''
    Survey MassBank records (in dict-format, from parse_json)
    Currently looking for InChIKey, InChI, and SMILES matches.
    '''
    inchikey_index, inchi_index, smiles_index = massbank_record_indexes(records)
    for i, (compound_name, compound_data) in enumerate(dictionary.items()):
        # do this
        fields = (('pos_count', 0), ('neg_count', 0), ('instruments', []),
                  ('accession', []), ('mb_recCount', 0), ('mb_recNames', []))
        for field, default in fields:
            if field not in compound_data:
                compound_data.setdefault(field, default)
                
        if isinstance(compound_data['pcQueried'], float):
            continue # skip compounds with no metadata
        
        used_accessions = set() # avoid duplicates
        hits = []
        for key, index in [
            (compound_data.get('inchikey'), inchikey_index),
            (compound_data.get('inchi'), inchi_index),
            (compound_data.get('smiles'), smiles_index)
        ]:
            if key and key in index:
                accession, record_data = index[key]
                if accession not in used_accessions:
                    hits.append((accession, record_data))
                    used_accessions.add(accession)
        
        for accession, record_data in hits:
            if instrument := record_data.get('instrument'):
                if instrument not in compound_data['instruments']:
                    compound_data['instruments'].append(instrument)
                    
            if (mode := record_data.get('mode')) in ('pos', 'neg'):
                compound_data[f'{mode}_count'] += 1
                
            compound_data['accession'].append(accession)
            compound_data['mb_recCount'] += 1
            if record_data['name'] not in compound_data['mb_recNames']:
                compound_data['mb_recNames'].append(record_data['name'])
        if i % 100 == 0 and i != 0:
            print(f'{i} compounds processed...')
        elif i == len(dictionary) - 1:
            print(f'done')
    return dictionary

# GNPS (.csv-format) functions
def load_gnps_csv(csv_path):
    try:
        gnps_records = pd.read_csv(csv_path)
        gnps_records = gnps_records.set_index('spectrum_id').to_dict(orient='index')
        return gnps_records
    except FileNotFoundError:
        print('Database record file not found.')

def gnps_record_indexes(records):
    '''
    Helper for survey_gnps_csv, builds indexes 
    that can be used for fast lookups.
    '''
    inchikey_index = {}
    smiles_index = {}
    for accession, data in records.items():
        # stores record data via identifiers in separate indices
        inchikeys = set()
        for inchikey_field in ('InChIKey_inchi', 'InChIKey_smiles'):
            val = data.get(inchikey_field)
            if val:
                inchikeys.add(val)
        for inchikey in inchikeys: 
            inchikey_index[inchikey] = (accession, data)
        if 'Smiles' in data and isinstance(data['Smiles'], str) and data['Smiles']:
            # make it lower case, GNPS is not consistent
            smiles_index[data['Smiles'].lower()] = (accession, data)
    return inchikey_index, smiles_index

def gnps_massBins(records, mass_field='Precursor_MZ', bin_size=0.5):
    '''
    Create mass bins from GNPS records to make 
    the mass-name matching process faster.
    '''
    mass_bins = defaultdict(list)
    for accession, data in records.items():
        rec_mass = data.get(mass_field)
        if rec_mass is not None:
            try:
                rec_mass = float(rec_mass)
                # key that will contain many entries 
                # of similar masses...
                bin_key = round(rec_mass / bin_size)
                mass_bins[bin_key].append((accession, data))
            except Exception:
                continue
    return mass_bins

def gnps_name_and_mass(name_list, monoisotopic_mass, mass_bins, delta=0.5, bin_size=0.5):
    '''
    Helper to find hits by name - hits are verified by similar-enough masses.
    GNPS has one mass field, Precursor_MZ, which (it seems) can hold
    either a monoisotopic/exact or an experimentally observed mass.
    '''
    matches = []
    target_bin = round(monoisotopic_mass / bin_size)
    candidate_bins = [target_bin-1, target_bin, target_bin+1]
    for bin_key in candidate_bins:
        for accession, data in mass_bins.get(bin_key, []):
            rec_name = str(data.get('Compound_Name', '')).lower()
            rec_mass = data.get('Precursor_MZ')
            if rec_mass is not None:
                try:
                    rec_mass = float(rec_mass)
                except Exception:
                    continue
                if any(name in rec_name for name in name_list) and abs(monoisotopic_mass - rec_mass) < delta:
                    matches.append((accession, data))
    return matches

def survey_gnps_csv(dictionary, records, name_and_mass=True):
    inchikey_index, smiles_index = gnps_record_indexes(records)
    mass_bins = gnps_massBins(records) if name_and_mass else None
    for i, (compound_name, compound_data) in enumerate(dictionary.items()):   
        fields = (('pos_count', 0), ('neg_count', 0), ('instruments', []),
                  ('accession', []), ('gnps_recCount', 0), ('gnps_recNames', []))
        for field, default in fields:
            if field not in compound_data:
                compound_data.setdefault(field, default)
                
        if isinstance(compound_data['pcQueried'], float):
            continue # skip compounds with no metadata
        
        # also matching names and masses for GNPS...
        name_list = [compound_name.lower()] + [
            name.lower() for name in compound_data.get('altNames', []) if is_valid_name(name)
        ]
        monoisotopic_mass = compound_data.get('monoisotopicMass')
        
        hits = []
        used_accessions = set()
        smiles = compound_data.get('smiles')
        smiles_key = smiles.lower() if isinstance(smiles, str) else None
        for key, index in [ # first, index matching (.lower() SMILES)
            (compound_data.get('inchikey'), inchikey_index),
            (compound_data.get(smiles_key), smiles_index)
        ]:
            if key and key in index:
                accession, record_data = index[key]
                if accession not in used_accessions:
                    hits.append((accession, record_data))
                    used_accessions.add(accession)
        # then, name & mass-matching -- OPTIONAL, though -- SLOW! O(NxM)
        if name_list and monoisotopic_mass and name_and_mass: 
            for accession, record_data in gnps_name_and_mass(name_list, monoisotopic_mass, mass_bins):
                if accession not in used_accessions:
                    hits.append((accession, record_data))
                    used_accessions.add(accession)
                    
        for accession, record_data in hits:
            instrument = record_data.get('Instrument')
            if instrument and instrument not in compound_data['instruments']:
                compound_data['instruments'].append(instrument)
            mode = record_data.get('Ion_Mode')
            if mode:
                if mode in ('positive', 'pos', '+'):
                    compound_data['pos_count'] += 1 
                if mode in ('negative', 'neg', '-'):
                    compound_data['neg_count'] += 1                                           
            compound_data['gnps_recCount'] += 1
            compound_data['accession'].append(accession)
            if record_data.get('Compound_Name') not in compound_data['gnps_recNames']:
                compound_data['gnps_recNames'].append(record_data.get('Compound_Name'))
        if i % 100 == 0 and i != 0:
            print(f'{i} compounds processed...')
        elif i == len(dictionary) - 1:
            print(f'done')
    return dictionary

# GNPS (.mgf-format) functions
def extract_name(input_line):
    match = re.search(r'=(.*?)\s*\[', input_line)
    return match.group(1).strip() if match else None

def parse_gnps_mgf(mgf_path):
    '''
    Read GNPS records in mgf-format (txt-like)
    Did this for MSNLIB - don't know if other mgf
    GNPS libraries will follow a similar structure.
    '''
    records = {}
    current_record = {}
    reading_data = False
    adduct_pattern = re.compile(r'(\[M[^\]]*\]\d*[+-])$')

    with open(mgf_path, 'r') as f:
        for line in f:
            current_line = line.strip()
            if current_line.startswith('BEGIN IONS'):
                reading_data = True
                current_record = {
                    'name': '',
                    'adduct': '',
                    'mode': '',
                    'smiles': '',
                    'inchikey': '',
                    'inchi': '',
                    'Instrument': '',
                    'SpectrumID': ''
                }
            elif current_line.startswith('END IONS') and reading_data:
                reading_data = False
                current_id = current_record.get('SpectrumID')
                if current_id and current_id not in records:
                    records[current_id] = current_record.copy()
            elif reading_data:
                if '=' in current_line:
                    key, value = current_line.split('=', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    if key == 'NAME':
                        current_record['name'] = value
                        match = adduct_pattern.search(value)
                        if match:
                            adduct = match.group(1)  # e.g., [M+2H]2+
                            current_record['adduct'] = adduct
                            current_record['mode'] = 'pos' if '+' in adduct else 'neg'
                    elif key == 'SMILES':
                        current_record['smiles'] = value
                    elif key == 'INCHIAUX':
                        current_record['inchikey'] = value
                    elif key == 'INCHI':
                        current_record['inchi'] = value
                    elif key == 'SOURCE_INSTRUMENT':
                        current_record['Instrument'] = value
                    elif key == 'SPECTRUMID':
                        current_record['SpectrumID'] = value
    return records

def survey_gnps_mgf(dictionary, records):
    '''
    Surveys GNPS records in .mgf format. This was written
    for MSNLIB - unsure if applicable to other .mgf GNPS libraries.
    Should be sufficient to match InChIKey, InChI & SMILES - more
    curated than GNPS at large.
    '''
    # can use mb indexing function, same variable names in records
    inchikey_index, inchi_index, smiles_index = massbank_record_indexes(records)
    for i, (compound_name, compound_data) in enumerate(dictionary.items()):
        fields = (('pos_count', 0), ('neg_count', 0), ('instruments', []),
                  ('accession', []), ('gnps_recCount', 0), ('gnps_recNames', []))
        for field, default in fields:
            if field not in compound_data:
                compound_data.setdefault(field, default)
        
        if isinstance(compound_data['pcQueried'], float):
            continue # skip compounds with no metadata
        
        used_accessions = set() # avoid duplicates
        hits = []
        for key, index in [
            (compound_data.get('inchikey'), inchikey_index),
            (compound_data.get('inchi'), inchi_index),
            (compound_data.get('smiles'), smiles_index)
        ]:
            if key and key in index:
                accession, record_data = index[key]
                if accession not in used_accessions:
                    hits.append((accession, record_data))
                    used_accessions.add(accession)
        
        for accession, record_data in hits:
            if instrument := record_data.get('instrument'):
                if instrument not in compound_data['instruments']:
                    compound_data['instruments'].append(instrument)
                    
            if (mode := record_data.get('mode')) in ('pos', 'neg'):
                compound_data[f'{mode}_count'] += 1
                
            # possible that MSNLIB is in the GNPS records we survey...
            # if so, don't add a duplicate hit from this run-through
            if accession not in compound_data['accession']:
                compound_data['accession'].append(accession)
                compound_data['gnps_recCount'] += 1
                if record_data['name'] not in compound_data['gnps_recNames']:
                    compound_data['gnps_recNames'].append(record_data['name'])
        if i % 100 == 0 and i != 0:
            print(f'{i} compounds processed...')
        elif i == len(dictionary) - 1:
            print('done')
    return dictionary

def safe_count(val):
    '''
    Helper for evaluate_instruments.
    '''
    if val is None:
        return 0
    try:
        # covers numpy.nan, float('nan')
        if math.isnan(val):
            return 0
    except TypeError:
        pass
    return val

# Instrument data functions
def evaluate_instruments(dictionary):
    orbitrap_keywords = ['orbitrap', 'q-exactive', 'ft', 'itft']
    tof_keywords = ['qtof', 'tof', 'impact']
    # gc_keywords = ['gc', 'ei-b', 'ci-b'] Maybe we need to do something with this later.

    for compound, data in dictionary.items():
        data.setdefault('hrms_records', False)
        data.setdefault('hrms_type', None)
        data.setdefault('only_lowRes', False)
        mb_count = safe_count(data.get('mb_recCount', 0))
        gnps_count = safe_count(data.get('gnps_recCount', 0))
        compound_has_entry = mb_count > 0 or gnps_count > 0
        current_instruments = data['instruments']
        
        if compound_has_entry:
            data['only_lowRes'] = True
            for instrument in current_instruments:
                current_instrument = instrument.lower()
                
                if any(keyword in current_instrument for keyword in orbitrap_keywords):
                    data['hrms_records'] = True
                    data['only_lowRes'] = False
                    data['hrms_type'] = 'Orbitrap'
                    break # Break if we encounter Orbitrap - that's all we need to know...
                
                elif any(keyword in current_instrument for keyword in tof_keywords):
                    data['hrms_records'] = True
                    data['only_lowRes'] = False
                    data['hrms_type'] = 'TOF'
    return dictionary

# For testing
#import genericUtilities

# MassBank
#json_records = load_json('files/survey/MassBank.json')
#records = parse_json(json_records)
#dictionary = genericUtilities.sheet_to_dict('output/prepOneSheet.xlsx', 'vendorName')
#dictionary = gu.sheet_to_dict('output/testOutput.csv')
#dictionary = survey_massbank(dictionary, records)

# GNPS
#gnps_pos = load_gnps_csv('files/survey/GNPS_pos.csv')
#dictionary = survey_gnps_csv(dictionary, gnps_pos)
#gnps_neg = load_gnps_csv('files/survey/GNPS_neg.csv')
#dictionary = survey_gnps_csv(dictionary, gnps_neg) 
#msnlib_pos = parse_gnps_mgf('files/survey/MSNLIB-POSITIVE.mgf')
#dictionary = survey_gnps_mgf(dictionary, msnlib_pos)
#msnlib_neg = parse_gnps_mgf('files/survey/MSNLIB-NEGATIVE.mgf')
#dictionary = survey_gnps_mgf(dictionary, msnlib_neg)

#sum(1 for compound, data in dictionary.items() if data['gnps_recCount'] > 0 or data['mb_recCount'] > 0)

#gu.dict_to_sheet(dictionary, 'afterSurvey')

#dictionary = evaluate_instruments(dictionary)
