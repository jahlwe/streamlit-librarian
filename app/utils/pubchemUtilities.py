# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 21:47:45 2025

@author: Jakob
"""

import os
import pubchempy as pcp
import pandas as pd
import requests
import re
from rdkit import Chem
from rdkit.Chem import Crippen
import utils.genericUtilities as gu
import time
from datetime import datetime 
import json
import math

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"
ENDPOINT_TEMPLATE = f"{BASE_URL}/compound/{{compound_cid}}/JSON"

def is_empty(val):
    '''
    Need this for checking 'nan' and None-values, 
    for reQuery in particular.
    '''
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False

def deep_get(d, keys, default=None):
    '''
    Helper for diving into HTTP GET .json responses
    Enter names of lists, dicts, etc to navigate to 
    where you want to go in the .json as 'keys'
    '''
    for key in keys:
        if callable(key): 
            d = key(d)
        elif isinstance(d, list):
            d = d[key] if len(d) > key else default
        elif isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
        if d is None:
            return default
    return d

def special_pcp_findParent(compound, compound_cid):
    '''
    Finds parent structures in PubChem entries.
    '''
    endpoint = ENDPOINT_TEMPLATE.format(compound_cid=compound_cid)
    response = requests.get(endpoint)
    data = response.json()
    parent_cid = None

    if 'Fault' in data:
        print(f'error retrieving PUG view data for compound {compound_cid}')
        return None
    else:
        try: # now using a helper function.
            parent_cid = int(
                deep_get(
                    data,
                    [
                        'Record', 'Section',
                        lambda l: next((i for i in l if i.get('TOCHeading') == 'Related Records'), {}),
                        'Section',
                        lambda l: next((i for i in l if i.get('TOCHeading') == 'Parent Compound'), {}),
                        'Information', 0, 'Value', 'StringWithMarkup', 0, 'String'
                    ]
                ).split(' ')[1]
            )
        except Exception:
            print(f'cannot find parent structure for {compound}')
            return None
    return parent_cid

def special_pcp_metadata(compound_cid):
    '''
    Get CAS & Comptox --- not available by calling Compound object.
    '''
    endpoint = ENDPOINT_TEMPLATE.format(compound_cid=compound_cid)
    response = requests.get(endpoint)
    data = response.json()
    dictionary = {}

    if 'Fault' in data:
        print(f'error retrieving PUG view data for compound {compound_cid}')
        return None
    else:
        dictionary['Name'] = deep_get(data, ['Record', 'RecordTitle']) # record title
        
        cas_number = deep_get(data, [
            'Record',
            'Reference',
            lambda l: next((i for i in l if i.get('SourceName') == 'CAS Common Chemistry'), {}), 
            'SourceID'])
        
        comptox_url = deep_get(data, [
            'Record',
            'Reference',
            lambda l: next((i for i in l if i.get('SourceName') == 'EPA DSSTox'), {}), 
            'SourceID'])
        
        dictionary['cas'] = cas_number if cas_number else None
        dictionary['comptoxURL'] = comptox_url if comptox_url else None
    return dictionary

def nameCleaner_special(compound_name):
    '''
    hardcore version for GC compounds.
    needs re-testing.
    '''
    no_prefix = str(compound_name)
    no_gc_suffix = re.sub(
        r';\s*(EI-B|MS|EI|CI|ESI|APCI|MALDI|FAB|FD|GC|LC|LC-MS|GC-MS|GC/EI|GC/CI|GC/MS|LC/MS|GC-TOF|LC-TOF|TOF|QTOF|HRMS|NMR|UV|IR)[^;]*',
        '', no_prefix, flags=re.IGNORECASE
    )
    no_tms_suffix = re.sub(r'([;,]\s*\d*\s*TMS[^;,]*)', '', no_gc_suffix, flags=re.IGNORECASE)
    no_bp_suffix = re.sub(r'[;,]\s*BP\s*$', '', no_tms_suffix, flags=re.IGNORECASE)
    no_suffixes = re.sub(r',\s*(?:[A-Z]-|tert\b)[^,]*$', '', no_bp_suffix, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*([;,])\s*', r'\1 ', no_suffixes).strip(' ;,')
    return cleaned.strip()

def nameCleaner(compound_name):
    '''
    basic version for PW-FDA.
    '''
    return re.sub(r'^(?:\([^)]+\)-?){1,2}', '', compound_name)

def pcQuery(compound, query_input, query_type='name', pc_data=None):
    '''
    Querying PubChem --- identifies parents via SMILES,
    and re-queries with non-salt form if a parent is found.
    '''
    # ORIGINAL QUERY
    pc_query = pcp.get_compounds(query_input, query_type)
    if pc_query:
        pc_data = pc_query[0] 
        smiles = pc_data.canonical_smiles
        cid = pc_data.cid
        if smiles and '.' in smiles:
            parent_cid = special_pcp_findParent(compound, cid)
            if parent_cid:
                pc_data = None
                # RE-QUERY (if salt)
                pc_query = pcp.get_compounds(parent_cid, 'cid')
                if pc_query:
                    pc_data = pc_query[0]
                    return pc_data
                else:
                    print(f'no parent found for salt {compound}')
                    pc_data = None
    return pc_data

def safe_getattr(obj, attr, default=None, cast=None):
    '''
    Helper for unloading variable data from PubChem results in a safe way.
    '''
    try:
        value = getattr(obj, attr)
        if cast:
            return cast(value)
        return value
    except Exception:
        return default
    
FIELDS = [ # things we get from the Compound objects.
    # (pubchem field name, storage name, default value, optional class)
    ('iupac_name', 'iupacName', ''),
    ('molecular_formula', 'molecularFormula', ''),
    ('monoisotopic_mass', 'monoisotopicMass', 0, float),
    #('canonical_smiles', 'smiles', ''), # ISOMERIC INSTEAD! MassBank error using canonical. (internal validation vs provided IUPAC name)
    ('isomeric_smiles', 'smiles', ''),
    ('inchi', 'inchi', ''),
    ('inchikey', 'inchikey', ''),
    ('cid', 'pubchemCID', '', int),
]

def pcQueries(dictionary, query_empty_only=True, progress_callback=None):
    n_compounds = len(dictionary.keys())
    for i, (compound, data) in enumerate(dictionary.items()):
        pc_data = None # added this to ensure no re-use of previous compound data when queries fail...
        if not is_empty(data.get('pcQueried')) and query_empty_only:
            continue
        if data.get('queryName', None): # use queryName if the column exists
            name = data.get('queryName')
        else:
            name = data.get('internalName')
        query_name = nameCleaner(name) if name and str(name) != 'nan' else nameCleaner(compound)
            
        # now helper function for querying
        try:
            pc_data = pcQuery(compound, query_name)
            if not pc_data:
                print(f'no data retrieved for {compound}')
                data['pcQueried'] = None
                continue
            
            data['pcQueryName'] = query_name
            data['pcQueried'] = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
            
            # synonyms
            current_synonyms = safe_getattr(pc_data, 'synonyms', [])
            data['altNames'] = [v.lower() for v in current_synonyms if not re.match(
                r'^(?=.*\d)(?=.*[A-Z\W])[\dA-Z\W]+$', v)]
            
            # others - now with helper
            for attr, key, default, *cast in FIELDS:
                data[key] = safe_getattr(pc_data, attr, default, *cast)
                
            cid = data.get('pubchemCID')
            
            if cid:
                try:
                    special_results = special_pcp_metadata(cid)
                    if special_results:
                        data.update(special_results)
                except Exception:
                    pass
        except Exception as e:
            print(f'{type(e).__name__} while querying {compound}, exiting.')
            fname = 'output/pcq_intermediate'
            gu.dict_to_sheet(dictionary, fname)
            break

        if i % 5 == 0 and i != 0:
            print(f'processed {i} of {n_compounds} compounds')
        if i == n_compounds - 1:
            print(f'processed {i+1} of {n_compounds} compounds')
            print('done')
        if progress_callback: # for streamlit...
            progress_callback(i + 1, n_compounds, compound)
    return dictionary

def reQuery_CID(dictionary, query_empty_only=True):
    '''
    Re-query PubChem for missing data after manually 
    adding a CID to compounds with otherwise missing 
    data in the output file from pcQueries.
    '''
    for i, (compound, data) in enumerate(dictionary.items()):
        if not is_empty(data.get('pcQueried')) and query_empty_only:
            continue
        cid = data.get('pubchemCID')
        if not cid:
            print(f'missing CID for {compound} --- cannot re-query')
            continue
        try:
            print(f'querying by CID for {compound}')
            pc_data = pcQuery(compound, cid, query_type='cid')
            if not pc_data:
                print(f'no data retrieved for {compound}')
                data['pcQueried'] = None
                continue
            
            data['pcQueryName'] = str(cid)
            data['pcQueried'] = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
            
            # synonyms
            current_synonyms = safe_getattr(pc_data, 'synonyms', [])
            data['altNames'] = [v.lower() for v in current_synonyms if not re.match(
                r'^(?=.*\d)(?=.*[A-Z\W])[\dA-Z\W]+$', v)]
            
            # others - now with helper
            for attr, key, default, *cast in FIELDS:
                data[key] = safe_getattr(pc_data, attr, default, *cast)
            
            if cid:
                try:
                    special_results = special_pcp_metadata(cid)
                    if special_results:
                        data.update(special_results)
                except Exception:
                    pass
        except Exception as e:
            print(f'{type(e).__name__} while querying {compound}, exiting.')
            fname = 'output/pcq_reQuery_intermediate'
            gu.dict_to_sheet(dictionary, fname)
            break
        
    return dictionary
        
#rq_test = gu.sheet_to_dict('output/testReQuery.csv')
#rq_test = reQuery_CID(rq_test)

#cpd = pcp.get_compounds(5904, 'cid')
#cpd_data = cpd[0]
#rq_test['Bephenium hydroxynaphthoate']

#dictionary = genericUtilities.sheet_to_dict('input/compoundList_short.xlsx')
#dictionary = gu.sheet_to_dict('input/compoundList.xlsx')
#dictionary = genericUtilities.sheet_to_dict('input/compoundListTest.xlsx')
#dictionary = prepOne_pcQueries(dictionary)
#gu.dict_to_sheet(dictionary, 'testOutput')

# nameCleaner
#nameCleaner_special('Nonadecanoic acid methyl ester; GC-EI-TOF; MS; 0 TMS; BP')
#nameCleaner('(+)-Levobunolol')
#nameCleaner('(+)-Levobunolol')

#dictionary = gu.sheet_to_dict('output/pcq_out.csv')