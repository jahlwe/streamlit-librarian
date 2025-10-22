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
import copy

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"
ENDPOINT_TEMPLATE = f"{BASE_URL}/compound/{{compound_cid}}/JSON"

CAS_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CAS_ENDPOINT_TEMPLATE = f"{CAS_BASE_URL}/compound/name/{{cas_number}}/cids/JSON"

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

def special_pcp_findParent(query_input, compound_cid):
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
            print(f'cannot find parent structure for {query_input}')
            return None
    return parent_cid

def special_pcp_metadata(compound_cid):
    '''
    Get CAS & Comptox --- not available by calling Compound object.
    '''
    endpoint = ENDPOINT_TEMPLATE.format(compound_cid=compound_cid)
    response = requests.get(endpoint)
    data = response.json()
    special_request_dict = {}

    if 'Fault' in data:
        print(f'error retrieving PUG view data for compound {compound_cid}')
        return None
    else:
        # get name here, while we're already doing a PUG request
        special_request_dict['name'] = data['Record']['RecordTitle']
        
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
        
        special_request_dict['cas'] = cas_number if cas_number else None
        special_request_dict['comptoxURL'] = comptox_url if comptox_url else None
    return special_request_dict

def casQuery_getCID(cas_input):
    '''
    Custom solution for querying CAS numbers.
    What we do is that we find the most relevant CID for a given
    CAS number, we return it and simply do a "normal" CID-based query.
    '''
    endpoint = CAS_ENDPOINT_TEMPLATE.format(cas_number=cas_input)
    response = requests.get(endpoint)
    data = response.json()
    
    if 'Fault' in data:
        print(f'no corresponding CID found for CAS input {cas_input}')
        return None
    else:
        cid_from_cas = deep_get(data, ['IdentifierList', 'CID'])[0]
    return cid_from_cas 

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
    
# we need to initialize these to make sure our re-query stuff works as it should
# just for... proper bookkeeping.
ALL_PCQ_FIELDS = [
    'library_id', 'queried_as', 'queried_at', 'synonyms', 'iupacName',
    'molecularFormula', 'monoisotopicMass', 'smiles', 'inchi', 'inchikey',
    'pubchemCID', 'name', 'cas', 'comptoxURL'
]
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

def pcQuery_expanded(query_input, query_type, pc_data=None):
    if not query_type in ('name', 'smiles', 'cid'):
        print(f'invalid query --- input: [{query_input}] type: [{query_type}]')
        return None
    pcq = pcp.get_compounds(query_input, query_type)
    if pcq:
        pc_data = pcq[0]
        # need to do this, something has changed with how PC stores SMILES
        smiles_list = [i.get('value').get('sval') for i in pc_data.to_dict().get('record').get('props') if i['urn']['label'] == 'SMILES' and i['urn']['name'] == 'Absolute']
        smiles = smiles_list[0] if smiles_list else None
        cid = pc_data.cid
        parent_cid = None
        if smiles and '.' in smiles:
            # we can also change special_pcp_findParent to not require
            # a 'compound' input --- it fills no major function, anyway
            parent_cid = special_pcp_findParent(query_input, cid)
            if parent_cid:
                pc_data = None # reset pc_data variable if parent
                pcq = pcp.get_compounds(parent_cid, 'cid')
                if pcq:
                    pc_data = pcq[0]
                else:
                    print(f'no parent found for salt --- input: {query_input}')
                    return None

        # used to get recordTitle here --- but do that in the special request instead.
        return pc_data
    else:
        print(f'no pubchem entry found --- input: [{query_input}] type: [{query_type}]')
        return None

def pcQueries(query_dict, query_empty_only=True, progress_callback=None):
    n_compounds = len(query_dict.keys())
    pcq_out = {} # create a dictionary to store data in an organized format
    print(f'running pcq for {n_compounds} compounds')
    # assume a query_dict structure key = number, data = query inputs
    for i, (idx, data) in enumerate(query_dict.items()):
        pc_data = None # initialize pc_data
        pcq_out[idx] = {field: None for field in ALL_PCQ_FIELDS} # initialize dict entry for query
        name_q, smiles_q, cid_q, cas_q = data.get('name_q'), data.get('smiles_q'), data.get('cid_q'), data.get('cas_q')
        # hierarchy of input types --- prioritize cid, then name, then smiles, then cas
        query_input = cid_q if cid_q else name_q if name_q else smiles_q if smiles_q else cas_q
        query_type = 'cid' if cid_q else 'name' if name_q else 'smiles' if smiles_q else 'cas' if cas_q else None
        
        # if we are querying with a name, clean it
        if name_q and not cid_q:
            query_input = nameCleaner(name_q) if str(name_q) != 'nan' else name_q
        
        # hmm... think about this one.
        if not is_empty(data.get('queried_at')) and query_empty_only:
            continue
        
        # little sidestep to deal with cas inputs. we get the CID, and change the query type.
        if query_type == 'cas' and cas_q:
            cas_from_cid = casQuery_getCID(cas_q)
            if cas_from_cid and str(cas_from_cid).isdigit():
                query_input, query_type = cas_from_cid, 'cid'
            else:
                query_input, query_type = None
        
        # now helper function for querying
        try:
            pc_data = pcQuery_expanded(query_input, query_type)
            if not pc_data:
                print(f'no pubchem entry found --- input: [{query_input}], type: [{query_type}]')
                pcq_out[idx]['queried_as'] = (query_input, query_type)
                continue
            
            # store query information
            pcq_out[idx]['queried_as'] = (query_input, query_type)
            pcq_out[idx]['queried_at'] = datetime.now().strftime('%H:%M:%S %d/%m/%Y')
            
            # synonyms
            current_synonyms = safe_getattr(pc_data, 'synonyms', [])
            pcq_out[idx]['synonyms'] = [v.lower() for v in current_synonyms if not re.match(
                r'^(?=.*\d)(?=.*[A-Z\W])[\dA-Z\W]+$', v)]
            
            # others - now with helper
            for attr, key, default, *cast in FIELDS:
                pcq_out[idx][key] = safe_getattr(pc_data, attr, default, *cast)
            
            # 250713 --- pubchem is messing around with SMILES.
            # it is None in pcp-objects. temporary solution:
            smiles_list = [
                i.get('value').get('sval') for i in pc_data.to_dict().get('record').get('props') if i['urn']['label'] == 'SMILES' and i['urn']['name'] == 'Absolute'
            ]
            pcq_out[idx]['smiles'] = smiles_list[0] if smiles_list else None
            
            # use CID for special requests
            cid = pcq_out[idx].get('pubchemCID')
            
            if cid:
                try:
                    special_results = special_pcp_metadata(cid)
                    if special_results:
                        pcq_out[idx].update(special_results)
                except Exception:
                    pass
                
            # decide library_id
            library_id = data.get('library_id')
            pcq_out[idx]['library_id'] = library_id if library_id else pcq_out[idx]['name']
                
        except Exception as e:
            # need to change this, later.
            # probably what will be useful when connection errors happen
            # need to make sure pcq_out will contain all query input types
            # in a list, even for those that didn't make the query
            print(f'{type(e).__name__} while querying {query_input}, exiting')
            fname = 'output/pcq_intermediate'
            gu.dict_to_sheet(pcq_out, fname)
            break

        if (i + 1) % 5 == 0 and idx != 0:
            print(f'processed {i+1} of {n_compounds} compounds')
        if i == n_compounds - 1:
            print(f'processed {i+1} of {n_compounds} compounds')
            print('done')
        if progress_callback: # for streamlit...
            progress_callback(i + 1, n_compounds, query_input)
            
    return pcq_out
        
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