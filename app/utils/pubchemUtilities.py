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
# import ctxpy as ctx

BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data"
ENDPOINT_TEMPLATE = f"{BASE_URL}/compound/{{compound_cid}}/JSON"

CAS_BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CAS_ENDPOINT_TEMPLATE = f"{CAS_BASE_URL}/compound/name/{{cas_number}}/cids/JSON"

CTX_CHEM_PROPERTIES = [
    'waterSolubilityTest', 'waterSolubilityOpera', 'viscosityCpCpTestPred',
    'vaporPressureMmhgTestPred', 'vaporPressureMmhgOperaPred', 'thermalConductivity',
    'tetrahymenaPyriformis', 'surfaceTension', 'soilAdsorptionCoefficient',
    'oralRatLd50Mol', 'operaKmDaysOperaPred', 'octanolWaterPartition', 'octanolAirPartitionCoeff',
    'meltingPointDegcTestPred', 'meltingPointDegcOperaPred', 'hrFatheadMinnow',
    'hrDiphniaLc50', 'henrysLawAtm', 'flashPointDegcTestPred', 'devtoxTestPred',
    'density', 'boilingPointDegcTestPred', 'boilingPointDegcOperaPred', 
    'biodegradationHalfLifeDays', 'bioconcentrationFactorTestPred', 'bioconcentrationFactorOperaPred',
    'atmosphericHydroxylationRate', 'amesMutagenicityTestPred', 'pkaaOperaPred',
    'pkabOperaPred'
    ]

def special_pcp_findParent(query_input, compound_cid):
    """
    Finds parent structures in PubChem entries. Performs an API request.
    
    Parameters & args:
        query_input (string): Name of queried structure. Printed to console when finding parent fails
        compound_cid (int): The salt CID, used to retrieve full PubChem entry
    Returns:
        parent_cid (int): CID of the parent structure
    """
    
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
                gu.deep_get(
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
    """
    Retrieves CAS & Comptox DTSXID data from full PubChem entries. Performs an API request.
    
    Parameters & args:
        compound_cid (int): Compound CID, used to retrieve full PubChem entry
    Returns:
        special_request_dict (dict): Dictionary with CID, Comptox DTSXID data
    """
    
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
        
        cas_number = gu.deep_get(data, [
            'Record',
            'Reference',
            lambda l: next((i for i in l if i.get('SourceName') == 'CAS Common Chemistry'), {}), 
            'SourceID'])
        
        # if not found, try here
        if not cas_number:
            cas_number = gu.deep_get(data, [
                'Record',
                'Section',
                lambda l: next((i for i in l if i.get('TOCHeading') == 'Names and Identifiers'), {}),
                'Section',
                lambda l: next((i for i in l if i.get('TOCHeading') == 'Other Identifiers'), {}),
                'Section',
                lambda l: next((i for i in l if i.get('TOCHeading') == 'CAS'), {}),
                'Information',
                0,
                'Value',
                'StringWithMarkup',
                0,
                'String'
            ])
            
        # also check if it is a string with only numbers and -'s?
        if cas_number and not all(c.isdigit() or c == '-' for c in cas_number):
            cas_number = None
        
        comptox_url = gu.deep_get(data, [
            'Record',
            'Reference',
            lambda l: next((i for i in l if i.get('SourceName') == 'EPA DSSTox'), {}), 
            'SourceID'])
        
        special_request_dict['cas'] = cas_number if cas_number else None
        special_request_dict['comptoxURL'] = comptox_url if comptox_url else None
    return special_request_dict

def casQuery_getCID(cas_input):
    """
    Function for querying PubChem using CAS numbers. Performs an API request.
    
    From a CAS number supplied by the user, the function finds the most relevant
    PubChem CID. This CID is returned and then used to do a standard query.
    This means that performing a CAS-based query for a compound will need
    one more API request to be completed, all things being equal.
    
    Parameters & args:
        cas_input (string): CAS number, given e.g. in web-app data frame
    Returns:
        special_request_dict (dict): Dictionary with CID, Comptox DTSXID data
    """
    
    endpoint = CAS_ENDPOINT_TEMPLATE.format(cas_number=cas_input)
    response = requests.get(endpoint)
    data = response.json()
    
    if 'Fault' in data:
        print(f'no corresponding CID found for CAS input {cas_input}')
        return None
    else:
        cid_from_cas = gu.deep_get(data, ['IdentifierList', 'CID'])[0]
    return cid_from_cas 

def nameCleaner_special(compound_name):
    """
    Not in use...
    """
    
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
    """
    Removes prefixes (e.g., certain stereochemistry specifications) from names to avoid query failures.
    May need to be removed or made optional. Currently used in pcQueries fn, below.
    """
    
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
    
# ---------- COMPTOX ----------

def get_DTXSID(chem_instance, data):
    try:
        # do a .search using the chem instance
        # prefer to use CAS. inchikey is also good, every cpd with pubchem data should have it.
        query = data['cas'] if data['cas'] else data['inchikey'] if data['inchikey'] else data['name'] if data['name'] else None
        if not query:
            return None
        res = chem_instance.search(by='equals', query=query)[0]
        if res and isinstance(res, dict):
            dtxsid = res.get('dtxsid', None)
            # CompTox seems to feature relevant cas numbers
            # Could pick that up while we're here
            ctx_cas = res.get('casrn', data['cas'])
            return dtxsid, ctx_cas
    except:
        return None
    
# ---------- WE NEED A NEW SOLUTION FOR COMPTOX ----------
# 260510 --- Temporarily or for longer term, ctxpy does not seem to resolve
# most of the properties anymore via the Chemical details path. 

CTX_BASE = 'https://comptox.epa.gov/ctx-api'

CTX_PROPNAMES_MAP = {
    # modelName: 'storageName' (on our end)
    # Variable data is stored inside propValue in the respective dict
    'TEST_TP_IGC50': '48H_T_Pyriformis_IGC50_TEST',
    'TEST_FHM_LC50': '96H_FatheadMinnow_LC50_TEST',
    'ACD_LogP_Consensus': 'LogKow_Percepta',
    'TEST_DM_LC50': '48H_D_Magna_LC50_TEST',
    'OPERA_PKA_A': 'pKa_A_Opera',
    'ACD_pKa_Apparent_MA': 'pKa_A_Percepta',
    'TEST_Mutagenicity': 'Ames_Mutagenicity_TEST',
    'OPERA_CoMPARA-Agonist': 'AR_Agonist_Opera',
    'OPERA_CoMPARA-Antagonist': 'AR_Antagonist_Opera',
    'OPERA_CoMPARA-Binding': 'AR_Binding_Opera',
    'TEST_V': 'Viscosity_TEST',
    'OPERA_PKA_B': 'pKa_B_Opera',
    'TEST_BCF': 'BioconcentrationFactor_TEST',
    'ACD_BP': 'BoilingPoint_Percepta',
    'TEST_BP': 'BoilingPoint_TEST',
    'OPERA_CACO2': 'Caco2Permeability_Opera',
    'ACD_Prop_Density': 'Density_Percepta',
    'TEST_D': 'Density_TEST',
    'ACD_Prop_Dielectric_Constant': 'DielectricConstant_Percepta',
    'OPERA_CERAPP-Agonist': 'ERa_Agonist_Opera',
    'OPERA_CERAPP-Antagonist': 'ERa_Antagonist_Opera',
    'TEST_DevTox': 'DevelopmentalToxicity_TEST',
    'OPERA_CERAPP-Binding': 'ERa_Binding_Opera',
    'ACD_FP': 'FlashPoint_Percepta',
    'TEST_FP': 'FlashPoint_TEST',
    'OPERA_FUB': 'UnboundFraction_HumanPlasma_Opera',
    'OPERA_Clint': 'IntrinsicClearance_HumanHepatic_Opera',
    'ACD_Prop_Index_Of_Refraction': 'RefractiveIndex_Percepta',
    'OPERA_RT': 'LC_RetentionTime_Opera',
    'OPERA_LOGD_PH_5_5': 'LogD_pH5.5_Opera',
    'ACD_LogD_5_5': 'LogD_pH5.5_Percepta',
    'OPERA_LOGD_PH_7_4': 'LogD_pH7.4_Opera',
    'ACD_LogD_7_4': 'LogD_pH7.4_Percepta',
    'TEST_MP': 'MeltingPoint_TEST',
    'ACD_Prop_Molar_Refractivity': 'MolarRefractivity_Percepta',
    'ACD_Prop_Molar_Volume': 'MolarVolume_Percepta',
    'TEST_Rat_LD50': 'Oral_Rat_LD50_TEST',
    'ACD_Prop_Polarizability': 'Polarizability_Percepta',
    'ACD_Prop_Surface_Tension': 'SurfaceTension_Percepta',
    'TEST_ST': 'SurfaceTension_TEST',
    'TEST_TC': 'ThermalConductivity_TEST',
    'ACD_VP': 'VaporPressure_Percepta',
    'TEST_VP': 'VaporPressure_TEST',
    'ACD_SolInPW': 'WaterSolubility_Percepta',
    'TEST_WS': 'WaterSolubility_TEST',
    'OPERA_AOH': 'AtmosphericHydroxylationRate_Opera',
    'OPERA_BCF': 'BioconcentrationFactor_Opera',
    'OPERA_BioDeg': 'BiodegradationHalfLife_Opera',
    'OPERA_BP': 'BoilingPoint_Opera',
    'OPERA_KM': 'FishBiotransformationHalfLife_km_Opera',
    'OPERA_HL': 'HenryVolatilityConstant_Opera',
    'OPERA_LogKOA': 'LogKoa_Opera',
    'OPERA_LogP': 'LogKow_Opera',
    'OPERA_MP': 'MeltingPoint_Opera',
    'OPERA_CATMoS-LD50': 'Oral_Rat_LD50_Opera',
    'OPERA_RBioDeg': 'ReadyBinaryBiodegradability_Opera',
    'OPERA_KOC': 'SoilAdsorptionCoefficient_Opero',
    'OPERA_VP': 'VaporPressure_Opera',
    'OPERA_WS': 'WaterSolubility_Opera'
}

def get_comptox_properties(
        dtxsid: str,
        api_key: str,
        prop_map: dict = CTX_PROPNAMES_MAP,
) -> dict:
    """
    Replacement for previous ctxpy functions.
    Retrieves predicted properties from the CTX API.
    """
    url = f'{CTX_BASE}/chemical/property/predicted/search/by-dtxsid/{dtxsid}'
    headers = {'x-api-key': api_key, 'accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 404:
            print(f'get_comptox_properties: DTXSID {dtxsid!r} not found')
            return {}
        response.raise_for_status()
        raw = response.json()
    except requests.exceptions.Timeout:
        print(f'get_comptox_properties: request timed out for {dtxsid!r}')
        return {}
    except requests.exceptions.RequestException as e:
        print(f'get_comptox_properties: request error for {dtxsid!r} — {e}')
        return {}

    # Initialize all props as None...
    result = {storage: None for storage in prop_map.keys()}
    #print(result)

    # The response is a list of dicts for the different props
    # We pick them out via modelName variables, taken from the prop_map keys
    if not isinstance(raw, list):
        print(f'get_comptox_properties: unexpected response type for {dtxsid!r} — {type(raw)}')
        return result

    for item in raw:
        #print(item)
        model_name = item.get('modelName')
        if model_name in prop_map:
            # DONT convert here, lets do it later
            result[model_name] = item.get('propValue')
    
    #print(result)
    return result

# get_comptox_properties('DTXSID2023270', '0c5e2d7b-b651-4a21-a798-98d853dc9859')

def search_comptox_raw(identifier: str, api_key: str) -> dict:
    """
    Sends a request to the CTX API to find DTXSIDs.
    SMILES or CAS should be used as identifiers.
    """
    url = f'{CTX_BASE}/chemical/search/equal/{requests.utils.quote(identifier)}'
    headers = {'x-api-key': api_key, 'accept': 'application/json'}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 404:
            print(f'search_comptox_raw: no match for {identifier!r}')
            return {}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print(f'search_comptox_raw: request timed out for {identifier!r}')
        return {}
    except requests.exceptions.RequestException as e:
        print(f'search_comptox_raw: request error for {identifier!r} — {e}')
        return {}
    
# search_comptox_raw('375-73-5', '0c5e2d7b-b651-4a21-a798-98d853dc9859')[0].get('dtxsid')

# ------------------ QUERIES ------------------
    
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
    """
    Helper for pcQueries function.
    Performs the actual querying of PubChem data via PubChemPy.
    Special metadata retrieval (CAS, DTSXID) happens in pcQueries proper.
    
    Parameters & args:
        query_input (string): Input used to query data for a single compound
        query_type (string): Holds information on what kind of query (name, CID etc) is being done
    Returns:
        pc_data (Compound object): PubChemPy Compound object with compound data
    """
    
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

def pcQueries(query_dict, query_empty_only=True, 
              query_comptox=False, api_key=None, ctx_query_specs=None,
              progress_callback=None
  ):
    """
    Organizes PubChem queries for the pcq module.
    
    Parameters & args:
        query_dict (dict): Dictionary created from input (e.g., sheet) to the pcq module
        query_empty_only (bool): Controls whether to perform queries for compounds with prior data
        progress_callback: Used for web-app progress bar visuals
    Returns:
        pcq_out (dict): Dictionary with column-indexed (e.g., library ID) data.
    """
    
    n_compounds = len(query_dict.keys())
    pcq_out = {} # create a dictionary to store data in an organized format
    print(f'running pcq for {n_compounds} compounds')
    # assume a query_dict structure key = number, data = query inputs
    for i, (idx, data) in enumerate(query_dict.items()):
        pc_data = None # initialize pc_data
        cas_retrieved = False # retrieval success flag for cas queries
        original_cas_q = None
        pcq_out[idx] = {field: None for field in ALL_PCQ_FIELDS} # initialize dict entry for query
        name_q, smiles_q, cid_q, cas_q = data.get('name_q'), data.get('smiles_q'), data.get('cid_q'), data.get('cas_q')
        # hierarchy of input types --- prioritize cid, then name, then smiles, then cas
        query_input = cid_q if cid_q else name_q if name_q else smiles_q if smiles_q else cas_q
        query_type = 'cid' if cid_q else 'name' if name_q else 'smiles' if smiles_q else 'cas' if cas_q else None
        
        # if we are querying with a name, clean it
        if name_q and not cid_q:
            query_input = nameCleaner(name_q) if str(name_q) != 'nan' else name_q
        
        # hmm... think about this one.
        if not gu.is_empty(data.get('queried_at')) and query_empty_only:
            continue
        
        # little sidestep to deal with cas inputs. we get the CID, and change the query type.
        if query_type == 'cas' and cas_q:
            cas_from_cid = casQuery_getCID(cas_q)
            if cas_from_cid is not None and str(cas_from_cid).isdigit():
                original_cas_q = (query_input, query_type)
                query_input, query_type = cas_from_cid, 'cid'
                cas_retrieved = True
            else:
                cas_retrieved = False
        
        # now helper function for querying
        try:
            # deal with non-cas and cas stuff
            if query_type != 'cas':
                pc_data = pcQuery_expanded(query_input, query_type)
            else:
                if cas_retrieved:
                    pc_data = pcQuery_expanded(query_input, query_type)
                else:
                    pc_data = None
                    
            if not pc_data:
                print(f'no pubchem entry found --- input: [{query_input}], type: [{query_type}]')
                pcq_out[idx]['queried_as'] = (query_input, query_type)
                continue
            
            # store query information
            pcq_out[idx]['queried_as'] = original_cas_q if original_cas_q else (query_input, query_type)
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
            
            # ---------- query comptox HERE? ---------- 
            if query_comptox and api_key and ctx_query_specs:   
                # use dtxsid from pubchem if we have it
                dtxsid = pcq_out[idx].get('comptoxURL', None)
                # if not...
                if not dtxsid:
                    # try w SMILES first
                    dtx_q = search_comptox_raw(pcq_out[idx]['smiles'], api_key)
                    dtxsid = dtx_q[0].get('dtxsid', None)
                    if not dtxsid:
                        dtx_q = search_comptox_raw(pcq_out[idx]['cas'], api_key)
                        dtxsid = dtx_q[0].get('dtxsid', None)

                if dtxsid:
                    try:
                        #print(dtxsid)
                        #print(api_key)
                        #print(ctx_query_specs)
                        ctx_data = get_comptox_properties(dtxsid, api_key)
                        if ctx_data:
                            #print(ctx_data)
                            for prop in ctx_query_specs:
                                if prop in ctx_data:
                                    # Variable names are already re-mapped
                                    # by the get_comptox_properties function
                                    pcq_out[idx][CTX_PROPNAMES_MAP[prop]] = ctx_data[prop]
                    except:
                        pass
                
        except Exception as e:
            # need to change this, later.
            # probably what will be useful when connection errors happen
            # need to make sure pcq_out will contain all query input types
            # in a list, even for those that didn't make the query
            print(f'{type(e).__name__} while querying {query_input}, exiting')
            #fname = 'output/pcq_intermediate'
            #gu.dict_to_sheet(pcq_out, fname)
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