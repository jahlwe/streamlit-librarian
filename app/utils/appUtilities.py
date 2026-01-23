# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:54:21 2025

@author: Jakob
"""

# app utilities 
# we need to adapt our functions for working with io bytes and so on

import pandas as pd
import glob
import os
import utils.genericUtilities as gu
import utils.pubchemUtilities as pu
import utils.compilerUtilities as cu
import utils.qcUtilities as qu
import utils.fragmentAnnotationNew as fa
import csv
import pandas as pd
from utils.spectrum import Spectrum
from utils.spectrum_type import SpectrumType
from utils.splash import Splash
from datetime import date
from datetime import datetime
import re
import time
import io
import zipfile
import streamlit as st
import plotly.graph_objects as go
import math

# --- PCQ SHEET TEMPLATE ---
def generate_pcq_template():
    """
    Generates a .csv pcq module template, downloadable via the web-app.
    """
    
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', lineterminator='\n')
    writer.writerow(['library_id', 'name_q', 'cas_q', 'smiles_q', 'cid_q'])
    return output.getvalue()

# --- PCQ RE-QUERY (sheet-based) HELPER ---
QUERY_FIELDS = ['library_id', 'name_q', 'cas_q', 'smiles_q', 'cid_q']

def query_dict_from_pcq_input(pcq_input):
    """
    Generates a dictionary from input to the pcq module, used during the query process.
    """
    
    query_dict = {}
    for idx, data in pcq_input.items():
        # only include entries not successfully queried yet
        if data.get('queried_at'):
            continue
        # get updated input & type
        query_input, query_type = (data.get('queried_as') or (None, None))
        query_type = query_type + '_q' if query_type else query_type
        # build dict
        entry = {field: None for field in QUERY_FIELDS}
        if query_type in QUERY_FIELDS:
            entry[query_type] = query_input
        # allow manual CID to override
        if data.get('pubchemCID'):
            entry['cid_q'] = data['pubchemCID']
        # also keep any assigned library_id
        if data.get('library_id'):
            entry['library_id'] = data['library_id']
        query_dict[idx] = entry
    return query_dict

# --- NEEDED STUFF ---
# use to convert less-complicated storage names of variables
# to the final massbank-format variable names.
# keeping names consistent with those from PubChem queries
# for simplified transfer from those sheets to the dictionary.
FIELD_CONVERSION = {
    # variable storage in python: massbank format-fields
    'accession': 'ACCESSION:',
    'title': 'RECORD_TITLE:',
    'date': 'DATE:',
    'authors': 'AUTHORS:',
    'license': 'LICENSE:',
    'copyright': 'COPYRIGHT:',
    'comment_1': 'COMMENT:', # add more as needed
    'comment_2': 'COMMENT:',
    'library_id': 'CH$NAME:',
    'iupacName': 'CH$NAME:',
    'class': 'CH$COMPOUND_CLASS:',
    'molecularFormula': 'CH$FORMULA:',
    'monoisotopicMass': 'CH$EXACT_MASS:',
    'smiles': 'CH$SMILES:',
    'inchi': 'CH$IUPAC:',
    'cas': 'CH$LINK: CAS',
    'pubchemCID': 'CH$LINK: PUBCHEM CID:',
    'inchikey': 'CH$LINK: INCHIKEY',
    'comptoxURL': 'CH$LINK: COMPTOX',
    'instrument': 'AC$INSTRUMENT:',
    'instrument_type': 'AC$INSTRUMENT_TYPE:',
    'ms_type': 'AC$MASS_SPECTROMETRY: MS_TYPE',
    'ion_mode': 'AC$MASS_SPECTROMETRY: ION_MODE',
    'ionization': 'AC$MASS_SPECTROMETRY: IONIZATION',
    'fragmentation_mode': 'AC$MASS_SPECTROMETRY: FRAGMENTATION_MODE',
    'collision_energy': 'AC$MASS_SPECTROMETRY: COLLISION_ENERGY',
    'resolution': 'AC$MASS_SPECTROMETRY: RESOLUTION',
    'column_name': 'AC$CHROMATOGRAPHY: COLUMN_NAME',
    'flow_gradient': 'AC$CHROMATOGRAPHY: FLOW_GRADIENT',
    'flow_rate': 'AC$CHROMATOGRAPHY: FLOW_RATE',
    'retention_time': 'AC$CHROMATOGRAPHY: RETENTION_TIME',
    'rti': 'AC$CHROMATOGRAPHY: UOA_RTI',
    'solvent_a': 'AC$CHROMATOGRAPHY: SOLVENT A',
    'solvent_b': 'AC$CHROMATOGRAPHY: SOLVENT B',
    'chromatography_comment_1': 'AC$CHROMATOGRAPHY: COMMENT', # same here
    'base_peak': 'MS$FOCUSED_ION: BASE_PEAK',
    'precursor_mz': 'MS$FOCUSED_ION: PRECURSOR_M/Z',
    'ion_type': 'MS$FOCUSED_ION: ION_TYPE',
    'data_processing': 'MS$DATA_PROCESSING:',
    'splash': 'PK$SPLASH:',
    'num_peak': 'PK$NUM_PEAK:',
    'ms2_peaks': 'PK$PEAK:',
    'ms2_annot': 'PK$ANNOTATION:',
}

# use this at the end of compilation to generate a mastersheet 
# (or, something that can be copied to the actual mastersheet),
# annoying, but must do this...
mastersheet_field_conversion = FIELD_CONVERSION.copy()
mastersheet_field_conversion['comment_2'] = 'COMMENT: (2)'
mastersheet_field_conversion['library_id'] = 'CH$NAME:'
mastersheet_field_conversion['iupacName'] = 'CH$NAME: (IUPAC)'
MASTERSHEET_COLUMNS = {
    'file_name': 'File_name',
    'short_accession': 'Short ACCESSION name:',
    **mastersheet_field_conversion,
    'submitted_to_MBEU': 'Submitted to MBEU'
}

# fields for msp.
MSP_FIELDS = {
    'precursor_mz': 'PRECURSORMZ:',
    'ion_type': 'PRECURSORTYPE:',
    'molecularFormula': 'FORMULA:',
    'class': 'Ontology:',
    'inchikey': 'INCHIKEY:',
    'smiles': 'SMILES:',
    'retention_time': 'RETENTIONTIME:',
    'ion_mode': 'IONMODE:',
    'instrument_type': 'INSTRUMENTTYPE:',
    'instrument': 'INSTRUMENT:',
    'collision_energy': 'COLLISIONENERGY:',
    'num_peak': 'Num Peaks:'    
}

# --- METADATA TEMPLATE ---
METADATA_TEMPLATE_FIELDS = {
    'authors': 'AUTHOR_NAME', 'license': 'LICENSE', 'copyright': 'COPYRIGHT_HOLDER',
    'comment_1': 'COMMENT_FIELD', 'instrument': 'INSTRUMENT_MODEL', 'instrument_type': 'INSTRUMENT_TYPE',
    'ms_type': 'MS2', 'ionization': 'IONIZATION', 'fragmentation mode': 'FRAGMENTATION_MODE',
    'collision_energy': 'CE', 'resolution': 'XXXXX', 'column_name': 'COLUMN_NAME',
    'flow_gradient': 'FLOW_GRADIENT', 'flow_rate': 'FLOW_RATE', 'solvent_a': 'SOLV_A', 
    'solvent_b': 'SOLV_B', 'chromatography_comment_1': 'CHR_COMMENT', 'data_processing': 'DATA_PROCESSING'
    }

def generate_metadata_template():
    """
    Generates a metadata template for pre-assembly, downloadable via the web-app.
    """
    
    output = io.StringIO()
    writer = csv.writer(output, delimiter='\t', lineterminator='\n')
    for field, value in METADATA_TEMPLATE_FIELDS.items():
        writer.writerow([field, value])
    return output.getvalue()

# --- MAT HANDLING ---

MAT_FIELDS = {
    # mat category: dictionary storage name
    # dictionary storage names for these
    # categories should of course match above
    'RETENTIONTIME:': 'retention_time',
    'PRECURSORMZ:': 'precursor_mz',
    'PRECURSORTYPE:': 'ion_type',
    'IONMODE:': 'ion_mode',
    'NAME:': 'library_id',
    'Num Peaks:': 'num_peak'
}

def get_charge(adduct):
    """
    Helper, gets the charge value of an adduct for calculations.
    """

    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        number = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * number
    return 0 # if no charge 

def read_archive(mat_archive, archive_type):
    """
    Reads .mat files from .zip archive, and returns a dictionary with the 
    file contents indexed by file names.
    Used in pre-assembly.
    """
    
    mat_files_dict = {}
    if archive_type == 'zip':
        with zipfile.ZipFile(mat_archive) as z:
            for name in z.namelist():
                if name.endswith('.mat'):
                    try:
                        mat_files_dict[name] = io.BytesIO(z.read(name))
                    except Exception as e:
                        st.warning(f'Failed to read {name}: {e}')
    #elif archive_type == 'rar': # RAR didn't work? File reading error?
    #    with rarfile.RarFile(mat_archive) as r:
    #        for name in r.namelist():
    #            if name.endswith('.mat'):
    #                try:
    #                    mat_files_dict[name] = io.BytesIO(r.read(name))
    #                except Exception as e:
    #                    st.warning(f'Failed to read {name}: {e}')
    return mat_files_dict

def read_archive_RTI(rti_archive, archive_type):
    """
    Reads RTI web app .csv files from .zip archive, and returns a dictionary 
    with the file contents indexed by file names.
    Used in pre-assembly.
    """
    
    rti_files_dict = {}
    if archive_type == 'zip':
        with zipfile.ZipFile(rti_archive) as z:
            for name in z.namelist():
                if name.endswith('.csv'):
                    try:
                        rti_files_dict[name] = io.BytesIO(z.read(name))
                    except Exception as e:
                        st.warning(f'Failed to read {name}: {e}')
    return rti_files_dict

def gather_RTIData_app(rti_files_dict):
    """
    Reads RTI data assembled in a dictionary by the read_archive_RTI fn.
    Returns a dictionary with RTI data indexed by compound names.
    """
    
    rti_dictionary = {}
    for i, (name, file) in enumerate(rti_files_dict.items()):
        if not hasattr(file, 'name'):
            file.name = name # need to do this for s2d to get it?
        current_dict = gu.sheet_to_dict(file, 'Compound Name')
        rti_dictionary.update(current_dict)
    return rti_dictionary

def add_cfData_app(dictionary, cf_data):
    """
    Adds chemont data from the Fiehn lab ClassyFire Batch portal. 
    https://cfb.fiehnlab.ucdavis.edu/
    
    Functionally redundant since MassBank update to automatically add chemont data to new entries.
    """
    
    for compound, data in dictionary.items():
        current_inchikey = data.get('inchikey')
        if current_inchikey and current_inchikey in cf_data.keys():
            class_data = cf_data[current_inchikey]
            # maybe think about this later...
            if str(class_data['Subclass']) == 'nan':
                class_string = '; '.join([
                    str(class_data['Superclass']), str(class_data['Class'])
                    ])
                dictionary[compound]['class'] = class_string
            else:
                class_string = '; '.join([
                    str(class_data['Class']), str(class_data['Subclass']),
                    str(class_data['Parent Level 1'])
                    ])
                dictionary[compound]['class'] = class_string
        else:
            print(f'no InChIKey match for {compound}')
    return dictionary

def parse_matFile_app(
    file, 
    dictionary, 
    mode,
    custom_mat_fields={},
    normalize_ms2=True
):
    """
    Function for reading single .mat files and storing the data in a dictionary.
    Adapted from the compilerUtilities fn to the Streamlit web-app context.

    Parameters & args:
        file (binary file-like): Single .mat file
        dictionary (dict): Dictionary to add compound information
        mode (string): Current mode --- pos/neg are compiled separately
        custom_mat_fields (dict): Dictionary w non-standard .mat fields to extract data from
        nonormalize_ms2 (bool): Controls whether ms2 data is normalized (separate column created)
        
    Returns:
        dictionary (dict): Dictionary w added compound information
    """
    
    current_compound = None
    current_record = {}
    reading_peaks = False
    peak_data = [] # MS2 container
    n_peaks = 0 # n MS2 peaks
    mode_sign = 1 if mode == 'pos' else -1
    
    with io.TextIOWrapper(file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('NAME:'):
                base_name = line.split(': ', 1)[1].strip()
                current_compound = base_name
                feature_index = 1
                
                while current_compound in dictionary:
                    feature_index += 1
                    current_compound = f'{base_name} feature no. {feature_index}'
                
                current_record = {'keyColumn': 'library_id'}
                continue
            
            for field, storage in MAT_FIELDS.items():
                if line.startswith(field):
                    value = line.split(': ', 1)[1].strip()
                    if storage in ['retention_time', 'precursor_mz']:
                        value = round(float(value), 2 if storage == 'retention_time' else 5)
                    current_record[storage] = value
                    
            if custom_mat_fields: # custom mat fields
                for field, storage in custom_mat_fields.items():
                    if line.startswith(field):
                        value = line.split(': ', 1)[1].strip()
                        current_record[storage] = value
                    
            if line.startswith('MSTYPE: MS2'):
                reading_peaks = True
                continue
            
            if reading_peaks:
                if line.startswith('Num'):
                    n_peaks = int(line.split(': ', 1)[1])
                    current_record['num_peak'] = n_peaks
                elif len(peak_data) < current_record.get('num_peak', 0):
                    mz, intensity = map(float, line.split())
                    peak_data.append((round(mz, 5), int(intensity)))
                    
                if len(peak_data) == current_record.get('num_peak', 0):
                    reading_peaks = False
                    current_record['ms2_data'] = peak_data
                    if normalize_ms2 and peak_data:
                        current_record['ms2_norm'] = qu.normalize_peaks(peak_data)
                    current_record['base_peak'] = round(max(peak_data, key=lambda x: x[1])[0], 5)
    
    # add records
    # EXTRA MODE VALIDATION!
    # Use get_charge on the ion type data, compare it to mode_sign
    # only if the mat file contains an ion type annotation though
    if current_record.get('ion_type', None):
        mode_agreement = get_charge(current_record['ion_type']) == mode_sign  
        if current_compound and current_record and mode_agreement:
            dictionary[current_compound] = current_record.copy()
    # if we don't have an ion type annotation, we have to leave it empty for now
    # autoassign it later...
    else:
        current_record['ion_type'] = None
        dictionary[current_compound] = current_record.copy()
        
    return dictionary

def gather_matData_app(mat_files_dict, mode, custom_mat_fields={}):
    """
    Helper, uses parse_matFile_app fn to gather data from all provided .mat files.
    """
    
    mat_dictionary = {}
    print(custom_mat_fields)
    for name, file_obj in mat_files_dict.items():
        if f'{mode}/' in name:
            # folder names are retained, so use it for mode filtering
            mat_dictionary = parse_matFile_app(file_obj, mat_dictionary, mode, custom_mat_fields)
    return mat_dictionary

def add_manual_metadata_app(dictionary, metadata_file):
    """
    Function to add manually provided experimental metadata to the compilation dictionary.
    This data is provided by the user in a separate .tsv file.
    
    Parameters & args:
        dictionary (dict): Pre-assembly dictionary to which data is added
        metadata_file (file): .tsv file with manual metadata
        
    Returns:
        dictionary (dict): Updated pre-assembly dictionary
    """
    
    manual_dictionary = {}
    metadata_file.seek(0)  # ensure "pointer" at start
    # need to wrap the file like we did for mat files
    with io.TextIOWrapper(metadata_file, encoding='utf-8') as m:
        reader = csv.reader(m, delimiter='\t')
        for key, value in reader: # store variables + values
            manual_dictionary[key] = value
    for compound in dictionary: # add them to each entry
        dictionary[compound].update(manual_dictionary)
    return dictionary

BR_ISOTOPE_DIFF = 1.99795 # need this for bromine stuff
CL_ISOTOPE_DIFF = 1.99705 # not implemented yet
# maybe do for Cl later --- polychlorinated might need it?

def adduct_assigner(compound, data):
    """
    Function to assign ion / adduct types to feature data that lack annotation.
    Evaluates whether experimental precursor m/z is within at least 10 ppm 
    of any of candidate adducts below. 

    Parameters & args:
        compound (string): Library ID for a compound in the pre-assembly dictionary
        data (dictionary): Compound variable data from the pre-assembly dictionary

    Returns:
        best_adduct (string): Assigned ion type/adduct
    """
    
    # if mat files lack adduct annotation...
    e_mass = 0.00054858
    h_mass = 1.00783
    adducts = {
        # pos
        # add 2x charge and neutral losses later
        '[M]+': - e_mass,
        '[M]2+': - (2 * e_mass),
        '[M+H]+': h_mass - e_mass,
        '[M+NH4]+': 14.00307 + (4 * h_mass) - e_mass,
        '[M+Na]+': 22.98977 - e_mass,
        '[M+K]+': 38.96371 - e_mass,
        '[M+2H]2+': (2 * h_mass) - (2 * e_mass),
        '[M+H-H2O]+': 17.00274 - e_mass,
        # neg
        '[M-H]-': - h_mass + e_mass,
        '[M+Cl]-': 34.96885 + e_mass,
        '[M+F]-': 18.99840 + e_mass,
        '[M-2H]2-': (-2 * h_mass) + (2 * e_mass),
        '[M-H2O-H]-': -19.01839 + e_mass
    }
    adduct = None
    monomass = data.get('monoisotopicMass', None)
    exp_mz = data.get('precursor_mz', None)
    
    # dealing with Br (Cl later, too...?)
    formula = data.get('molecularFormula', None)
    nBr = 0
    if formula and 'Br' in formula:
        atom_count = fa.parse_formula(formula)
        nBr = atom_count.get('Br', 0)
    
    # subset adducts for the current mode
    # don't know if we will ever have to worry about not having ion_mode data.
    charge_indicator = '+' if data.get('ion_mode', None).lower() == 'positive' else '-' if data.get('ion_mode', None).lower() == 'negative' else None
    adduct_subset = {k: v for k, v in adducts.items() if k.endswith(charge_indicator)}
    
    best_ppm = float('inf')
    best_adduct = None
    
    if monomass is None or exp_mz is None:
        return None
    
    for adduct in adduct_subset.keys():
        charge = get_charge(adduct)
        print('outside loop')
        for i in range(nBr + 1 if nBr > 0 else 1):
            print('inside loop')
            theo_mz = (monomass + adduct_subset[adduct] + i * BR_ISOTOPE_DIFF) / abs(charge)
            ppm_dev = abs(((theo_mz - exp_mz)/ theo_mz) * 1e6)
            if ppm_dev < 10 and ppm_dev < best_ppm:
                best_ppm = ppm_dev
                best_adduct = adduct
        
    return best_adduct if best_adduct else None

def adduct_checker(compound, data):
    """
    Function to validate given ion type/adduct data.
    Evaluates whether experimental precursor m/z is within expected bounds
    given the ion type and the theoretical monoisotopic mass.
    Results of the validation are provided in the pre-assembly module output sheet.

    Parameters & args:
        compound (string): Library ID for a compound in the pre-assembly dictionary
        data (dictionary): Compound variable data from the pre-assembly dictionary

    Returns:
        adduct (string): Ion type/adduct
    """
    
    # in particular Br-containing compounds might need to calculate 
    # two monomasses, one for M and one for M+2 etc
    e_mass = 0.00054858
    h_mass = 1.00783
    adducts = {
        # pos
        # add 2x charge and neutral losses later
        '[M]+': - e_mass,
        '[M]2+': - (2 * e_mass),
        '[M+H]+': h_mass - e_mass,
        '[M+NH4]+': 14.00307 + (4 * h_mass) - e_mass,
        '[M+Na]+': 22.98977 - e_mass,
        '[M+K]+': 38.96371 - e_mass,
        '[M+2H]2+': (2 * h_mass) - (2 * e_mass),
        '[M+H-H2O]+': 17.00274 - e_mass,
        # neg
        '[M-H]-': - h_mass + e_mass,
        '[M+Cl]-': 34.96885 + e_mass,
        '[M+F]-': 18.99840 + e_mass,
        '[M-2H]2-': (-2 * h_mass) + (2 * e_mass),
        '[M-H2O-H]-': -19.01839 + e_mass
    }
    adduct = data.get('ion_type', None)
    monomass = data.get('monoisotopicMass', None)
    exp_mz = data.get('precursor_mz', None)
    
    # dealing with Br and Cl
    formula = data.get('molecularFormula', None)
    nBr = 0
    nCl = 0
    if formula and ('Br' in formula or 'Cl' in formula):
        atom_count = fa.parse_formula(formula)
        nBr = atom_count.get('Br', 0)
        nCl = atom_count.get('Cl', 0)
    
    # subset adducts for the current mode
    charge_indicator = adduct[-1]
    adduct_subset = {k: v for k, v in adducts.items() if k.endswith(charge_indicator)}
    
    def matches(adduct):
        charge = get_charge(adduct)
        for i in range(nBr + 1 if nBr > 0 else 1):
            for j in range(nCl + 1 if nCl > 0 else 1):
                iso_shift = i * BR_ISOTOPE_DIFF + j * CL_ISOTOPE_DIFF
                theo_mz = (monomass + adduct_subset[adduct] + iso_shift) / abs(charge)
                ppm_dev = abs((theo_mz - exp_mz) / theo_mz * 1e6)
                if ppm_dev < 10:
                    return True
        return False
    
    # look through the subset for matches
    if adduct in adduct_subset and matches(adduct):
        return adduct, True
    
    for other_adduct in adduct_subset:
        if other_adduct == adduct:
            continue
        if matches(other_adduct):
            print(f'more suitable ion type {other_adduct} for {compound} --- discarding {adduct}')
            return other_adduct, True
    print(f'unsuitable adduct for {compound} but no alternative found --- validate manually')
    return adduct, False

# combined assembling preComp dict with the preparation steps.
def preCompile_app(
    mode,
    pcq_data, 
    metadata_tsv,
    mat_data,
    storage_fields,
    rti_data=None,
    cf_data=None,
    # True for annotation, false for remove unannotated as basic settings
    annotate_fragments=(True, False),
    progress_callback=None
):
    """
    Organizes pre-assembly for the web-app.

    Parameters & args:
        mode (string): Current mode, pos/neg
        pcq_data (dict): Dictionary containing compound chemical metadata
        metadata_tsv (dict): Dictionary containing experimental metadata
        mat_data (dict): Dictionary containing compound feature data
        storage_fields (list): List of variables to store for each compound
        rti_data (dict): Dictionary containing RTI data
        cf_data (dict): Dictionary containing chemont data from ClassyFire
        annotate_fragments (bool): Controls whether fragment formula annotation is performed
        progress_callback: Used for web-app progress bar visuals

    Returns:
        dictionary (dict): Pre-assembly dictionary
    """
    
    if mat_data:
        # we do call for create compilation dictionary here
        # now need to bring the full, possibly custom storage fields in explicitly
        dictionary = cu.create_compilation_dictionary(mat_data, storage_fields)
        dictionary = cu.add_chemical_metadata(dictionary, pcq_data)
        dictionary = add_manual_metadata_app(dictionary, metadata_tsv)
        
        # optional stuff
        if rti_data:
            dictionary = cu.add_RTIData(dictionary, rti_data)
        if cf_data:
            dictionary = add_cfData_app(dictionary, cf_data)
            
    validate = True # for ion type stuff below...
    
    len_before_drop = len(dictionary)
    # need to do this to skip compounds without pcq data
    # use a few of the common metadata types to check that sufficient data is present
    dictionary = {c: d for c, d in dictionary.items() if d.get('monoisotopicMass') and d.get('molecularFormula') and d.get('smiles')}
    len_after_drop = len(dictionary)
    print(f'dropped {len_before_drop-len_after_drop} compounds lacking chemical metadata')
    
    if dictionary:
        total = len(dictionary)
        for i, (compound, data) in enumerate(dictionary.items()):
            # first! calculate ion types for those that lack it
            if not data.get('ion_type'):
                data['ion_type'] = adduct_assigner(compound, data)
                data['adduct_validated'] = 'assigned'
                validate = False
            else:
                validate = True
            #print('past assigner')
            # generate record_title and save sheet
            short_name = re.sub(r' feature no\. \d+$', '', compound)
            record_title = '; '.join(
                [short_name, data['instrument_type'], data['ms_type'], 
                 data['collision_energy'], data['resolution'], data['ion_type']])
            data['title'] = record_title
            # also! run the adduct checker. IF there is prior ion type data.
            if validate:
                data['ion_type'], data['adduct_validated'] = adduct_checker(compound, data)
            # also! annotate fragments.
            if annotate_fragments[0]: # if we should annotate, this is True
                try:
                    print(f'annotating {compound} MS2')
                    loss_fragments = fa.generate_ref_fragments(data)
                    loss_fragments = fa.generate_more_fragments(data, loss_fragments)
                    match_list = fa.match_loss_fragments(data, loss_fragments)
                    data['frag_annot'] = fa.format_annotation(data, match_list)
                    if annotate_fragments[1]: # if we should discard, this is True
                        print('discarding unannotated fragments')
                        ms2_trimmed = []
                        fa_trimmed = []
                        for j, (mz, abs_int, norm_int) in enumerate(data.get('ms2_norm')):
                            # fragment formula is None if unassigned
                            # this is the data that corresponds to current peak
                            frag_annot = data['frag_annot'][j]
                            if frag_annot[1] is not None:
                                ms2_trimmed.append((mz, abs_int, norm_int))
                                fa_trimmed.append(frag_annot)
                        print(f'trimmed from {len(data.get("ms2_norm", []))} to {len(ms2_trimmed)} peaks')
                        data['ms2_norm'] = ms2_trimmed
                        data['frag_annot'] = fa_trimmed
                except Exception as e:
                    data['frag_annot'] = None
                    print(f'failed fragment annotation for {compound}: {e}')
                finally: # this is a thing?
                    if progress_callback:
                        progress_callback(i+1, total, compound)
                        
    return dictionary

# --- COMPILATION ---

def filter_preComp_app(
    dictionary, 
    mode
):
    """
    Optional fn to filter the pre-assembly sheet.
    
    Compared to the CLI verison, only deals with experimental features with
    identical names. Compounds that there is more than one experimental feature for
    and that have the sane name (i.e. two .mat files both with the name Caffeine)
    have previously had the latter feature given the name "Caffeine feature n."
    In this step, these features are merged into one, and information about
    the latter feature appended to the first as a comment.
    
    Deprecated...? Filtering behavior can be avoided by having unique
    names for all feature data.

    Parameters & args:
        dictionary (dict): Pre-assembly dictionary
        mode (string): Current mode, pos/neg

    Returns:
        filtered_dict (dict): Pre-assembly dictionary, filtered
    """
    
    filtered_dict = {}
            
    # add info about (presumed) enantiomer peaks in comments of one of the entries
    for compound, data in dictionary.items():
        if 'Candidate' not in compound:           
            if 'feature' in compound:
                # only convert 'feature' entries to comments in related 
                # compounds if they are of the same adduct type
                short_name = re.sub(r' feature no\. \d+$', '', compound)
                same_adduct = True if dictionary[short_name]['ion_type'] == dictionary[compound]['ion_type'] else False
                if short_name in dictionary.keys() and same_adduct:
                    # no support for deciding which enantiomer is major - maybe implement later
                    comment_line = f'Enantiomer MS1 peak at RT {dictionary[compound]["retention_time"]}'
                    dictionary[short_name]['comment_2'] = comment_line
                elif short_name in dictionary.keys() and not same_adduct:
                    # unique adducts get their own entries
                    filtered_dict[compound] = data
            else:
                filtered_dict[compound] = data      
                
    return filtered_dict

def compileLib_app(
    dictionary,
    acc_start,
    acc_full,
    acc_short,
    mode
):
    """
    Organizes assembly.
    Adds final necessary variables for each compound to the pre-assembly dictionary and returns it.

    Parameters & args:
        dictionary (dict): Pre-assembly dictionary
        acc_start (int): Accession numbering, starting value
        acc_full (string): Full MassBank-format accession prefix
        acc_short (string): Short MassBank-format accession prefix
        mode (string): Current mode, pos/neg
        
    Returns:
        dictionary (dict): Assembly dictionary
    """
    
    # fill in the the final data fields using provided acc info
    for i, (compound, data) in enumerate(dictionary.items()):
        acc_n = f'{acc_start + i:06d}'
        current_acc = f'{acc_full}{acc_n}'
        short_acc = f'{acc_short}{acc_n}'
        current_file = f'{compound}_{short_acc}'

        data['acc'] = current_acc
        data['short_acc'] = short_acc
        # MassBank wants dates in this format! yyyy.mm.dd
        data['date'] = str(date.today()).replace('-', '.')
        data['file_name'] = current_file

        # splash and MS2
        peak_data_splash = [(mz, abs_int) for mz, abs_int in data['ms2_data']]
        current_ms2 = Spectrum(peak_data_splash, SpectrumType.MS)
        data['splash'] = str(Splash().splash(current_ms2))
        peak_line = 'm/z int. rel.int.\n' + ''.join(
            f'  {mz} {abs_int} {norm_int}\n' for mz, abs_int, norm_int in data['ms2_norm']
        )
        data['ms2_peaks'] = peak_line
        
        if data.get('frag_annot', None):
            annot_line = 'm/z tentative_formula formula_count mass error(ppm)\n' + ''.join(
                f'  {theo_mz} {t_f} {f_count} {exp_mz} {ppm}\n' for theo_mz, t_f, f_count, exp_mz, ppm in data['frag_annot'] if theo_mz is not None
            )
            data['ms2_annot'] = annot_line
        
        # here we can create a reconciled peak list for display
        # this will of course be unique for the app
        # and the contents of this will not be given in the compSheet
        if data.get('frag_annot', None):
            annot_lookup = {}
            # collect all annotated peaks
            for theo_mz, formula, count, exp_mz, ppm in data.get('frag_annot', []):
                key = round(exp_mz, 4)  # round to 4 decimals to avoid floating point issues
                annot_lookup[key] = {
                    'theo_mz': theo_mz,
                    'formula': formula,
                    'count': count,
                    'ppm': ppm
                }
        # reconcile with full peak list
        ms2_display = []
        # run through each MS2 peak, w/ or w/out annotation
        for mz, abs_int, norm_int in data['ms2_norm']:
            key = round(mz, 4)
            # add data from annotated lookup dict
            if annot_lookup:
                annotation = annot_lookup.get(key, {})
                ms2_display.append({
                    'exp_mz': mz,
                    'abs_int': abs_int,
                    'norm_int': norm_int,
                    'theo_mz': annotation.get('theo_mz'),
                    'formula': annotation.get('formula'),
                    'count': annotation.get('count'),
                    'ppm': annotation.get('ppm')
                })
            else: # if there are no annotations at all, we have to do this
                ms2_display.append({
                    'exp_mz': mz,
                    'abs_int': abs_int,
                    'norm_int': norm_int
                })

            data['ms2_display'] = ms2_display

    # now we simply return this and do the .txt files later
    return dictionary

def write_compSheet(dictionary):
    """
    Writes a summary sheet for the compiled library.
    """
    
    output = io.StringIO()
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df = df.rename(columns=MASTERSHEET_COLUMNS)
    df = df.reindex(columns=list(MASTERSHEET_COLUMNS.values()))
    df.to_csv(output, index=False)
    return output.getvalue()

# MassBank wants a PARTICULAR FORMAT for natively charged mol formulas
# lets actually fix it when writing txt files, we want it to be intact
# when doing stuff prior, like formula annotation
def reformat_charged_formula(formula):
    """
    Helper, converts molecular formula strings to MassBank-formatted ones.
    (Same fn as under compilerUtilities...)
    """
    
    match = re.match(r'^([A-Za-z0-9]+)([+-])(\d*)$', formula)
    if match:
        formula_base, sign, number = match.groups()
        charge = (number if number else '') + sign
        return f'[{formula_base}]{charge}'
    else: # if for some reason a non-charged formula is put through the function
        return formula

def write_txtFile_app(compound, data, field_mapping, field_order=None):
    """
    Web app-adapted helper from compilerUtilities.
    Returns the full MassBank-format .txt file for one compound as a string.
    """
    
    data['library_id'] = compound # do we still need this? keep 4 now...
    output = io.StringIO()
    fields = field_order if field_order else field_mapping.keys()
    for field in fields:
        mb_field = field_mapping.get(field)
        if not mb_field:
            continue
        value = data.get(field)
        if field == 'library_id' and value:
            value = re.sub(r' feature no\. \d+$', '', value)
        if value is None or str(value) == 'nan':
            # on per-compound basis --- 
            # skip all compSheet columns that don't have values
            continue 
        if field == 'molecularFormula':
            output.write(f'{mb_field} {reformat_charged_formula(value)}\n')
            continue
        if field in ('ms2_peaks', 'ms2_annot'):
            output.write(f'{mb_field} {value}')
            continue
        output.write(f'{mb_field} {value}\n')
    output.write('//\n')
    return output.getvalue()

# adapted...
def write_mspFile_app(comp_dict, mode):
    """
    Helper, writes and returns the .msp library file as a string.
    """
    
    msp_output = io.StringIO()
    for compound, fields in comp_dict.items():
        msp_output.write(f'NAME: {compound}\n')
        for excel_col, msp_field in MSP_FIELDS.items():
            value = fields.get(excel_col, '')
            if value != '':
                msp_output.write(f'{msp_field} {value}\n')
        # peak handling
        peak_data = fields.get('ms2_norm', '')
        if peak_data:
            # OLD CODE --- think we changed format/timing of data since
            #peak_data = peak_data.split(' ')
            #for i in range(3, len(peak_data)-2, 3):
            #    msp_output.write(f'{peak_data[i]}\t{peak_data[i+2]}\n')
            for peak in peak_data:
                mz = peak[0]
                norm_int = peak[2]
                msp_output.write(f'{mz}\t{norm_int}\n')
        msp_output.write('\n')
    msp_output.seek(0)
    return msp_output.getvalue()

# create everything inside the zip file is a good solution
def create_compZip(comp_data, mode, field_mapping):
    """
    Organizes the library assembly files for download in a .zip archive.
    """
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        # add .txt files
        for compound, data in comp_data.items():
            txt_content = write_txtFile_app(compound, data, field_mapping)
            txt_filename = f'{compound}_{data["short_acc"]}.txt'
            zipf.writestr(f'{mode}/txt/{txt_filename}', txt_content)
        
        # add .csv
        csv_content = write_compSheet(comp_data)
        zipf.writestr(f'{mode}/compiled_{mode}.csv', csv_content)
        
        # add .msp
        msp_content = write_mspFile_app(comp_data, mode)
        zipf.writestr(f'{mode}/msp/library_{mode}.msp', msp_content)
    
    zip_buffer.seek(0)
    return zip_buffer

def formula_to_subscript(formula):
    """
    Helper for visuals following assembly.
    """
    
    # unicode subscripts, apparently that is a thing
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    result = []
    for char in formula:
        result.append(subscript_map.get(char, char))
    return ''.join(result)

def plot_MS2(data, ms2_display, precursor_mz, title='placeholder'):
    """
    Function used to plot spectra in the web-app.
    """
    
    mzs = [peak['exp_mz'] for peak in ms2_display]
    intensities = [peak['norm_int'] for peak in ms2_display]
    formulas = [peak.get('formula') for peak in ms2_display]
    ppms = [peak.get('ppm') for peak in ms2_display]
    int_explained = round(sum(
        peak['norm_int'] for peak in ms2_display if peak.get('formula')) / sum(peak['norm_int'] for peak in ms2_display) * 100, 1)
    
    base_formula_formatted = formula_to_subscript(data['molecularFormula'])
    fig = go.Figure()

    hovertexts = []
    for mz, intensity, formula, ppm in zip(mzs, intensities, formulas, ppms):
        text = f'm/z: {mz}<br>Rel. Int: {intensity}'
        if formula: # only add these if present
            text += f'<br>Formula: {formula_to_subscript(formula)}'
        if ppm is not None and not math.isinf(ppm):
            text += f'<br>ppm dev: {0.0 + ppm}'
        hovertexts.append(text)
    
    # make bars
    fig.add_trace(go.Bar(
        x=mzs,
        y=intensities,
        width=[0.25]*len(mzs),
        marker_color='red',
        hovertext=hovertexts,
        hoverinfo='text',
        name='MS2 Peaks'
    ))

    fig.update_layout(
        title={
            'text': f'{title}<br><sup>MS1: {precursor_mz} • Formula: {base_formula_formatted} • int. explained: {int_explained}%</sup>',
            'font': dict(size=18)
        },
        hoverlabel=dict(font_size=16),
        xaxis_title='m/z',
        yaxis_title='Relative Intensity',
        bargap=0.1,
        showlegend=False,
        template='simple_white'
    )
    return fig

# ---- UTILITY STUFF ----

# ---- RTI ---
def generate_rtiSheets_app(compound_dict):
    """
    Function to generate spreadsheets in the input format of the RTI web app.
    Adapted from the CLI version, see compilerUtilities.py.
    
    Parameters & args:
        compound_dict (dict): Pre-assembly dictionary with necessary compound data
        
    Returns:
        sheet_dict (dict): Dictionary with RTI web app input sheets 
    """
    
    # store sheets here
    sheet_dict = {}
    
    # setup
    batch_size = 50
    n_sheets = 1
    
    # "legacy" filtering, but keepit for now...
    compounds = [k for k in compound_dict if 'candidate' not in k.lower()]

    for batch_start in range(0, len(compounds), batch_size):
        batch = compounds[batch_start:batch_start + batch_size]
        rows = []
        for i, compound in enumerate(batch, start=batch_start + 1):
            entry = compound_dict[compound]
            smiles = entry.get('smiles', '')
            if smiles == '':
                print(f'no SMILES found for {compound}, sheet {n_sheets}, please add manually')
            row = {
                'MolID': i,
                'Compound Name': compound,
                'CAS_RN': entry.get('cas', ''),
                'SMILES': smiles,
                'tR(min)': entry.get('retention_time', '')
            }
            rows.append(row)
        
        # generate sheet, do the buffer thing
        rti_sheet = pd.DataFrame(rows, columns=['MolID', 'Compound Name', 'CAS_RN', 'SMILES', 'tR(min)'])
        
        output = io.StringIO()
        rti_sheet.to_csv(output, index=False)
        csv_content = output.getvalue()
        output.close()
        
        # store the content
        sheet_dict[f'sheet_{n_sheets}'] = csv_content
        n_sheets += 1
        
    return sheet_dict

# ---- FILE CONV ----
MGF_MAT_FIELDS = { # field correspondence; 'MGF field': 'MAT field'
    'RTINSECONDS': 'RETENTIONTIME', 'PEPMASS': 'PRECURSORMZ',
    'ADDUCT': 'PRECURSORTYPE', 'IONMODE': 'IONMODE',
    'COLLISION_ENERGY': 'COLLISIONENERGY', # can .mat natively provide CE? need to know
    'FRAGMENTATION_METHOD': 'FRAGMENTATIONMODE', # same q as for CE
    'INSTRUMENT_TYPE': 'INSTRUMENT', # same q as for CE
    'Num peaks': 'Num Peaks'
    # we deal with the name separately
}

def parse_mgf_app(mgf_input, custom_mgf_fields={}):
    """
    Function to convert .mgf files to Librarian-adapted .mat files.
    
    Parameters & args:
        mgf_input (string): Contents of .mgf file in string format
        custom_mgf_fields (dict): Dictionary with non-standard tags to extract
        
    Returns:
        feature_dict (dict): Dictionary with compound data
    """
    
    # basic stuff, we read the mgf and turn it into a dictionary
    # then we flip the dictionary into a set of mat files
    feature_dict = {}
    current_compound = None
    feature_data = {}
    peak_data = []
    reading_peaks = False
    
    # deal with app input type
    if isinstance(mgf_input, str):
        lines = mgf_input.splitlines()
    elif hasattr(mgf_input, 'read'):
        lines = mgf_input.read().decode('utf-8').splitlines()
    else:
        # file path stuff --- actually irrelevant for app but whatever
        with open(mgf_input, 'r') as f:
            lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line == 'BEGIN IONS':
            feature_data = {}
            peak_data = []
            reading_peaks = False
            
        elif line == 'END IONS':
            if peak_data:
                feature_data['Num peaks'] = len(peak_data)
                feature_data['peak_data'] = peak_data
            if current_compound is None:
                current_compound = f'compound_{len(feature_dict)}'
            feature_dict[current_compound] = feature_data
            current_compound = None
            
        elif reading_peaks:
            parts = line.split()
            if len(parts) == 2:
                peak_data.append(tuple(map(float, parts)))
                
        else:
            if '=' in line:
                field, value = line.split('=', 1)
                field = field.strip()
                value = value.strip()
                if field == 'NAME':
                    current_compound = value
                if field in MGF_MAT_FIELDS.keys():                            
                    feature_data[field] = value
                if custom_mgf_fields:
                    if field in custom_mgf_fields.keys():
                        feature_data[field] = value
                if field == 'Num peaks':
                    reading_peaks = True
                
    return feature_dict

def dict2mat_zip(feature_dict, custom_mgf_fields={}):
    """
    Function to create .mat files from dictionary containing .mgf file data.
    Created files are placed into a .zip archive that is downloaded via the web app.
    
    Parameters & args:
        feature_dict (dict): Dictionary with compound data
        custom_mgf_fields (dict): Dictionary with non-standard tags to write
        
    Returns:
        zip_buffer (file-like): Buffer
    """
    
    # create .mat files
    if not feature_dict:
        return
    
    # bytes buffer for zip-file
    zip_buffer = io.BytesIO()
    
    date_folder = datetime.now().strftime('%Y-%m-%d')
    
    # get field stuff in order    
    basic_mgf_fields = MGF_MAT_FIELDS.copy()
        
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for compound, data in feature_dict.items():
            # get mode here already to handle folder placement
            ion_mode = data.get('IONMODE', '').upper()
            if ion_mode == 'POSITIVE':
                mode_folder = 'pos'
            elif ion_mode == 'NEGATIVE':
                mode_folder = 'neg'
            else:
                mode_folder = 'NA'
            
            # folder stuff
            zip_path = f"{date_folder}/{mode_folder}/{compound}.mat"
            
            # .mat file content as a string
            lines = []
            lines.append(f'NAME: {compound}')
            
            for mgf_key in basic_mgf_fields.keys():
                if mgf_key in data:
                    if mgf_key == 'Num peaks':
                        continue
                    mat_key = basic_mgf_fields.get(mgf_key, mgf_key)
                    value = data[mgf_key]
                    if mgf_key == 'IONMODE':
                        value = value.capitalize()
                    if mgf_key == 'COLLISION_ENERGY':
                        val_str = value.strip('[]')
                        try:
                            val_float = float(val_str)
                            val_int = int(round(val_float))
                            value = f'{val_int}%'
                        except ValueError:
                            value = value
                    lines.append(f'{mat_key}: {value}')
            
            if custom_mgf_fields:
                for mgf_key in custom_mgf_fields.keys():
                    if mgf_key in data:
                        mat_key = custom_mgf_fields.get(mgf_key, mgf_key)
                        value = data[mat_key]
                        lines.append(f'{mat_key}: {value}')
                
            lines.append('MSTYPE: MS2') # need this 
            num_peaks = data.get('Num peaks', len(data.get('peak_data', [])))
            lines.append(f'Num Peaks: {num_peaks}')
            
            peak_data = data.get('peak_data', [])
            for mz, intensity in peak_data:
                lines.append(f'{mz}\t{intensity}')
                    
            # add it all up
            mat_text = '\n'.join(lines) + '\n'
            # and write it
            zf.writestr(zip_path, mat_text)
    
    zip_buffer.seek(0)
    return zip_buffer
    