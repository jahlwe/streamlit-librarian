# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:54:21 2025

@author: Jakob
"""

# app utilities 
# we need to adapt our functions for working with io bytes and so on
# lets try it...

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
import re
import time
import io
import zipfile
import streamlit as st
import plotly.graph_objects as go
import math

# --- PCQ SHEET TEMPLATE ---
def generate_pcq_template():
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', lineterminator='\n')
    writer.writerow(['internal_id', 'name_q', 'cas_q', 'smiles_q', 'cid_q'])
    return output.getvalue()

# --- PCQ RE-QUERY (sheet-based) HELPER ---
QUERY_FIELDS = ['internal_id', 'name_q', 'cas_q', 'smiles_q', 'cid_q']

def query_dict_from_pcq_input(pcq_input):
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
        # also keep any assigned internal_id
        if data.get('internal_id'):
            entry['internal_id'] = data['internal_id']
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
    'internal_id': 'CH$NAME:',
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
mastersheet_field_conversion['internal_id'] = 'CH$NAME:'
mastersheet_field_conversion['iupacName'] = 'CH$NAME: (IUPAC)'
MASTERSHEET_COLUMNS = {
    'file_name': 'File_name',
    'short_accession': 'Short ACCESSION name:',
    **mastersheet_field_conversion,
    'submitted_to_MBEU': 'Submitted to MBEU'
}

# fields for msp.
MSP_FIELDS = {
    'MS$FOCUSED_ION: PRECURSOR_M/Z': 'NAME:',
    'MS$FOCUSED_ION: ION_TYPE': 'PRECURSORMZ:',
    'CH$FORMULA:': 'PRECURSORTYPE:',
    'CH$COMPOUND_CLASS:': 'Ontology:',
    'CH$LINK: INCHIKEY': 'INCHIKEY:',
    'CH$SMILES:': 'SMILES:',
    'AC$CHROMATOGRAPHY: RETENTION_TIME': 'RETENTIONTIME:',
    'AC$MASS_SPECTROMETRY: ION_MODE': 'IONMODE:',
    'AC$INSTRUMENT_TYPE:': 'INSTRUMENTTYPE:',
    'AC$INSTRUMENT:': 'INSTRUMENT:',
    'AC$MASS_SPECTROMETRY: COLLISION_ENERGY': 'COLLISIONENERGY:',
    'PK$NUM_PEAK:': 'Num Peaks:'    
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
    'NAME:': 'internal_id',
    'Num Peaks:': 'num_peak'
}

def get_charge(adduct):
    # defines groups to match in a string 
    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        number = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * number
    return 0 # if no charge 

def read_archive(mat_archive, archive_type):
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
    rti_dictionary = {}
    for i, (name, file) in enumerate(rti_files_dict.items()):
        if not hasattr(file, 'name'):
            file.name = name # need to do this for s2d to get it?
        current_dict = gu.sheet_to_dict(file, 'Compound Name')
        rti_dictionary.update(current_dict)
    return rti_dictionary

def add_cfData_app(dictionary, cf_data):
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
    normalize_ms2=True
):
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
                
                current_record = {'keyColumn': 'internal_id'}
                continue
            
            for field, storage in MAT_FIELDS.items():
                if line.startswith(field):
                    value = line.split(': ', 1)[1].strip()
                    if storage in ['retention_time', 'precursor_mz']:
                        value = round(float(value), 2 if storage == 'retention_time' else 5)
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
    # Use get_charge on the ion type data, compare it to 
    mode_agreement = get_charge(current_record['ion_type']) == mode_sign  
    if current_compound and current_record and mode_agreement:
        dictionary[current_compound] = current_record.copy()
        
    return dictionary

def gather_matData_app(mat_files_dict, mode):
    mat_dictionary = {}
    for name, file_obj in mat_files_dict.items():
        if f'{mode}/' in name:
            # folder names are retained, so use it for mode filtering
            mat_dictionary = parse_matFile_app(file_obj, mat_dictionary, mode)
    return mat_dictionary

def add_manual_metadata_app(dictionary, metadata_file):
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

def adduct_checker(compound, data):
    # for Br-containing (Cl also sometimes? No?) compounds -- 
    # might need to calculate two monomasses, one for M and one for M+2
    # ...
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
    
    # subset adducts for the current mode
    charge_indicator = adduct[-1]
    adduct_subset = {k: v for k, v in adducts.items() if k.endswith(charge_indicator)}
    
    if adduct in adduct_subset.keys() and adduct and monomass and exp_mz:
        charge = get_charge(adduct)
        theo_mz = (monomass + adduct_subset[adduct]) / abs(charge)
        # calculate how near the MSDIAL mass is to the expected mass for that adduct
        ok_diff = True if abs(theo_mz - exp_mz) < 0.1 else False
        if not ok_diff:
            for other_adduct in adduct_subset.keys():
                if other_adduct != adduct:
                    charge = get_charge(other_adduct)
                    candidate_theo_mz = (monomass + adduct_subset[other_adduct]) / abs(charge)
                    ok_diff = True if abs(candidate_theo_mz - exp_mz) < 0.1 else False
                    if ok_diff:
                        print(f'more suitable ion type {other_adduct} for {compound} --- discarding {adduct}')
                        return other_adduct, True
            print(f'unsuitable adduct for {compound} but no alternative found --- validate manually')
            return adduct, False
        else:
            return adduct, True
    elif adduct not in adduct_subset.keys() and adduct and monomass and exp_mz:
        for other_adduct in adduct_subset.keys():
            charge = get_charge(other_adduct)
            candidate_theo_mz = (monomass + adduct_subset[other_adduct]) / abs(charge)
            ok_diff = True if abs(candidate_theo_mz - exp_mz) < 0.1 else False
            if ok_diff:
                print(f'more suitable ion type {other_adduct} found for {compound} --- discarding {adduct}')
                return other_adduct, True
        print(f'unsuitable adduct for {compound} but no alternative found --- validate manually')
        return adduct, False
    else:
        print(f'data missing to validate adduct for {compound} --- validate manually')
        return adduct, False

# combined assembling preComp dict with the preparation steps.
def preCompile_app(
    mode, 
    pcq_data, 
    metadata_tsv,
    mat_data,
    rti_data=None,
    cf_data=None,
    annotate_fragments=True,
    progress_callback=None
):
    if mat_data:
        # we do call for create compilation dictionary here
        # so we don't need to have a STORAGE_FIELDS version in this file
        dictionary = cu.create_compilation_dictionary(mat_data)
        dictionary = cu.add_chemical_metadata(dictionary, pcq_data)
        dictionary = add_manual_metadata_app(dictionary, metadata_tsv)
        
        # optional stuff
        if rti_data:
            dictionary = cu.add_RTIData(dictionary, rti_data)
        if cf_data:
            dictionary = add_cfData_app(dictionary, cf_data)
            
    if dictionary:
        total = len(dictionary)
        for i, (compound, data) in enumerate(dictionary.items()):        
            # generate record_title and save sheet
            short_name = re.sub(r' feature no\. \d+$', '', compound)
            record_title = '; '.join(
                [short_name, data['instrument_type'], data['ms_type'], 
                 data['collision_energy'], data['resolution'], data['ion_type']])
            data['title'] = record_title
            # also! run the adduct checker.
            data['ion_type'], data['adduct_validated'] = adduct_checker(compound, data)
            # also! annotate fragments.
            if annotate_fragments:
                try:
                    print(f'annotating {compound} MS2')
                    loss_fragments = fa.generate_ref_fragments(data)
                    loss_fragments = fa.generate_more_fragments(data, loss_fragments)
                    match_list = fa.match_loss_fragments(data, loss_fragments)
                    data['frag_annot'] = fa.format_annotation(data, match_list)
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
    '''
    Optional function to filter the preComp-sheet when assembling.
    Besides using a .txt file to skip compounds by name as listed
    in the .txt file, it also makes comments of enantiomer features
    for one of the features, while discarding the other.
    '''
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
            else: # if there is no annotations at all, we have to do this
                ms2_display.append({
                    'exp_mz': mz,
                    'abs_int': abs_int,
                    'norm_int': norm_int
                })

            data['ms2_display'] = ms2_display

    # now we simply return this and do the .txt files later
    return dictionary

def write_compSheet(dictionary):
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
    match = re.match(r'^([A-Za-z0-9]+)([+-])(\d*)$', formula)
    if match:
        formula_base, sign, number = match.groups()
        charge = (number if number else '') + sign
        return f'[{formula_base}]{charge}'
    else: # if for some reason a non-charged formula is put through the function
        return formula

def write_txtFile_app(compound, data, field_order=None):
    """
    Adapted helper from compilerUtilities...
    Returns full .txt file layout as a string.
    """
    data['internal_id'] = compound # do we still need this? keep 4 now...
    output = io.StringIO()
    fields = field_order if field_order else FIELD_CONVERSION.keys()
    for field in fields:
        mb_field = FIELD_CONVERSION.get(field)
        if not mb_field:
            continue
        value = data.get(field)
        if field == 'internal_id' and value:
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
    msp_output = io.StringIO()
    for compound, fields in comp_dict.items():
        msp_output.write(f'NAME: {compound}\n')
        for excel_col, msp_field in MSP_FIELDS.items():
            value = fields.get(excel_col, '')
            if value != '':
                msp_output.write(f'{msp_field} {value}\n')
        # peak handling
        peak_data = fields.get('ms2_peaks', '')
        if peak_data:
            peak_data = peak_data.split(' ')
            for i in range(3, len(peak_data)-2, 3):
                msp_output.write(f'{peak_data[i]}\t{peak_data[i+2]}\n')
        msp_output.write('\n')
    msp_output.seek(0)
    return msp_output.getvalue()

# create everything inside the zip file is a good solution
def create_compZip(comp_data, mode):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        # add .txt files
        for compound, data in comp_data.items():
            txt_content = write_txtFile_app(compound, data)
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
    # unicode subscripts, never knew that was a thing
    subscript_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    result = []
    for char in formula:
        result.append(subscript_map.get(char, char))
    return ''.join(result)

def plot_MS2(data, ms2_display, precursor_mz, title='placeholder'):
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
def generate_rtiSheets_app(compound_dict):
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
    