# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 11:07:27 2025

@author: martingroup
"""

import pandas as pd
import glob
import os
import utils.genericUtilities as gu
import utils.pubchemUtilities as pu
#import utils.classyfireUtilities as cu
#import utils.qcUtilities as qu
import utils.fragmentAnnotationNew as fa
import csv
import pandas as pd
from utils.spectrum import Spectrum
from utils.spectrum_type import SpectrumType
from utils.splash import Splash
from datetime import date
import re
import time

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
    # add collision energy support --- what's the name?
    # we dont need to know. with the web-app, we can manage it manually
}

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
    # moved these up here now --- were at the bottom earlier
    # which didn't make sense. 
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
    'ms2_annot': 'PK$ANNOTATION:', # added recently --- SHOULD BE ABOVE NUM PEAK!
    'num_peak': 'PK$NUM_PEAK:',
    'ms2_peaks': 'PK$PEAK:',
}

STORAGE_FIELDS = [
    'keyColumn',
    'file_name',
    'short_accession',
    *FIELD_CONVERSION.keys(),
    'ms2_data',
    'ms2_norm',
    'frag_annot',
    'submitted_to_MBEU'
]

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

# need this here already to do mode validation in parse_matFile
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

# Functions for dealing with MS-DIAL .mat-files
def parse_matFile(
        file_path, 
        dictionary, 
        mode,
        normalize_ms2=True
):
    """
    Function for reading single .mat files and storing the data in a dictionary.

    Parameters & args:
        file_path (string): Path to .mat file
        dictionary (dict): Dictionary to add compound information
        mode (string): Current mode --- pos/neg are compiled separately
        normalize_ms2 (bool): Controls whether ms2 data is normalized (separate column created)

    Returns:
        dictionary (dict): Dictionary w added compound information
    """
    
    current_compound = None
    current_record = {}
    reading_peaks = False
    peak_data = [] # MS2 container
    n_peaks = 0 # n MS2 peaks
    mode_sign = 1 if mode == 'pos' else -1
    
    with open(file_path, 'r') as f:
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
                        current_record['ms2_norm'] = normalize_peaks(peak_data)
                    current_record['base_peak'] = round(max(peak_data, key=lambda x: x[1])[0], 5)
    # add records
    # EXTRA MODE VALIDATION!
    # Use get_charge on the ion type data, compare it to
    current_charge = 1 if get_charge(current_record['ion_type']) > 0 else -1
    mode_agreement = current_charge == mode_sign
    if not mode_agreement:
        print(f'mode agreement check failed for {current_compound}')
    if current_compound and current_record and mode_agreement:
        dictionary[current_compound] = current_record.copy()
        
    return dictionary

# We have to do one mode at a time. Basically. Dictionary key issues.
def gather_matData(mode, data_dir):
    """
    Helper, uses parse_matFile fn to gather data from all provided .mat files.
    """
    
    # A bit of error handling...
    if mode not in ['pos', 'neg']:
        raise ValueError('Invalid mode provided.')
    mat_dictionary = {}
    # This should work to look in all subdirectories
    mat_files = glob.glob(os.path.join(data_dir, '**/*.mat'), recursive=True)
    mode_filtered_files = [file for file in mat_files if f"/{mode}/" in file.replace("\\", "/")]
    
    for i, file in enumerate(mode_filtered_files):
        print(f'parsing {file}...')
        # let's implement a 2nd mode check, HERE. Or rather, inside parse_matFile.
        # but we need to supply the mode.
        mat_dictionary = parse_matFile(file, mat_dictionary, mode)
    
    return mat_dictionary

#dictionary = gather_matData('pos')
#parse_matFile('input/mat/run6_exports/non-prio/25/pos/ID10760_8.29_287.16.mat', dictionary, 'pos')
#dictionary = gather_matData('neg')

def create_compilation_dictionary(mat_dictionary, storage_fields):    
    """
    Helper, creates dictionary for use in the pre-assembly submodule.
    Compound data from the different sources (experimental, PubChem metadata etc)
    are gathered in this dictionary in subsequent steps.
    Here, as a first addition, the gathered .mat data is transferred.
    """       
    
    # create dictionary with all storage fields and transfer mat data
    dictionary = {key: {col: None for col in storage_fields} for key in mat_dictionary.keys()}
    for compound in dictionary:
        for col in dictionary[compound]:
            if compound in mat_dictionary and col in mat_dictionary[compound]:
                dictionary[compound][col] = mat_dictionary[compound][col]
        dictionary[compound]['library_id'] = compound
    return dictionary

#dictionary = create_compilation_dictionary(dictionary)

# Now, we want to read our prep sheet, if we have one, and use the data
# from there, instead of querying PubChem all over again.
#ref_dictionary = gu.sheet_to_dict('output/prepOneSheet.csv', 'internalName')

def add_chemical_metadata(dictionary, ref_dictionary):
    """
    Function to add chemical metadata (pcq output) to the compilation dictionary.
    
    Parameters & args:
        dictionary (dict): Pre-assembly dictionary to which data is added for each cpd
        ref_dictionary (dict): PubChem data in dictionary format (via sheet-to-dict)
        
    Returns:
        dictionary (dict): Updated pre-assembly dictionary
    """
    
    for compound in dictionary.keys():
        # maybe need to add flexibility later for different name columns
        lookup_name = re.sub(r' feature no\. \d+$', '', compound)
        if lookup_name in ref_dictionary.keys():
            ref_data = ref_dictionary.get(lookup_name)
        else:
            print(f'no PubChem data found for {compound}')
            continue
        if ref_data:
            for storage in STORAGE_FIELDS:
                if storage in ref_data and storage in dictionary[compound]:
                    if storage == 'monoisotopicMass':
                        dictionary[compound][storage] = round(float(ref_data[storage]), 5)
                    else:
                        dictionary[compound][storage] = ref_data[storage]
            # fixing natively charged ions, missing in MS-DIAL
            if '+' in ref_data['molecularFormula']:
                current_formula = ref_data['molecularFormula']
                charge = 1 if current_formula.endswith('+') else int(current_formula[-1])
                if charge == 1:
                    dictionary[compound]['ion_type'] = '[M]+'
                elif charge > 1:
                    dictionary[compound]['ion_type'] = f'[M]{charge}+'
    return dictionary

#dictionary = add_chemical_metadata(dictionary, ref_dictionary)

def add_manual_metadata(dictionary, tsv_path):
    """
    Function to add manually provided metadata to the compilation dictionary.
    This data is provided by the user in a separate .tsv file.
    
    Parameters & args:
        dictionary (dict): Pre-assembly dictionary to which data is added
        manual_metadata (string): Path/name for the .tsv file
        
    Returns:
        dictionary (dict): Updated pre-assembly dictionary
    """
    
    manual_dictionary = {}
    with open(tsv_path, 'r', newline='', encoding='utf-8') as m:
        reader = csv.reader(m, delimiter='\t')
        for key, value in reader:
            manual_dictionary[key] = value
    for compound in dictionary:
        # only fill in values that are currently None --- this preserves any data
        # already present from .mat files (e.g. resolution, retention_time)
        # while still applying the general metadata to fields that are empty
        for k, v in manual_dictionary.items():
            if dictionary[compound].get(k) is None:
                dictionary[compound][k] = v
    return dictionary

#dictionary = add_manual_metadata(dictionary)

# sheets for the RTI website.
# needs .mat files -- obviously -- for collecting RT and stuff.
def generate_rtiSheet(dictionary, mode, save_path='output/RTI/'):
    """
    Function to generate spreadsheets in the input format of the RTI web app.
    Used to convert recorded retention times to RTI values.
    For normal users of the RTI web app (?) there is a maximum batch size of 50 compounds.
    E.g., for 525 compounds, 10 sheets á 50 cpds will be created and one sheet á 25.
    
    Parameters & args:
        dictionary (dict): Pre-assembly dictionary with necessary compound data
        mode (string): Current mode --- pos/neg files are created separately
        
    Returns:
        dictionary (dict): Updated pre-assembly dictionary
    """
    
    # Ensure output directory exists
    mode_path = os.path.join(save_path, mode)
    os.makedirs(mode_path, exist_ok=True)

    # Filter out 'candidate' features
    compounds = [k for k in dictionary if 'candidate' not in k.lower()]
    
    batch_size = 50
    n_sheets = 1

    for batch_start in range(0, len(compounds), batch_size):
        batch = compounds[batch_start:batch_start + batch_size]
        rows = []
        for i, compound in enumerate(batch, start=batch_start + 1):
            entry = dictionary[compound]
            smiles = entry.get('smiles', '')
            if smiles == '':
                print(f"no SMILES found for {compound}, sheet {n_sheets}, please add manually")
            row = {
                'MolID': i,
                'Compound Name': compound,
                'CAS_RN': entry.get('cas', ''),
                'SMILES': smiles,
                'tR(min)': entry.get('retention_time', '')
            }
            rows.append(row)
        rti_sheet = pd.DataFrame(rows, columns=['MolID', 'Compound Name', 'CAS_RN', 'SMILES', 'tR(min)'])
        file_name = os.path.join(mode_path, f'RTISheet_{n_sheets}.csv')
        rti_sheet.to_csv(file_name, index=False)
        print(f'saving RTI sheet {file_name}')
        n_sheets += 1
    return None

#generate_rtiSheet(dictionary, 'pos')

def gather_RTIData(mode, folder_path='input/RTI/'):
    """
    Function to collect data from spreadsheets output by the RTI web app.
    
    Parameters & args:
        mode (string): Current mode
        folder_path (string): Location of the RTI spreadsheets
        
    Returns:
        rti_dictionary (dict): Dictionary with RTI data for each compound
    """
    
    if mode not in ['pos', 'neg']:
        raise ValueError('invalid mode provided')
    rti_dictionary = {}
    file_path = folder_path + mode + '/'
    rti_files = glob.glob(os.path.join(file_path, '*.csv'))

    for i, file in enumerate(rti_files):
        current_dict = gu.sheet_to_dict(file, 'Compound Name')
        rti_dictionary.update(current_dict)
    return rti_dictionary

#rti_dictionary = gather_RTIData('pos')
#rti_dictionary = gather_RTIData('neg')

def add_RTIData(dictionary, rti_dictionary):
    """
    Helper, adds RTI data to the pre-assembly dictionary.
    """
    
    for compound in dictionary.keys():
        lookup_name = re.sub(r' feature no\. \d+$', '', compound)
        if lookup_name in rti_dictionary.keys():
            dictionary[compound]['rti'] = rti_dictionary[lookup_name]['Exp. RTI']
        elif lookup_name not in rti_dictionary.keys():
            print(f'no RTI data found for compound {lookup_name}')
    
    return dictionary

#dictionary = add_RTIData(dictionary, rti_dictionary)

# FORGET ABOUT THIS FOR NOW - CLASSYFIRE SERVERS ARE DOWN.
#def query_ClassyFire(dictionary, use_ref_sheet=False, sheet_path=None, no_queries=False):
#    """
#    Not in use. Also --- redundant! MassBank (EU) automatically adds chemont data to new entries.
#    """
#    
#    n_compounds = len(dictionary.keys())
#    # this was added if we already have queried ClassyFire before and
#    # want to reuse that half-filled sheet.
#    if use_ref_sheet and sheet_path:
#        print(f'transferring class information from {sheet_path}')
#        ref_dictionary = gu.sheet_to_dict(sheet_path)
#        for compound in dictionary.keys():
#            if compound in ref_dictionary.keys() and ref_dictionary[compound]['class']:
#                print(ref_dictionary[compound]['class'])
#                dictionary[compound]['class'] = ref_dictionary[compound]['class']
#    
#    if not no_queries:
#        print(f'querying ClassyFire --- n = {n_compounds} compounds')
#        for i, compound in enumerate(dictionary.keys()):
#            print(compound)
#            if dictionary[compound]['CH$COMPOUND_CLASS:'] == None or str(dictionary[compound]['CH$COMPOUND_CLASS:']) == 'nan':
#                current_smiles = dictionary[compound]['CH$SMILES:']
#                print(current_smiles)
#                try:
#                    class_data = cu.get_classyfire(current_smiles)
#                    print(class_data)
#                    if class_data:
#                        dictionary[compound]['CH$COMPOUND_CLASS:'] = class_data
#                except:
#                    pass
#                
#                # DELAY TO AVOID HTTPERROR
#                time.sleep(10)
#                
#            # This is not helpful if queries are not being made for every compound
#            # Move it outside the conditional and it's always helpful.
#            if i % 10 == 0 and i != 0:
#                print(f'processed {i} of {n_compounds} compounds')
#                if i == n_compounds - 1:
#                    print('processed {i+1} of {n_compounds} compounds')
#                    print('done')
#            
#    return dictionary

#dictionary = add_compoundClasses(dictionary, True, 'output/compiler/classref_pos.xlsx', True)
#dictionary = add_compoundClasses(dictionary, True, 'output/compiler/classref_neg.xlsx', True)

# Load preComp-sheets into dictionaries to avoid having to redo above steps...
#dictionary = genericUtilities.sheet_to_dict('output/compiler/preCompilationSheet_pos.xlsx', 'CH$NAME:')
#dictionary = genericUtilities.sheet_to_dict('output/compiler/preCompilationSheet_neg.xlsx', 'CH$NAME:')

def manual_classyfire(dictionary, ref_sheet_path):
    """
    Function to add ClassyFire chemical ontology data to the pre-assembly dictionary.
    The type of input read by this function comes from the Fiehn lab ClassyFire Batch portal.
    https://cfb.fiehnlab.ucdavis.edu/
    
    Functionally redundant since MassBank update to automatically add chemont data to new entries.

    Parameters & args:
        dictionary (dict): Pre-assembly dictionary
        ref_sheetpath (string): Location of the ClassyFire Batch spreadsheet

    Returns:
        dictionary (dict): Pre-assembly dictionary updated w chemont data
    """
    
    # these queries use InChIKeys, so that is the link to our compounds
    class_dict = gu.sheet_to_dict(ref_sheet_path, 'InChIKey')
    for compound, data in dictionary.items():
        current_inchikey = data['inchikey']
        if current_inchikey in class_dict.keys():
            class_data = class_dict[current_inchikey]
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

#dictionary = manual_classyfire(dictionary, 'extras/classyfire_1200.csv')

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

#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv')
#adduct_checker('Diphenidol', dictionary['Diphenidol'])
#adduct_checker('Abacavir', dictionary['Abacavir'])
#(dictionary['Abacavir']['monoisotopicMass'] + 1.00783 - e_mass) - dictionary['Abacavir']['precursor_mz']

def prepare_preCompilationSheet( # deprecated...
    dictionary, mode, 
    annotate_fragments=True, fragment='brute',
    file_name='preComp'
):
    """
    Converts pre-assembly dictionary into a spreadsheet, to complete pre-assembly.

    Parameters & args:
        dictionary (dict): Pre-assembly dictionary
        mode (string): Current mode, pos/neg
        annotate_fragments (bool): Controls whether fragment formula annotation is performed
        fragment (string): Deprecated...
        file_name (string): Name of output pre-assembly sheet

    Returns:
        Nothing
    """
    
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
                
    gu.dict_to_sheet(dictionary, f'compiler/{file_name}_{mode}')
    return None

#prepare_preCompilationSheet(dictionary, 'neg')
#prepare_preCompilationSheet(dictionary, 'pos')

def filter_preComp( # DEPRECATED
        sheet_path, 
        mode, 
        exclude_path='files/compiler/exclude_compounds.txt'
    ):
    """
    Optional fn to filter the pre-assembly sheet.
    
    Allows an 'exclude_compounds' text file to be provided for compounds to skip.
    
    Also, compounds that there is more than one experimental feature for
    and that have the sane name (i.e. two .mat files both with the name Caffeine)
    have previously had the latter feature given the name "Caffeine feature n."
    In this step, these features are merged into one, and information about
    the latter feature appended to the first as a comment.
    
    Deprecated...? Filtering behavior can be avoided by having unique
    names for all feature data.

    Parameters & args:
        sheet_path (string): Name of the pre-assembly sheet
        mode (string): Current mode, pos/neg
        exclude_path (string): Path for the exclude_compounds .txt file

    Returns:
        filtered_dict (dict): Pre-assembly dictionary, filtered
    """
    dictionary = gu.sheet_to_dict(sheet_path)
    filtered_dict = {}
    
    # names to be excluded
    exclude_names = []
    with open(exclude_path, 'r') as exclude_list:
        for line in exclude_list:
            exclude_names.append(line.strip())
            
    # add info about (presumed) enantiomer peaks in comments of one of the entries
    for compound, data in dictionary.items():
        if compound not in exclude_names:           
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

# MassBank wants a PARTICULAR FORMAT for natively charged mol formulas
# lets actually fix it when writing txt files, we want it to be intact
# when doing stuff prior, like formula annotation
def reformat_charged_formula(formula):
    """
    Helper, converts molecular formula strings to MassBank-formatted ones.
    """
    
    match = re.match(r'^([A-Za-z0-9]+)([+-])(\d*)$', formula)
    if match:
        formula_base, sign, number = match.groups()
        charge = (number if number else '') + sign
        return f'[{formula_base}]{charge}'
    else: # if for some reason a non-charged formula is put through the function
        return formula

def write_txtFile(compound, data, save_path, field_order=None):
    """
    Helper for create_txtFiles.
    Writes a single MassBank-format .txt file to disk.
    """
    
    data['library_id'] = compound # need to make this explicit
    with open(save_path, 'w', encoding='utf-8') as f:
        fields = field_order if field_order else FIELD_CONVERSION.keys()
        for field in fields:
            mb_field = FIELD_CONVERSION.get(field)
            if not mb_field:
                continue
            value = data.get(field)
            if field == 'library_id' and value:
                value = re.sub(r' feature no\. \d+$', '', value)
            if field == 'ion_mode' and value: # mode should be CAPITALIZED. for MassBank.
                f.write(f'{mb_field} {value.upper()}\n')
                continue
            if field == 'molecularFormula':
                if value:
                    f.write(f'{mb_field} {reformat_charged_formula(value)}\n')
                else:
                    f.write(f'{mb_field}\n')
                continue
            if value is None or str(value) == 'nan':
                continue
            if field in ('ms2_peaks', 'ms2_annot'): # do this to avoid double \n at the end
                f.write(f'{mb_field} {value}')
                continue
            f.write(f'{mb_field} {value}\n')
        f.write('//\n')

def create_txtFiles(
        precomp_sheet_path,
        output_dir,
        accession_long,
        accession_short,
        accession_start, 
        mode, 
        massbank_fields=FIELD_CONVERSION,
        do_filter=True,
    ):
    """
    Organizes writing of MassBank-format .txt files.
    Also creates a final .csv data sheet for all compounds in the assembly.
    
    Parameters & args:
        accession_start (int): Start of accession numbering
        mode (string): Current mode, pos/neg
        sheet_path (string): Path to pre-assembly output sheet
        massbank_fields (string): Path to .txt file with tag order
        txt_path (string): Path to where .txt files will be placed
        do_filter (bool): Pass pre-assembly sheet through filter_preComp fn, True/False
        
    Returns:
        Nothing
    """
    
    if mode not in ['pos', 'neg']:
        raise ValueError('Invalid mode provided.')
    
    dictionary = filter_preComp(precomp_sheet_path, mode) if do_filter else gu.sheet_to_dict(precomp_sheet_path) 
    
    for i, (compound, data) in enumerate(dictionary.items()):
        acc_n = f'{accession_start + i:06d}'
        current_accession = f'MSNBNK-{accession_long}-{accession_short}{acc_n}'
        short_accession = f'{accession_short}{acc_n}'
        current_file = f'{compound}_{short_accession}'
        save_path = os.path.join(output_dir, mode, f'{current_accession}.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        data['accession'] = current_accession
        data['short_accession'] = short_accession
        # MassBank wants dates in this format! yyyy.mm.dd
        data['date'] = str(date.today()).replace('-', '.')
        data['file_name'] = current_file

        # splash and MS2
        peak_data_splash = [(mz, abs_int) for mz, abs_int in data['ms2_data']]
        current_ms2 = Spectrum(peak_data_splash, SpectrumType.MS)
        data['splash'] = str(Splash().splash(current_ms2))
        peak_line = 'm/z int. rel.int.\n' + ''.join(
            # TWO leading spaces
            f'  {mz} {abs_int} {norm_int}\n' for mz, abs_int, norm_int in data['ms2_norm']
        )
        data['ms2_peaks'] = peak_line
        # now also fragment annotations...
        if data.get('frag_annot'):
            annot_line = 'm/z tentative_formula formula_count mass error(ppm)\n' + ''.join(
                f'  {theo_mz} {t_f} {f_count} {exp_mz} {ppm}\n' for theo_mz, t_f, f_count, exp_mz, ppm in data['frag_annot']
            )
            data['ms2_annot'] = annot_line
        
        # write txt file
        write_txtFile(compound, data, save_path)
    
    # create postComp sheet --- with nice formatting!
    output_sheet_path = os.path.join(output_dir, f'postComp_{mode}.csv')
    df = pd.DataFrame.from_dict(dictionary, orient='index')
    df = df.rename(columns=MASTERSHEET_COLUMNS)
    df = df.reindex(columns=list(MASTERSHEET_COLUMNS.values()))
    df.to_csv(output_sheet_path, index=False)
    print(f'final accession number used --- {acc_n}')
    return None

#create_txtFiles(272, 'pos')
#create_txtFiles(1232, 'neg')

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

# can we also try to get a MGF file out for the library
MGF_FIELDS = {
    # MGF field name: postComp CSV column name
    'RTINSECONDS':      'AC$CHROMATOGRAPHY: RETENTION_TIME',
    'PEPMASS':          'MS$FOCUSED_ION: PRECURSOR_M/Z',
    'COLLISION_ENERGY': 'AC$MASS_SPECTROMETRY: COLLISION_ENERGY',
    'INSTRUMENT_TYPE':  'AC$INSTRUMENT_TYPE:',
    'IONMODE':          'AC$MASS_SPECTROMETRY: ION_MODE',
}

def compSheet_to_msp(output_dir, mode):
    """
    Creates an .msp file from the final assembly sheet.

    Parameters & args:
        output_dir (string): Folder containing the postComp_{mode}.csv written by create_txtFiles
        mode (string): Current mode, pos/neg

    Returns:
        Nothing
    """
    # postComp sheet name is fixed — derived from the same output_dir used by create_txtFiles
    postcomp_sheet_path = os.path.join(output_dir, f'postComp_{mode}.csv')
    
    dictionary = gu.sheet_to_dict(postcomp_sheet_path, 'CH$NAME:')
    msp_path = os.path.join(output_dir, f'{str(date.today())}_{mode}.msp')
    with open(msp_path, 'w') as msp:
        for compound, fields in dictionary.items():
            msp.write(f'NAME: {compound}\n')
            for excel_col, msp_field in MSP_FIELDS.items():
                value = fields.get(excel_col, '')
                if value != '':
                    msp.write(f'{msp_field} {value}\n')
            # Handle peaks separately
            peak_data = fields.get('PK$PEAK:', '')
            if peak_data:
                peak_data = peak_data.split(' ')
                for i in range(3, len(peak_data)-2, 3):
                    msp.write(f"{peak_data[i]}\t{peak_data[i+2]}\n")
            msp.write('\n')
    return None
    

#compSheet_to_msp('output/compiler/postComp_pos.csv', 'pos')
#compSheet_to_msp('output/compiler/postComp_neg.xlsx', 'neg')

def compSheet_to_mgf(output_dir, mode):
    """
    Creates an .mgf file from the final assembly sheet.
    Analogous to compSheet_to_msp, for CLI use.

    Parameters & args:
        output_dir (string): Folder containing the postComp_{mode}.csv written by create_txtFiles
        mode (string): Current mode, pos/neg

    Returns:
        Nothing
    """
    postcomp_sheet_path = os.path.join(output_dir, f'postComp_{mode}.csv')
    dictionary = gu.sheet_to_dict(postcomp_sheet_path, 'CH$NAME:')
    mgf_path = os.path.join(output_dir, f'{str(date.today())}_{mode}.mgf')

    with open(mgf_path, 'w') as mgf:
        for i, (compound, fields) in enumerate(dictionary.items(), start=1):
            mgf.write('BEGIN IONS\n')
            mgf.write(f'SPECTRUMID={i}\n')
            mgf.write(f'NAME={compound}\n')
            mgf.write(f'FEATURE_ID={i}\n')
            mgf.write(f'MSLEVEL=2\n')
            # dynamic fields in order matching app version
            for mgf_field, col in MGF_FIELDS.items():
                if mgf_field == 'PEPMASS':
                    # insert CHARGE immediately after PEPMASS
                    value = fields.get(col, '')
                    if value != '':
                        mgf.write(f'PEPMASS={value}\n')
                    ion_type = fields.get('MS$FOCUSED_ION: ION_TYPE', '')
                    if ion_type:
                        charge = abs(get_charge(ion_type))
                        mgf.write(f'CHARGE={charge if charge else "none"}\n')
                    else:
                        mgf.write('CHARGE=none\n')
                    mgf.write('FEATURE_MS1_HEIGHT=none\n')
                else:
                    value = fields.get(col, '')
                    if value != '':
                        mgf.write(f'{mgf_field}={value}\n')
            mgf.write('FILENAME=none\n')
            # peaks --- PK$PEAK: is a formatted string: header line + '  mz abs_int norm_int' lines
            peak_data = fields.get('PK$PEAK:', '')
            if peak_data:
                peak_lines = [
                    l.strip() for l in peak_data.strip().split('\n')
                    if l.strip() and not l.strip().startswith('m/z')
                ]
                mgf.write(f'Num peaks={len(peak_lines)}\n')
                for line in peak_lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        mgf.write(f'{parts[0]} {parts[1]}\n')
            mgf.write('END IONS\n')

    return None

#compSheet_to_mgf('output/compiler', 'pos')
#compSheet_to_mgf('output/compiler', 'neg')

# -------------------- CLI SUPPORT --------------------

BR_ISOTOPE_DIFF = 1.99795

def adduct_assigner(compound, data):
    """
    Assigns an ion/adduct type to feature data that lacks annotation.
    Evaluates whether experimental precursor m/z is within 10 ppm of any
    candidate adduct. Ported from appUtilities for CLI use.
    """
    e_mass = 0.00054858
    h_mass = 1.00783
    adducts = {
        '[M]+': -e_mass,
        '[M]2+': -(2 * e_mass),
        '[M+H]+': h_mass - e_mass,
        '[M+NH4]+': 14.00307 + (4 * h_mass) - e_mass,
        '[M+Na]+': 22.98977 - e_mass,
        '[M+K]+': 38.96371 - e_mass,
        '[M+2H]2+': (2 * h_mass) - (2 * e_mass),
        '[M+H-H2O]+': 17.00274 - e_mass,
        '[M-H]-': -h_mass + e_mass,
        '[M+Cl]-': 34.96885 + e_mass,
        '[M+F]-': 18.99840 + e_mass,
        '[M-2H]2-': (-2 * h_mass) + (2 * e_mass),
        '[M-H2O-H]-': -19.01839 + e_mass,
    }
    monomass = data.get('monoisotopicMass')
    exp_mz = data.get('precursor_mz')
    formula = data.get('molecularFormula')

    nBr = 0
    if formula and 'Br' in formula:
        atom_count = fa.parse_formula(formula)
        nBr = atom_count.get('Br', 0)

    ion_mode = data.get('ion_mode', '')
    charge_indicator = '+' if ion_mode.lower() == 'positive' else '-' if ion_mode.lower() == 'negative' else None
    if not charge_indicator:
        return None
    adduct_subset = {k: v for k, v in adducts.items() if k.endswith(charge_indicator)}

    if monomass is None or exp_mz is None:
        return None

    best_ppm = float('inf')
    best_adduct = None
    for adduct, shift in adduct_subset.items():
        charge = get_charge(adduct)
        for i in range(nBr + 1 if nBr > 0 else 1):
            theo_mz = (monomass + shift + i * BR_ISOTOPE_DIFF) / abs(charge)
            ppm_dev = abs(((theo_mz - exp_mz) / theo_mz) * 1e6)
            if ppm_dev < 10 and ppm_dev < best_ppm:
                best_ppm = ppm_dev
                best_adduct = adduct
    return best_adduct if best_adduct else None

# ---------- MORE RECORD VALIDATION FUNCTIONS ----------

# Monoisotopic adduct mass shifts, mirrors adduct_checker
# Defined once here so validate_record can use it without redefining
_ADDUCT_MASSES = {
    '[M]+':       -0.00054858,
    '[M]2+':      -(2 * 0.00054858),
    '[M+H]+':     1.00783 - 0.00054858,
    '[M+NH4]+':   14.00307 + (4 * 1.00783) - 0.00054858,
    '[M+Na]+':    22.98977 - 0.00054858,
    '[M+K]+':     38.96371 - 0.00054858,
    '[M+2H]2+':   (2 * 1.00783) - (2 * 0.00054858),
    '[M+H-H2O]+': 1.00783 - 18.01056 - 0.00054858,  # H - H2O - e
    '[M-H]-':     -1.00783 + 0.00054858,
    '[M+Cl]-':    34.96885 + 0.00054858,
    '[M+F]-':     18.99840 + 0.00054858,
    '[M-2H]2-':   (-2 * 1.00783) + (2 * 0.00054858),
    '[M-H2O-H]-': -19.01839 + 0.00054858,
}

_ISOTOPE_SPACING = 1.003355  # 13C − 12C, Da

def validate_record(
    compound,
    data,
    ppm_tol=10,
    isolation_window=1.0,
    min_peaks=3,
    annot_threshold=20.0,
):
    """
    Validates a single record during pre-assembly and returns a validation string. 
    Returns 'passed' if no issues are found, otherwise a list of issue descriptions.

    Validation steps:
        - Adduct not validated by adduct_checker
        - Precursor mass deviation beyond ppm_tol
        - Fewer than min_peaks MS2 peaks (NOT CURRENTLY IN USE)
        - Base peak at precursor m/z (spectrum may be unfragmented)
        - Peaks detected above precursor m/z (NOT CURRENTLY IN USE)
        - Co-isolated ions within isolation_window Da
        - Molecular ion peak not identified (annotation-dependent)
        - Fragment annotation coverage below annot_threshold % (annotation-dependent)

    Parameters:
        compound (str): Compound name, used in printed warnings only
        data (dict): Pre-assembly record dictionary
        ppm_tol (float): PPM tolerance for mass accuracy and isotope checks
        isolation_window (float): Da window used for co-isolation check
        min_peaks (int): Minimum acceptable MS2 peak count
        annot_threshold (float): Minimum acceptable annotation coverage (%)
    """
    issues = []

    adduct      = data.get('ion_type')
    monomass    = data.get('monoisotopicMass')
    precursor   = data.get('precursor_mz')
    ms2_data    = data.get('ms2_data')
    ms2_norm    = data.get('ms2_norm')
    frag_annot  = data.get('frag_annot')

    # 1. Adduct validation outcome -------------------------------------------
    adduct_validated = data.get('adduct_validated')
    if adduct_validated is False:
        issues.append('adduct not validated')

    # 2. Precursor mass accuracy ---------------------------------------------
    if adduct and monomass and precursor and adduct in _ADDUCT_MASSES:
        charge = get_charge(adduct)
        if charge != 0:
            theo_mz = (monomass + _ADDUCT_MASSES[adduct]) / abs(charge)
            ppm_dev = abs((theo_mz - precursor) / precursor) * 1e6
            if ppm_dev > ppm_tol:
                issues.append(
                    f'precursor mass deviation {ppm_dev:.1f} ppm exceeds tolerance'
                )

    if ms2_data:
        ppm_tol_da = (precursor * ppm_tol * 1e-6) if precursor else 0

        # SKIP THIS FOR NOW, actually.....
        # 3. Minimum peak count ----------------------------------------------
        #if len(ms2_data) < min_peaks:
        #    issues.append(f'fewer than {min_peaks} MS2 peaks')

        # Skip this too, should be caught in pre-processing...
        # 4. Base peak at/near precursor -------------------------------------
        #if precursor:
        #    base_mz = max(ms2_data, key=lambda x: x[1])[0]
        #    if abs(base_mz - precursor) <= ppm_tol_da:
        #        issues.append(
        #            'base peak at precursor m/z (spectrum may be unfragmented)'
        #        )

        # 5. Peaks above precursor -------------------------------------------
        if precursor:
            n_above = sum(1 for mz, _ in ms2_data if mz > precursor + ppm_tol_da)
            if n_above:
                issues.append(f'{n_above} peak(s) detected above precursor m/z')

        # 6. Co-isolation ----------------------------------------------------
        if adduct and precursor:
            charge = get_charge(adduct)
            if charge != 0:
                window_peaks = [
                    mz for mz, _ in ms2_data
                    if abs(mz - precursor) <= isolation_window
                    and abs(mz - precursor) > ppm_tol_da
                ]
                co_isolated = []
                for peak_mz in window_peaks:
                    delta = peak_mz - precursor
                    if delta < 0:
                        co_isolated.append(peak_mz)
                    else:
                        is_isotope = any(
                            abs(delta - n * _ISOTOPE_SPACING / abs(charge))
                            <= peak_mz * ppm_tol * 1e-6
                            for n in range(1, 4)
                        )
                        if not is_isotope:
                            co_isolated.append(peak_mz)
                if co_isolated:
                    issues.append(
                        f'co-isolated ions detected within {isolation_window} Da window'
                    )

    # Annotation-dependent checks --------------------------------------------
    if frag_annot is not None:

        # 7. Molecular ion not identified ------------------------------------
        mol_formula = data.get('molecularFormula')
        if mol_formula and adduct:
            charge = get_charge(adduct)
            mol_ion_formula = fa.regenerate_formula_hill(
                fa.apply_adduct(fa.parse_formula(mol_formula), adduct)
            )
            expected_fmt = fa.format_formula(data, mol_ion_formula, charge)
            if not any(entry[1] == expected_fmt for entry in frag_annot):
                issues.append('molecular ion peak not identified in MS2')

        # 8. Low annotation coverage ----------------------------------------
        if ms2_norm:
            total_int = sum(norm_int for _, _, norm_int in ms2_norm)
            if total_int > 0:
                annot_int = sum(
                    norm_int
                    for (_, _, norm_int), (_, formula, _, _, _)
                    in zip(ms2_norm, frag_annot)
                    if formula is not None
                )
                coverage = annot_int / total_int * 100
                if coverage < annot_threshold:
                    issues.append(
                        f'annotation coverage {coverage:.1f}% below threshold '
                        f'({annot_threshold:.0f}%)'
                    )

    return 'passed' if not issues else ' | '.join(issues)


def validate_preComp(
    dictionary,
    ppm_tol=10,
    isolation_window=1.0,
    min_peaks=3,
    annot_threshold=20.0,
):
    """
    Applies validate_record to every compound in the pre-assembly dictionary
    and writes the result to data['validation'].

    Parameters:
        dictionary (dict): Pre-assembly dictionary
        ppm_tol (float): PPM tolerance passed to validate_record
        isolation_window (float): Da window for co-isolation check
        min_peaks (int): Minimum acceptable MS2 peak count
        annot_threshold (float): Minimum acceptable annotation coverage (%)
    """
    for compound, data in dictionary.items():
        data['validation'] = validate_record(
            compound, data,
            ppm_tol=ppm_tol,
            isolation_window=isolation_window,
            min_peaks=min_peaks,
            annot_threshold=annot_threshold,
        )
    return dictionary


def preCompile_CLI(
    dictionary,
    mode,
    output_path,
    annotate_fragments=True,
    ppm_tol=10,
    ppm_tol_validation=10,
    isolation_window=1.0,
):
    """
    Organizes pre-assembly for CLI use. Mirrors preCompile_app logic.

    Parameters & args:
        dictionary (dict): Pre-assembly dictionary (output of add_chemical_metadata etc.)
        mode (string): Current mode, pos/neg
        output_path (string): Full output file path including extension
        annotate_fragments (bool): Whether to perform fragment formula annotation
        ppm_tol (float): PPM tolerance for MS2 fragment annotation (default: 10)
        ppm_tol_validation (float): PPM tolerance for MS1 validation (default: 10)
        isolation_window (float): Isolation window width in Da for co-isolation check (default: 1.0)

    Returns:
        Nothing --- saves pre-assembly sheet to output_path
    """
    # drop compounds lacking essential chemical metadata
    len_before = len(dictionary)
    dictionary = {
        c: d for c, d in dictionary.items()
        if d.get('monoisotopicMass') and d.get('molecularFormula') and d.get('smiles')
    }
    len_after = len(dictionary)
    if len_before - len_after > 0:
        print(f'dropped {len_before - len_after} compounds lacking chemical metadata')

    total = len(dictionary)
    for i, (compound, data) in enumerate(dictionary.items()):
        # assign ion type if missing
        if not data.get('ion_type'):
            data['ion_type'] = adduct_assigner(compound, data)
            data['adduct_validated'] = 'assigned'
            validate = False
        else:
            validate = True

        # record title
        short_name = re.sub(r' feature no\. \d+$', '', compound)
        record_title = '; '.join([
            short_name,
            str(data.get('instrument_type', '')),
            str(data.get('ms_type', '')),
            str(data.get('collision_energy', '')),
            str(data.get('resolution', '')),
            str(data.get('ion_type', '')),
        ])
        data['title'] = record_title

        # adduct validation
        if validate:
            data['ion_type'], data['adduct_validated'] = adduct_checker(compound, data)

        # fragment annotation
        if annotate_fragments:
            try:
                print(f'annotating {compound} MS2')
                candidates = fa.generate_subformulas(data, ppm_tol=ppm_tol)
                candidates = fa.match_iso_patterns(data, candidates)
                result = fa.finalize_annotation(data, candidates, ppm_tol=ppm_tol)
                data['frag_annot'] = fa.format_annotation(data, result)
            except Exception as e:
                data['frag_annot'] = None
                print(f'failed fragment annotation for {compound}: {e}')

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f'processed {i + 1} of {total} compounds')

    print('running validation checks')
    dictionary = validate_preComp(dictionary, ppm_tol=ppm_tol_validation, isolation_window=isolation_window)

    gu.dict_to_sheet(dictionary, output_path)
    print(f'pre-assembly sheet saved to {output_path}')
    return None
