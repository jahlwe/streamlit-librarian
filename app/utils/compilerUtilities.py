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
import utils.classyfireUtilities as cu
import utils.qcUtilities as qu
import utils.fragmentAnnotation as fa
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
    'NAME:': 'internalName',
    'Num Peaks:': 'num_peak'
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
    'internalName': 'CH$NAME:',
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
    'num_peak': 'PK$NUM_PEAK:',
    'ms2_peaks': 'PK$PEAK:',
    'ms2_annot': 'PK$ANNOTATION:', # added recently
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
mastersheet_field_conversion['internalName'] = 'CH$NAME:'
mastersheet_field_conversion['iupacName'] = 'CH$NAME: (IUPAC)'
MASTERSHEET_COLUMNS = {
    'file_name': 'File_name',
    'short_accession': 'Short ACCESSION name:',
    **mastersheet_field_conversion,
    'submitted_to_MBEU': 'Submitted to MBEU'
}

# need this here already to do mode validation in parse_matFile
def get_charge(adduct):
    # defines groups to match in a string 
    pattern = re.search(r'(\d*)([+-])$', adduct)
    if pattern:
        sign = 1 if pattern.group(2) == '+' else -1
        number = int(pattern.group(1)) if pattern.group(1) else 1
        return sign * number
    return 0 # if no charge 

# Functions for dealing with MS-DIAL .mat-files
def parse_matFile(
        file_path, 
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
                
                current_record = {'keyColumn': 'internalName'}
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
    current_charge = 1 if get_charge(current_record['ion_type']) > 0 else -1
    mode_agreement = current_charge == mode_sign
    if not mode_agreement:
        print(f'mode agreement check failed for {current_compound}')
    if current_compound and current_record and mode_agreement:
        dictionary[current_compound] = current_record.copy()
        
    return dictionary

# We have to do one mode at a time. Basically. Dictionary key issues.
def gather_matData(mode, folder_path='input/mat'):
    # A bit of error handling...
    if mode not in ['pos', 'neg']:
        raise ValueError('Invalid mode provided.')
    mat_dictionary = {}
    # This should work to look in all subdirectories
    mat_files = glob.glob(os.path.join(folder_path, '**/*.mat'), recursive=True)
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

def create_compilation_dictionary(mat_dictionary):           
    # create dictionary with all storage fields and transfer mat data
    dictionary = {key: {col: None for col in STORAGE_FIELDS} for key in mat_dictionary.keys()}
    for compound in dictionary:
        for col in dictionary[compound]:
            if compound in mat_dictionary and col in mat_dictionary[compound]:
                dictionary[compound][col] = mat_dictionary[compound][col]
        dictionary[compound]['internalName'] = compound
    return dictionary

#dictionary = create_compilation_dictionary(dictionary)

# Now, we want to read our prep sheet, if we have one, and use the data
# from there, instead of querying PubChem all over again.
#ref_dictionary = gu.sheet_to_dict('output/prepOneSheet.csv', 'internalName')

def add_chemical_metadata(dictionary, ref_dictionary):
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

def add_manual_metadata(dictionary, manual_metadata='files/compiler/manual_metadata.tsv'):
    manual_dictionary = {}
    with open(manual_metadata, 'r', newline='', encoding='utf-8') as m:
        reader = csv.reader(m, delimiter='\t')
        for key, value in reader:
            manual_dictionary[key] = value
    for compound in dictionary:
        dictionary[compound].update(manual_dictionary)
    return dictionary

#dictionary = add_manual_metadata(dictionary)

# sheets for the RTI website.
# needs .mat files -- obviously -- for collecting RT and stuff.
def generate_rtiSheet(dictionary, mode, save_path='output/RTI/'):
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
    for compound in dictionary.keys():
        lookup_name = re.sub(r' feature no\. \d+$', '', compound)
        if lookup_name in rti_dictionary.keys():
            dictionary[compound]['rti'] = rti_dictionary[lookup_name]['Exp. RTI']
        elif lookup_name not in rti_dictionary.keys():
            print(f'no RTI data found for compound {lookup_name}')
    
    return dictionary

#dictionary = add_RTIData(dictionary, rti_dictionary)

# FORGET ABOUT THIS FOR NOW - CLASSYFIRE SERVERS ARE DOWN.
def query_ClassyFire(dictionary, use_ref_sheet=False, sheet_path=None, no_queries=False):
    n_compounds = len(dictionary.keys())
    # this was added if we already have queried ClassyFire before and
    # want to reuse that half-filled sheet.
    if use_ref_sheet and sheet_path:
        print(f'transferring class information from {sheet_path}')
        ref_dictionary = gu.sheet_to_dict(sheet_path)
        for compound in dictionary.keys():
            if compound in ref_dictionary.keys() and ref_dictionary[compound]['class']:
                print(ref_dictionary[compound]['class'])
                dictionary[compound]['class'] = ref_dictionary[compound]['class']
    
    if not no_queries:
        print(f'querying ClassyFire --- n = {n_compounds} compounds')
        for i, compound in enumerate(dictionary.keys()):
            print(compound)
            if dictionary[compound]['CH$COMPOUND_CLASS:'] == None or str(dictionary[compound]['CH$COMPOUND_CLASS:']) == 'nan':
                current_smiles = dictionary[compound]['CH$SMILES:']
                print(current_smiles)
                try:
                    class_data = cu.get_classyfire(current_smiles)
                    print(class_data)
                    if class_data:
                        dictionary[compound]['CH$COMPOUND_CLASS:'] = class_data
                except:
                    pass
                
                # DELAY TO AVOID HTTPERROR
                time.sleep(10)
                
            # This is not helpful if queries are not being made for every compound
            # Move it outside the conditional and it's always helpful.
            if i % 10 == 0 and i != 0:
                print(f'processed {i} of {n_compounds} compounds')
                if i == n_compounds - 1:
                    print('processed {i+1} of {n_compounds} compounds')
                    print('done')
            
    return dictionary

#dictionary = add_compoundClasses(dictionary, True, 'output/compiler/classref_pos.xlsx', True)
#dictionary = add_compoundClasses(dictionary, True, 'output/compiler/classref_neg.xlsx', True)

# Load preComp-sheets into dictionaries to avoid having to redo above steps...
#dictionary = genericUtilities.sheet_to_dict('output/compiler/preCompilationSheet_pos.xlsx', 'CH$NAME:')
#dictionary = genericUtilities.sheet_to_dict('output/compiler/preCompilationSheet_neg.xlsx', 'CH$NAME:')

def manual_classyfire(dictionary, ref_sheet_path):
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
    # for Br-containing (Cl also sometimes? No?) compounds -- 
    # might need to calculate two monomasses, one for M and one for M+2 etc
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

#dictionary = gu.sheet_to_dict('output/compiler/preComp_pos.csv')
#adduct_checker('Diphenidol', dictionary['Diphenidol'])
#adduct_checker('Abacavir', dictionary['Abacavir'])
#(dictionary['Abacavir']['monoisotopicMass'] + 1.00783 - e_mass) - dictionary['Abacavir']['precursor_mz']

def prepare_preCompilationSheet(
    dictionary, mode, 
    annotate_fragments=True, fragment='brute',
    file_name='preComp'
):
    '''
    Turn dictionary into a sheet before making entries.
    Allows final manual edits to be made to the "preComp"-sheet,
    which is later used to create .txt files from
    '''
    for i, (compound, data) in enumerate(dictionary.items()):        
        # generate record_title and save sheet
        short_name = re.sub(r' feature no\. \d+$', '', compound)
        record_title = '; '.join(
            [short_name, data['instrument_type'], data['ms_type'], 
             data['collision_energy'], data['resolution'], data['ion_type']])
        data['title'] = record_title
        # apparently, MassBank does NOT want charge indicators in molecular formulas.
        # adjusting that here.
        if data.get('molecularFormula'):
            data['molecularFormula'] = re.sub(r"[+\-]\d*$|[+\-]$", '', data['molecularFormula'])
        # also! run the adduct checker.
        data['ion_type'], data['adduct_validated'] = adduct_checker(compound, data)
        # also! annotate fragments.
        if annotate_fragments:
            try:
                print(f'annotating {compound} MS2')
                data['frag_annot'] = fa.annotate_spectrum(
                    compound, data, fragment
                )
            except Exception as e:
                data['frag_annot'] = None
                print(f'failed fragment annotation for {compound}: {e}')  
                
    gu.dict_to_sheet(dictionary, f'compiler/{file_name}_{mode}')
    return None

#prepare_preCompilationSheet(dictionary, 'neg')
#prepare_preCompilationSheet(dictionary, 'pos')

def filter_preComp(
        sheet_path, 
        mode, 
        exclude_path='files/compiler/exclude_compounds.txt'
    ):
    '''
    Optional function to filter the preComp-sheet when assembling.
    Besides using a .txt file to skip compounds by name as listed
    in the .txt file, it also makes comments of enantiomer features
    for one of the features, while discarding the other.
    '''
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

def write_txtFile(compound, data, save_path, field_order=None):
    '''
    Helper for create_txtFiles --- writes a single .txt file from dictionary data.
    '''
    data['internalName'] = compound # need to make this explicit
    with open(save_path, 'w', encoding='utf-8') as f:
        fields = field_order if field_order else FIELD_CONVERSION.keys()
        for field in fields:
            mb_field = FIELD_CONVERSION.get(field)
            if not mb_field:
                continue
            value = data.get(field)
            if field == 'internalName' and value:
                value = re.sub(r' feature no\. \d+$', '', value)
            if field == 'ion_mode' and value: # mode should be CAPITALIZED. for MassBank.
                f.write(f'{mb_field} {value.upper()}\n')
                continue
            if value is None or str(value) == 'nan':
                continue
            if field in ('ms2_peaks', 'ms2_annot'): # do this to avoid double \n at the end
                f.write(f'{mb_field} {value}')
                continue
            f.write(f'{mb_field} {value}\n')
        f.write('//\n')

def create_txtFiles(
        accession_start, 
        mode, 
        sheet_path='output/compiler/',
        massbank_fields='files/compiler/massbank_fields.txt',
        txt_path=f'output/compiler/{str(date.today())}',
        do_filter=True,
    ):
    if mode not in ['pos', 'neg']:
        raise ValueError('Invalid mode provided.')
        
    input_sheet_path = f'output/compiler/preComp_{mode}.csv'
    dictionary = filter_preComp(input_sheet_path, mode) if do_filter else gu.sheet_to_dict(input_sheet_path) 
    
    for i, (compound, data) in enumerate(dictionary.items()):
        acc_n = f'{accession_start + i:06d}'
        current_accession = f'MSBNK-ACES_SU-AS{acc_n}'
        short_accession = f'AS{acc_n}'
        current_file = f'{compound}_{short_accession}'
        save_path = os.path.join(txt_path, mode, f'{current_accession}.txt')
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
                f' {theo_mz} {t_f} {f_count} {exp_mz} {ppm}\n' for theo_mz, t_f, f_count, exp_mz, ppm in data['frag_annot']
            )
            data['ms2_annot'] = annot_line
        
        # write txt file
        write_txtFile(compound, data, save_path)
    
    # create postComp sheet --- with nice formatting!
    output_sheet_path = os.path.join(sheet_path, f'postComp_{mode}.csv')
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

def compSheet_to_msp(sheet_path, mode):
    dictionary = gu.sheet_to_dict(sheet_path, 'CH$NAME:')
    msp_path = f'output/compiler/{str(date.today())}_{mode}.msp'
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
