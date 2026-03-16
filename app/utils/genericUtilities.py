# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 21:17:20 2025

@author: Jakob
"""

import pandas as pd
import ast
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import inchi
from typing import Dict, Any, Optional
import os
import re
import math
import io

def is_empty(val):
    """
    Helper for checking missing value variants.
    """
    
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == '':
        return True
    return False

def deep_get(d, keys, default=None):
    """
    Helper for navigating nested dictionaries, e.g. PubChem entry .json data.
    
    Parameters & args:
        d (dict): A dictionary
        keys (list): A list containing the sequence of elements to navigate through
    Returns:
        Value associated with the final element
    """
    
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

def convert_value(value):
    """
    Helper for value conversions when reading a sheet and creating a dictionary from it. 
    Empty cells become None rather than float 'nan'.
    """
    
    # handle NA-variants
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value_stripped = value.strip()
        if value_stripped in ('', 'nan', 'NaN', 'None'):
            return None
        if (value_stripped.startswith('[') and value_stripped.endswith(']')) or \
            (value_stripped.startswith('(') and value_stripped.endswith(')')):
            # need this to deal with numpy floats
            value_stripped = re.sub(r'np\.float64\(([^)]+)\)', r'\1', value_stripped)
            # deal with inf in lists
            value_stripped = value_stripped.replace('inf', '1e999')
            try:
                result = ast.literal_eval(value_stripped)
                # replace 1e999 with inf again
                if isinstance(result, list):
                    result = [
                        tuple(float('inf') if x == 1e999 else x for x in tup)
                        if isinstance(tup, tuple) else tup
                        for tup in result
                    ]
                elif isinstance(result, tuple):
                    result = tuple(float('inf') if x == 1e999 else x for x in result)
                return result
            except Exception:
                return value_stripped
        # try integer conversion
        if value_stripped.isdigit():
            return int(value_stripped)
        # try float conversion
        try:
            float_val = float(value_stripped)
            return float_val
        except ValueError:
            pass
        return value_stripped
    return value

INTEGER_COLUMNS = ['pubchemCID','pos_count','neg_count','mb_recCount',
                   'gnps_recCount','cid_q']

def sheet_to_dict(sheet, preferred_key='library_id'):
    """
    Creates a dictionary from a spreadsheet (.csv, .xlsx) input file.
    Used to turn Librarian-format spreadsheets into dictionaries for module operations.
    
    Parameters & args:
        sheet (string): Name/path for spreadsheet
        preferred_key (string): Spreadsheet column to use as dictionary keys
    Returns:
        dictionary (dict): A dictionary.
    """
    
    try:
        # need this to work with streamlit
        if hasattr(sheet, 'name'):
            if sheet.name.endswith('.csv'):
                in_sheet = pd.read_csv(sheet)
            elif sheet.name.endswith('.xlsx'):
                in_sheet = pd.read_excel(sheet)
            else:
                return 'invalid file type.'
        # this is for supplying a standard path; CLI
        elif isinstance(sheet, str):
            if sheet.endswith('.csv'):
                in_sheet = pd.read_csv(sheet)
            elif sheet.endswith('.xlsx'):
                in_sheet = pd.read_excel(sheet)
            else:
                return 'invalid file path/type.'
    except FileNotFoundError:
        print('input file not found.')
        return None
    
    possible_keys = ['library_id', 'internalName', 'vendorName', 'Name']
    key_column = preferred_key if preferred_key in in_sheet.columns else next((col for col in possible_keys if col in in_sheet.columns), None)
    if key_column is None:
        print('no valid name column found (library_id/internalName/vendorName/Name)')
        return None
    
    dictionary = {} # convert to dictionary
    for _, row in in_sheet.iterrows():
        key = row[key_column]
        sub_dict = {}
        for col, val in row.drop(key_column).items():
            # Convert integer columns
            if col in INTEGER_COLUMNS:
                if isinstance(val, float) and val.is_integer():
                    sub_dict[col] = int(val)
                elif isinstance(val, str) and val.isdigit():
                    sub_dict[col] = int(val)
                else:
                    sub_dict[col] = convert_value(val)
            else:
                sub_dict[col] = convert_value(val)
        sub_dict['keyColumn'] = key_column
        dictionary[key] = sub_dict
        
    return dictionary

#test = sheet_to_dict('output/compiler/preCompilationSheet_pos.xlsx', 'CH$NAME:')
#test = sheet_to_dict('output/prepTwoSheet.xlsx', 'vendorName')
#dictionary = sheet_to_dict('output/pcq_out.csv')

def dict_to_sheet(dictionary, file_name=None, fmat='.csv', buffer=None):
    """
    Creates a spreadsheet (.csv, .xlsx) from a dictionary.
    Output files created in chosen directories (CLI) or as downloadables (web-app)
    
    Parameters & args:
        dictionary (dict): Dictionary to use
    Returns:
        Nothing (in-memory object buffer in web-app)
    """
    
    if not dictionary:
        print('input dictionary is empty')
        return None
    # get the key column name
    first_item = next(iter(dictionary.values()))
    key_column = first_item.get('keyColumn', 'key')
    # convert to dataframe, get rid of key column
    out_sheet = pd.DataFrame.from_dict(dictionary, orient='index')
    if key_column in out_sheet.columns:
        out_sheet = out_sheet.drop(columns=[key_column])
    elif 'key' in out_sheet.columns:
        out_sheet = out_sheet.drop(columns=['key'])
    # organize columns
    out_sheet = out_sheet.reset_index().rename(columns={'index': key_column})    
    cols = [key_column] + [col for col in out_sheet.columns if col != key_column]
    out_sheet = out_sheet[cols]    
    
    # dealing with files through streamlit 
    if buffer is not None:
        if fmat == '.csv':
            out_sheet.to_csv(buffer, index=False)
        elif fmat in ['.xlsx', '.xls']:
            out_sheet.to_excel(buffer, index=False)
        else:
            print('unsupported format, use .csv or .xlsx')
            return None
        buffer.seek(0)
        return buffer
    else:
        os.makedirs('output', exist_ok=True)
        save_path = os.path.join('output', file_name + fmat)
        
        try:
            if fmat == '.csv':
                out_sheet.to_csv(save_path, index=False)
            elif fmat in ['.xlsx', '.xls']:
                out_sheet.to_excel(save_path, index=False)
            else:
                print("unsupported format, use .csv or .xlsx")
                return None
            print(f"{save_path} saved.")
        except Exception as e:
            print(f"failed to save file: {e}")

# new helpers for integer-indexed dictionaries.
# used for pcq.

PCQ_INIT_COLUMNS = [
    'library_id', 'name_q', 'cas_q', 'smiles_q', 'cid_q' 
]

def sheet_to_idx_dict(sheet):
    """
    Modified version of sheet-to-dict fn, needed for the initial Librarian module.
    Uses integer indexing rather than later column (e.g., library ID) indexing.

    Parameters & args:
        sheet (string): Name/path for spreadsheet
    Returns:
        idx_dict (dict): An integer-indexed dictionary.
    """
    
    try:
        # need this to work with streamlit
        if hasattr(sheet, 'name'):
            if sheet.name.endswith('.csv'):
                in_sheet = pd.read_csv(sheet)
            elif sheet.name.endswith('.xlsx'):
                in_sheet = pd.read_excel(sheet)
            else:
                return None
        # this is for supplying a standard path; CLI
        elif isinstance(sheet, str):
            if sheet.endswith('.csv'):
                in_sheet = pd.read_csv(sheet)
            elif sheet.endswith('.xlsx'):
                in_sheet = pd.read_excel(sheet)
            else:
                return None
    except FileNotFoundError:
        print('input file not found.')
        return None
    
    idx_dict = {} # convert to dictionary
    for idx, row in in_sheet.iterrows():
        key = idx
        # no need to condition on pcq --- we only use this for pcq.
        sub_dict = {pcq_col: None for pcq_col in PCQ_INIT_COLUMNS}
        for col, val in row.items():
            # convert integer columns
            if col in INTEGER_COLUMNS:
                if isinstance(val, float) and val.is_integer():
                    sub_dict[col] = int(val)
                elif isinstance(val, str) and val.isdigit():
                    sub_dict[col] = int(val)
                else:
                    sub_dict[col] = convert_value(val)
            else:
                sub_dict[col] = convert_value(val)
        idx_dict[key] = sub_dict
        
    return idx_dict

# this is probably not needed. actually it is needed. to save the pcq.
# we just use an 'idx_dict'-format for pcq stuff, then we can revert
# to indexing by name or ID.

def idx_dict_to_sheet(
    dictionary, file_name=None, fmat='.csv', buffer=None
):
    """
    Modified version of dict-to-sheet fn, needed for the initial Librarian module.

    Parameters & args:
        dictionary (string): An integer-indexed dictionary.
    Returns:
        Nothing (in-memory object buffer in web-app)
    """
    
    if not dictionary:
        print('input dictionary is empty')
        return None
    
    out_sheet = pd.DataFrame.from_dict(dictionary, orient='index')
    
    # only thing we need is to move name up front
    cols = list(out_sheet.columns)
    if 'library_id' in cols:
        cols.remove('library_id')
        cols = ['library_id'] + cols
        out_sheet = out_sheet[cols]
    
    # dealing with files through streamlit 
    if buffer is not None:
        if fmat == '.csv':
            out_sheet.to_csv(buffer, index=False)
        elif fmat in ['.xlsx', '.xls']:
            out_sheet.to_excel(buffer, index=False)
        else:
            print('unsupported format, use .csv or .xlsx')
            return None
        buffer.seek(0)
        return buffer
    else:
        os.makedirs('output', exist_ok=True)
        save_path = os.path.join('output', file_name + fmat)
        
        try:
            if fmat == '.csv':
                out_sheet.to_csv(save_path, index=False)
            elif fmat in ['.xlsx', '.xls']:
                out_sheet.to_excel(save_path, index=False)
            else:
                print("unsupported format, use .csv or .xlsx")
                return None
            print(f"{save_path} saved.")
        except Exception as e:
            print(f"failed to save file: {e}")
    # end

# For testing.
#dictionary = sheet_to_dict('pcq_out.csv')
#dictionary = sheet_to_dict('output/pcq_out.csv')
#dict_to_sheet(dictionary, 'testOut')

# More stuff.
def monoisotopic_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    monoisotopic_mass = Descriptors.ExactMolWt(mol)
    return monoisotopic_mass

def inchikey_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchikey = inchi.MolToInchiKey(mol)
    return inchikey

def inchi_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi_str = inchi.MolToInchi(mol)
    return inchi_str

# Test this stuff. Even though it really isnt necessary.
#monoisotopic_from_smiles('CC12C3=CC=CC=C3CC(N1)C4=CC=CC=C24')
#inchikey_from_smiles('CC12C3=CC=CC=C3CC(N1)C4=CC=CC=C24')
#inchi_from_smiles('CC12C3=CC=CC=C3CC(N1)C4=CC=CC=C24')