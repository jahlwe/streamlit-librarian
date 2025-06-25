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

def deep_get(d, keys, default=None):
    '''
    Helper for diving into .json or other nested
    structures. Enter names of lists, dicts, etc 
    to navigate through to get where you want to 
    go in the .json (or other structure) as 'keys'
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

def convert_value(value):
    '''
    Helper for value conversions when reading 
    a sheet and creating a dictionary from it.
    Empty cells become None rather than float 'nan'.
    '''
    # handle NA-variants
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value_stripped = value.strip()
        if value_stripped in ('', 'nan', 'NaN', 'None'):
            return None
        if value_stripped.startswith('[') and value_stripped.endswith(']'):
            # need this to deal with numpy floats
            value_stripped = re.sub(r'np\.float64\(([^)]+)\)', r'\1', value_stripped)
            try:
                return ast.literal_eval(value_stripped)
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
                   'gnps_recCount']

def sheet_to_dict(sheet, preferred_key='internalName'):
    '''
    Create a dictionary from a sheet-style 
    (.csv, .xlsx) input file.
    '''
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
    
    possible_keys = ['internalName', 'vendorName', 'Name']
    key_column = preferred_key if preferred_key in in_sheet.columns else next((col for col in possible_keys if col in in_sheet.columns), None)
    if key_column is None:
        print('no valid name column found (internalName/vendorName/Name)')
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
    '''
    Create a sheet-style output file (.csv, .xlsx) 
    from a dictionary.
    '''
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
    
# For testing.
#dictionary = sheet_to_dict('input/compoundList.xlsx')
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