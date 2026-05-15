# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:37:17 2025

@author: Jakob
"""

import utils.genericUtilities as gu
import utils.fragmentAnnotationNew as fa
import zipfile
import io
import csv
import os

DDA_COLUMNS = [
    'Compound',
    'Formula',
    'Adduct',
    'm/z',
    'z',
    'RT Time (min)',
    'Window (min)'
]

def natively_charged_adduct(
    mol_formula, monoisotopic_mass
):
    """
    Helper, returns expected m/z for compounds based on (native) charge state.
    """
    
    z = 1 if mol_formula.endswith('+') else int(mol_formula[-1])
    return (monoisotopic_mass), z if z == 1 else (monoisotopic_mass / z, z)

def group_by_mixture(dictionary):
    """
    Helper, groups compounds by mixture assignments for creating DDA lists.
    """
    
    mixtures = {}
    for name, data in dictionary.items():
        try:
            mixtures.setdefault(data['assignedMixture'], []).append(name)
        except KeyError as e:
            print(f'missing mixture column: {e}')
            break
    return mixtures

def write_rows(
    writer, compound_name, formula, adducts, data, 
    adducts_formatted, z_values, rt, rt_window
):
    """
    Helper, writes a complete row for one compound-adduct combination.
    """
    
    for adduct, formatted, z in zip(adducts, adducts_formatted, z_values):
        mass = data.get(adduct, '') if adduct else data['monoisotopicMass']
        row = [compound_name, formula if formatted else '', formatted, mass, z, rt, rt_window]
        writer.writerow(row)

def create_targetDDA_app(dictionary, settings, mode):
    """
    Organizes creation of Thermo XCalibur-format targeted DDA inclusion lists.
    
    Parameters & args:
        dictionary (dict): Mix module output sheet
        settings (tuple): User settings, fed forward from the web app
        mode (string): Current mode, pos/neg
        
    Returns:
        zip_buffer (file-like): Zipfile buffer with DDA lists, downloadable via web app
    """
    
    mixtures = group_by_mixture(dictionary)
    max_mz, double_charge_limit, rt_baseline, rt_window = settings
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for mode in ('pos', 'neg'):
            for mix, compounds in mixtures.items():
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow([
                    'Compound', 'Formula', 'Adduct', 'm/z', 'z', 'RT Time (min)', 'Window (min)']
                )
                
                for compound in compounds:
                    data = dictionary[compound]
                    formula = data.get('molecularFormula', '')
                    mass = data.get('monoisotopicMass', 0)
                    
                    if mode == 'pos' and formula.endswith('+'):
                        mass, z = natively_charged_adduct(formula, mass)
                        writer.writerow([compound, formula, '', mass, z, rt_baseline, rt_window])
                        continue
                    if mode == 'neg' and '+' in formula:
                        continue
                    if mode == 'pos':
                        if mass > max_mz:
                            adducts = ['[M+2H]2+']
                            formatted = ['']
                            z_vals = [2]
                        elif mass > double_charge_limit:
                            adducts = ['[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+2H]2+']
                            formatted = ['+H', '+NH4', '+Na', '']
                            z_vals = [1, 1, 1, 2]
                        else:
                            adducts = ['[M+H]+', '[M+NH4]+', '[M+Na]+']
                            formatted = ['+H', '+NH4', '+Na']
                            z_vals = [1, 1, 1]
                    elif mode == 'neg':
                        if mass > max_mz:
                            adducts = ['[M-2H]2-']
                            formatted = ['']
                            z_vals = [2]
                        elif mass > double_charge_limit:
                            adducts = ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-', '[M-2H]2-']
                            formatted = ['-H', '', '', '']
                            z_vals = [1, 1, 1, 2]
                        else:
                            adducts = ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-']
                            formatted = ['-H', '', '']
                            z_vals = [1, 1, 1]
                        
                    for adduct, fmt, z_val in zip(adducts, formatted, z_vals):
                        mz_val = data.get(adduct, mass if not adduct else '')
                        writer.writerow([compound, formula, fmt, mz_val, z_val, rt_baseline, rt_window])
                        
                csv_content = csv_buffer.getvalue()
                filename = f'{mode}/ddaList_{mode}_mixture_{mix}.csv'
                zf.writestr(filename, csv_content)
            
    zip_buffer.seek(0)
    return zip_buffer

# Testing.
#natively_charged_adduct('C12H30N2+2', 202.240898965)
#create_targetDDA('output/prepThreeSheet.csv', 'pos')
#create_targetDDA('output/prepThreeSheet.csv', 'neg')

# -------------------- CLI SUPPORT --------------------

default_settings = {
    'max_mz': 950,
    'double_charge_limit': 600,
    'baseline_rt': 8,
    'baseline_rt_window': 15,
}

def create_targetDDA(sheet_path, mode, output_dir, settings=default_settings):
    """
    Organizes creation of Thermo XCalibur-format targeted DDA inclusion lists.
    CLI version: reads from file, writes CSVs directly to output/ddaLists/{mode}/.
    
    Parameters & args:
        sheet_path (string): Path to mix module output sheet
        mode (string): Current mode, pos/neg
        settings (dict): Optional settings dict; falls back to default_settings
    """
    
    dictionary = gu.sheet_to_dict(sheet_path)
    mixtures = group_by_mixture(dictionary)
    max_mz = settings.get('max_mz', 950)
    double_charge_limit = settings.get('double_charge_limit', 600)
    rt_baseline = settings.get('baseline_rt', 8)
    rt_window = settings.get('baseline_rt_window', 15)
    
    mode_settings = {
        'pos': {
            'adducts': [
                (max_mz, ['[M+2H]2+'], ['',], [2]),
                (double_charge_limit, ['[M+H]+','[M+NH4]+','[M+Na]+', '[M+2H]2+'], ['+H', '+Na', '+NH4', ''], [1, 1, 1, 2]),
                (0, ['[M+H]+','[M+NH4]+','[M+Na]+'], ['+H', '+Na', '+NH4'], [1, 1, 1])
            ]
        },
        'neg': {
            'adducts': [
                (max_mz, ['[M-2H]2-'], ['',], [2]),
                (double_charge_limit, ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-', '[M-2H]2-'], ['-H', '', '', ''], [1, 1, 1, 2]),
                (0, ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-'], ['-H', '', ''], [1, 1, 1])
            ]
        }
    }

    os.makedirs(output_dir, exist_ok=True)

    for mix, compounds in mixtures.items():
        with open(f"{output_dir}ddaList_{mode}_mixture_{mix}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(DDA_COLUMNS)
            for compound in compounds:
                data = dictionary[compound]
                formula = data['molecularFormula']
                mass = data['monoisotopicMass']
                if mode == 'pos' and '+' in formula:
                    mass, z = natively_charged_adduct(formula, mass)
                    writer.writerow([compound, formula, '', mass, z, rt_baseline, rt_window])
                    continue
                if mode == 'neg' and '+' in formula:
                    continue
                for threshold, adducts, adducts_formatted, z_values in mode_settings[mode]['adducts']:
                    if mass > threshold:
                        write_rows(writer, compound, formula, adducts, data,
                                   adducts_formatted, z_values, rt_baseline, rt_window)
                        break
    return None
