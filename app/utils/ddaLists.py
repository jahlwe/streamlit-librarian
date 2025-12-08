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
    z = 1 if mol_formula.endswith('+') else int(mol_formula[-1])
    return (monoisotopic_mass), z if z == 1 else (monoisotopic_mass / z, z)

def group_by_mixture(dictionary):
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
    for adduct, formatted, z in zip(adducts, adducts_formatted, z_values):
        mass = data.get(adduct, '') if adduct else data['monoisotopicMass']
        row = [compound_name, formula if formatted else '', formatted, mass, z, rt, rt_window]
        writer.writerow(row)

def create_targetDDA_app(dictionary, settings, mode):
    mixtures = group_by_mixture(dictionary)
    max_mz, double_charge_limit, rt_baseline, rt_window = settings
    
    # we condition what adducts we provide based on the m/z
    # more info --- the first adducts are "normal" formatting, the second [] is XCalibur formatting
    # this can also be empty, if there isn't a native format for an adduct in XCalibur
    # the final column contains charge values
    # the columns we need to consider are
    # compound - formula - adduct - m/z - z - RT - window
    
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
                    
                    # make adducts customizable...? later.
                    
                    # if native charge, we only use that one
                    if mode == 'pos' and formula.endswith('+'):
                        mass, z = natively_charged_adduct(formula, mass)
                        writer.writerow([compound, formula, '', mass, z, rt_baseline, rt_window])
                        continue
                    if mode == 'neg' and '+' in formula:
                        continue  # skip native cations in neg mode
                    # rules for others
                    if mode == 'pos':
                        if mass > max_mz: # only double charge if mass is above top end
                            adducts = ['[M+2H]2+']
                            formatted = ['']
                            z_vals = [2]
                        elif mass > double_charge_limit: # incl double charge if...
                            adducts = ['[M+H]+', '[M+NH4]+', '[M+Na]+', '[M+2H]2+']
                            formatted = ['+H', '+NH4', '+Na', '']
                            z_vals = [1, 1, 1, 2]
                        else: # and if not, use the basics
                            adducts = ['[M+H]+', '[M+NH4]+', '[M+Na]+']
                            formatted = ['+H', '+NH4', '+Na']
                            z_vals = [1, 1, 1]
                    elif mode == 'neg':
                        if mass > max_mz: # only double charge if mass is above top end
                            adducts = ['[M-2H]2-']
                            formatted = ['']
                            z_vals = [2]
                        elif mass > double_charge_limit: # incl double charge if...
                            adducts = ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-', '[M-2H]2-']
                            formatted = ['-H', '', '', '']
                            z_vals = [1, 1, 1, 2]
                        else: # and if not, use the basics
                            adducts = ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-']
                            formatted = ['-H', '', '']
                            z_vals = [1, 1, 1]
                        
                        
                    for adduct, fmt, z_val in zip(adducts, formatted, z_vals):
                        mz_val = data.get(adduct, mass if not adduct else '')
                        writer.writerow([compound, formula, fmt, mz_val, z_val, rt_baseline, rt_window])
                        
                # add csv to zip
                csv_content = csv_buffer.getvalue()
                filename = f'{mode}/ddaList_{mode}_mixture_{mix}.csv'
                zf.writestr(filename, csv_content)
            
    zip_buffer.seek(0)
    return zip_buffer

# Testing.
#natively_charged_adduct('C12H30N2+2', 202.240898965)
#create_targetDDA('output/prepThreeSheet.csv', 'pos')
#create_targetDDA('output/prepThreeSheet.csv', 'neg')
