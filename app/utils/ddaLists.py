# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:37:17 2025

@author: Jakob
"""

import utils.genericUtilities as gu
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

def read_settings(path='files/ddaLists/settings.tsv'):
    settings = {}
    with open(path, 'r') as s:
        reader = csv.reader(s, delimiter='\t')
        for row in reader:
            if len(row) == 2:
                key, value = row
                try:
                    settings[key.strip()] = int(value.strip())
                except ValueError:
                    settings[key.strip()] = value.strip()
    return settings

def natively_charged_adduct(mol_formula, monoisotopic_mass):
    z = 1 if mol_formula.endswith('+') else int(mol_formula[-1])
    return (monoisotopic_mass), z if z == 1 else (monoisotopic_mass / z, z)

def group_by_mixture(dictionary):
    mixtures = {}
    for name, data in dictionary.items():
        mixtures.setdefault(data['assignedMixture'], []).append(name)
    return mixtures

def write_rows(writer, compound_name, formula, adducts, data, 
               adducts_formatted, z_values, rt, rt_window):
    for adduct, formatted, z in zip(adducts, adducts_formatted, z_values):
        mass = data.get(adduct, '') if adduct else data['monoisotopicMass']
        row = [compound_name, formula if formatted else '', formatted, mass, z, rt, rt_window]
        writer.writerow(row)

def create_targetDDA(sheet_path, mode):
    dictionary = gu.sheet_to_dict(sheet_path)
    mixtures = group_by_mixture(dictionary)
    settings = read_settings(path='files/ddaLists/settings.tsv')
    max_mz = settings.get('max_mz', 950)
    doubly_charged_limit = settings.get('doubly_charged_limit', 600)
    rt_baseline = settings.get('baseline_rt', 8)
    rt_window = settings.get('baseline_rt_window', 15)
    
    mode_settings = {
        'pos': {
            'output_dir': 'output/ddaLists/pos/',
            'adducts': [
                (max_mz, ['[M+2H]2+'], ['',], [2]),
                (doubly_charged_limit, ['[M+H]+','[M+NH4]+','[M+Na]+', '[M+2H]2+'], ['+H', '+Na', '+NH4', ''], [1, 1, 1, 2]),
                (0,   ['[M+H]+','[M+NH4]+','[M+Na]+'], ['+H', '+Na', '+NH4'], [1, 1, 1])
            ]
        },
        'neg': {
            'output_dir': 'output/ddaLists/neg/',
            'adducts': [
                (max_mz, ['[M-2H]2-'], ['',], [2]),
                (doubly_charged_limit, ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-', '[M-2H]2-'], ['-H', '', '', ''], [1, 1, 1, 2]),
                (0,   ['[M-H]-','[M+CH3COOH-H]-','[M+Cl]-'], ['-H', '', ''], [1, 1, 1])
            ]
        }
    }

    for mix, compounds in mixtures.items():
        with open(f"{mode_settings[mode]['output_dir']}ddaList_{mode}_mixture_{mix}.csv", 'w', newline='') as f:
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
                    continue  # skip natively charged cations in neg mode
                for threshold, adducts, adducts_formatted, z_values in mode_settings[mode]['adducts']:
                    if mass > threshold:
                        write_rows(writer, compound, formula, adducts, data, 
                                   adducts_formatted, z_values, rt_baseline, rt_window)
                        break
    return None

# Testing.
#natively_charged_adduct('C12H30N2+2', 202.240898965)
#create_targetDDA('output/prepThreeSheet.csv', 'pos')
#create_targetDDA('output/prepThreeSheet.csv', 'neg')
