# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 21:51:36 2025

@author: Jakob
"""

import utils.genericUtilities as gu
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

def generate_adducts(dictionary):
    """
    Generates theoretical adduct m/z as metadata for mixture distribution.
    Currently, few of these are actually used for automated mixture distribution considerations.
    May need to be updated later.

    Parameters & args:
        dictionary (dict): Dictionary with compound data to fill with adduct information
    Returns:
        dictionary (dict): The same input dictionary, now incl adduct information
    """
    
    pos_adducts = [ # this will do for now...
        ("[M]+", -0.00055, 1),
        ("[M+H]+", 1.00728, 1),
        ("[M+NH4]+", 18.03437, 1),
        ("[M+Na]+", 22.98977, 1),
        ("[M+K]+", 38.96371, 1),
        ("[M+2H]2+", 1.00728, 0.5)
    ]
    neg_adducts = [
        ("[M-H]-", -1.00728, 1),        
        ("[M+CH3COOH-H]-", 59.01385, 1),
        ("[M+Cl]-", 34.9694, 1),        
        ("[M+F]-", 18.9984, 1),         
        ("[M-2H]2-", -1.00728, 0.5)       
    ]
    for compound in dictionary.keys():
        current_mass = dictionary[compound]['monoisotopicMass']
        for adduct, mass_shift, charge_modifier in pos_adducts:
            mz = (current_mass + mass_shift) * charge_modifier
            dictionary[compound][adduct] = mz
        for adduct, mass_shift, charge_modifier in neg_adducts:
            mz = (current_mass + mass_shift) * charge_modifier
            dictionary[compound][adduct] = mz
    return dictionary

def expected_mz(dictionary):
    """
    Function that decides, for each compound, what m/z to consider for mixture distribution.
    Currently, singly charged ions are the default expected ion type.
    Natively charged cations are an exception, where the [M]+ (with appropriate charge) is expected.
    
    Only positive mode is considered --- the distribution of [M-H]- ions would not be different, only offset.
    
    Parameters & args:
        dictionary (dict): Dictionary with compound data to fill with expected m/z information
    Returns:
        dictionary (dict): The same input dictionary, now incl expected m/z information
    """
    
    for compound in dictionary.keys():
        current_formula = str(dictionary[compound]['molecularFormula'])
        current_mass = dictionary[compound]['monoisotopicMass']
        if current_formula != 'nan':
            if '+' in current_formula:
                charge_modifier = current_formula[-1]
                charge_modifier = int(charge_modifier) if charge_modifier.isdigit() else 1
                # we expect [M]+, divided by the charge
                dictionary[compound]['expected_mz_pos'] = (dictionary[compound]['[M]+'] / charge_modifier)
                # it won't show in neg, but throw in for completions sake
                dictionary[compound]['expected_mz_neg'] = dictionary[compound]['[M-H]-']
            elif '+' not in current_formula and current_mass > 900:
                # we 'expect' double charges
                dictionary[compound]['expected_mz_pos'] = dictionary[compound]['[M+2H]2+']
                dictionary[compound]['expected_mz_neg'] = dictionary[compound]['[M-2H]2-']
            elif '+' not in current_formula and current_mass < 900:
                # keep it simple for now, just basic adducts
                dictionary[compound]['expected_mz_pos'] = dictionary[compound]['[M+H]+']
                dictionary[compound]['expected_mz_neg'] = dictionary[compound]['[M-H]-']
        
    return dictionary

def calculate_xlogp(dictionary):
    """
    Calculates logp for compounds from SMILES using RDKit; atom-based approach of Crippen.
    
    Parameters & args:
        dictionary (dict): Dictionary with compound data to fill with xlogp information
    Returns:
        dictionary (dict): The same input dictionary, now incl xlogp information
    """
    
    for compound in dictionary.keys():
        if dictionary[compound]['smiles'] and str(dictionary[compound]['smiles']) != 'nan':
            current_smiles = dictionary[compound]['smiles']
            current_mol = Chem.MolFromSmiles(current_smiles)
            dictionary[compound]['xlogp'] = Crippen.MolLogP(current_mol)
        else:
            dictionary[compound]['xlogp'] = 0 # Maybe inappropriate. But NAs = annoying.
    return dictionary

# functions related to making mixes.
def is_unique_mass(group, new_compound, mass_tol=0.01):
    """
    Helper to check whether a compound mass in a mixture is considered unique
    given a specified mass proximity tolerance (Da).
    """
    
    if len(group) == 0:
        return True
    # what's an appropriate minimum difference to call something unique.
    return not np.any(np.abs(group[:, 0] - new_compound[0]) < mass_tol)

def min_xlogp_difference(group):
    """
    Helper to find the smallest difference in xlogp within a mixture.
    """
    
    if len(group) < 2:
        return float('inf')
    sorted_xlogp = np.sort(group[:, 1])
    return np.min(np.diff(sorted_xlogp))

def prepare_data(dictionary):
    """
    Helper that prepares datasets for the mixture distribution function.
    Filters out compounds that have m/z and xlogp data, and normalizes these.
    Both the filtered and normalized data are returned to keep track of assignments.
    """
    
    # let's focus on expected pos mz's for now...
    data = pd.DataFrame.from_dict(dictionary, orient='index')[['expected_mz_pos', 'xlogp']]
    filtered = data.dropna(subset=['expected_mz_pos', 'xlogp'])
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(filtered[['expected_mz_pos', 'xlogp']])
    return normalized_data, filtered, filtered.index

def dict_to_sheet_prepThree(dictionary):
    sheet = pd.DataFrame.from_dict(dictionary, orient='index')
    return sheet

def sheet_to_dict_prepThree(working_sheet):
    dictionary = working_sheet.to_dict(orient='index')    
    return dictionary

# need to break up assignment scheme into helpers
def assign_with_mass_diff(
        labels, data, n_groups, xlogp_order, min_diff, min_compounds, 
        max_compounds, n_larger_groups, enforce, index
):
    """
    Performs the default mixture assignment of compounds by to m/z.
    
    Parameters & args:
        labels (np.array): Contains mixture assignments for each compound (initialized as -1 for all compounds)
        data (dataframe): Contains m/z and xlogp values for each compound
        n_groups (int): Specified number of mixtures
        xlogp_order (series): xlogp column from 'data', ordered (ascending)
        min_diff (float): Specified minimum accepted within-mixture m/z difference
        min_compounds (int): Minimum n compounds per mixture given n_groups and total n compounds
        max_compounds (int): Maximum n compounds per mixture given n_groups and total n compounds
        n_larger_groups (int): The n of groups with +1 compound, if n compounds does not evenly split among grps
        enforce (bool): Controls whether assignment is allowed to finish when m/z assignment fails
                        i.e., if True, failure to assign by m/z given parameters will halt execution
                        Fed forward from the main assignment function, see below
        index (list): Compound index
        
    Returns:
        labels (np.array): Contains mixture assignments for each compound
    """
    n_samples = len(data)
    for i in range(n_samples):
        best_group = -1
        best_min_diff = -float('inf')

        for j in range(n_groups):
            current_group_size = np.sum(labels == j)
            max_size = max_compounds if j < n_larger_groups else min_compounds

            if current_group_size < max_size:
                temp_group = data[labels == j]
                new_compound = data[xlogp_order[i]]

                if is_unique_mass(temp_group, new_compound, min_diff):
                    temp_group = np.vstack((temp_group, new_compound))
                    this_min_diff = min_xlogp_difference(temp_group)

                    if this_min_diff > best_min_diff:
                        best_min_diff = this_min_diff
                        best_group = j

        if best_group == -1:
            for j in range(n_groups):
                current_group_size = np.sum(labels == j)
                max_size = max_compounds if j < n_larger_groups else min_compounds
                if current_group_size < max_size and is_unique_mass(data[labels == j], data[xlogp_order[i]], min_diff):
                    best_group = j
                    break

            if best_group == -1:
                if enforce:
                    raise Exception(f'cannot place compound {index[xlogp_order[i]]} in any mix without violating min mass diff')
                else:
                    print(f'cannot place compound {index[xlogp_order[i]]} in any mix without violating min mass diff')
                    labels[xlogp_order[i]] = -1
                    continue

        labels[xlogp_order[i]] = best_group
    return labels

def auto_assign_unplaced(
        labels, data, n_groups, xlogp_order, index
):
    """
    Performs assignment of compounds to mixtures by xlogp.
    
    Parameters & args:
        labels (np.array): Contains mixture assignments for each compound 
        data (dataframe): Contains m/z and xlogp values for each compound
        n_groups (int): Specified number of mixtures
        xlogp_order (series): xlogp column from 'data', ordered (ascending)
        index (list): Compound index
        
    Returns:
        labels (np.array): Contains mixture assignments for each compound
    """
    unassigned = list(np.where(labels == -1)[0])
    
    while len(unassigned) > 0:
        assigned_this_round = set()
        groups_assigned_this_round = set()
        np.random.shuffle(unassigned)
        for idx in unassigned:
            best_group = None
            best_min_gap = -np.inf
            compound_data = data[xlogp_order[idx]]
            for group in range(n_groups):
                if group in groups_assigned_this_round:
                    continue
                group_indices = np.where(labels == group)[0]
                current_group = data[group_indices]
                temp_group = np.vstack((current_group, compound_data))
                temp_min_gap = min_xlogp_difference(temp_group)
                if temp_min_gap > best_min_gap:
                    best_min_gap = temp_min_gap
                    best_group = group
            if best_group is not None:
                labels[idx] = best_group
                assigned_this_round.add(idx)
                groups_assigned_this_round.add(best_group)
                print(f'auto-assigning compound {index[idx]} to group {best_group+1}')
            if len(groups_assigned_this_round) == n_groups:
                break
        unassigned = [idx for idx in unassigned if idx not in assigned_this_round]
        
    return labels

def distribute_compounds(
        dictionary, working_sheet, data, n_groups, 
        min_diff=0.01, enforce=False, auto_assign=False, index=None
):
    """
    Organizes mixture distribution for the mix module.
    
    Parameters & args:
        dictionary (dict): Dictionary with all compound information
        working_sheet (dataframe): DataFrame with compound m/z and xlogp
        data (dataframe): DataFrame with normalized m/z and xlogp data
        n_groups (int): Specified number of mixtures
        min_diff (float): Specified minimum accepted within-mixture m/z difference
        enforce (bool): Controls whether assignment is allowed to finish when m/z assignment fails
                        i.e., if True, failure to assign by m/z given parameters will halt execution
        auto_assign (bool): Controls whether failed m/z assignments should automatically be assigned by xlogp
        index (list): Compound index
        
    Returns:
        dictionary (dict): Input dictionary updated w mixture assignments
        working_sheet (dataframe): DataFrame with m/z, xlogp, assignments for stats calculation (see below)
    """
    
    # now broken up assignment procedures into helpers
    n_samples = len(data)
    labels = np.full(n_samples, -1)
    min_compounds = n_samples // n_groups
    max_compounds = min_compounds + 1
    n_larger_groups = n_samples % n_groups
    xlogp_order = np.argsort(data[:, 1])

    labels = assign_with_mass_diff(
        labels, data, n_groups, xlogp_order, 
        min_diff, min_compounds, max_compounds, n_larger_groups, enforce, index
    )

    if auto_assign and not enforce:
        print('auto-assign enabled --- processing unassigned compounds...')
        labels = auto_assign_unplaced(labels, data, n_groups, xlogp_order, index)

    working_sheet['assignedMixture'] = labels + 1
    for k, compound in enumerate(working_sheet.index):
        if compound in dictionary:
            dictionary[compound]['assignedMixture'] = int(working_sheet['assignedMixture'].iloc[k])
    return dictionary, working_sheet

def mixture_stats(working_sheet, save_path = 'output/', streamlit=False):
    """
    Function to generate basic statistics that describe the mixture assignment results.
        
    Parameters & args:
        working_sheet (dataframe): DataFrame with m/z, xlogp, assignments
        save_path (string): String that directs all CLI-created files to output folder
        
    Returns:
        For CLI: Nothing --- saves a spreadsheet to disk
        For web-app: returns mixture_stats (dataframe), contains calculated statistics
    """
    
    n_groups = working_sheet['assignedMixture'].max()
    mixture_stats = pd.DataFrame()
    for group in range(1, n_groups + 1):
        group_data = working_sheet[working_sheet['assignedMixture'] == group]
        new_row = pd.DataFrame(columns=('mixture','n_compounds','min_expectedMass',
                                        'max_expectedMass','min_xlogp','max_xlogp',
                                        'min_diff_expectedMass','min_diff_xlogp'))
        new_row.at[0, 'mixture'] = group
        new_row.at[0, 'n_compounds'] = len(group_data)
        new_row.at[0, 'min_expectedMass'] = group_data['expected_mz_pos'].min()
        new_row.at[0, 'max_expectedMass'] = group_data['expected_mz_pos'].max()
        new_row.at[0, 'min_xlogp'] = group_data['xlogp'].min()
        new_row.at[0, 'max_xlogp'] = group_data['xlogp'].max()
        new_row.at[0, 'min_diff_expectedMass'] = np.min(np.diff(np.sort(group_data['expected_mz_pos'])))
        new_row.at[0, 'min_diff_xlogp'] = np.min(np.diff(np.sort(group_data['xlogp'])))
        mixture_stats = pd.concat([mixture_stats, new_row], ignore_index=True)
    # Save it
    if not streamlit:
        mixture_stats = mixture_stats.set_index('mixture')
        file_name = str(save_path + 'mixtureStats.xlsx')
        mixture_stats.to_excel(file_name)
    if streamlit:
        mixture_stats = mixture_stats.set_index('mixture')
        return mixture_stats
    
def visual_summary(dictionary, group_key='assignedMixture', mz_key='monoisotopicMass', xlogp_key='xlogp'):
    """
    Generates visual summaries of the mixture assignment results.
        
    Parameters & args:
        dictionary (dict): Dictionary with all compound information
        group_key (string): Name of variable that contains mixture assignments
        mz_key (string): Name of variable that contains monoisotopic mass data
        xlogp_key (string): Name of variable that contains xlogp data
        
    Returns:
        ...
    """
    
    # collect data
    groups = []
    masses = []
    xlogps = []

    for compound, data in dictionary.items():
        group = data.get(group_key)
        mass = data.get(mz_key)
        xlogp = data.get(xlogp_key)
        if group is not None and mass is not None and xlogp is not None:
            groups.append(int(group))
            masses.append(mass)
            xlogps.append(xlogp)

    # shift xlogp for plotting (all positive)
    min_xlogp = min(xlogps)
    xlogp_shifted = [x + (1 - min_xlogp) for x in xlogps]
    sizes = np.array(xlogp_shifted)
    if sizes.max() != sizes.min():
        sizes = 4 + 24 * (sizes - sizes.min()) / (sizes.max() - sizes.min())
    else:
        sizes = np.full_like(sizes, 10)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(masses, groups, s=sizes, c='#4d79ff', alpha=0.7, edgecolors='k', linewidth=0.5)

    ax.set_xlabel('Monoisotopic mass (Da)', fontsize=12)
    ax.set_ylabel('Mixture', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.set_xscale('log')

    # custom tick values
    tick_candidates = [100, 250, 500, 750, 1000, 1500, 2000]
    min_mass = min(masses)
    max_mass = max(masses)
    # only keep ticks within data range
    if min_mass < 100:
        xticks = [tick for tick in tick_candidates if min_mass * 0.95 <= tick <= max_mass * 1.05]
    else:
        xticks = [tick for tick in tick_candidates if tick <= max_mass * 1.05]
    # include min tick for niceness --- if it does not result in a cluttered addition
    if min_mass < 85:
        xticks = [int(min_mass)] + xticks
    # skip extra max mass tick
    #if xticks and xticks[-1] < max_mass * 1.05:
    #    xticks = xticks + [int(max_mass)]
    
    if min_mass < 100:
        ax.set_xlim(min_mass * 0.95, max_mass * 1.05)
    else:
        ax.set_xlim(95, max_mass * 1.05)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks])  # Set labels manually
    
    # Optionally turn off minor ticks (which can cause duplicate labels)
    ax.xaxis.set_minor_formatter(NullFormatter())

    # set y-axis ticks
    unique_groups = sorted(set(groups))
    ax.set_yticks(unique_groups)
    ax.set_ylim(min(unique_groups) - 0.5, max(unique_groups) + 0.5)

    # hide axes top & right
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    n_compounds = len(groups)
    n_groups = len(unique_groups)
    ax.set_title(f'Compound distribution (n = {n_compounds} compounds; n = {n_groups} mixtures)', 
                 fontsize=10, fontweight='bold', loc='left')

    plt.tight_layout()
    plt.show()

# Testing 
#dictionary = gu.sheet_to_dict('output/sdb_out.csv')
#dictionary = gu.sheet_to_dict('output/pcq_test.csv')
#dictionary = generate_adducts(dictionary)
#dictionary = expected_mz(dictionary)
#dictionary = calculate_xlogp(dictionary)
#normalized_data, working_sheet, ws_index = prepare_data(dictionary)
#dictionary, working_sheet = distribute_compounds(
#    dictionary, working_sheet, normalized_data, 
#    30, auto_assign=True, index=ws_index)
#dictionary, working_sheet = distribute_compounds(
#    dictionary, working_sheet, normalized_data, 
#    2, auto_assign=True, index=ws_index)
#mixture_stats(working_sheet)
#visual_summary(dictionary)
