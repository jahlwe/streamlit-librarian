# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:54:05 2025

@author: Jakob
"""

import os
import gc
import argparse
import utils.genericUtilities as gu
import utils.pubchemUtilities as pu
import utils.surveyUtilities as su
import utils.mixtureUtilities as mu
import utils.compilerUtilities as cu
import utils.ddaLists as dda

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# qpc ; query_PubChem
# sdb ; survey_DBs
# mix ; generate mixtures

###############
# PREPARATORY #
###############

def query_PubChem(
    sheet_path, 
    save_path = 'output/', 
    preferred_key = 'internalName', 
    save_name = 'pcq_out'
):
    dictionary = gu.sheet_to_dict(sheet_path, preferred_key)
    if dictionary:
        dictionary = pu.pcQueries(dictionary)
        gu.dict_to_sheet(dictionary, save_name)
    
def reQuery_PubChem(
    sheet_path, 
    save_path = 'output/', 
    preferred_key = 'internalName', 
    save_name = 'repcq_out'
):
    dictionary = gu.sheet_to_dict(sheet_path, preferred_key)
    if dictionary:
        dictionary = pu.reQuery_CID(dictionary)
        gu.dict_to_sheet(dictionary, save_name)

def survey_DBs(
    sheet_path, 
    json=None, # massbank .json path
    csv_pos=None, # gnps .csv pos path
    csv_neg=None, # gnps .csv neg path
    mgf_pos=None, # gnps .mgf pos path
    mgf_neg=None, # gnps .mgf neg path
    save_path = '', # dict_to_sheet always uses /output 
    save_name= 'sdb_out'
):
    # Fixing paths. A little annoying.
    folder_path = 'files/survey/'
    path_dict = {
        'json': json,
        'csv_pos': csv_pos,
        'csv_neg': csv_neg,
        'mgf_pos': mgf_pos,
        'mgf_neg': mgf_neg
    }
    updated_paths = {
        key: os.path.join(folder_path, path) 
        for key, path in path_dict.items() 
        if path is not None
    }
    json = updated_paths.get('json', json)
    csv_pos = updated_paths.get('csv_pos', csv_pos)
    csv_neg = updated_paths.get('csv_neg', csv_neg)
    mgf_pos = updated_paths.get('mgf_pos', mgf_pos)
    mgf_neg = updated_paths.get('mgf_neg', mgf_neg)
    
    dictionary = gu.sheet_to_dict(sheet_path)
    print(f'running database survey --- n {len(dictionary)} compounds')
    if json:
        try:
            mb_json = su.load_json(json)
            mb_dict = su.parse_json(mb_json)
            # delete and release memory
            del mb_json
            gc.collect()
            # survey MB
            print('surveying massbank (.json)')
            dictionary = su.survey_massbank(dictionary, mb_dict)
            del mb_dict
            gc.collect()
        except FileNotFoundError:
            return f'File {json} not found.'
    if csv_pos and csv_neg:
        try:
            gnps_csv_pos = su.load_gnps_csv(csv_pos)
            print('surveying gnps (.csv, pos)')
            dictionary = su.survey_gnps_csv(dictionary, gnps_csv_pos)
            del gnps_csv_pos
            gc.collect()
            gnps_csv_neg = su.load_gnps_csv(csv_neg)
            print('surveying gnps (.csv, neg)')
            dictionary = su.survey_gnps_csv(dictionary, gnps_csv_neg) 
            del gnps_csv_neg
            gc.collect()
        except FileNotFoundError:
            return f'Files {csv_pos}, {csv_neg} not found.'
    if mgf_pos and mgf_neg:
        try:
            gnps_mgf_pos = su.parse_gnps_mgf(mgf_pos)
            print('surveying gnps (.mgf, pos)')
            dictionary = su.survey_gnps_mgf(dictionary, gnps_mgf_pos)
            del gnps_mgf_pos
            gc.collect()
            gnps_mgf_neg = su.parse_gnps_mgf(mgf_neg)
            print('surveying gnps (.mgf, neg)')
            dictionary = su.survey_gnps_mgf(dictionary, gnps_mgf_neg) 
            del gnps_mgf_neg
            gc.collect()
        except FileNotFoundError:
            return f'Files {mgf_pos}, {mgf_neg} not found.'
    # Do the instrument thing
    dictionary = su.evaluate_instruments(dictionary)
    if dictionary:
        gu.dict_to_sheet(dictionary, f'{save_name}')

def create_mixtures(
    sheet_path,
    n_mixes, # n mixes desired
    min_diff = 0.01, # mass proximity limit
    enforce = False, # enforce assignment of all compounds without fail if True
    save_path = '', # dict_to_sheet always uses /output
    save_name = 'mix_out'
):
    dictionary = gu.sheet_to_dict(sheet_path)
    if dictionary:
        try:
            dictionary = mu.generate_adducts(dictionary)
            dictionary = mu.expected_mz(dictionary)
            dictionary = mu.calculate_xlogp(dictionary)
            normalized_data, working_sheet, index = mu.prepare_data(dictionary)
            dictionary, result_sheet = mu.distribute_compounds(
                dictionary, working_sheet, normalized_data, 
                n_mixes, min_diff, enforce, index
            )
            mu.mixture_stats(working_sheet)
            gu.dict_to_sheet(dictionary, f'{save_name}')
        except Exception as e:
            print(f'{type(e).__name__}, "{e}", exiting.')
            
def ddaLists(
    sheet_path,
    mode        
):
    dda.create_targetDDA(sheet_path, mode)

#dictionary = gu.sheet_to_dict('output/prepOneSheet.csv')
#reQuery_PubChem('output/prepOneSheet.csv')

#survey_DBs('output/prepOneSheet.csv', json='MassBank.json', csv_pos='GNPS_pos.csv',
#           csv_neg='GNPS_neg.csv', mgf_pos='MSNLIB-POSITIVE.mgf', mgf_neg='MSNLIB-NEGATIVE.mgf')

#create_mixtures('output/prepTwoSheet.csv', 40, 30)

############
# COMPILER #
############

def get_rtiSheet(mode):
    try:
        dictionary = cu.gather_matData(mode)
        if dictionary:
            cu.generate_rtiSheet(dictionary, mode)
    except Exception as e:
        print(f'{type(e).__name__}, "{e}", exiting.')

def preCompile(
    mode, 
    ref_path,
    annot_fragments=True,
    rti=False,
    classyfire=None
):
    try:
        dictionary = cu.gather_matData(mode)
    except Exception as e:
        print(f'{type(e).__name__}, "{e}", exiting.')
    if dictionary:
        dictionary = cu.create_compilation_dictionary(dictionary)
        ref_dictionary = gu.sheet_to_dict(ref_path, 'internalName')
        dictionary = cu.add_chemical_metadata(dictionary, ref_dictionary)
        del ref_dictionary
        gc.collect()
        dictionary = cu.add_manual_metadata(dictionary)
        if rti:
            rti_dictionary = cu.gather_RTIData(mode)
            dictionary = cu.add_RTIData(dictionary, rti_dictionary)
            del rti_dictionary
            gc.collect()
        if classyfire:
            if classyfire == 'm':
                dictionary = cu.manual_classyfire(
                    dictionary, 
                    'input/classyfire/cf_manual.csv'
                )
            # elif classyfire == 'a':
                # CF servers don't work, change this when they do.
    cu.prepare_preCompilationSheet(dictionary, mode, annot_fragments)
    print('done')

def compile_library(
    accession_start,
    mode,
    do_filter=True
):
    sheet_path=f'output/compiler/postComp_{mode}.csv'
    print(f'compiling library ({mode})')
    cu.create_txtFiles(accession_start, mode)
    cu.compSheet_to_msp(sheet_path, mode)
    print('done')
    
################
# COMMAND-LINE #
################

def run_pcq(args):
    query_PubChem(
    sheet_path=args.sheet_path,
    save_path=args.save_path,
    preferred_key=args.preferred_key,
    save_name=args.save_name
)
    
def run_repcq(args):
    reQuery_PubChem(
    sheet_path=args.sheet_path,
    save_path=args.save_path,
    preferred_key=args.preferred_key,
    save_name=args.save_name
)
    
def run_sdb(args):
    survey_DBs(
    sheet_path = args.sheet_path,
    json = args.mb_json,
    csv_pos = args.csv_pos,
    csv_neg = args.csv_neg,
    mgf_pos = args.mgf_pos,
    mgf_neg = args.mgf_neg,
    save_path = args.save_path,
    save_name = args.save_name
)

def run_mix(args):
    create_mixtures(
    sheet_path = args.sheet_path,
    n_mixes = args.n_mixes,
    min_diff = args.min_diff,
    enforce = args.enforce,
    save_path = args.save_path,
    save_name = args.save_name
)
    
def run_dda(args):
    ddaLists(
    sheet_path = args.sheet_path, 
    mode = args.mode
)

def get_rti(args):
    get_rtiSheet(
    mode = args.mode
)

def precomp(args):
    preCompile(
    mode = args.mode,
    ref_path = args.ref_path,
    annot_fragments = args.fa,
    rti = args.rti,
    classyfire = args.cf
)

def run_compile(args):
    compile_library(
    accession_start = args.acc_start,
    mode = args.mode,
    do_filter = args.filter    
)

def main():
    parser = argparse.ArgumentParser(description='Librarian command-line interface')
    subparsers = parser.add_subparsers(dest='step', required=True, help='Modules')
    
    # pcq
    parser_pcq = subparsers.add_parser('pcq', help='Query PubChem for chemical metadata')
    parser_pcq.add_argument('sheet_path', help='Input sheet path (required)')
    parser_pcq.add_argument('-s', '--save_path', default='output/', help='') 
    parser_pcq.add_argument('-k', '--preferred_key', default='internalName', help='')
    parser_pcq.add_argument('-n', '--save_name', default='pcq_out', help='')
    parser_pcq.set_defaults(func=run_pcq)
    
    # repcq
    parser_repcq = subparsers.add_parser('repcq', help='Re-query PubChem with manually added CIDs')
    parser_repcq.add_argument('sheet_path', help='Input sheet path (required)')
    parser_repcq.add_argument('-s', '--save_path', default='output/', help='') 
    parser_repcq.add_argument('-k', '--preferred_key', default='internalName', help='')
    parser_repcq.add_argument('-n', '--save_name', default='repcq_out', help='')
    parser_repcq.set_defaults(func=run_repcq)
    
    # sdb
    parser_sdb = subparsers.add_parser('sdb', help='Survey one or more databases for matching entries')
    parser_sdb.add_argument('sheet_path', help='Input sheet path (required)')
    parser_sdb.add_argument('-mb', '--mb_json', help='MassBank .json file path')
    parser_sdb.add_argument('-cp', '--csv_pos', help='GNPS positive .csv file path')
    parser_sdb.add_argument('-cn', '--csv_neg', help='GNPS negative .csv file path')
    parser_sdb.add_argument('-mp', '--mgf_pos', help='GNPS positive .mgf file path')
    parser_sdb.add_argument('-mn', '--mgf_neg', help='GNPS negative .mgf file path')
    parser_sdb.add_argument('-s', '--save_path', default='output/', help='Output directory')
    parser_sdb.add_argument('-n', '--save_name', default='sdb_out', help='Output file name')
    parser_sdb.set_defaults(func=run_sdb)
    
    # mix
    parser_mix = subparsers.add_parser('mix', help='Create mixtures from a (pcq) data sheet')
    parser_mix.add_argument('sheet_path', help='Input sheet path (required)')
    parser_mix.add_argument('n_mixes', type=int, help='Number of mixtures to create')
    parser_mix.add_argument('-d', '--min_diff', type=float, default=0.01, help='Minimum mass difference allowed in mixtures')
    parser_mix.add_argument('-e', '--enforce', default=False, help='Require assignment of all compounds by algorithm to finish')
    parser_mix.add_argument('-s', '--save_path', default='output/', help='Output directory')
    parser_mix.add_argument('-n', '--save_name', default='mix_out', help='Output file name')
    parser_mix.set_defaults(func=run_mix)
    
    # ddalists
    parser_dda = subparsers.add_parser('dda', help='Generate DDA lists from a data sheet. Requires "assignedMixture"-column from mix module')
    parser_dda.add_argument('mode', choices=['pos', 'neg'], help='Mode of experimental data (pos/neg)')
    parser_dda.add_argument('sheet_path', help='Input sheet path (required)')
      
    ##### compiler stuff #####
    
    # rti
    parser_rti = subparsers.add_parser('rti', help='Create sheets for RTI webapp. Requires .mat files in /input folder')
    parser_rti.add_argument('mode', choices=['pos', 'neg'], help='Mode of experimental data (pos/neg)')
    parser_rti.set_defaults(func=get_rti)
    
    # precomp
    parser_precomp = subparsers.add_parser(
        'precomp', help='Assemble all data required for final .txt files in one sheet. This sheet is later used to create MassBank-style .txt files')
    parser_precomp.add_argument('mode', choices=['pos', 'neg'], help='Mode of experimental data (pos/neg)')
    parser_precomp.add_argument('ref_path', help='Sheet to pull chemical metadata from. E.g., pcq_out')
    parser_precomp.add_argument('-fa', action='store_true', help='Perform fragment annotation')
    parser_precomp.add_argument('-rti', default=False, help='Include RTI data (True/False). Requires output sheets from RTI webapp in input/RTI folder')
    parser_precomp.add_argument('-cf', default=None, help='Include ClassyFire data. "m" to add "manual" data from .csv (path and name input/classyfire/cf_manual.csv). "a" to make new queries (CF servers are down! Not working!)')
    parser_precomp.set_defaults(func=precomp)
    
    # compile
    parser_compile = subparsers.add_parser('compile', help='Create .txt & .msp from preCompilation sheet')
    parser_compile.add_argument('acc_start', type=int, help='Number from which to start making new accession IDs')
    parser_compile.add_argument('mode', choices=['pos', 'neg'], help='Mode of experimental data (pos/neg)')
    parser_compile.add_argument('-f', '--filter', default=True, help='Filters preComp data --- keeps only one of enantiomer features and makes comment in the remaining entry. Also skips compounds whose names match those in files/compiler/exclude_compounds.txt')
    parser_compile.set_defaults(func=run_compile)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':    
    main()