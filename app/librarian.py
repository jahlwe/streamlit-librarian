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
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

###############
# PREPARATORY #
###############

def query_PubChem(
    sheet_path,
    output_path,
    preferred_key='library_id',
):
    ip, op = Path(sheet_path), Path(output_path)
    if ip.suffix.lower() not in ('.csv', '.xlsx', '.xls'):
        raise ValueError('unsupported input format')
    if op.suffix.lower() not in ('.csv', '.xlsx', '.xls'):
        raise ValueError('unsupported output format')
        
    dictionary = gu.sheet_to_dict(sheet_path, preferred_key)
    if dictionary:
        dictionary = pu.pcQueries_CLI(dictionary)
        gu.dict_to_sheet(dictionary, output_path)

def survey_DBs(
    sheet_path,
    output_path,
    json=None,       # massbank .json path
    csv_pos=None,    # gnps .csv pos path
    csv_neg=None,    # gnps .csv neg path
    mgf_pos=None,    # gnps .mgf pos path
    mgf_neg=None,    # gnps .mgf neg path
):
    dictionary = gu.sheet_to_dict(sheet_path)
    print(f'running database survey --- n {len(dictionary)} compounds')
    if json:
        try:
            mb_json = su.load_json(json)
            mb_dict = su.parse_json(mb_json)
            del mb_json
            gc.collect()
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
    dictionary = su.evaluate_instruments(dictionary)
    if dictionary:
        gu.dict_to_sheet(dictionary, output_path)

def create_mixtures(
    sheet_path,
    n_mixes,
    output_path,
    min_diff=0.01,
    enforce=False,
    auto_assign=False,
):
    dictionary = gu.sheet_to_dict(sheet_path)
    if dictionary:
        try:
            dictionary = mu.generate_adducts(dictionary)
            dictionary = mu.expected_mz(dictionary)
            dictionary = mu.calculate_xlogp(dictionary)
            normalized_data, working_sheet, index = mu.prepare_data(dictionary)
            dictionary, working_sheet = mu.distribute_compounds(
                dictionary, working_sheet, normalized_data,
                n_groups=n_mixes, min_diff=min_diff,
                enforce=enforce, auto_assign=auto_assign, index=index
            )
            stats_dir = (os.path.dirname(output_path) or '.') + os.sep
            mu.mixture_stats(working_sheet, save_path=stats_dir)
            gu.dict_to_sheet(dictionary, output_path)
        except Exception as e:
            print(f'{type(e).__name__}, "{e}", exiting.')

def ddaLists(
    sheet_path,
    mode,
    output_dir,
    settings=None
):
    dda.create_targetDDA(sheet_path, mode, output_dir, settings or dda.default_settings)

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
    data_dir,
    ref_path,
    tsv_path,
    output_path,
    annot_fragments=True,
    rti=False,
    classyfire=None,
):
    try:
        dictionary = cu.gather_matData(mode, data_dir)
    except Exception as e:
        print(f'{type(e).__name__}, "{e}", exiting.')
        return
    if dictionary:
        dictionary = cu.create_compilation_dictionary(dictionary, cu.STORAGE_FIELDS)
        ref_dictionary = gu.sheet_to_dict(ref_path, 'library_id')
        dictionary = cu.add_chemical_metadata(dictionary, ref_dictionary)
        del ref_dictionary
        gc.collect()
        dictionary = cu.add_manual_metadata(dictionary, tsv_path)
        if rti:
            rti_dictionary = cu.gather_RTIData(mode)
            dictionary = cu.add_RTIData(dictionary, rti_dictionary)
            del rti_dictionary
            gc.collect()
        if classyfire:
            dictionary = cu.manual_classyfire(dictionary, classyfire)
        # updated precomp function...
        cu.preCompile_CLI(dictionary, mode, output_path, annot_fragments)
    print('done')

def compile_library(
    precomp_sheet_path,
    output_dir,
    accession_long,
    accession_short,
    accession_start,
    mode,
    do_filter,
):
    print(f'compiling library ({mode})')
    cu.create_txtFiles(
        precomp_sheet_path, output_dir, accession_long, accession_short,
        accession_start, mode, do_filter=False) # keeping filter fn for now but dont use it
    cu.compSheet_to_msp(output_dir, mode)
    print('done')

################
# COMMAND-LINE #
################

def run_pcq(args):
    query_PubChem(
        sheet_path=args.sheet_path,
        output_path=args.output_path,
        preferred_key=args.preferred_key,
    )

def run_sdb(args):
    survey_DBs(
        sheet_path=args.sheet_path,
        output_path=args.output_path,
        json=args.mb_json,
        csv_pos=args.csv_pos,
        csv_neg=args.csv_neg,
        mgf_pos=args.mgf_pos,
        mgf_neg=args.mgf_neg,
    )

def run_mix(args):
    create_mixtures(
        sheet_path=args.sheet_path,
        n_mixes=args.n_mixes,
        output_path=args.output_path,
        min_diff=args.min_diff,
        enforce=args.enforce,
        auto_assign=args.auto_assign,
    )

def run_dda(args):
    settings = {
        'max_mz': args.max_mz,
        'double_charge_limit': args.double_charge_limit,
        'baseline_rt': args.baseline_rt,
        'baseline_rt_window': args.baseline_rt_window,
    }
    ddaLists(
        sheet_path=args.sheet_path,
        mode=args.mode,
        output_dir=args.output_dir,
        settings=settings,
    )

def get_rti(args):
    get_rtiSheet(
        mode=args.mode,
    )

def precomp(args):
    preCompile(
        mode=args.mode,
        data_dir=args.data_dir,
        ref_path=args.ref_path,
        tsv_path=args.tsv_path,
        output_path=args.output_path,
        annot_fragments=args.fa,
        rti=args.rti,
        classyfire=args.cf,
    )

def run_compile(args):
    compile_library(
        precomp_sheet_path=args.precomp_sheet_path,
        output_dir=args.output_dir,
        accession_long=args.acc_long,
        accession_short=args.acc_short,
        accession_start=args.acc_start,
        mode=args.mode,
        do_filter=args.filter,
    )


def main():
    parser = argparse.ArgumentParser(description='Librarian command-line interface')
    subparsers = parser.add_subparsers(dest='step', required=True, help='Modules')

    # pcq
    parser_pcq = subparsers.add_parser('pcq', help='Query PubChem for chemical metadata')
    parser_pcq.add_argument('sheet_path', help='Input sheet path (e.g. data/compounds.xlsx)')
    parser_pcq.add_argument('output_path', help='Output file path including extension (e.g. results/pcq_out.csv)')
    parser_pcq.add_argument('-k', '--preferred_key', default='library_id', help='Column to use as dictionary key (default: library_id)')
    parser_pcq.set_defaults(func=run_pcq)

    # sdb
    parser_sdb = subparsers.add_parser('sdb', help='Survey one or more databases for matching entries')
    parser_sdb.add_argument('sheet_path', help='Input sheet path (e.g. results/pcq_out.csv)')
    parser_sdb.add_argument('output_path', help='Output file path including extension (e.g. results/sdb_out.csv)')
    parser_sdb.add_argument('-mb', '--mb_json', help='MassBank .json file path')
    parser_sdb.add_argument('-cp', '--csv_pos', help='GNPS positive mode .csv file path')
    parser_sdb.add_argument('-cn', '--csv_neg', help='GNPS negative mode .csv file path')
    parser_sdb.add_argument('-mp', '--mgf_pos', help='GNPS positive mode .mgf file path')
    parser_sdb.add_argument('-mn', '--mgf_neg', help='GNPS negative mode .mgf file path')
    parser_sdb.set_defaults(func=run_sdb)

    # mix
    parser_mix = subparsers.add_parser('mix', help='Create mixtures from a (pcq) data sheet')
    parser_mix.add_argument('sheet_path', help='Input sheet path (e.g. results/sdb_out.csv)')
    parser_mix.add_argument('n_mixes', type=int, help='Number of mixtures to create')
    parser_mix.add_argument('output_path', help='Output file path including extension (e.g. results/mix_out.csv)')
    parser_mix.add_argument('-d', '--min_diff', type=float, default=0.01, help='Minimum mass difference allowed within a mixture (default: 0.01)')
    parser_mix.add_argument('-e', '--enforce', default=False, help='Require all compounds to be assigned before finishing (default: False)')
    parser_mix.add_argument('-a', '--auto_assign', action='store_true', help='Auto-assign unplaced compounds by xlogp if mass diff constraint cannot be met')
    parser_mix.set_defaults(func=run_mix)

    # dda
    parser_dda = subparsers.add_parser('dda', help='Generate targeted DDA inclusion lists. Requires "assignedMixture" column from mix module')
    parser_dda.add_argument('sheet_path', help='Input sheet path (e.g. results/mix_out.csv)')
    parser_dda.add_argument('mode', choices=['pos', 'neg'], help='Ionisation mode (pos/neg)')
    parser_dda.add_argument('output_dir', help='Output directory for DDA list CSVs (e.g. results/dda/)')
    parser_dda.add_argument('max_mz', type=float, nargs='?', default=950, help='Upper m/z limit; only doubly charged adduct above this (default: 950)')
    parser_dda.add_argument('double_charge_limit', type=float, nargs='?', default=600, help='m/z above which doubly charged adduct is included (default: 600)')
    parser_dda.add_argument('baseline_rt', type=float, nargs='?', default=8, help='Baseline retention time in minutes (default: 8)')
    parser_dda.add_argument('baseline_rt_window', type=float, nargs='?', default=15, help='Retention time window in minutes (default: 15)')
    parser_dda.set_defaults(func=run_dda)

    ##### compiler stuff #####

    # rti
    parser_rti = subparsers.add_parser('rti', help='Create sheets for RTI webapp. Requires .mat files in /input folder')
    parser_rti.add_argument('mode', choices=['pos', 'neg'], help='Ionisation mode (pos/neg)')
    parser_rti.set_defaults(func=get_rti)

    # precomp
    parser_precomp = subparsers.add_parser(
        'precomp', help='Assemble all data into a pre-compilation sheet for final .txt file generation')
    parser_precomp.add_argument('mode', choices=['pos', 'neg'], help='Ionisation mode (pos/neg)')
    parser_precomp.add_argument('data_dir', help='Folder containing .mat file folders (whom should be named pos and/or neg respectively)')
    parser_precomp.add_argument('ref_path', help='Chemical metadata reference sheet path (e.g. data/pcq_out.csv)')
    parser_precomp.add_argument('tsv_path', help='Instrumental metadata file path (e.g. data/manual_metadata.tsv)')
    parser_precomp.add_argument('output_path', help='Output file path including extension (e.g. data/preComp_pos.csv)')
    parser_precomp.add_argument('-fa', action='store_true', help='Perform fragment annotation')
    parser_precomp.add_argument('-rti', default=False, help='Include RTI data (True/False)')
    parser_precomp.add_argument('-cf', default=None, help='Path to ClassyFire .csv file for manual classification data')
    parser_precomp.set_defaults(func=precomp)

    # compile
    parser_compile = subparsers.add_parser('compile', help='Create .txt & .msp files from a pre-compilation sheet')
    parser_compile.add_argument('precomp_sheet_path', help='Path to pre-assembly .csv sheet (e.g. data/preComp_pos.csv)')
    parser_compile.add_argument('output_dir', help='Folder where .txt files, postComp sheet, and .msp are written')
    parser_compile.add_argument('acc_long', help='Long accession prefix (format MSBNK-USERID-SHORTACC)')
    parser_compile.add_argument('acc_short', help='Short accession prefix (format SHORTACC)')
    parser_compile.add_argument('acc_start', type=int, help='Starting accession number for new entries')
    parser_compile.add_argument('mode', choices=['pos', 'neg'], help='Ionisation mode (pos/neg)')
    parser_compile.add_argument('-f', '--filter', default=False, help='Filter enantiomers and excluded compounds (default: True)')
    parser_compile.set_defaults(func=run_compile)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
