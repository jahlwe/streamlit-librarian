# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 09:24:55 2025

@author: Jakob
"""

import streamlit as st
import utils.librarian as lib
import utils.appUtilities as au
import utils.genericUtilities as gu
import utils.pubchemUtilities as pu
import utils.mixtureUtilities as mu
import utils.compilerUtilities as cu
import argparse
import pandas as pd
import io
import matplotlib.pyplot as plt
import zipfile
import rarfile
import filetype
import csv
import plotly.graph_objects as go
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

# streamlit
def app():
    st.set_page_config(page_title='librarian', layout='wide')
    st.logo('static/logo.png')
    
    # --- MODULES ---
    st.sidebar.title('modules')
    module = st.sidebar.radio('Select:', ['pcq', 'mix', 'lib', 'utilities', 'readme'])
    # submodules for lib
    if module == 'lib':
        submodule = st.sidebar.radio('Sub-module:', ['pre-assembly', 'assembly'])
    else: 
        submodule = None

    # --- SESSION RESET LOGIC ---
    # track which module is active
    if 'current_module' not in st.session_state:
        st.session_state['current_module'] = module
    if 'current_submodule' not in st.session_state:
        st.session_state['current_submodule'] = submodule
        
    # track changes to either
    module_changed = module != st.session_state['current_module']
    submodule_changed = submodule != st.session_state['current_submodule']

    # clear session state except for current_(sub)module if module changes
    if module_changed or submodule_changed:
        keys_to_keep = ['current_module', 'current_submodule']
        for key in list(st.session_state.keys()):
            if key not in keys_to_keep:
                del st.session_state[key]
        st.session_state['current_module'] = module
        st.session_state['current_submodule'] = submodule
        st.rerun()
        
    # --- MODULE ROUTING ---
    if module == 'pcq':
        render_pcq()
    elif module == 'mix':
        render_mix()
    elif module == 'lib':
        if submodule == 'pre-assembly':
            render_lib_precomp()
        elif submodule == 'assembly':
            render_lib_compile()
    elif module == 'utilities':
        render_utilities()
    elif module == 'readme':
        render_readme()
    
def render_pcq():
    st.header('pcq module')
    st.caption('Batch query of chemical metadata via PubChem')
    
    st.markdown(
        """
        - Query chemical metadata via __name__, __SMILES__ or __CAS__
            - If a suitable entry is known a priori, __CID__ queries are also supported
        - Query inputs are supplied directly in-browser or by uploading a sheet (.csv, .xlsx)
        - No need for salt (i.e., **X HCl**) pre-cleaning – salts are recognized and parent compound (i.e., **X**) data retrieved instead
        - Use the `internal_id` field if a specific "internal" name should be used for a compound during library assembly
            - If left blank, `internal_id` defaults to PubChem entry title names
            - ___N.B.!___ To ensure proper downstream management of chemical metadata, `internal_id` values should be unique
        """
    )
    
    pcq_submodule = st.radio(
        'Select',
        ('in-browser', 'from sheet'),
        horizontal=True,
        label_visibility='collapsed'
    )
    
    # ---- FROM SHEET ----
    if pcq_submodule == 'from sheet':
        # initialize session state variables
        session_keys = ['pcq_input', 'pcq_output', 'pcq_success']
        for key in session_keys + ['pcq_dict', 'sheet_name']:
            if key not in st.session_state:
                if key == 'pcq_dict':
                    st.session_state[key] = {}
                else:
                    st.session_state[key] = None
        
        col1, col2 = st.columns([1,5], vertical_alignment='center')
        with col1:
            sheet_template = au.generate_pcq_template()
            st.download_button(
                label='Download template',
                data=sheet_template,
                file_name='pcq_template.csv',
                mime='text/comma-separated-values'
            )
        with col2:
            sheet = st.file_uploader(label='sheet', type=['csv', 'xlsx'], 
                                     label_visibility='collapsed')
    
        # clears state when an uploaded file is removed
        if sheet is None:
            if any(st.session_state[key] is not None for key in session_keys):
                for key in session_keys:
                    st.session_state[key] = None
                st.rerun()
        else:
            try:
                st.session_state['pcq_input'] = gu.sheet_to_idx_dict(sheet)
                # reset logic if a new sheet is uploaded (without clearing the uploader first)
                current_name = sheet.name if hasattr(sheet, 'name') else str(sheet)
                if current_name != st.session_state['sheet_name']:
                    for key in 'pcq_output', 'pcq_success':
                        st.session_state[key] = None
                    st.session_state['sheet_name'] = current_name
                    if any('queried_as' in data for data in st.session_state['pcq_input'].values()):
                        # if we jump from one module to the next, we will clear the
                        # stored pcq_dict state --- but we need to retain it to
                        # make the re-query work nicely. solution --- if we input
                        # a pcq output sheet as an input, just rebuild the pcq_dict
                        # accordingly. should work?
                        # we probably have to specify not to include name_q etc.
                        EXCLUDE_KEYS = ['name_q', 'cas_q', 'smiles_q', 'cid_q']
                        st.session_state['pcq_dict'] = {
                            idx: {k: v for k, v in data.items() if k not in EXCLUDE_KEYS} for idx, data in st.session_state['pcq_input'].items()
                        }
            except Exception as e:
                st.error(f'Error reading file: {str(e)}')
    
        if st.session_state['pcq_input']:
            # prepare query dict based on what type of sheet has been uploaded --- fresh template or pcq output
            if any('queried_as' in data for data in st.session_state['pcq_input'].values()):
                # we should also support changing 'queried_as' information for re-queries
                # which is a little... involved. made a helper in au.
                query_dict = au.query_dict_from_pcq_input(st.session_state['pcq_input'])
                st.info(f'{len(query_dict)} compound(s) recognized (re-query)')
                # old version that only supports manually entered CIDs and not changes to 'queried_as'
                #query_dict = {
                #    idx: {'name_q': None, 'cas_q': None, 'smiles_q': None, 'cid_q': data.get('pubchemCID')} for idx, data in st.session_state['pcq_input'].items() if not data.get('queried_at')}
            else:
                query_dict = st.session_state['pcq_input']
                st.info(f'{len(query_dict)} compound(s) recognized')
                
            with st.form(key='pcq_form'):
                submitted = st.form_submit_button('run pcq')
                if submitted:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    def progress_callback(current, total, compound):
                        progress_bar.progress(current / total)
                        status_text.text(f'Processing {current}/{total} : {compound}')
                    try:
                        pcq_result = pu.pcQueries(query_dict, progress_callback=progress_callback)
                        st.session_state['pcq_dict'].update(pcq_result)
                        output = io.BytesIO()
                        gu.idx_dict_to_sheet(st.session_state['pcq_dict'], buffer=output)
                        output.seek(0)
                        st.session_state['pcq_output'] = output
                        st.session_state['pcq_success'] = True
                    except Exception as e:
                        st.error(f'Error: {str(e)}')
                        st.session_state['pcq_success'] = False
    
        # Display download button if processing was successful
        if st.session_state.get('pcq_success'):
            st.success('pcq complete!')
            if st.session_state['pcq_output']:
                st.download_button('download results', st.session_state['pcq_output'].getvalue(),
                                   file_name='pcq_out.csv')
                
    # ---- IN-BROWSER ---
    elif pcq_submodule == 'in-browser':
        if 'in_browser_df' not in st.session_state or not isinstance(st.session_state['in_browser_df'], pd.DataFrame):
            st.session_state['in_browser_df'] = pd.DataFrame(
                [{'internal_id': None, 'name_q': None, 'cas_q': None, 'smiles_q': None, 'cid_q': None, 'pcq_success': False}]
            )
        if 'pcq_dict' not in st.session_state or not isinstance(st.session_state['pcq_dict'], dict):
            st.session_state['pcq_dict'] = {}
        # and other stuff
        session_keys = ['pcq_output', 'pcq_success']
        for key in session_keys:
            if key not in st.session_state:
                st.session_state[key] = None
            
        df = st.data_editor(
            st.session_state['in_browser_df'],
            num_rows='dynamic',
            key='ibdf'
        )
        
        st.session_state['in_browser_df_current'] = df
        has_data = any(row for row in df.to_dict(orient='records') if any(row.get(field) for field in ['name_q', 'cas_q', 'smiles_q','cid_q']))
        # use this as a counter
        entries_with_content = sum(1 for row in df.to_dict(orient='records') if any(row.get(field) for field in ['name_q', 'cas_q', 'smiles_q','cid_q']))

        if has_data:
            st.info(f'{entries_with_content} compounds recognized')
            with st.form(key='pcq_form'):
                submitted = st.form_submit_button('run_pcq')
                if submitted:
                    all_rows = df.to_dict(orient='records')
                    query_rows = [
                        (idx, row) for idx, row in enumerate(all_rows) if any([row.get('name_q'), row.get('cas_q'), row.get('smiles_q'), row.get('cid_q')]) and not row.get('pcq_success')
                    ]
                    if query_rows:
                        query_dict = {
                            idx: {**row} for idx, row in query_rows
                        }
                        print(query_dict)
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        def progress_callback(current, total, compound):
                            progress_bar.progress(current / total)
                            status_text.text(f'Processing {current}/{total} : {compound}')
                        try:
                            dictionary = pu.pcQueries(query_dict, progress_callback=progress_callback)
                            st.session_state['pcq_dict'].update(dictionary)
                            output = io.BytesIO()
                            gu.idx_dict_to_sheet(st.session_state['pcq_dict'], buffer=output)
                            output.seek(0)
                            st.session_state['pcq_output'] = output
                            st.session_state['pcq_success'] = True
                            
                            # update rows with query results
                            updated_rows = []                            
                            for idx, row in enumerate(all_rows):
                                entry = st.session_state['pcq_dict'].get(idx)
                                if entry:
                                    if 'internal_id' in entry:
                                        row['internal_id'] = entry.get('internal_id', row.get('internal_id'))
                                    cid = entry.get('pubchemCID')
                                    row['pcq_success'] = isinstance(cid, int)
                                else:
                                    row['pcq_success'] = False
                                updated_rows.append(row)
                            
                            # create updated df and feed it into the data_editor frame
                            updated_df = pd.DataFrame(updated_rows)            
                            st.session_state['in_browser_df'] = updated_df
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f'Error: {str(e)}')
                            st.session_state['pcq_success'] = False
                            
        # display dl button if query was successful
        if st.session_state.get('pcq_success'):
            st.success('pcq complete!')            
            if st.session_state['pcq_output']:
                st.download_button('download results', st.session_state['pcq_output'].getvalue(),
                                   file_name='pcq_out.csv')
    # end

def render_mix():
    st.header('mix module')
    st.caption('Prepare compound mixtures for HRMS acquisition')
    st.markdown(
        """
        - Requires an output sheet from the pcq module
        - In fields below, provide a desired `Number of mixes` and `Minimum mass distance`
        - logK$_{ow}$ and expected ___m/z___ in positive mode (in most cases, **[M+H]+**) are calculated and used for algorithmic sorting of compounds
        """
    )
    
    sheet = st.file_uploader(label='pcq sheet', type=['csv', 'xlsx'],
                             label_visibility='collapsed')
    preferred_key = 'internal_id'
    dictionary = None
    
    # need to initialize "session state variables" to keep things
    # intact when we interact with the buttons later on
    session_keys = ['dictionary', 'working_sheet', 'output', 'stats_df', 'visual_buf', 'mix_success']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # clear state when file is removed
    if sheet is None:
        if any(st.session_state[key] is not None for key in session_keys):
            for key in session_keys:
                st.session_state[key] = None
            st.rerun()
    else:
        try:
            dictionary = gu.sheet_to_dict(sheet, preferred_key)
            dictionary = mu.generate_adducts(dictionary)
            dictionary = mu.expected_mz(dictionary)
            dictionary = mu.calculate_xlogp(dictionary)
            st.info('pcq sheet loaded --- adducts & xlogp calculated')
            st.session_state['dictionary'] = dictionary
        except Exception as e:
            st.error(f'Error reading file: {str(e)}')
            st.session_state['dictionary'] = None
       
    if dictionary:
        with st.form('mix_form'):
            col1, col2 = st.columns(2)
            with col1:
                num_mixes = st.number_input(
                    'Number of mixes',
                    min_value=2,
                    max_value=1000,
                    value=2,
                    step=1
                )
            with col2:
                min_mass_dist = st.number_input(
                    'Minimum mass distance (Da)',
                    min_value=0.001,
                    max_value=1.0,
                    value=0.01,
                    step=0.001,
                    format="%.3f"
                )
            submitted = st.form_submit_button('run mix')

            if submitted:
                try:
                    normalized_data, working_sheet, index = mu.prepare_data(dictionary)
                    dictionary, working_sheet = mu.distribute_compounds(
                        dictionary, working_sheet, normalized_data,
                        num_mixes, min_mass_dist, enforce=False,
                        auto_assign=True, index=index
                    )
                    output = io.BytesIO()
                    gu.dict_to_sheet(dictionary, buffer=output)
                    output.seek(0)
                    st.session_state['output'] = output
                    st.session_state['working_sheet'] = working_sheet
                    
                    # do mix stats
                    stats_df = mu.mixture_stats(working_sheet, streamlit=True)
                    st.session_state['stats_df'] = stats_df
                    
                    # do vis 
                    buf = io.BytesIO()
                    mu.visual_summary(dictionary)
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    st.session_state['visual_buf'] = buf
                    plt.close()
                    
                    st.session_state['mix_success'] = True
                except Exception as e:
                    st.error(f'Error: {str(e)}')
                    st.session_state['mix_success'] = False
    
    if st.session_state.get('mix_success'):
        st.success('mix complete!')
        if st.session_state['output']:
            st.download_button('download results', st.session_state['output'].getvalue(),
                               file_name='mix_out.csv')
        if st.session_state['visual_buf']:
            st.image(st.session_state['visual_buf'], caption='')
            st.download_button(
                label='download visual summary (.png)',
                data=st.session_state['visual_buf'].getvalue(),
                file_name='mix_out.png',
                mime='image/png'
            )
        if st.session_state['stats_df'] is not None:
            st.dataframe(st.session_state['stats_df'])
            stats_csv = st.session_state['stats_df'].to_csv().encode('utf-8')
            st.download_button(
                label='download mixture report (.csv)',
                data=stats_csv,
                file_name='mixture_report.csv',
                mime='text/csv'
            )
    # end

def render_lib_precomp():
    st.header('lib module; pre-assembly')
    st.caption('Two-step assembly of MassBank-format spectral library')
    st.markdown(
        """
        - **Pre-assembly**
            - Collates metadata (pcq module-format), experimental settings (manually supplied) and experimental data (.mat)
            - Outputs an editable sheet for use in the assembly sub-module
        """
    )
    # init keys for what we need
    input_keys = ['mode', 'pcq_data', 'metadata_tsv', 'mat_data']
    optional_input_keys = ['rti_data', 'cf_data']
    for key in input_keys + optional_input_keys + ['output', 'precomp_ready', 'prev_inputs']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # --- SELECT MODE ---
    mode = st.radio(
        'ionization mode',
        options=['pos', 'neg'],
        index=0,  # default selection
        horizontal=True  # this looks nicer here
    )
    st.session_state['mode'] = mode
    
    # --- UPLOADS ---
    # columns for horizontal layout
    req_col1, req_col2, req_col3 = st.columns(3, vertical_alignment='top')
    
    with req_col1:
        pcq_sheet = st.file_uploader('pcq sheet (.csv)', type=['csv', 'xlsx'])
        preferred_key = 'internal_id'
        if pcq_sheet is not None:
            try:
                pcq_data = gu.sheet_to_dict(pcq_sheet, preferred_key)
                st.session_state['pcq_data'] = pcq_data
            except Exception as e:
                st.error(f'Error reading file: {str(e)}')
                st.session_state['pcq_data'] = None
        else:
            st.session_state['pcq_data'] = None
            
    with req_col2:
        metadata_tsv = st.file_uploader('experimental metadata (.tsv)', type=['tsv'])
        if metadata_tsv is not None:
            st.session_state['metadata_tsv'] = metadata_tsv
        else:
            st.session_state['metadata_tsv'] = None
        
        # also - a template that can be downloaded!
        tsv_content = au.generate_metadata_template()
        st.download_button(
            label='Download template',
            data=tsv_content,
            file_name='metadata_template.tsv',
            mime='text/tab-separated-values'
        )
    
    # .mat files in .zip format (.rar did not cooperate.)
    with req_col3:
        mat_archive = st.file_uploader('.mat files (in .zip)', type=['zip'])
        if mat_archive is not None:
            archive_type = filetype.guess(mat_archive.getvalue()).extension
            mat_files_dict = au.read_archive(mat_archive, archive_type)
            mat_files_dict = {k: v for k, v in mat_files_dict.items() if (k.endswith('.mat') and f'{mode}/' in k)}
            st.info(f'{len(mat_files_dict)} .mat ({mode}) files recognized')
            if len(mat_files_dict) > 0:
            # process with app-specific gather_matData
                mat_data = au.gather_matData_app(mat_files_dict, mode)
                st.session_state['mat_data'] = mat_data
                print(mat_data)
            else:
                st.session_state['mat_data'] = None
        else:
            st.session_state['mat_data'] = None
        
    # RTI & ClassyFire
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    
    with opt_col1:
        rti_box = st.checkbox('RTI')
        if rti_box:
            rti_archive = st.file_uploader('RTI sheets (.csv, in .zip)', type=['zip'])
            if rti_archive is not None:
                archive_type = filetype.guess(rti_archive.getvalue()).extension
                rti_files_dict = au.read_archive_RTI(rti_archive, archive_type)
                rti_files_dict = {k: v for k, v in rti_files_dict.items() if (k.endswith('.csv') and f'{mode}/' in k)}
                st.info(f'{len(rti_files_dict)} RTI sheets ({mode}) recognized')
                if len(rti_files_dict) > 0:
                    # process.
                    rti_data = au.gather_RTIData_app(rti_files_dict)
                    st.session_state['rti_data'] = rti_data
            else:
                st.session_state['rti_data'] = None
        else:
            st.session_state['rti_data'] = None
            
    with opt_col2:
        cf_box = st.checkbox('ClassyFire')
        if cf_box:
            cf_sheet = st.file_uploader('ClassyFire sheet (.csv)', type=['csv'])
            if cf_sheet is not None:
                cf_data = gu.sheet_to_dict(cf_sheet, 'InChIKey')
                st.session_state['cf_data'] = cf_data
            else:
                st.session_state['cf_data'] = None
        else:
            st.session_state['cf_data'] = None
            
    # --- RESET OUTPUT/STATE IF INPUTS CHANGED ---
    current_inputs = { # track these...
        'mode': st.session_state['mode'],
        'pcq_data': pcq_sheet,
        'metadata_tsv': metadata_tsv,
        'mat_data': mat_archive
    }
    # if any uploaded file is removed --- we reset the whole thing.
    if st.session_state['prev_inputs'] is not None:
        for k in current_inputs:
            if current_inputs[k] != st.session_state['prev_inputs'].get(k):
                st.session_state['output'] = None
                st.session_state['precomp_ready'] = False
                break
    st.session_state['prev_inputs'] = current_inputs
    
    # --- PRE-ASSEMBLY BLOCK ---
    # ready check. need to adapt if we add optional data like RTI, CF.
    all_ready = all(st.session_state[k] is not None for k in input_keys)
    if all_ready:
        st.success('All required data loaded. Ready to proceed!')
        if st.button('Perform pre-assembly'):
            # get a progress bar here, because annnotation can take a long time
            progress_bar = st.progress(0)
            status_text = st.empty()
            def progress_callback(current, total, compound):
                progress_bar.progress(current / total)
                status_text.text(f'Annotating MS2 {current}/{total} : {compound}')
            try:
                precomp_dict = au.preCompile_app(
                    mode=st.session_state['mode'],
                    pcq_data=st.session_state['pcq_data'],
                    metadata_tsv=st.session_state['metadata_tsv'],
                    mat_data=st.session_state['mat_data'],
                    # optional RTI and CF
                    rti_data=st.session_state.get('rti_data'),
                    cf_data=st.session_state.get('cf_data'),
                    annotate_fragments=True,
                    progress_callback=progress_callback
                )
                output_buffer = io.StringIO()
                gu.dict_to_sheet(precomp_dict, file_name=None, fmat='.csv', buffer=output_buffer)
                output_buffer.seek(0)
                st.session_state['output'] = output_buffer
                st.session_state['precomp_ready'] = True
            except Exception as e:
                st.session_state['output'] = None
                st.session_state['precomp_ready'] = False
                st.error(f"Error during pre-assembly: {e}")

    # show download and success if ready, regardless of rerun
    if st.session_state.get('precomp_ready', False) and st.session_state['output']:
        st.success('Pre-assembly complete!')
        st.download_button(
            'Download pre-assembly sheet', 
            st.session_state['output'].getvalue(),
            file_name=f'preAssembly_{st.session_state["mode"]}.csv',
            mime='text/csv'
        )
    # end
    
def render_lib_compile():
    st.header('lib module; assembly')
    st.caption('Two-step assembly of MassBank-format spectral library')
    st.markdown(
        """
        - **Assembly**
            - Generates individual .txt files (MassBank-format) and an .msp library file from the pre-assembly sheet
        """
    )
    input_keys = ['mode', 'precomp_data', 'acc_start', 'acc_full', 'acc_short']
    for key in input_keys + ['comp_data', 'output_sheet']:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # --- SELECT MODE ---
    mode = st.radio(
        'ionization mode',
        options=['pos', 'neg'],
        index=0,  # default selection
        horizontal=True  # this looks nicer here
    )
    st.session_state['mode'] = mode
    
    # --- INPUT ---
    precomp_sheet = st.file_uploader('pre-assembly sheet', type=['csv', 'xlsx'])
    preferred_key = 'internal_id'
    if precomp_sheet is not None:
        try:
            precomp_data = gu.sheet_to_dict(precomp_sheet, preferred_key)
            print(precomp_data)
            # filter here, by default --- for now.
            precomp_data = au.filter_preComp_app(precomp_data, mode) 
            st.session_state['precomp_data'] = precomp_data
        except Exception as e:
            st.error(f'Error reading file: {str(e)}')
            st.session_state['precomp_data'] = None
    else:
        st.session_state['precomp_data'] = None
        
    col1, col2, col3 = st.columns(3)
    with col1:
        acc_start = st.number_input("start accession number", min_value=1, value=1, step=1)
        st.session_state['acc_start'] = acc_start
    with col2:
        acc_full = st.text_input("full accession prefix", value="MSBNK-UNI_LAB-SHORTACC")
        st.session_state['acc_full'] = acc_full
    with col3:
        acc_short = st.text_input("short accession prefix", value="SHORTACC")
        st.session_state['acc_short'] = acc_short
    
    # --- ASSEMBLY BLOCK ---
    all_ready = all(st.session_state[k] is not None for k in input_keys)
    if all_ready:
        st.success('All required data loaded. Ready to proceed!')
        if st.button('Perform assembly'):
            try:
                # MAYBE think about how we do this ---
                # right now we store both precomp and comp in memory
                comp_data = au.compileLib_app(
                    st.session_state['precomp_data'],
                    st.session_state['acc_start'],
                    st.session_state['acc_full'],
                    st.session_state['acc_short'],
                    st.session_state['mode']
                )
                st.session_state['comp_data'] = comp_data
                st.success('Assembly complete!')
            except Exception as e:
                st.error(f"Error during assembly: {e}")
                st.session_state['comp_data'] = None
        
    # --- INTERACTIVE ---
    if st.session_state.get('comp_data', None):
        comp_data = st.session_state['comp_data']
        
        # create zip --- function creates txts, csv, msp
        zip_buffer = au.create_compZip(
            st.session_state['comp_data'],
            st.session_state['mode']
        )
        st.download_button(
            'Download assembly (.zip)',
            zip_buffer,
            file_name=f'full_export_{st.session_state["mode"]}.zip',
            mime='application/zip'
        )
        
        # --- PLOTTING ---
        compound_options = []
        compound_keys = []
        for compound, data in st.session_state['comp_data'].items():
            display_name = f"{data.get('acc', 'NOACC')} - {compound}"
            compound_options.append(display_name)
            compound_keys.append(compound)
    
        # Let the user select a compound
        selected_idx = st.selectbox(
            "View MS2",
            options=range(len(compound_options)),
            format_func=lambda i: compound_options[i]
        )
        selected_compound = compound_keys[selected_idx]
        selected_data = comp_data[selected_compound]
        # PLOT.
        col1, col2 = st.columns([0.75, 0.25], vertical_alignment='center')
        with col1:
            fig = au.plot_MS2(
                selected_data,
                selected_data['ms2_display'],
                selected_data['precursor_mz'],
                title=f'{selected_compound} {selected_data["ion_type"]}')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mol = Chem.MolFromSmiles(selected_data['smiles'])
            if mol is not None:
                img = Draw.MolToImage(mol)
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                st.image(img_bytes)
            else:
                st.write(f'Invalid SMILES: {selected_data["smiles"]}')

    # end
                
# --- UTILITY RENDERS ---
def render_utilities():
    st.header('utilities')
    # initialize session variables
    rti_keys = ['rti_pcq_data', 'rti_exp_data']
    for key in rti_keys:
        if key not in st.session_state:
            st.session_state[key] = None
    
    # first, RTI
    st.markdown(
        """
        **RTI web app sheet generator**  
        Upload a pcq sheet and a .zip archive of experimental data
        to generate .csv sheets for input into the RTI web app
        """
    )
    
    rti_col1, rti_col2, rti_col3 = st.columns([1,2,2], vertical_alignment='top')
    
    with rti_col1:
        mode = st.radio(
            'ionization mode',
            options=['pos', 'neg'],
            index=0,  # default selection
            horizontal=False
        )
        st.session_state['mode'] = mode
        
    with rti_col2:
        pcq_sheet = st.file_uploader(label='pcq sheet', type=['csv', 'xlsx'],
                                 label_visibility='collapsed')
        if pcq_sheet is not None:
            try:
                pcq_data = gu.sheet_to_dict(pcq_sheet)
                st.session_state['rti_pcq_data'] = pcq_data
            except:
                pass
            
    with rti_col3:
        exp_archive = st.file_uploader('experimental data (.mat, in .zip)', type=['zip'],
                                       label_visibility='collapsed')
        if exp_archive is not None:
            archive_type = filetype.guess(exp_archive.getvalue()).extension
            exp_files_dict = au.read_archive(exp_archive, archive_type)
            exp_files_dict = {k: v for k, v in exp_files_dict.items() if (k.endswith('.mat') and f'{mode}/' in k)}
            st.info(f'{len(exp_files_dict)} .mat ({mode}) files recognized')
            if len(exp_files_dict) > 0:
            # process with app-specific gather_matData
                exp_data = au.gather_matData_app(exp_files_dict, mode)
                st.session_state['rti_exp_data'] = exp_data
                print(exp_data)
            else:
                st.session_state['rti_exp_data'] = None
        else:
            st.session_state['rti_exp_data'] = None
            
    # ready check for RTI sheet generation
    rti_ready = all(st.session_state[k] is not None for k in rti_keys)
    if rti_ready:
        if st.button('generate RTI sheet(s)'):
            try:
                rti_dict = cu.create_compilation_dictionary(st.session_state['rti_exp_data'])
                rti_dict = cu.add_chemical_metadata(rti_dict, st.session_state['rti_pcq_data'])
                csv_sheet_dict = au.generate_rtiSheets_app(rti_dict)
                if len(csv_sheet_dict) > 0:
                    # get csv files into an archive and make it downloadable
                    zip_buffer = io.BytesIO() # use this
                    with zipfile.ZipFile(zip_buffer, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                        for filename, csv_content in csv_sheet_dict.items():
                            fname = f'{filename}.csv' if not filename.endswith('.csv') else filename
                            zf.writestr(fname, csv_content)
                    # return buffer cursor after wrting
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label='download RTI sheet(s) (.zip)',
                        data=zip_buffer,
                        file_name='rti_sheets.zip',
                        mime='application/zip')
                else:
                    st.warning('error --- no .csv sheets generated')
            except Exception as e:
                st.error(f"Error during assembly: {e}")
                st.session_state['rti_output'] = None
                
    # ----
    
    # end

def render_readme():
    # "header"
    st.image(
        'static/logo.png',
        width=360
    )
    st.caption('A web application for high-resolution tandem mass spectral library assembly')
    
    # main descriptive stuff    
    st.markdown(
        """
        Welcome to the Librarian web application!
        \n
        Librarian provides in-browser data management utilities for assembly 
        of high-resolution tandem mass spectral records in the MassBank format.  
        The three modules ___pcq___, ___mix___ and ___lib___ support batch query of chemical metadata,
        distribution of compounds to mixtures for HRMS acquisition and (following data acquisition) library assembly, respectively.
        \n
        Complete source code for the web application is available via https://github.com/jahlwe/streamlit-librarian  
        A command-line version of Librarian is available via https://github.com/jahlwe/librarian
        \n
        For first-time users, a simple worked example of library assembly using Librarian is given below.
        
        """
    )
    # end

if __name__ == '__main__':
    app()
    
# streamlit run stapp.py 