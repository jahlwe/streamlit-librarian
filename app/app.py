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
    module = st.sidebar.radio('Select:', ['pcq', 'mix', 'lib'])
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
    
def render_pcq():
    st.header('pcq module')
    st.caption('Batch query of chemical metadata via PubChem')
    st.markdown(
        """
        - Requires an `internalName` column in sheet with unique compound names (i.e., avoid duplicate names)
        - A column `queryName` can be provided if a name other than the internalName should be used for querying
        - No need for salt (i.e., **X HCl**) pre-cleaning – salts are recognized and parent compound (i.e., **X**) data retrieved instead
        """
    )

    # initialize session state variables
    session_keys = ['pcq_dictionary', 'pcq_output', 'pcq_success']
    for key in session_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    sheet = st.file_uploader(label='sheet', type=['csv', 'xlsx'], 
                             label_visibility='collapsed')

    preferred_key = 'internalName'

    # clears state when an uploaded file is removed
    if sheet is None:
        if any(st.session_state[key] is not None for key in session_keys):
            for key in session_keys:
                st.session_state[key] = None
            st.rerun()
    else:
        try:
            dictionary = gu.sheet_to_dict(sheet, preferred_key)
            st.info(f'{len(dictionary)} unique compound names recognized')
            st.session_state['pcq_dictionary'] = dictionary
        except Exception as e:
            st.error(f'Error reading file: {str(e)}')
            st.session_state['pcq_dictionary'] = None

    dictionary = st.session_state['pcq_dictionary']

    if dictionary:
        with st.form(key='pcq_form'):
            submitted = st.form_submit_button('run pcq')
            if submitted:
                progress_bar = st.progress(0)
                status_text = st.empty()
                def progress_callback(current, total, compound):
                    progress_bar.progress(current / total)
                    status_text.text(f'Processing {current}/{total} : {compound}')
                try:
                    dictionary = pu.pcQueries(dictionary, progress_callback=progress_callback)
                    output = io.BytesIO()
                    gu.dict_to_sheet(dictionary, buffer=output)
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
    preferred_key = 'internalName'
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
        preferred_key = 'internalName'
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
            mat_files_dict = {k: v for k, v in mat_files_dict.items() if (k.endswith('.mat') and f'/{mode}/' in k)}
            st.info(f'{len(mat_files_dict)} .mat ({mode}) files recognized')
            if len(mat_files_dict) > 0:
            # process with app-specific gather_matData
                mat_data = au.gather_matData_app(mat_files_dict, mode)
                st.session_state['mat_data'] = mat_data
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
    preferred_key = 'internalName'
    if precomp_sheet is not None:
        try:
            precomp_data = gu.sheet_to_dict(precomp_sheet, preferred_key)
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
def render_RTI():
    st.markdown(
        """
            - Generate input sheets for RTI WebApp
        """
    )
    # end

if __name__ == '__main__':
    app()
    
# streamlit run stapp.py 