import subprocess
import sys
import os


if 'BBStudies/Executables/py_BB/bin' not in os.environ.get('PATH').split(':')[0]:
    raise Exception('Wrong Python Distribution, use BBStudies/Executables/py_BB')



# Running pymask
configuration = None
for mode in ['b1_with_bb','b4_from_b2_with_bb']:

    OUTPUTFOLDER = f'../Checks/{mode}_no_coupling'

    cwd = os.getcwd()
    os.chdir('./')
    template = open("000_mask_template_rich.py").read()
    template = template.replace("mode = configuration['mode']",f"mode = '{mode}'")
    template = template.replace("folder_name = './xsuite_lines'",f"folder_name = '{OUTPUTFOLDER}'")
    exec(template)
    opticsFile = configuration['optics_file'].split('/')[-1]
    os.chdir(cwd)

    # NOTE: Make sure you add: 
    # pm.install_lenses_in_sequence(mad_tra[]ck, bb_dfs['b2'], 'lhcb2')
    # at line 438 of '000_pymask_rich.py'{}
    # Saving sequences and BB dfs

    for seq in ['lhcb1','lhcb2']:
        mad_track.input(f'use, sequence={seq};')
        mad_track.twiss()
        mad_track.survey()
        
        twiss = mad_track.table.twiss.dframe()
        survey = mad_track.table.survey.dframe()

        twiss.to_pickle(f"{OUTPUTFOLDER}/twiss_opticsfile{opticsFile.split('.')[-1]}_{seq}.pkl")
        survey.to_pickle(f"{OUTPUTFOLDER}/survey_opticsfile{opticsFile.split('.')[-1]}_{seq}.pkl")
        
    bb_dfs['b1'].to_pickle(f"{OUTPUTFOLDER}/bb_dfs_opticsfile{opticsFile.split('.')[-1]}_lhcb1.pkl")
    bb_dfs['b2'].to_pickle(f"{OUTPUTFOLDER}/bb_dfs_opticsfile{opticsFile.split('.')[-1]}_lhcb2.pkl")