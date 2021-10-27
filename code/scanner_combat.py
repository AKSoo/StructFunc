#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import pandas as pd

from utils import abcd
from utils.combat import combat


covariates = abcd.load_covariates(covars=['interview_age', 'sex', 'hisp'],
                                  simple_race=True)

fcon, fc_extra = abcd.load_fcon(dropna=True)
scon, sc_extra = abcd.load_scon(dropna=True)

def save_scan_combat(filename, con, extra, scan_confounds):
    batch = extra[scan_confounds].apply(tuple, axis=1)
    covars = extra[['meanmotion']].join(covariates)
    adjusted = combat(con, batch, covars, n_procs=4)
    adjusted.to_csv(Path('outputs') / filename)

save_scan_combat('fcon-device.csv', fcon, fc_extra,
                 ['mri_info_deviceserialnumber'])
save_scan_combat('fcon-device-software.csv', fcon, fc_extra,
                 ['mri_info_deviceserialnumber', 'mri_info_softwareversion'])
save_scan_combat('scon-device.csv', scon, sc_extra,
                 ['mri_info_deviceserialnumber'])
save_scan_combat('scon-device-software.csv', scon, sc_extra,
                 ['mri_info_deviceserialnumber', 'mri_info_softwareversion'])
