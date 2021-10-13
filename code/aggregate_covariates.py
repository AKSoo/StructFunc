#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from functools import reduce

from utils import abcd

NA_VALUES = [777, 999]


# INPUTS
dem_long = pd.read_csv(abcd.PATH / 'abcd_lpds01.txt', sep='\t',
                       skiprows=[1], index_col=abcd.INDEX)
dem_site = pd.read_csv(abcd.PATH / 'abcd_lt01.txt', sep='\t',
                       skiprows=[1], index_col=abcd.INDEX)
dem_base = pd.read_csv(abcd.PATH / 'pdem02.txt', sep='\t',
                       skiprows=[1], index_col=abcd.INDEX)
dem_acs = pd.read_csv(abcd.PATH / 'acspsw03.txt', sep='\t',
                      skiprows=[1], index_col=abcd.INDEX)

# start aggregate
covariates = dem_long[['interview_date', 'interview_age']].copy()

covariates['sex'] = dem_long['sex'].astype('category')
covariates['site_id'] = dem_site['site_id_l']

def combine_dems(dems, na_values=None):
    combined = reduce(
        lambda l,r: l.combine_first(r),
        [dem.replace(na_values, np.nan) for dem in dems]
    )
    return combined

# income
comb_income = combine_dems([dem_base['demo_comb_income_v2'],
                            dem_long['demo_comb_income_v2_l']],
                           na_values=NA_VALUES)

def map_income_3level(x):
    if x in np.arange(1,7):
        return '[<50K]'
    elif x in [7,8]:
        return '[>=50K & <100K]'
    elif x in [9,10]:
        return '[>=100K]'
    else:
        return np.nan

covariates['comb_income.3level'] = comb_income.apply(map_income_3level).astype('category')

# education
prnt_ed = combine_dems([dem_base['demo_prnt_ed_v2'],
                        dem_long['demo_prnt_ed_v2_l'],
                        dem_long['demo_prnt_ed_v2_2yr_l']],
                       na_values=NA_VALUES)
prtnr_ed = combine_dems([dem_base['demo_prtnr_ed_v2'],
                         dem_long['demo_prtnr_ed_v2_l'],
                         dem_long['demo_prtnr_ed_v2_2yr_l']],
                        na_values=NA_VALUES)
highest_ed = prnt_ed.fillna(-1).combine(prtnr_ed.fillna(-1), max)

def map_ed_5level(x):
    if x in np.arange(0,13):
        return '< HS Diploma'
    elif x in [13,14]:
        return 'HS Diploma/GED'
    elif x in [15,16,17,22,23]:
        return 'Some College'
    elif x == 18:
        return 'Bachelor'
    elif x in [19,20,21]:
        return 'Post Graduate Degree'
    else:
        return np.nan

covariates['highest_ed.5level'] = highest_ed.apply(map_ed_5level).astype('category')

# marital
marital = combine_dems([dem_base['demo_prnt_marital_v2'],
                        dem_long['demo_prnt_marital_v2_l']],
                       na_values=NA_VALUES)

def map_marital(x):
    if x == 1:
        return 'Yes'
    elif x in np.arange(2,7):
        return 'No'
    else:
        return np.nan

covariates['married'] = marital.apply(map_marital).astype('category')

# race
white = dem_base['demo_race_a_p___10']
black = dem_base['demo_race_a_p___11']
aian = dem_base['demo_race_a_p___12'] | dem_base['demo_race_a_p___13']
nhpi = reduce(lambda l,r: l | r,
              [dem_base['demo_race_a_p___' + str(n)] for n in range(14,18)])
asian = reduce(lambda l,r: l | r,
               [dem_base['demo_race_a_p___' + str(n)] for n in range(18,25)])
other = dem_base['demo_race_a_p___25']
mixed = white + black + asian + aian + nhpi + other > 1

race = pd.Series(np.nan, index=mixed.index, name='race.6level')
race[white == 1] = 'White'
race[black == 1] = 'Black'
race[asian == 1] = 'Asian'
race[(aian == 1) | (nhpi == 1)] = 'AIAN/NHPI'
race[other == 1] = 'Other'
race[mixed] = 'Mixed'

covariates = covariates.join(race.droplevel('eventname').astype('category'))

covariates = covariates.join(dem_base['demo_ethn_v2'].map({1:'Yes', 2:'No'}).rename('hisp')
                             .droplevel('eventname').astype('category'))

# family ID
covariates = covariates.join(
    dem_acs.xs('baseline_year_1_arm_1', level='eventname')['rel_family_id'].astype(int)
)

# OUTPUTS
covariates.to_csv(abcd.OUT_PATH / 'abcd_covariates.csv')
