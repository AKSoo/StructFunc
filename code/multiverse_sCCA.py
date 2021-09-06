#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

import pandas as pd
idx = pd.IndexSlice
SEED = 69

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from statsmodels.regression.linear_model import OLS
from gemmr.estimators import SparseCCA

import joblib
from itertools import product

from utils import abcd, combat, residual


out_path = abcd.OUT_PATH / 'multiverse'

covariates = abcd.load_covariates(covars=['interview_age', 'sex', 'hisp'],
                                  simple_race=True)

shell = ['base', 'full']
covar = ['', 'm', 'r', 'mr']
scan = ['dev', 'devsoft']
metric = ['fa', 'famd']
out = ['all', 'noout']

verses = ['-'.join(c) for c in product(shell, covar, scan, metric, out)]

def verse_inputs(verse, test=False, random_state=SEED):
    """
    Parse experiment verse name and get analysis inputs. 

    Params:
        verse: str experiment settings
        test: or train?

    Returns:
        dFC, SC, FC, FCSC: ndarrays
    """
    # parse params
    params = verse.split('-')
    full = params[0] == 'full'
    sc_motion = 'm' in params[1]
    covars = ['interview_age', 'sex'] + ['race', 'hisp'] * ('r' in params[1])
    scan_confounds = [abcd.SCAN_INFO[2]] + [abcd.SCAN_INFO[3]] * ('soft' in params[2])
    metrics = ['fa'] + ['md'] * ('md' in params[3])
    noout = params[4] == 'noout'

    # ComBat
    fcon, fc_extra = abcd.load_fcon()
    scon, sc_extra = abcd.load_scon(metrics=metrics, full=full)
    fc_covariates = fc_extra[['meanmotion']].join(covariates[covars])
    sc_covariates = sc_extra[['meanmotion'] * sc_motion].join(covariates[covars])

    fcon = combat.combat(fcon, fc_extra[scan_confounds].apply(tuple, axis=1), fc_covariates)
    scon = combat.combat(scon, sc_extra[scan_confounds].apply(tuple, axis=1), sc_covariates)

    # sample 1 per family
    fcon = abcd.filter_siblings(fcon.loc[fcon.index.intersection(scon.index)],
                                random_state=random_state)
    scon = scon.loc[fcon.index]
    subs = fcon.index.get_level_values(0).unique()

    # train-test split
    subs_train, subs_test = train_test_split(subs, test_size=.2,
                                             random_state=random_state)
    if test:
        subs = subs_test
    else:
        subs = subs_train
    fc, sc = fcon.loc[subs], scon.loc[subs]

    # outlier
    if noout:
        fc_med, fc_mad = fc.median(axis=0), stats.median_abs_deviation(fc)
        sc_med, sc_mad = sc.median(axis=0), stats.median_abs_deviation(sc)
        fc_noout = fc.mask((fc - fc_med).abs().divide(fc_mad) > 3)
        sc_noout = sc.mask((sc - sc_med).abs().divide(sc_mad) > 3)

        mask_noout = ((fc_noout.notna().sum(axis=1) > fc_noout.shape[1] * .8)
                      & (sc_noout.notna().sum(axis=1) > sc_noout.shape[1] * .8))
        subs_noout = mask_noout.groupby(level=0).filter(lambda g: g.all()).index
        fc, sc = fc_noout.loc[subs_noout], sc_noout.loc[subs_noout]

    # regress out confounds
    fc = residual.residualize(fc, OLS, fc_covariates.drop(columns='interview_age'))
    sc = residual.residualize(sc, OLS, sc_covariates.drop(columns='interview_age'))

    # input features
    age = fc_covariates.loc[fc.index, 'interview_age'] / 12
    age_diff = age.groupby(level=0).diff().dropna().droplevel(1)
    dFC = (fc.groupby(level=0).diff().xs(abcd.EVENTS[1], level=1)
           .divide(age_diff, axis=0))
    SC = sc.loc[idx[dFC.index, abcd.EVENTS[0]], :].droplevel(1)
    FC = fc.loc[idx[dFC.index, abcd.EVENTS[0]], :].droplevel(1)
    FCSC = FC.join(SC)

    # if no outliers, impute
    if noout:
        imputer = KNNImputer(n_neighbors=3)
        dFC = imputer.fit_transform(dFC)
        SC = imputer.fit_transform(SC)
        FC = imputer.fit_transform(FC)
        FCSC = imputer.fit_transform(FCSC)
    else:
        dFC = dFC.to_numpy()
        SC = SC.to_numpy()
        FC = FC.to_numpy()
        FCSC = FCSC.to_numpy()

    return dFC, SC, FC, FCSC

def scca_analysis(verse):
    (out_path / verse).mkdir(parents=True, exist_ok=True)

    dFC, SC, FC, FCSC = verse_inputs(verse)

    # sCCA
    scca_SC = SparseCCA(n_components=10, scale=True)
    scca_SC.fit(SC, dFC)
    joblib.dump(scca_SC, out_path / verse / 'scca_SC.joblib.gz')

    scca_FC = SparseCCA(n_components=10, scale=True)
    scca_FC.fit(FC, dFC)
    joblib.dump(scca_FC, out_path / verse / 'scca_FC.joblib.gz')

    scca_FCSC = SparseCCA(n_components=10, scale=True)
    scca_FCSC.fit(FCSC, dFC)
    joblib.dump(scca_FCSC, out_path / verse / 'scca_FCSC.joblib.gz')

    return scca_SC.corrs_.max(), scca_FC.corrs_.max(), scca_FCSC.corrs_.max()


if __name__ == '__main__':
    summary = joblib.Parallel(n_jobs=8)(joblib.delayed(scca_analysis)(v)
                                        for v in verses)
    summary = pd.DataFrame(summary, index=verses, columns=['SC', 'FC', 'FCSC'])
    summary.to_csv(out_path / 'summary.csv')
