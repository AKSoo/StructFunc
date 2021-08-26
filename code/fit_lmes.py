#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import sys, warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.tools as sm_tools
SEED = 69

from utils import abcd, residual


# load data
covariates = abcd.load_covariates(covars=['interview_age', 'sex', 'hisp'],
                                  simple_race=True)

path = Path(sys.argv[1])
data, extra = abcd.load_mri_data(path.name[:4], path=path)
data = abcd.filter_siblings(data, random_state=SEED)
extra = extra.loc[data.index]

# inputs
exog = sm_tools.add_constant(
    extra[['meanmotion']].join(covariates['interview_age'] / 12)
    .join(covariates[['sex', 'race', 'hisp']])
)
groups = data.index.to_frame()[abcd.INDEX[0]]

with warnings.catch_warnings():
    warnings.simplefilter('ignore', sm_tools.sm_exceptions.ConvergenceWarning)
    lmes = residual.residualize(
        data, MixedLM, exog,
        groups=groups, exog_re=exog[['const', 'interview_age']],
        method=['bfgs', 'cg', 'nm'],
        return_results=True, n_procs=4
    )

# output
lmes.to_pickle(path.parent / (path.stem + '_lmes.pkl.gz'))
