#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

import pandas as pd
SEED = 69

import joblib
from gemmr.estimators import SparseCCA

from utils import permutation
import multiverse_sCCA


def scca_pvals(verse, n_perm=4999, test=False,
               random_state=SEED, n_procs=1):
    dFC, SC, _, _ = multiverse_sCCA.verse_inputs(verse, test=test)
    scca = joblib.load(multiverse_sCCA.out_path / verse / 'scca_SC.joblib.gz')

    def quick_scca(X, Y):
        tmp_scca = SparseCCA(scale=True, optimize_penalties=False,
                             penaltyxs=scca.penaltyx_, penaltyys=scca.penaltyy_)
        tmp_scca.fit(X, Y)
        return tmp_scca.corrs_[0]

    distrib = permutation.permute_func(quick_scca, SC, dFC, n_perm=n_perm,
                                       random_state=random_state, n_procs=n_procs)
    p = ((distrib >= scca.corrs_.max()).sum() + 1) / (len(distrib) + 1)
    return p

if __name__ == '__main__':
    pvals = joblib.Parallel(n_jobs=8)(joblib.delayed(scca_pvals)(v)
                                      for v in multiverse_sCCA.verses)
    pvals = pd.DataFrame(pvals, index=multiverse_sCCA.verses, columns=['p_SC'])
    pvals.to_csv(multiverse_sCCA.out_path / 'pvals.csv')
