#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=UserWarning)

import numpy as np
import pandas as pd
SEED = 69

import joblib
from scipy import stats
from gemmr.estimators import SparseCCA

from utils import permutation
import multiverse_sCCA


def scca_test(verse, n_perm=99, random_state=SEED):
    """
    Calculate p value for a given experiment verse by permutation.

    Params:
        verse: str experiment settings
        n_perm: int number of permutations

    Returns:
        p: float p value
    """
    dFC, SC, _, _ = multiverse_sCCA.verse_inputs(verse, test=True)
    scca_SC = joblib.load(multiverse_sCCA.out_path / verse / 'scca_SC.joblib.gz')

    # permutations
    def quick_scca(X, Y):
        tmp_scca = SparseCCA(scale=True, optimize_penalties=False,
                             penaltyxs=scca_SC.penaltyx_, penaltyys=scca_SC.penaltyy_)
        tmp_scca.fit(X, Y)
        return tmp_scca.corrs_[0]

    distrib = permutation.permute_func(quick_scca, SC, dFC, n_perm=n_perm,
                                       random_state=random_state)

    # correlations
    SC_score, dFC_score = scca_SC.transform(SC, dFC)
    corrs = np.array([stats.pearsonr(SC_score[:, i], dFC_score[:, i])[0]
                      for i in range(len(scca_SC.corrs_))])

    corr = corrs.max()
    p = ((distrib >= corr).sum() + 1) / (len(distrib) + 1)
    return corr, p

if __name__ == '__main__':
    test = joblib.Parallel(n_jobs=8)(joblib.delayed(scca_test)(v)
                                      for v in multiverse_sCCA.verses)
    test = pd.DataFrame(test, index=multiverse_sCCA.verses, columns=['SC_test', 'p_SC'])
    test.to_csv(multiverse_sCCA.out_path / 'test.csv')
