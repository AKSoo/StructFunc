import os
import pandas as pd
idx = pd.IndexSlice

from statsmodels.tools import tools as sm_tools


INDEX = ['src_subject_id', 'eventname']
EVENTS = ['baseline_year_1_arm_1', '2_year_follow_up_y_arm_1']

FCON_TEMPLATE = 'rsfmri_c_ngd_{0}_ngd_{1}'
FCON = {
    'ad': 'auditory',
    'cgc': 'cingulo-opercular',
    'ca': 'cingulo-parietal',
    'dt': 'default',
    'dla': 'dorsal attention',
    'fo': 'fronto-parietal',
    'n': None,
    'rspltp': 'retrosplenial temporal',
    'smh': 'sensorimotor hand',
    'smm': 'sensorimotor mouth',
    'sa': 'salience',
    'vta': 'ventral attention',
    'vs': 'visual'
}

SCON_TEMPLATE = 'dmri_dtifa_fiberat_{0}'


def load_data(abcd_path, data, include_rec=None, exclude_n=None):
    """
    Load a longitudinally ordered ABCD dataset.
    * fcon: Gordon network correlations
    * scon: DTI atlas tract fractional anisotropy averages

    Params:
        abcd_path: ABCD dataset directory
        data: dataset to load
        include_rec: [fcon, scon] Filter by recommended inclusion?
        exclude_n: [fcon] Ignore None "network"?

    Returns:
        dataset: DataFrame indexed and sorted by (subject, event)
        extra: DataFrame of relevant supplementary data
            * fcon, scon: mean motion
    """
    if data == 'fcon':
        filename = 'abcd_betnet02.tsv'
    elif data == 'scon':
        filename = 'abcd_dti_p101.tsv'
    else:
        raise ValueError('Unknown dataset ' + data)

    raw = pd.read_csv(os.path.join(abcd_path, filename), sep='\t',
                      skiprows=[1], index_col=INDEX)
    raw = raw.loc[~raw.index.duplicated(keep='last')]

    # rows
    if include_rec:
        imgincl = pd.read_csv(os.path.join(abcd_path, 'abcd_imgincl01.tsv'), sep='\t',
                              skiprows=[1], index_col=INDEX)
        imgincl = imgincl.dropna(subset=['visit'])
        imgincl = imgincl.loc[~imgincl.index.duplicated(keep='last')]

        if data == 'fcon':
            included = imgincl.loc[imgincl['imgincl_rsfmri_include'] == 1]
        elif data == 'scon':
            included = imgincl.loc[imgincl['imgincl_dmri_include'] == 1]
        else:
            raise ValueError('No recommended inclusion for ' + data)
    else:
        included = raw

    subs_included = included.groupby(level='src_subject_id').size()
    subs_long = subs_included.index[subs_included == len(EVENTS)]

    # columns
    if data == 'fcon':
        fcon_codes = list(FCON.keys())
        if exclude_n:
            fcon_codes.remove('n')

        columns = []
        for i in range(len(fcon_codes)):
            for j in range(i+1):
                columns.append(FCON_TEMPLATE.format(fcon_codes[i], fcon_codes[j]))

        extra_columns = ['rsfmri_c_ngd_meanmotion']
    elif data == 'scon':
        columns = raw.columns.str.startswith(SCON_TEMPLATE.format(''))

        extra_columns = ['dmri_dti_meanmotion']
    else:
        columns = slice(None)

        extra_columns = []

    dataset = raw.loc[idx[subs_long, EVENTS], columns]
    extra = raw.loc[dataset.index, extra_columns]
    return dataset, extra


def get_scon_dict(abcd_path):
    """
    Builds and returns a dict of DTI atlas tract descriptions.

    Params:
        abcd_path: ABCD dataset directory

    Returns:
        scon_dict: dict with (code, description) for each tract
    """
    dti_labels = pd.read_csv(os.path.join(abcd_path, 'abcd_dti_p101.tsv'), sep='\t', nrows=1)
    code_start = SCON_TEMPLATE.format('')
    description_starts = ('Average fractional anisotropy within ', 'DTI atlas tract ')

    scon_labels = dti_labels.loc[0, dti_labels.columns.str.startswith(code_start)]
    codes = scon_labels.index.str.replace(code_start, '')
    descriptions = (scon_labels.str.replace(description_starts[0], '')
                    .str.replace(description_starts[1], '').values)

    scon_dict = dict(zip(codes, descriptions))
    return scon_dict


def load_covariates(path, simple_race=False):
    """
    Load ABCD covariates.

    Params:
        path: Covariates file
        simple_race: Simpler race with [White, Black, Asian, Other, Mixed].
            Other includes missing.

    Returns:
        covariates: DataFrame indexed by (subject, event)
    """
    covariates = pd.read_csv(path, index_col=INDEX)

    if simple_race:
        covariates['race'] = covariates['race.6level'].replace('AIAN/NHPI', 'Other').fillna('Other')
        covariates = covariates.drop('race.6level', axis=1)

    return covariates


def confound_residuals(feature, model=None, regressors=None, groups=None, verbose=False,
                       **kwargs):
    """
    Regress out confounds from a feature with a statsmodels model.

    Params:
        feature: Series
        model: statsmodels model (OLS, MixedLM)
        regressors: DataFrame
        groups: Series, for mixed effects model
        verbose: print model fit results?
        **kwargs: passed to the model

    Returns:
        resid: Series, same index as feature
    """
    if model is None or regressors is None:
        raise ValueError('Model or regressors not specified.')

    # MixedLM(missing='drop') doesn't work
    na_filter = feature.notna()
    endog = feature.loc[na_filter]
    exog = sm_tools.add_constant(pd.get_dummies(regressors, drop_first=True)).loc[na_filter]
    if groups is not None:
        groups = groups.loc[na_filter]

    result = model(endog, exog, groups=groups, **kwargs).fit()

    if verbose:
        print(result.summary())
    
    resid = result.resid.reindex(feature.index)
    return resid
