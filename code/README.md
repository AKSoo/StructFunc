# Code

* `utils/`: Several helpful functions.

## Scripts

1. `aggregate_covariates.py`: Aggregate demographic covariates of interest
   (based on script used by DEAP).
2. `scanner_combat.py`: Use ComBat to harmonize connectivity data across
   scanners and software versions.
3. `multiverse_sCCA.py`: Run sparse canonical correlation analyses with many
   different choices in data preprocessing methods.
4. `multiverse_test.py`: Test associations from `multiverse_sCCA.py` with
   permutation tests.

## Notebooks

1. `ConfoundEffects.ipynb`: Effects of various scanner confounds and covariates.
2. `Change.ipynb`: Test whether early adolescent data show known changes in
   connectivity during adolescent development.
3. `StructFunc.ipynb`: Replication of previous results on structural
   connectivity (SC) prediction of functional connectivity (FC) changes.
4. `SparseCCA.ipynb`: Find associations between SC and FC changes.
5. `Multiverse.ipynb`: Results of the `multiverse_sCCA.py`.
