#!/bin/bash
set -e

dataset=../Nifti/derivatives/fmriprep
cohorts=../work
design=../work/fc-36p_despike-PNC.dsn
outdir=../output/xcpengine
logdir=../log/xcpengine
mkdir -p $outdir $logdir

xcpengine () {
    docker run --rm -u $(id -u):$(id -g) --cpus="1.5" \
        -v $dataset:/data:ro \
        -v $1:/cohort/cohort.csv \
        -v $design:/xcpEngine/designs/fc-36p_despike-PNC.dsn \
        -v $2:/out \
        pennbbl/xcpengine \
        -r /data \
        -c /cohort/cohort.csv \
        -d /xcpEngine/designs/fc-36p_despike-PNC.dsn \
        -o /out
}


for i in {0..3}; do
    mkdir -p $outdir/cohort_$i
    xcpengine $cohorts/cohort_$i.csv $outdir/cohort_$i &> $logdir/cohort_$i.log &
done
wait
