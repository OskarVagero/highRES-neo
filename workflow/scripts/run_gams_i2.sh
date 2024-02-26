#!/usr/bin/env bash

# TODO: export environment variable to then have gams read the path from there.
# TODO: also call gams from here and then set the log file path here too


cd ${snakemake_params[modelpath]}
pwd -P
${snakemake_params[gamspath]}gams highres_i2.gms logOption=2