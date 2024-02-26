import calendar
import datetime
import pathlib

import pandas as pd

from data2dd_funcs import euro_demand2dd, scen2dd, temporal2dd, trans_links

root = pathlib.Path(snakemake.output[0]).parent
data_root = root
out = pathlib.Path(".")

pscens = ["BASE"]


for psys in pscens:
    psys_scen = psys
    esys_scen = "BASE"

    scen_db = snakemake.input[1]
    f_techno = snakemake.input[2]

    params_to_write = {}

    params_to_write["gen"] = {}
    params_to_write["gen"]["set"] = [
        "g",
        "non_vre",
        "vre",
        "hydro_res",
        "uc_int",
        "uc_lin",
    ]
    params_to_write["gen"]["parameter"] = {}
    params_to_write["gen"]["parameter"]["all"] = [
        "emis fac",
        "max ramp",
        "min down",
        "min up",
        "startup cost",
        "inertia",
        "unit size",
        "capex2050",
        "varom",
        "fom",
        "fuelcost2050",
        "cap2area",
        "af",
        "min gen",
        "peak af",
    ]

    params_to_write["store"] = {}
    params_to_write["store"]["set"] = ["s", "uc_lin"]
    params_to_write["store"]["parameter"] = {}
    params_to_write["store"]["parameter"]["all"] = [
        "max_freq",
        "max_res",
        "eff_in",
        "eff_out",
        "loss_per_hr",
        "p_to_e",
        "varom",
        "e capex2050",
        "p capex2050",
        "fom",
        "af",
        "min down",
        "min up",
        "startup cost",
        "inertia",
        "unit size",
        "min gen",
        "max ramp",
    ]

    zones = pd.read_csv(snakemake.input[0]).loc[:, "zone"]

    scen2dd(
        snakemake.output[1],
        root,
        f_techno,
        params_to_write,
        scen_db,
        psys_scen,
        esys_scen,
        zones,
        out=out,
        esys_cap=False,
        exist_cap=True,
    )

    trans_links(
        root,
        f_techno,
        aggregated_regions=snakemake.params.aggregated_regions,
        out=out,
    )


years = [snakemake.wildcards.year]

for yr in years:
    yr = int(yr)

    dstart = datetime.datetime(yr, 1, 1, 0)
    dstop = datetime.datetime(yr, 12, 31, 23)

    if calendar.isleap(yr):
        rleap = False
    else:
        rleap = True

    print(rleap)
    temporal2dd(dstart, dstop, root / out, snakemake.output[0])

    euro_demand2dd(
        snakemake.input["europedemandcsvlocation"],
        snakemake.input["europecountriescsvlocation"],
        root,
        root / out,
        dstart,
        dstop,
        scen_db,
        esys_scen,
        yr,
        "norescale",
    )
