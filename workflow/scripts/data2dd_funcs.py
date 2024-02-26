import calendar
import datetime
import itertools
import re
import warnings

import numpy as np
import pandas as pd


def wrapdd(d, out_par, otype, outfile=""):
    d = np.atleast_2d(d)

    if otype == "set":
        top = np.array([["set"], [out_par + " /"]])
        bottom = np.array([["/"], [""]])

    if otype == "scalar":
        top = np.array([["scalar"], [out_par + " /"]])
        bottom = np.array([["/"], [""]])

    if otype == "parameter":
        top = np.array([["parameter", ""], [out_par + " /", ""]])
        bottom = np.array([["/", ""], ["", ""]])

    d = np.concatenate((top, d, bottom), axis=0)

    if outfile != "":
        np.savetxt(outfile, d, delimiter=" ", fmt="%s")
    else:
        return d


def data2dd(data, sets, outfile="", all_combin=False, rounddp=8):
    # rounddp=5

    # each set in sets needs to be 1D at the moment

    # sets and data must be in correct order -> last set must be column
    # headers, previous sets are rows

    if type(data) != "numpy.ndarray":
        data = np.array(data)
    sets = np.array(sets, dtype="object")

    if len(sets) > 0:
        data = np.round(data.astype(float), rounddp)
        if len(sets) == 1:
            sets_out = sets[0]
            sets_new = np.array(sets_out).astype(str)

            data_out = np.hstack((sets_new.reshape(-1, 1), data.reshape(-1, 1)))
        else:
            if all_combin:
                sets_out = [item for item in list(itertools.product(*sets))]

                sets_out = np.char.array(sets_out)

                sets_new = sets_out[:, 0]

                for i in range(sets_out.shape[1] - 1):
                    sets_new = sets_new + "." + sets_out[:, i + 1]

                data_out = np.hstack((sets_new.reshape(-1, 1), data.reshape(-1, 1)))

            else:
                lens = np.array([item.shape[0] for item in sets])

                sets_new = []
                for i, s in enumerate(sets):
                    if lens[i] != lens.max():
                        sets_new.append(np.repeat(s, lens.max()))
                    else:
                        sets_new.append(s)

                sets_new = np.array(
                    [".".join(item) for item in np.char.array(list(zip(*sets_new)))]
                )

                data_out = np.hstack((sets_new.reshape(-1, 1), data.reshape(-1, 1)))

    else:
        data_out = data.reshape(-1, 1)
    if outfile != "":
        np.savetxt(outfile, data_out, fmt="%s", delimiter=" ")
    else:
        return data_out


def temporal2dd(dstart, dend, opath, temporaloutputpath):
    # nyears=(dend.year-dstart.year)+1
    years = np.arange(dstart.year, dend.year + 1)
    ntime = ((dend - dstart).total_seconds() / 3600) + 1

    hr2yr = []
    for nyr, yr in enumerate(years):
        shour = ((datetime.datetime(yr, 1, 1, 0) - dstart).total_seconds() / 3600) + 1
        if dend.year == yr:
            ehour = ((dend - dstart).total_seconds() / 3600) + 1
        else:
            ehour = (
                (datetime.datetime(yr, 12, 31, 23) - dstart).total_seconds() / 3600
            ) + 1

        hrs = np.arange(shour - 1, ehour).astype(int)
        hr2yr.append(list(zip(np.repeat(nyr, hrs.shape[0]).astype(int), hrs)))

    hr2yr = np.char.array(np.vstack(hr2yr).astype(str))

    hr2yr = (hr2yr[:, 0] + "." + hr2yr[:, 1]).reshape(-1, 1)

    if dstart.year != dend.year:
        yr = str(dstart.year) + "-" + str(dend.year)
    else:
        yr = str(dstart.year)

    out = []

    out.append(wrapdd(np.arange(ntime).reshape(-1, 1).astype(int), "h", "set"))
    out.append(
        wrapdd(np.arange(years.shape[0]).reshape(-1, 1).astype(int), "yr", "set")
    )
    out.append(wrapdd(hr2yr, "hr2yr_map", "set"))

    np.savetxt(temporaloutputpath, np.concatenate(out, axis=0), fmt="%s")


def trans_links(root, f, aggregated_regions, out="work"):
    tech_type = "trans"

    links_allowed = pd.read_excel(
        f, sheet_name="transmission_allowed", skiprows=1
    ).query("Zone1 == @aggregated_regions and Zone2 == @aggregated_regions")

    links_out = np.array(
        links_allowed["Zone1"]
        + "."
        + links_allowed["Zone2"]
        + "."
        + links_allowed["Tech"]
    )

    links_tech = pd.unique(links_allowed["Tech"])

    set_outdd = []
    param_outdd = []

    set_outdd.append(wrapdd(data2dd(links_tech, []), "trans", "set"))
    set_outdd.append(wrapdd(data2dd(links_out, []), "trans_links", "set"))

    # data2dd(links_out,[],outfile=root/out/(tech_type+"_links.dd"))

    out_data = ["links_dist", "links_cap"]

    for od in out_data:
        out_par = tech_type + "_" + od

        param_outdd.append(
            wrapdd(data2dd(links_allowed[od].values, [links_out]), out_par, "parameter")
        )

    params = pd.read_excel(f, sheet_name="transmission", skiprows=1)

    nans = params.isnull()
    params = params.where(~nans, other=0.0)

    out_data = ["loss", "line_capex", "sub_capex", "varom"]

    #    trans_set=params["Technology Name (highRES)"]

    params = params[params["Technology Name (highRES)"].isin(links_tech)]

    for od in out_data:
        out_par = tech_type + "_" + od

        param_outdd.append(
            wrapdd(
                data2dd(params[od], [params["Technology Name (highRES)"]]),
                out_par,
                "parameter",
            )
        )

    param_outdd = np.concatenate(param_outdd, axis=0)
    set_outdd = np.concatenate(set_outdd, axis=0)

    pad = np.repeat(np.array(""), set_outdd.shape[0]).reshape(set_outdd.shape[0], 1)

    outdd = np.concatenate((np.hstack((set_outdd, pad)), param_outdd), axis=0)

    #    outdd=np.concatenate(outdd,axis=0)

    np.savetxt(root / out / (tech_type + ".dd"), outdd, delimiter=" ", fmt="%s")


def co2lim2dd(co2budgetddlocation, root, run, esys, scen_db, out=""):
    scen = pd.read_excel(scen_db, sheet_name="scenario_co2_cap", skiprows=1)

    dout = scen.loc[scen["Esys Scenario"] == esys, 2050].values

    wrapdd(dout, "co2_budget", "scalar", outfile=co2budgetddlocation)


def getzlims(lim, techs, zones):
    lim = lim.loc[(lim["Year"] == 2050) & (lim["Technology"].isin(techs)), :]

    if lim.empty:
        return np.array([])

    have_lim = ~lim.loc[:, "limtype"].isnull()

    if (~have_lim).any():
        warnings.warn(
            "Warning, missing zonal new capacity limits for:"
            + ", ".join(lim.loc[~have_lim, "Technology"])
        )

    lim = lim.loc[have_lim, :]
    para_lim = lim["parameter"].drop_duplicates()

    outlims = []

    for p_lim in para_lim:
        nl = []
        for _, row in lim.loc[lim["parameter"] == p_lim, :].iterrows():
            zones = row[zones].index.values
            limval = row[zones].values.astype(float) / 1e3
            limval[np.isnan(limval)] = 0.0

            tech = np.atleast_1d(row["Technology"])
            limtype = np.atleast_1d(row["limtype"])

            nl.append(data2dd(limval.T, [zones, tech, limtype]))

        outlims.append(wrapdd(np.concatenate(nl, axis=0), p_lim, "parameter"))

    return np.concatenate(outlims, axis=0)


def scen2dd(
    co2budgetddlocation,
    root,
    fin,
    params2write,
    scen_db,
    run,
    esys,
    zones,
    out="work",
    esys_cap=False,
    exist_cap=False,
):
    co2lim2dd(co2budgetddlocation, root, run, esys, scen_db, out=out)

    scen = pd.read_excel(scen_db, sheet_name="scenario_tech_definition", skiprows=0)

    scen = scen.loc[(scen["Psys Scenario"] == run), :]

    techs = scen["Technology Name (highRES)"]
    tech_class = ["gen", "store"]

    for tech_type in tech_class:
        set_outdd = []
        param_outdd = []

        params = pd.read_excel(fin, sheet_name=None, skiprows=1)[tech_type]

        params = params[params["Technology Name (highRES)"].isin(techs)]

        sets = params2write[tech_type]["set"]
        data_out = params2write[tech_type]["parameter"]

        params = params.merge(scen, on="Technology Name (highRES)")

        # new_lim=pd.read_excel(fin,sheet_name=tech_type+"_lim_z",skiprows=0)

        param_outdd.append(
            getzlims(
                pd.read_excel(fin, sheet_name=tech_type + "_lim_z", skiprows=0),
                techs,
                zones,
            )
        )

        if exist_cap:
            lims = getzlims(
                pd.read_excel(fin, sheet_name=tech_type + "_exist_z", skiprows=0),
                techs,
                zones,
            )

            if lims.size != 0:
                param_outdd.append(
                    getzlims(
                        pd.read_excel(
                            fin, sheet_name=tech_type + "_exist_z", skiprows=0
                        ),
                        techs,
                        zones,
                    )
                )

        if esys_cap:
            dout = params.loc[
                (~params["Esys_capacity~2050"].isnull()), "Esys_capacity~2050"
            ]

            if dout.empty:
                continue
            else:
                wrapdd(
                    data2dd(
                        dout.values,
                        [
                            params[
                                (params["Technology Name (highRES)"] != "pgen")
                                & (~params["Esys_capacity~2050"].isnull())
                            ]["Technology Name (highRES)"].values
                        ],
                        rounddp=2,
                    ),
                    tech_type + "_fx_natcap",
                    root / out / (esys + "_" + tech_type + "_fx_natcap.dd"),
                )

        for s in sets:
            if s == "g" or s == "s":
                vals = params["Technology Name (highRES)"]

                set_outdd.append(wrapdd(data2dd(vals, []), s, "set"))
                continue

                # outsets.append(wrapdd(data2dd(vals,[])))
                # set_outdd.append(wrapdd(data2dd(vals,[],s,"set"))
            elif "uc" in s:
                vals = params.loc[(~params[s].isnull()), "Technology Name (highRES)"]

                # if vals.empty:
                #    continue

                set_outdd.append(
                    wrapdd(
                        data2dd(vals, []),
                        tech_type + "_" + s + "(" + sets[0] + ")",
                        "set",
                    )
                )
                # else:
                #   set_outdd.append(wrapdd(data2dd(
                #       vals,
                #       [],
                #       root/out/(run+"_"+tech_type+"_set_"+s+".dd"))))

            else:
                vals = params[params["set"] == s]["Technology Name (highRES)"]

                set_outdd.append(
                    wrapdd(data2dd(vals, []), s + "(" + sets[0] + ")", "set")
                )

                # data2dd(vals,[],root/out/(run+"_"+tech_type+"_set_"+s+".dd"))

        for s in data_out:
            for p in data_out[s]:
                if "varom" in p or "fuel" in p:
                    # rounddp=5
                    rounddp = 8
                elif "emis" in p:
                    # rounddp=3
                    rounddp = 8
                else:
                    # rounddp=2
                    rounddp = 8

                out_par = tech_type + "_" + str.replace(p, " ", "")

                if "capex" in p or "fuelcost" in p:
                    out_par = re.sub(r"\d+", "", out_par)

                vals = params.loc[~(params[p].isnull()), p].values
                t_out = params.loc[
                    ~(params[p].isnull()), "Technology Name (highRES)"
                ].values

                param_outdd.append(
                    wrapdd(
                        data2dd(vals, [t_out], rounddp=rounddp), out_par, "parameter"
                    )
                )

        param_outdd = np.concatenate(param_outdd, axis=0)
        set_outdd = np.concatenate(set_outdd, axis=0)

        pad = np.repeat(np.array(""), set_outdd.shape[0]).reshape(set_outdd.shape[0], 1)

        outdd = np.concatenate((np.hstack((set_outdd, pad)), param_outdd), axis=0)

        np.savetxt(
            root / out / (run + "_" + tech_type + ".dd"), outdd, delimiter=" ", fmt="%s"
        )

        # param_outdd=np.concatenate(param_outdd,axis=0)
        # set_outdd=np.concatenate(set_outdd,axis=0)

        # np.savetxt(
        #     root/out/(run+"_"+tech_type+"_parameters.dd"), param_outdd,
        #     delimiter=" ",
        #     fmt="%s")
        # np.savetxt(
        #     root/out/(run+"_"+tech_type+"_sets.dd"),
        #     set_outdd, delimiter=" ",
        #     fmt="%s")


def euro_demand2dd(
    europedemandcsvlocation,
    europecountriescsvlocation,
    dpath,
    opath,
    dstart,
    dstop,
    scen_db,
    esys_scen,
    yr,
    rescale="annual",
    zones="yes",
):
    # 2010 demand is ~ 3216 TWh over the 31 countries in ETM
    #

    countries = pd.read_csv(europecountriescsvlocation)
    # pd.read_csv(dpath/"zonal_def"/"europe_countries.csv")

    c_sel = countries.loc[countries["ETM"] == 1, "ISO2"]

    d = pd.read_csv(europedemandcsvlocation)

    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.set_index("datetime")

    d = d.loc[:, d.columns.isin(c_sel)]

    d = d[dstart:dstop]

    if d.shape[1] != c_sel.shape[0]:
        print("Countries missing...")

    d[(d == 0)] = np.nan

    d.interpolate(limit=2, inplace=True)

    if pd.isnull(d).any().any():
        for c in d.columns[pd.isnull(d).any(axis=0)]:
            print(d.loc[pd.isnull(d[c]), c])

        print("Zero demands for: ", d.columns[pd.isnull(d).any(axis=0)])

    if calendar.isleap(yr):
        d = pd.concat((d, d.iloc[0:24, :]))

    if rescale == "annual":
        out_flg = "annual"
        if d.shape[0] >= 8760.0:
            scen = pd.read_excel(scen_db, sheet_name="scenario_annual_dem", skiprows=0)

            euro31_dem = scen[scen["Esys Scenario"] == esys_scen][
                "Annual demand (2050)"
            ].iloc[0]

            d = d * (euro31_dem * 1e6 / d.sum().sum())

    #        else:
    #            dem_scaling=1.511
    #            demand=demand*dem_scaling

    else:
        out_flg = "norescale"

    t = np.arange(d.shape[0])
    z = d.columns.values

    wrapdd(
        data2dd(d.values.T, [z, t], all_combin=True),
        "demand",
        "parameter",
        outfile=opath / (esys_scen + "_" + out_flg + "_demand_" + str(yr) + ".dd"),
    )
