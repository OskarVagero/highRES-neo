# -*- coding: utf-8 -*-
"""
Mapping of .ods file to .csv file
all_simple_translate() translate all sheets directly (results instead of formulas)
proc_transmission() translate the transmission* sheets more sophisticatedly
transmission_recreate() recreates the transmission sheet info from the raw csv files
This file can be also run as a script. It then runs all of the above for now.
"""
import os
import re
from types import SimpleNamespace

import numpy as np
import numpy_financial as npf
import pandas as pd
import yaml


############
# Simple translation to csv of all sheets
def simple_conversion(fname, sheets):
    """
    Convert all pure input excel sheets to csv

    This here: Different function
    dont atumate this I think, exception if you see repeating stuff 
    also rename them to _INPUT.csv (after forumals, _OUTPUT.csv)

    Convert from ods to something else
    https://stackoverflow.com/questions/15257032/python-convert-excel-file-xls-or-xlsx-to-from-ods/73865904#73865904
    https://github.com/pyexcel/pyexcel
    https://openpyxl.readthedocs.io/en/stable/
    So that you can parse all the ods sheets and put FORMULA_HERE, where there is formulas or math
    """
    # Straight convert the lim and exist sheets
    # list of sheets names and output names (dict?), for loop with excel read

    sheets = {"gen_lim_z":"gen_lim_z_INPUT",
            "gen_exist_z":"gen_exist_z_INPUT",
            "store_lim_z":"store_lim_z_INPUT",
            "store_exist_z":"store_exist_z_INPUT"}


    print("Starting simple transfer of pure input sheets (no formulas):")
    for sheet, lines_to_skip in sheets.items():
        print(f"Converting {sheet}") 
        df = pd.read_excel(fname, sheet_name=sheet)
        df.to_csv(sheet + '.csv',  index=False)

    # Then convert the transmission allowed sheet
    sheets["transmission allowed"] = "transmission_allowed_INPUT"
    print("Converting transmission allowed")
    df = pd.read_excel(fname, sheet_name=sheets["transmission allowed"], usecols="C:K")
    df.to_csv(sheets["transmission allowed"] + ".csv", index=False)

    # Then convert uktm_emis_factors sheet
    # excel read og output name as var
    # Row 4 til and including 7, columns D:Z
    sheets["uktm_emis_factors"] = "uktm_emis_factors_INPUT" 
    print("Converting uktm_emis_factors")
    df = pd.read_excel(fname, sheet_name=sheets["uktm_emis_factors"], usecols="D:Z", skiprows=3, nrows=4)
    df.to_csv(sheets["uktm_emis_factors"] + ".csv", index=False)


def fuelcost_default_scenarious_formulas(df_in, config = None):
    df_out = df_in.copy(deep=True) 
    params = config["fuelcosts"]
    year_2050 = params["EU average elc price (euro/MWh)"]["value"] * 0.9 * params["pounds scaling"]
    Continental_Europe = year_2050
    Ireland = year_2050
    
    # M8 was calculated using N8 * c16 * 1000, with N8 = 0.01
    M8 = 2.77778
    # M10, changed value, from 20.3188878976133 to 3.81

    # In £k/MWh (k, is 1000)
    # For the commodity Coal, 0.01 (N8 in the Excel sheet) was an input value
    df_out["£k/MWh"] = (df_out["2050"]/params["energy scaling (GJ to MWh)"]) / 1000 
    """
    Comment from the Excel sheet for the commodity Natural gas:
    James Price: BEIS - Average prices of fuels purchased by the major UK power producers
    """
    return df_out


def generator_costs_formulas(df_input, config = None):
    params = config["generator costs"]
    
    # NCAP_2050+IDC, =-FV(C4,ABS(G4),O4/ABS(G4)), rows 4 and down
    df_input.loc["NuclearEPR":"SynCon", "NCAP_2050+IDC"] = (
            -npf.fv(df_input.loc["NuclearEPR":"SynCon", "Discount rate"], np.abs(df_input.loc["NuclearEPR":"SynCon", "NCAP_ILED"]),
                    df_input.loc["NuclearEPR":"SynCon", "NCAP_COST~2050"] / np.abs(df_input.loc["NuclearEPR":"SynCon", "NCAP_ILED"]))
        )

    # Capex_annual~2015 to Capex_annual~2045, same formula, not including row 26
    # H4 * $C4 / (1 - (1+ $C4)^ - $F4)
    technologies_capex = ["NuclearEPR",
        "NuclearSW",
        "Hydrogen CCGT (new)",
        "Hydrogen OCGT (new)",
        "Natural gas CCGT with CCS (new)",
        "Natural gas CCGT (new)",
        "Natural gas OCGT (new)",
        "Manufactured fuels (OCGT)",
        "Biomass",
        "Biomass CCS",
        "Solar",
        "Tidal ",
        "Wind onshore",
        "Wind offshore",
        "Wave",
        "Geothermal",
        "HydroRoR",
        "HydroRes",
        "Coal",
        "Coal CCS"]

    
    df_input.loc[technologies_capex, "Capex_annual~2015":"Capex_annual~2045"] = (
                df_input.loc[technologies_capex, "NCAP_COST~2015":"NCAP_COST~2045"] * df_input.loc[technologies_capex, "Discount rate"] / 
                (1 - (1 + df_input.loc[technologies_capex, "Discount rate"])**(-df_input.loc[technologies_capex, "NCAP_ELIFE"]))
            )

    # ACT_COST_kWh, AA, several rows, but not all rows in the column
    # =Z4/277.7778
    technologies_variable_costs_kWh = ["NuclearEPR",
        "NuclearSW", 
        "Hydrogen CCGT (new)", 
        "Hydrogen OCGT (new)", 
        "Natural gas CCGT (existing)", 
        "Natural gas OCGT (existing)", 
        "Manufactured fuels (OCGT)", 
        "Biomass", 
        "Biomass CCS", 
        "Solar", 
        "Tidal ", 
        "Wind onshore", 
        "Wind offshore", 
        "Wave", 
        "Geothermal", 
        "Coal"]
    df_input.loc[technologies_variable_costs_kWh, "ACT_COST"] = df_input.loc[technologies_variable_costs_kWh, "ACT_COST_kWh"] / 277.7778

    # FOM. AB, math, EXCEL comment in that column
    df_input.loc["NuclearEPR", "FOM"] = 72.9 / (1 + 0.03686)**4
    df_input.loc["NuclearSW", "FOM"] = 72.9 / (1 + 0.03686)**4

    # LCOE (2010), AH, one formula several rows, but empty cells in them
    technologies_LCOE_2010 = ["NuclearEPR", "NuclearSW", "Natural gas CCGT with CCS (new)"]
    # (AG9*8760*(AA9+(AF9/AE9))+AB9+Y9)*1000/(8760*AG9)
    df_input.loc[technologies_LCOE_2010, "LCOE (2010)"] = (
            (df_input.loc[technologies_LCOE_2010, "Capacity factor"] * 8760 * (df_input.loc[technologies_LCOE_2010, "ACT_COST_kWh"] + 
                (df_input.loc[technologies_LCOE_2010, "Fuel cost"] / df_input.loc[technologies_LCOE_2010, "Unknown constants"])) + 
                 df_input.loc[technologies_LCOE_2010, "FOM"] + df_input.loc[technologies_LCOE_2010, "Capex_annual~2050"])
            * 1000 / (8760 * df_input.loc[technologies_LCOE_2010, "Capacity factor"]) 
            )
    # =(AG16*8760*AA16+AB16+Y16)*1000/(8760*AG16)
    technologies_LCOE_2010_v2 = ["Solar", "Wind onshore", "Wind offshore"]
    df_input.loc[technologies_LCOE_2010_v2, "LCOE (2010)"] = (
            (df_input.loc[technologies_LCOE_2010_v2, "Capacity factor"] * 8760 * df_input.loc[technologies_LCOE_2010_v2, "ACT_COST_kWh"] + 
             df_input.loc[technologies_LCOE_2010_v2, "FOM"] + df_input.loc[technologies_LCOE_2010_v2, "Capex_annual~2050"]) * 1000 / 
             (8760 * df_input.loc[technologies_LCOE_2010_v2, "Capacity factor"])
            )


    # LCOE (2018), AI, same as AH
    technologies_LCOE_2018 = ["NuclearEPR",
        "NuclearSW",
        "Solar",
        "Wind onshore",
        "Wind offshore"]
    df_input.loc[technologies_LCOE_2018, "LCOE (2018)"] = df_input.loc[technologies_LCOE_2018, "LCOE (2010)"] * params["AI2"]

    # LCOE (2012), AJ, get hold of only 2 rows in one column
    df_input.loc["NuclearEPR", "LCOE (2012)"] = df_input.loc["NuclearEPR", "LCOE (2010)"] * params["AJ2"]
    df_input.loc["NuclearSW", "LCOE (2012)"] = df_input.loc["NuclearSW", "LCOE (2010)"] * params["AJ2"]


def gen_formulas(df_input, config = None):
    params = config["gen"]
    params_fuelcost = config["fuelcosts"]

    # Column af (D)
    df_input.loc["Natural gas CCGT with CCS (new) OT", "af"] = np.round(df_input.loc["Natural gas CCGT with CCS (new)", "af"] * (1 - df_input.loc["Natural gas CCGT with CCS (new) OT", "cooling load"]), 2)

    # Column peak af (C)
    row_label_list = ["Natural gas CCGT with CCS (new) OT", "Natural gas CCGT with CCS", "Natural gas OCGT (new)", "Solar", "HydroRoR", "HydroRes", "NuclearEPR", "p gen"]
    df_input.loc[row_label_list, "peak af"] = df_input.loc[row_label_list, "af"]

    # max ramp (F)
    df_input.loc["NuclearEPR", "max ramp"] = df_input.loc["NuclearEPR", "unit size"] * df_input.loc["NuclearEPR", "max ramp (%)"]


    # max ramp (%), H
    row_label_list = ["Natural gas CCGT with CCS (new) OT", "Natural gas CCGT with CCS", "Natural gas OCGT (new)"]
    df_input.loc[row_label_list, "max ramp (%)"] = df_input.loc[row_label_list, "max ramp"] / df_input.loc[row_label_list, "unit size"]

    # emis fac, J
    row_label_list = ["Natural gas CCGT with CCS (new) OT", "Natural gas CCGT with CCS", "Natural gas OCGT (new)"]
    df_input.loc[row_label_list, "emis fac"] = (df_uktm["ELCCO2N", "ELCTRANSGAS"] / df_input.loc[row_label_list, "efficiency"]) * (1000 / 27778) * (1 - df_input.loc[row_label_list, "capture efficiency"])

    # efficiency, L
    df_input.loc["Natural gas CCGT with CCS (new) OT", "efficiency"] = df_input.loc["Natural gas CCGT with CCS (new)", "efficiency"] * (1 - df_input.loc["Natural gas CCGT with CCS (new) OT", "cooling load"])
    # cap2area, M, EXCEL comment not formulas here


    # cooling, Q
    df_input.loc["Natural gas CCGT with CCS (new) OT", "cooling"] = df_input.loc["Natural gas CCGT with CCS (new) OT", "cooling capex"] * df_input.loc["Natural gas CCGT with CCS (new) OT", "abstract"]

    # capex2015, W
    technologies_capex = ["Natural gas CCGT with CCS (new)", "Natural gas OCGT (new)", "Solar", "Wind onshore", "HydroRoR", "HydroRes", "NuclearEPR"]
    df_input.loc[technologies_capex, "capex2015":"capex2050"] = (
            np.round(df_generator_costs.loc[technologies_capex, "Capex_annual~2015":"Capex_annual~2050"], params["capex rounding index"]) 
        )
    df_input.loc["Natural gas CCGT with CCS (new) OT", "capex2015":"capex2050"] = (
        np.round(df_input.loc["Natural gas CCGT with CCS (new)", "capex2015":"capex2050"] + df_input.loc["Natural gas CCGT with CCS (new) OT", "cooling"], params["capex rounding index"])
        )

    # fom, AE
    df_input.loc[technologies_capex, "fom"] = np.round(df_generator_costs.loc[technologies_capex, "FOM"], params["capex rounding index"])
    df_input.loc["Natural gas CCGT with CCS (new) OT", "fom"] = df_input.loc["Natural gas CCGT with CCS (new)", "fom"]

    # varom, AF, EXCEL comment in this column
    df_input.loc[technologies_capex, "varom"] = df_generator_costs.loc[technologies_capex, "ACT_COST_kWh"] 
    df_input.loc["Natural gas CCGT with CCS (new) OT", "varom"] = df_input.loc["Natural gas CCGT with CCS (new)", "varom"]

    # fuelcost2010, AH
    technologies_with_fuels = ["Natural gas CCGT with CCS (new) OT", "Natural gas CCGT with CCS (new)", "Natural gas OCGT (new)", "NuclearEPR"]
    for tech in technologies_with_fuels: 
        fuel = df_input.loc[tech, "fuel"]
        df_input.loc[tech, "fuelcost2010":"fuelcost2050"] = (
            df_fuelcosts.loc[fuel, "2010":"2050"] * params_fuelcosts["pounds scaling"] / (df_input.loc[row_list_TEMP_HERE, "efficiency"] * params_fuelcosts["energy scaling (GJ to MWh)"])
            )
    return df_input


def storage_costs_forumlas(df_input, config = None):
    df_input.loc["PumpedHydro", "Discharge Time"] = np.round(9000/1600, 2)

    # Comment, excel TODO
    df_input.loc["NaS", "Discharge Time"] = 300/50 

    # PCS (£k/MW), K
    # =J3*B3/(B3+C3*G3)
    df_input.loc["PumpedHydro":"CAES (underground)", "PCS (£k/MW)"] = (
            df_input.loc["PumpedHydro":"CAES (underground)", "NCAP_COST~2050"] * df_input.loc["PumpedHydro":"CAES (underground)", "PCS"] / 
            (df_input.loc["PumpedHydro":"CAES (underground)", "PCS"] + df_input.loc["PumpedHydro":"CAES (underground)", "Storage"] * df_input.loc["PumpedHydro":"CAES (underground)", "Discharge Time"]
                )
    df_input.loc["H2-Salt-OCGT", "PCS (£k/MW)"] = 308 + 131
    df_input.loc["H2-Salt-CCGT", "PCS (£k/MW)"] = 560 + 131

    # PCS+IDC, M 
    technologies = ["Li-ion",
         "H2-Salt-OCGT", 
         "H2-Salt-CCGT"]
    df_input.loc[technologies, "Storage+IDC"] = (
        -npf.fv(df_input.loc[technologies, "Discount rate"], df_input.loc[technologies, "ILED PCS"], 
                df_input.loc[technologies, "PCS (£k/MW)"] / df_input.loc[technologies, "ILED PCS"]) 
        )


    # Storage (£k/MWh), N, four rows formula, EXCEL Comment in this column
    df_input.loc["PumpedHydro":"CAES (underground)", "PCS (£k/MW)"] = (
            df_input.loc["PumpedHydro":"CAES (underground)", "NCAP_COST~2050"] * 
            (df_input.loc["PumpedHydro":"CAES (underground)", "Storage"] * df_input.loc["PumpedHydro":"CAES (underground)", "Discharge Time"] / 
             (df_input.loc["PumpedHydro":"CAES (underground)", "PCS"] + df_input.loc["PumpedHydro":"CAES (underground)", "Storage"] * 
              df_input.loc["PumpedHydro":"CAES (underground)", "Discharge Time"])) / df_input.loc["PumpedHydro":"CAES (underground)", "Discharge Time"] 

    # Storage+IDC, P, FV formula
    technologies = ["Li-ion",
         "H2-Salt-OCGT", 
         "H2-Salt-CCGT"]
    df_input.loc[technologies, "Storage+IDC"] = (
        -npf.fv(df_input.loc[technologies, "Discount rate"], df_input.loc[technologies, "ILED Storage"], 
                df_input.loc[technologies, "PCS (£k/MW)"] / df_input.loc[technologies, "ILED Storage"]) 
        )

    # TCC Check, Q
    df_input.loc["PumpedHydro":"CAES (underground)", "PCS (£k/MW)"] = (
            df_input.loc["PumpedHydro":"CAES (underground)", "PCS (£k/MW)"] + df_input.loc["PumpedHydro":"CAES (underground)", "Storage (£k/MWh)"] * 
            df_input.loc["PumpedHydro":"CAES (underground)", "Discharge Time"]
            )

    # PCS (annualised), R
    technologies = ["Li-ion",
         "H2-Salt-OCGT", 
         "H2-Salt-CCGT"]
    df_input.loc[technologies, "PCS (annualised)"] = (
        df_input.loc[technologies, "PCS+IDC"] * df_input.loc[technologies, "Discount rate"] / 
         (1 - (1 + df_input.loc[technologies, "Discount rate"])**(-df_input.loc[technologies, "Power economic lifetime"]))
        )
    
    # Storage (annualised), S
    technologies = ["Li-ion",
         "H2-Salt-OCGT", 
         "H2-Salt-CCGT"]
    df_input.loc[technologies, "Storage (annualised)"] = (
        df_input.loc[technologies, "Storage+IDC"] * df_input.loc[technologies, "Discount rate"] / 
         (1 - (1 + df_input.loc[technologies, "Discount rate"])**(-df_input.loc[technologies, "Energy economic lifetime"]))
        )

    # power FOM, T, just an EXCEL comment no formulas here

    # power varom, V
    df_input.loc["PumpedHydro", "power varom"] = np.round((42/1000) / (1 + 0.0368)**4, 3) 
    return df_input


def store_formulas(df_input, config = None):
    params = config["store"]
    # startup cost, O, EXCEL comment in column no formulas here
    # eff_in, T
    technologies = ["PumpedHydro",
        "NaS",
        "VRFB",
        "VRFB",
        "VRFB",
        "Li-ion",
        "Li-ion"]
    df_input.loc[technologies, "eff_in"] = np.round(np.sqrt(df_input.loc[technologies, "roundtrip efficiency"]), 2)
    
    # eff_out, U
    df_input.loc[technologies, "eff_out"] = df_input.loc[technologies, "eff_in"]

    # loss_per_hr, V, EXCEL comment in column no formulas here
    # p_to_e, W, EXCEL comment in column no formulas here

    # varom, X, EXCEL comment, here column B is used in the Match statement in the Excel file
    # TODO: write that into notebook  
    # For-loop, for each row and check if there is a row in the storage cost dataframe
    # Or https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#selection-by-callable combined with https://pandas.pydata.org/pandas-docs/stable/user_guide/gotchas.html#using-the-in-operator
    # Or a feature not a bug, right now the rows in store and storage costs need to be identical
    df_input.loc["PumpedHydro":"H2-Salt-CCGT", "varom"] = df_storage_costs.loc["PumpedHydro":"H2-Salt-CCGT", "power varom"]
    here crash the program, the line above

    # fom, Y
    # TODO: write that into notebook
    # Also this can't handle if there is any missing values in the storage costs sheet, maybe the easiest is just to hard code a list of row labels?
    df_input.loc["PumpedHydro":"H2-Salt-CCGT", "fom"] = np.round(df_storage_costs.loc["PumpedHydro":"H2-Salt-CCGT", "power FOM"])

    # Same as for varom and fom, PumpedHydro:H2-Salt-CCGT
    # e capex2050, AH
    df_input.loc["PumpedHydro":"H2-Salt-CCGT", "e capex2050"] = np.round(df_storage_costs.loc["PumpedHydro":"H2-Salt-CCGT", "Storage (annualised)"], params["e capex rounding index"])

    # Same as for varom and fom, PumpedHydro:H2-Salt-CCGT
    # p capex2050, AQ
    df_input.loc["PumpedHydro":"H2-Salt-CCGT", "p capex2050"] = np.round(df_storage_costs.loc["PumpedHydro":"H2-Salt-CCGT", "PCS (annualised)"], params["e capex rounding index"])
    return df_input

def transmission_costs_formulas(df_input, config = None):
    params = config["transmission_costs"]
    # Line Capex £m (Fixed + variable build costs + O&M), H
    df_input.loc["HVDCSubsea", "Line Capex £m (Fixed + variable build costs + O&M)"] = 837.1 + 358.8 + 221.7
    df_input.loc["HVAC", "Line Capex £m (Fixed + variable build costs + O&M)"] = 118.2 + 4.3

    # Capex £k per MW-100km, I
    # =(H5/C5/E5)*(1000*100)
    df_input.loc["HVDCSubsea":"HVAC", "Capex £k per MW-100km"] = (
            (df_input.loc["HVDCSubsea":"HVAC", "Line Capex £m (Fixed + variable build costs + O&M)"] / df_input.loc["HVDCSubsea":"HVAC", "Distance"] / df_input.loc["HVDCSubsea":"HVAC", "Capacity"]) *
            (1000 * 100)
            )
    df_input.loc["HVDCSubsea", "Capex £k per MW-100km"] = 240 * params["EUR16 to GBP16"] * params["GBP16 to GBP10"] / 10

    # Annualised, J (Line Capex?)
    df_input.loc["HVDCSubseaOLD":"HVDCSubsea", ] = (
            df_input.loc["HVDCSubseaOLD":"HVDCSubsea", "Capex £k per MW-100km"] * params["discount rate"] / (1 - (1 + params["discount rate"])**(-df_input.loc["HVDCSubseaOLD":"HVDCSubsea", "Lifetime"]))
            )

    # Sub Capex (£k/MW)
    df_input.loc["HVAC":"HVDCSubsea", "Sub Capex (£k/MW)"] = 38.8 * params["EUR16 to GBP16"] * params["GBP16 to GBP10"]
    df_input.loc["HVAC":"HVDCSubsea", "Sub Capex (£k/MW)"] = 121 * params["EUR16 to GBP16"] * params["GBP16 to GBP10"]
    df_input.loc["HVDCSubseaOLD", "Sub Capex (£k/MW)"] = df_input.loc["HVDCSubsea", "Sub Capex (£k/MW)"]

    # Annualised, L (Sub Capex?)
    df_input.loc["HVDCSubseaOLD":"HVDCSubsea", ] = (
            df_input.loc["HVDCSubseaOLD":"HVDCSubsea", "Sub Capex (£k/MW)"] * params["discount rate"] / (1 - (1 + params["discount rate"])**(-df_input.loc["HVDCSubseaOLD":"HVDCSubsea", "Lifetime"]))
            )

    # FOM, M
    df_input.loc["HVAC":"HVDCSubsea", "FOM"] = df_input.loc["HVAC":"HVDCSubsea", "Capex £k per MW-100km"] * params["unknown constant"]
    return df_input


def transmission_formulas(df_input, config = None):
    # Formulas from ods
    # line_capex, C
    df_input.loc['HVAC Overhead line':'HVDC Subsea', 'line_capex'] = np.round(df_transmission_costs.loc['HVAC':'HVDCSubsea', 'Annualised'], 2)

    # sub_capex, D
    df_input.loc['HVAC Overhead line':'HVDC Subsea', 'sub_capex'] = np.round(df_transmission_costs.loc['HVAC':'HVDCSubsea', 'sub_capex'], 2)
    return df_input
    
def transmission_recreate(fpath, fpath_costs):
    # Recreate spreadsheet from raw csv versions
    # fpath is the path to transmission.csv
    # fpath_costs to transmission_costs.csv
    df = pd.read_csv(fpath)
    df_costs = pd.read_csv(fpath_costs)
    df, df_costs = transmission_formulas(df, df_costs)
    return df, df_costs
    

def proc_transmission(fname, sheets, recalculate=False, dir = None):
    # transmission* sheets and dependencies
    # Recalculate True is for testing purposes
    print('Transferring transmission data from ods to csv. Recalulate =', recalculate)

    out_dir = dir + '/transmission/'
    print('Writing to', out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    ############
    sheet = 'transmission_allowed'
    print('Transfering', sheet, 'sheet to .csv' )
    df_tc = pd.read_excel(fname, sheet_name=sheet, usecols="A:E")
    df_tc.to_csv(out_dir + sheet + '.csv')
    ############
    
    sheet = 'transmission_costs'
    print('Transfering', sheet, 'sheet to .csv' )
    df_tc = pd.read_excel(fname, sheet_name=sheet, skiprows=sheets[sheet])
    
    df_tc = df_tc.set_index('Type')

    if recalculate:
        # Formulas from ods
        df_tc = transmission_costs_formulas(df_tc)
        df_tc_write = df_tc
    else:
        # Remove derived columns in spreadsheet
        df_tc_write = df_tc.drop(columns=['Capex £k per MW-100km',
                                         'Sub Capex (£k/MW)', 'Annualised.1', 'FOM'])
    
    df_tc_write.to_csv(out_dir + sheet + '.csv')
    ############
    
    ############
    sheet = 'transmission'
    print('Transfering', sheet, 'sheet to .csv' )
    df = pd.read_excel(fname, sheet_name=sheet, header=1, index_col=0)
    if recalculate:
        # Spreadsheet formulas
        df, df_tc = transmission_formulas(df, df_tc)
        df_write = df
    else:
        # Remove derived columns
        df_write = df.drop(columns=['line_capex', 'sub_capex'])
    df_write.to_csv(out_dir + sheet + '.csv')
    return df_tc, df
    ############


def proc_fuelcosts(fname, dir = None):
    print('Mapping the fuelcosts sheet to multiple tables')

    outdir = dir + '/fuelcosts/'
    print('Writing to', outdir)
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    sheet='fuelcosts'
    
    df = pd.read_excel(fname, sheet_name=sheet,
                       skiprows=4,
                       usecols='B:N',
                       nrows=7,
                       index_col=0
                       )
    outname = '/fuelcost20xx.csv'
    df.to_csv(outdir+outname)
    
    df = pd.read_excel(fname, sheet_name=sheet,
                       skiprows=4,
                       usecols='B, Q:AA',
                       nrows=7,
                       index_col=0
                       )
    outname = '/fuelcostNoCCS.csv'
    df.columns = df.columns.str.replace('\.1', '', regex=True)
    df.to_csv(outdir+outname)
    
    # The rest have no row index, but seem to be the same shape as above 
    df = pd.read_excel(fname, sheet_name=sheet,
                       skiprows=list(range(0,4)) + list(range(5,15)),
                       usecols='Q:AA',
                       nrows=7,
                       )
    outname = '/fuelcostUKTM_1.2.2BAU.csv'
    df.columns = df.columns.str.replace('\.1', '', regex=True)
    df.to_csv(outdir+outname)

    df = pd.read_excel(fname, sheet_name=sheet,
                       skiprows=list(range(0,4)) + list(range(5,25)),
                       usecols='Q:AA',
                       nrows=7,
                       )
    outname = '/fuelcostImportPrice.csv'
    df.columns = df.columns.str.replace('\.1', '', regex=True)
    df.to_csv(outdir+outname)

    df = pd.read_excel(fname, sheet_name=sheet,
                       skiprows=list(range(0,35)),
                       usecols='Q:AA',
                       nrows=7,
                       )
    outname = '/fuelcostNOCCS_WRONG.csv'
    df.to_csv(outdir+outname)

    
def main():
    # Where the input data are
    wd_initial = os.getcwd()
    mainpath = "/home/vetle/sommerjobb2023/joint-wind/"
    # Source .ods file
    fname = mainpath + "highres_gb_ext_database.ods"

    sheets = {'gen': 1,
    'store': 1,
    'gen_lim_z': 0,
    'gen_exist_z': 0,
    'store_lim_z': 0,
    'store_exist_z': 0,
    'transmission': 1,
    'transmission_costs': 3,
    'gen_2': 1,
    'transmission_allowed': 1,
    'transmission_exist_TODO': 1,
    'economic_conversion_factors': 1,
    'storage_costs': 1,
    'generator_costs': 3, #????????
    'uktm_emis_factors': 3, #???????
    'fuelcosts': 2}
    # sheetname: lines_to_skip
    # This number is a bit of a guess, but correct for sheets read by data2dd
    # Some sheets consists of multiple tables

    # Sheets that are actually read in data2dd, just noting it down, not really used here:
    read_sheets = ["gen", "store", "*_lim_z", "*_exist_z", "transmission_allowed", "transmission"]

    # Read parameters from yaml file
    # Import transmission_costs_params as tcp
    with open("ODSparams.yml", 'r') as file:
        ODSparams = yaml.safe_load(file)
    # print(ODSparams)

    # My stuff:
    fuelcost_default_scenarious_formulas()
    generator_costs_formulas()
    gen_formulas()
    storage_costs_forumlas()
    store_formulas()


if __name__ == "__main__":
    main()
    
