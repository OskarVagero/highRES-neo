"""
Change different parameters in the GAMS code
"""


def build_gams(year):
    """
    Crudely modify the GAMS code textfile, will break if line numbers change
    """
    with open(snakemake.input[0], "r", encoding="utf8") as file:
        list_of_lines = file.readlines()
    list_of_lines[
        45
    ] = f'$setglobal codefolderpath "{snakemake.params.sharedcodepath}"\n'
    list_of_lines[70] = f'$setglobal weather_yr "{year}"\n'
    list_of_lines[71] = f'$setglobal dem_yr "{year}"\n'
    list_of_lines[
        636
    ] = f"sum((z,h)$(hr2yr_map(yr,h)),demand(z,h))*{snakemake.params.co2intensity}"
    with open(snakemake.output[0], "w", encoding="utf8") as file:
        file.writelines(list_of_lines)


build_gams(snakemake.wildcards.year)
