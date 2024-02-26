import pandas as pd
resultspath = snakemake.params.modelpath + '/gen_results.tsv'

norway_counties = [
    'NO020',
    'NO060',
    'NO071',
    'NO074',
    'NO081',
    'NO082',
    'NO091',
    'NO092',
    'NO0A1',
    'NO0A2',
    'NO0A3',
]

def extract_gen_sum_nordic(country):
    """
    Extract the total electricity generated from iteration 1 for non-Norwegian
    Nordic countries. 
    """
    x = (
            pd.read_csv(resultspath,sep='\t')
            .query("zone == @country")
            .electricity_generation_GWh.values[0]
        )
    return x

def extract_gen_sum_norway():
    """
    Extract the total electricity generated from iteration 1 for Norway.
    """
    x = (
            pd.read_csv(resultspath,sep='\t')
            .query("zone.str.contains('NO')")
            .electricity_generation_GWh.sum()
        )
    return x

FI_gen = extract_gen_sum_nordic('FI')
SE_gen = extract_gen_sum_nordic('SE')
DK_gen = extract_gen_sum_nordic('DK')
NO_gen = extract_gen_sum_norway()

def build_gams():
    """
    Crudely modify the GAMS code textfile, will break if line numbers change
    """
    with open(snakemake.input[0], "r", encoding="utf8") as file:
        list_of_lines = file.readlines()

    list_of_lines[80] = '$setglobal outname "results_i2"\n'
    list_of_lines.insert(
        174,
        'set norway_count(z) / NO020,NO060,NO071,NO074,NO081,NO082,NO091,NO092,NO0A1,NO0A2,NO0A3 /;\n'
    )

    list_of_lines.insert(
        464,
        'eq_neo_export_SE \neq_neo_export_FI \neq_neo_export_DK \neq_neo_export_NO \n'
    )
    
    list_of_lines.insert(
        639,
        '\neq_neo_export_SE.. sum((h,gen_lim("SE",g)),var_gen(h,"SE",g)) =G= ' + str(round(SE_gen,1)) + '+ 5000;\n'
        )
    list_of_lines.insert(
        640,
        'eq_neo_export_FI.. sum((h,gen_lim("FI",g)),var_gen(h,"FI",g)) =G= ' + str(round(FI_gen,1)) + ' + 5000;\n'
        )
    list_of_lines.insert(
        641,
        'eq_neo_export_DK.. sum((h,gen_lim("DK",g)),var_gen(h,"DK",g)) =G= ' + str(round(DK_gen,1)) + ' + 10000;\n'
        )
    list_of_lines.insert(
        642,
        'eq_neo_export_NO.. sum((h,gen_lim(norway_count(z),g)),var_gen(h,z,g)) =G= ' + str(round(NO_gen,1)) + ' + 10000;\n'
        )

    with open(snakemake.output[0], "w", encoding="utf8") as file:
        list_of_lines = "".join(list_of_lines)
        file.write(list_of_lines)

build_gams()
