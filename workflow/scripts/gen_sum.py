import sqlite3
import pandas as pd

#print(snakemake.params.modelpath + 'results.db')
resultspath = snakemake.params.modelpath + '/results.db'
nordics = ['SE','DK','FI']
gen = []
con = sqlite3.connect(resultspath)
gen.append(
    pd
    .read_sql_query("SELECT * from var_gen", con)
    .drop(columns={'lo','up','marginal'})
    .set_index(['z','h','g'])
    .stack()
    .reset_index()
    .rename(columns={
                "z": "zone",
                "g": "technology",
                "h": "hour",
                "level_3" : "type",
                0: 'electricity_generation_GWh'
            })
    .astype({'hour' : int})
    .set_index(['zone','technology','hour','type'])
    .sort_index())
con.close()
df_gen = pd.concat(gen)
del(gen)

(
    df_gen
    .query("zone == @nordics | zone.str.contains('NO')")
    .groupby('zone').sum()
    .to_csv(snakemake.output.gen_results,sep='\t')
)
