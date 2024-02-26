import pandas as pd

pd.read_csv(snakemake.input[0], index_col=["time", "technology", "spatial"]).to_parquet(
    snakemake.output[0], compression="zstd"
)
