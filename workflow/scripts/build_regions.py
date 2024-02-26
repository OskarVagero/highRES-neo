with open(snakemake.input["zonescsv"], "r", encoding="utf8") as f:
    list_of_lines = f.readlines()
    list_of_lines.pop(0)

if snakemake.wildcards.spatial == "grid":
    with open(snakemake.input["indreg"], "r", encoding="utf8") as f:
        list_of_lines.extend(f.readlines())

with open(snakemake.output["regionsdd"], "w", encoding="utf8") as f:
    f.writelines(sorted(list_of_lines))
