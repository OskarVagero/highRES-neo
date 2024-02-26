with open(snakemake.input[0], "r") as f:
    list_of_lines = f.readlines()
    list_of_lines.pop(0)

with open(snakemake.output[0], "w") as f:
    f.writelines(sorted(list_of_lines))
