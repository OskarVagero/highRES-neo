with open(snakemake.input[0], "r") as f:
    list_of_lines = f.readlines()
    list_of_lines.pop(0)

newlines = []
for line in list_of_lines:
    newlines.extend(["HydroRoR." + line[:-1] + "." + line[:-1] + " inf\n"])

with open(snakemake.output["areashydro"], "w") as f:
    f.writelines(sorted(newlines))
