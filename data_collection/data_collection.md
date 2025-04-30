# Data Collection

This [Zenodo link](https://zenodo.org/records/15313069) hosts the train/validation/test datasets we used in our publication. If you want to build off this work, downloading those datasets is the easiest way to get started. If instead you want to process the data upstream of those pre-partitioned datasets, this document partial instructions for downloading and wrangling the data.

## Data Sources

The source for the data is https://elifesciences.org/articles/73983. The inferred genotype and phenotype data is available for download here: https://datadryad.org/dataset/doi:10.5061/dryad.1rn8pk0vd.

## Steps

Download all of the files from https://datadryad.org/dataset/doi:10.5061/dryad.1rn8pk0vd and place them into a directory.

In order to have the same dataset used in the original publication the

- geno_data_1.txt.gz
- geno_data_2.txt.gz
- geno_data_3.txt.gz
- geno_data_4.txt.gz
- geno_data_5.txt.gz

files need to be combined into a single file. Then they need to be converted into a numpy format using the script `convert_geno_data.py`.

For example, let's say we have all of the file in `example_data`. Then we would run the following commands:

```bash
cat example_data/geno_data_*.txt.gz > example_data/geno_data.txt.gz

python convert_geno_data.py -i example_data/geno_data.txt.gz -o example_data/merged_geno_data.npy
```

The final step is to run the https://github.com/Emergent-Behaviors-in-Biology/GenoPhenoMapAttention/blob/main/obtain_independent_loci.ipynb notebook to obtain the "independent" loci, which creates the `ind_loci_list_3.npy` file.

The resulting files directly plug into the notebook files provided by Rijal et al: https://github.com/Emergent-Behaviors-in-Biology/GenoPhenoMapAttention/tree/main/experiment, which let us begin reproducing their work.

In their notebooks, Rijal et al partition the datasets into train/validation/test datasets on-the-fly, relying on consistent RNG seeding. We instead ran one of their notebooks, using the same RNG seeds, up until the data partitioning and standard normalization, then wrote the resulting numpy arrays to file, which is what we uploaded to Zenodo.
