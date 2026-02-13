[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/seqwin/README.html)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/seqwin/badges/downloads.svg)](https://anaconda.org/bioconda/seqwin)

# Seqwin

**Seqwin** is a lightning‑fast, memory‑efficient toolkit for discovering **signature sequences** (genomic markers) that balance **high sensitivity** with **high specificity**. It builds a minimizer‑based pan‑genome graph across target and neighboring non‑target genomes and extracts signature sequences using a novel graph algorithm. Signatures can be used for downstream assay design such as qPCR, ddPCR, amplicon sequencing and hybrid capture probes. 

> [!NOTE]
> **Citation:** If you use Seqwin in your research, please cite: 
> 
> **Michael X. Wang, Bryce Kille, Michael G. Nute, Siyi Zhou, Lauren B. Stadler, and Todd J. Treangen** ["Seqwin: Ultrafast identification of signature sequences in microbial genomes"](https://doi.org/10.1101/2025.11.07.687294). *bioRxiv* (2025).

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Key parameters](#key-parameters)
4. [Outputs](#outputs)

## Installation

### Bioconda (recommended)

**Prerequisites**
- Linux, macOS, or Windows via [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- x64 or ARM64
- conda (install with [miniforge](https://github.com/conda-forge/miniforge#install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions))

**1. Create a new Conda environment "seqwin" and install Seqwin via Bioconda**
```bash
conda create -n seqwin seqwin \
  --channel conda-forge \
  --channel bioconda \
  --strict-channel-priority
```
> [!TIP]
> Setting channel priority is important for Bioconda packages to function properly. You may also persist channel priority settings for all package installation by modifying your `~/.condarc` file. For more information, check the [Bioconda documentation](https://bioconda.github.io/). 

**2. Activate the environment and verify the install**
```bash
conda activate seqwin
seqwin --help
```

### Manual installation

**1. Install dependencies**

> [python](https://www.python.org/) >=3.10  
> [numpy](https://numpy.org/) >=2  
> [numba](https://numba.pydata.org/)  
> [pandas](https://pandas.pydata.org/) >=2  
> [networkx](https://networkx.org/)  
> [pydantic](https://docs.pydantic.dev/latest/)  
> [typer](https://typer.tiangolo.com/)  
> [btllib](https://github.com/bcgsc/btllib)  
> [mash](https://github.com/marbl/Mash)  
> [blast](https://www.ncbi.nlm.nih.gov/books/NBK279690/)  
> [ncbi-datasets-cli](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/)

**2. Clone this repository and install with `pip`**
```bash
git clone https://github.com/treangenlab/Seqwin.git
cd Seqwin
pip install .
seqwin --help
```

## Quick start

Identify signatures by providing one or more target taxa and non-target neighboring taxa. 
```bash
seqwin \
  -t "Salmonella enterica subsp. diarizonae" \
  -n "Salmonella enterica subsp. salamae" \
  -n "Salmonella bongori" \
  --threads 8
```
Outputs are written to `seqwin-out/` in your working directory (see [Outputs](#outputs)). Taxa names must be exact matches to [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/datasets/taxonomy/tree/). 

Alternatively, a list of target or non-target genomes can be provided as a text file of file paths. Each line of the text file should be the path to a genome file in FASTA format (plain text or compressed in gzip). 
```bash
seqwin --tar-paths targets.txt --neg-paths non-targets.txt
```
Below is an example of `targets.txt` or `non-targets.txt`
```bash
./genomes/GCA_003718275.1_ASM371827v1_genomic.fna
/data/genomes/GCA_000389055.1_46.E.09_genomic.fna
/data/genomes/GCA_008363955.1_ASM836395v1_genomic.fna.gz
```

Expected runtime (with `--threads 8` or `-p 8`): 
- ~5min and 2.5GB peak RAM for ~500 bacterial genomes with default settings. 
- ~12min and 23GB peak RAM for ~15k bacterial genomes with `--no-blast` and `--no-mash`. 

Run `seqwin --help` or `seqwin -h` to see the full command line interface. 

## Key parameters

### Node penalty threshold
The node penalty threshold (`--penalty-th`) controls the sensitivity and specificity of output signatures. Higher values allow longer / more signatures, but might reduce sensitivity and/or specificity. 

When `--penalty-th` is not specified, it is automatically estimated with k-mer sketches. MinHash sketches (calculated with [Mash](https://doi.org/10.1186/s13059-016-0997-x)) are used by default. If `--no-mash` is provided, minimizer sketches are used instead (faster but might be biased). Use `--stringency` or `-s` to tune this auto-estimated threshold (0-10, default 5, higher stringency lowers the threshold). 

### Signature evaluation

By default, output signatures are BLAST checked against target genomes for sensitivity (`conservation`), and non-target genomes for specificity (`divergence`). Signatures are sorted by `conservation` and `divergence`, which can be found in `signatures.csv`. Evaluation can be turned off with `--no-blast` for shorter running time. In that case, output signatures are still very likely to be sensitive and specific, but without second validation of BLAST. 

### Minimizer sketch
`--kmerlen` or `-k` (default 21): shorter k‑mers might be helpful for genomes with more sequence variations (e.g. set to 17 for viruses). 

`--windowsize` or `-w` (default 200): smaller windows generate more minimizers and increase resolution (e.g., to find shorter signatures), at the cost of runtime & memory. 

### Performance tuning
Use `--threads` or `-p` to leverage multiple CPU cores (4 by default). When the number of input genomes is large, add `--no-mash` and `--no-blast` for fastest running time. 
- [Mash](https://doi.org/10.1186/s13059-016-0997-x) takes quadratic time as the number of genomes grows. With `--no-mash`, minimizer sketches are used to calculate node penalty threshold. 
- Signatures are evaluated by building a BLAST database of all input genomes. With `--no-blast`, evaluation is skipped and `conservation` and `divergence` won't be calculated in `signatures.csv`. 

## Outputs
Seqwin creates the following files/directories inside the directory specified by `--title` or `-o` (default `seqwin-out/`):
| Name | Description|
| :-------  | :-------- |
| `signatures.fasta`| Signature sequences (top candidates are listed first) |
| `signatures.csv`| Tabulated metrics for each signature |
| `assemblies.csv`| Mapping of internal genome IDs to file paths (used in `signatures.fasta`) |
| `blastdb/`| BLAST database built from all input genomes |
| `assemblies/`| Genomes downloaded from NCBI |
| `results.seqwin`| Serialized run snapshot (Python pickle) |
| `config.json`| Full run configuration |
| `seqwin.log`| Execution log |
