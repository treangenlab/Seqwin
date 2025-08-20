# Seqwin

**Seqwin** is a lightning‑fast, memory‑efficient toolkit for discovering **signature sequences** (genomic markers) that balance **high sensitivity** with **high specificity**. It builds a minimizer‑based pan‑genome graph across target and neighboring non‑target genomes and extracts signature sequences using a novel graph algorithm. 

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Parameters](#parameters)
4. [Outputs](#outputs)
5. [Python APIs](#python-apis)
5. [License](#license)

## Installation

### Bioconda (pending)

#### Prerequisites
- Linux, MacOS or [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- x64 or ARM64
- conda (install with [miniforge](https://github.com/conda-forge/miniforge#install) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions))

#### 1. Create a new Conda environment called "seqwin" and install Seqwin via Bioconda
```bash
conda create -n seqwin seqwin \
  --channel conda-forge \
  --channel bioconda \
  --strict-channel-priority
```
> [!TIP]
> Setting channel priority is important for Bioconda packages to function properly. You may also persist channel priority settings for all package installation by modifying your `~/.condarc` file. For more information, check the [Bioconda documentation](https://bioconda.github.io/). 

#### 2. Activate the environment and verify the install
```bash
conda activate seqwin
seqwin --help
```

### PypI

#### 1. Install dependencies
```
python >=3.10
numpy
numba
pandas
networkx
pydantic
typer
btllib
mash
blast
ncbi-datasets-cli
```

#### 2. Clone this repository and install with `pip`
```bash
git clone https://github.com/treangenlab/Seqwin.git
cd Seqwin
pip install .
seqwin --help
```

## Quick start

Identify markers by providing a target taxonomy name and one or more non-target neighbors. 
```bash
seqwin \
  -t "Salmonella enterica subsp. enterica" \
  -n "Salmonella enterica subsp. salamae" \
  -n "Salmonella bongori" \
  -p 8
```
Outputs are written to `seqwin-out/` in your working directory. Names must be exact matches (search via [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/datasets/taxonomy/tree/)). 

Alternatively, a list of target or non-target genomes can be provided as a text file of file paths. Each line of the text file should be a path to a genome file in FASTA format (plain text or compressed in gzip). 
```bash
seqwin --tar-paths targets.txt --neg-paths neighbors.txt
```

Expected runtime: ~10min for 550+ bacterial genomes using 20 threads. 

Run `seqwin --help` to see the full command line interface. 

## Parameters

### Node penalty threshold (`--penalty-th`)
Automatically calculated with [Mash](https://doi.org/10.1186/s13059-016-0997-x) by default. 
Controls the sensitivity and specificity of output markers (default 0.2). Higher values allow longer / more markers, but might reduce sensitivity and/or specificity. Note that there's no direct mathematical relationship between sensitivity / specificity and the penalty threshold. Recommended range: 0 - 0.2. 

### Minimizer sketch
`--kmerlen` (default 21): shorter k‑mers may help highly variable genomes (e.g. viruses). 

`--windowsize` (default 200): smaller windows generate more minimizers and increase resolution at the cost of runtime & memory. 

### Performance tuning
Use `--threads / -p` to leverage multiple CPU cores. You can also disable BLAST entirely (`--no-blast`) or limit it to non‑targets (`--fast-blast`) for quicker marker evaluation. 

## Outputs
Seqwin creates the following files/directories inside the directory specified by `--title` (default `seqwin-out/`):
| Name &nbsp; &nbsp; &nbsp; | Description|
| :-------  | :-------- | 
| `markers.fasta`| Candidate marker sequences sorted by `purity + divergence` |
| `markers.csv`| Tabulated metrics for each marker |
| `assemblies.csv`| Mapping of internal genome IDs to file paths (used in `markers.fasta`) |
| `blastdb/`| Generated BLAST database. |
| `assemblies/`| Genomes downloaded from NCBI |
| `results.seqwin`| Serialized run snapshot (pickle) |
| `config.json`| Full run configuration |
| `seqwin.log`| Execution log |

## Python APIs

## License

Seqwin is released under the **GPL 3.0**. See [LICENSE](LICENSE) for details. 
