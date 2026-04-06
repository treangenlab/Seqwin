[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](https://bioconda.github.io/recipes/seqwin/README.html)
[![conda downloads](https://img.shields.io/conda/d/bioconda/seqwin)](https://anaconda.org/bioconda/seqwin)
[![pypi version](https://img.shields.io/pypi/v/seqwin?color=blue)](https://pypi.org/project/seqwin/)
[![supported platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-blue)](https://pypi.org/project/seqwin/)
[![Build and Test](https://github.com/treangenlab/Seqwin/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/treangenlab/Seqwin/actions/workflows/main.yml)

# Seqwin

**Seqwin** is a lightning‑fast, memory‑efficient toolkit for discovering **signature sequences** (genomic markers) that balance **high sensitivity** with **high specificity**. It builds a minimizer‑based pan‑genome graph across target and neighboring non‑target genomes and extracts signature sequences using a novel graph algorithm. Signatures can be used for downstream assay design such as qPCR, ddPCR, amplicon sequencing, and hybrid capture probes. 

Seqwin computes minimizers with [ntHash](https://doi.org/10.1093/bioinformatics/btw397), using code adopted from [btllib](https://github.com/bcgsc/btllib) (licensed under the GNU General Public License v3.0). 

---

## Table of contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Citation](#citation)

See the [Seqwin Wiki](https://github.com/treangenlab/Seqwin/wiki) for full documentation. 

## Installation
Seqwin can be installed from **Bioconda** or **PyPI**. 

- **Bioconda** is the recommended installation method because it installs Seqwin with all dependencies, but it requires Conda and supports only Linux and macOS. 
- **PyPI** (`pip install seqwin`) supports Windows (x64), Linux, and macOS, but installs only Seqwin and its Python dependencies. Non-Python dependencies can be installed separately if needed. 

### Bioconda (recommended)
Works on Linux (x64 / arm64) and macOS (Intel / Apple Silicon). 

If Conda is not installed, install it with [Miniforge](https://github.com/conda-forge/miniforge#install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions). 

**1. Create a new Conda environment named `seqwin` and install Seqwin via [Bioconda](https://bioconda.github.io/index.html)**
```bash
conda create -n seqwin seqwin \
  --channel conda-forge \
  --channel bioconda \
  --strict-channel-priority
```

**2. Activate the environment and verify the install**
```bash
conda activate seqwin
seqwin --help
```

### PyPI
Works on Windows (x64), Linux (x64 / arm64), and macOS (Intel / Apple Silicon). Requires Python >= 3.10. 

**1. Install Seqwin from PyPI**
```bash
python -m pip install --upgrade pip
python -m pip install --prefer-binary seqwin
seqwin --help
```

**2. Install non-Python dependencies (optional)**  
Seqwin can run without these tools, but some features will be unavailable or skipped. See the [Command Line Parameters](https://github.com/treangenlab/Seqwin/wiki/Command-Line-Parameters) for details. 
- [Mash](https://github.com/marbl/Mash) (minimizer sketches are used if it is not installed)
- [NCBI BLAST+](https://www.ncbi.nlm.nih.gov/books/NBK279690/) (needed for signature evaluation)
- [NCBI Datasets CLI](https://www.ncbi.nlm.nih.gov/datasets/docs/v2/reference-docs/command-line/datasets/) (needed for downloading NCBI genomes)

## Quick start

Identify signatures by providing one or more target taxa (`-t`) and neighboring non-target taxa (`-n`). 
```bash
seqwin \
  -t "Salmonella enterica subsp. diarizonae" \
  -n "Salmonella enterica subsp. salamae" \
  -n "Salmonella bongori" \
  --threads 8
```
**Taxa names must be exact matches to [NCBI Taxonomy](https://www.ncbi.nlm.nih.gov/datasets/taxonomy/tree/)**. Genomes under each taxon will be downloaded automatically. 

Outputs are written to `seqwin-out/` in your working directory (see [Description of Outputs](https://github.com/treangenlab/Seqwin/wiki/Description-of-Outputs)). 

Alternatively, a list of target or non-target genomes can be provided as a text file of file paths. Each line should be the path to a genome FASTA file (plain text or gzipped). 
```bash
seqwin --tar-paths targets.txt --neg-paths non-targets.txt
```
Examples can be found under [`test/`](test/). Use the [test script](test/run_test.py) to download and run the test dataset. 
```bash
git clone https://github.com/treangenlab/Seqwin.git
cd Seqwin/test/
python run_test.py
```

Expected runtime (with `--threads 8` or `-p 8`): 
- ~5 min and 2.5 GB peak RAM for ~500 bacterial genomes with default settings. 
- ~5 min and 23 GB peak RAM for ~15k bacterial genomes with `--no-blast` and `--no-mash`. 

Run `seqwin --help` or `seqwin -h` to see the full command line interface. 

## Citation

If you use Seqwin in your research, please cite: 

**Michael X. Wang, Bryce Kille, Michael G. Nute, Siyi Zhou, Lauren B. Stadler, and Todd J. Treangen** ["Seqwin: Ultrafast identification of signature sequences in microbial genomes"](https://doi.org/10.1101/2025.11.07.687294). *Proceedings of ISMB 2026*, accepted (2026). 

Benchmarking datasets, outputs, and scripts are available on [Zenodo](https://doi.org/10.5281/zenodo.19176444). 
