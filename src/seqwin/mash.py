"""
Mash
====

Assembly distance estimation with `Mash <https://doi.org/10.1186/s13059-016-0997-x>`__. 

Dependencies:
------------
- pandas
- mash
- .utils

Functions:
----------
- sketch
- dist
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import logging, shutil, subprocess
from pathlib import Path
from io import StringIO
from collections.abc import Iterable, Generator

logger = logging.getLogger(__name__)

import pandas as pd
from .utils import log_and_raise, file_to_write, run_cmd

_MASH_SKETCH_EXT = '.msh' # File extension of Mash output. [.msh]
_STDIN = Path('/dev/stdin')
_DIST_COL = ('ref', 'query', 'dist', 'pval', 'jaccard') # output columns of `mash dist`

if shutil.which('mash') is None:
    raise ImportError('Mash is not installed (`mash` is not found in your PATH).')


def sketch(
    assembly_path: Path | Iterable[Path], 
    kmerlen: int=21, 
    sketchsize: int=1000, 
    out_path: Path | None=None, 
    overwrite: bool=False, 
    n_cpu: int=1
) -> Path:
    """Create a Mash / MinHash sketch for a single assembly, or a merged sketch for multiple assemblies. 

    Args:
        assembly_path (Path | list[Path]): Path to the assembly file in FASTA format, or a list of assembly paths. 
        kmerlen (int, optional): k-mer length. [21]
        sketchsize (int, optional): Sketch size (number of non-redundant k-mers). [1000]
        out_path (Path | None, optional): Output path for the sketch file. 
            If None, output to `assembly_path + '.msh'` or `assembly_path[0] + '.msh'`. [None]
        overwrite(bool, optional): If True, overwrite the existing Mash sketch file. [False]
        n_cpu (int, optional): Number of processes to run in parallel. [1]

    Returns:
        Path: Path to the Mash sketch file (.msh). 
    """
    args = [
        'mash', 'sketch', 
        '-k', str(kmerlen), 
        '-s', str(sketchsize), 
        '-p', str(n_cpu)
    ]

    # determine input type
    if isinstance(assembly_path, Path):
        # single assembly
        args.append(assembly_path)
        stdin = None
        log_text = f' - Generating MinHash sketch with Mash for {assembly_path}'
    elif isinstance(assembly_path, Iterable):
        assembly_path = list(assembly_path)
        # multiple assemblies (feed all paths to stdin)
        args += ['-l', _STDIN]
        stdin = '\n'.join(map(str, assembly_path))
        log_text = f' - Generating MinHash sketches with Mash for {len(assembly_path)} assemblies...'
        assembly_path = assembly_path[0]
    else:
        log_and_raise(ValueError, 'Invalid assembly_path for mash_sketch, must be str or list.')

    # determine output path (Mash appends .msh to the output path)
    if out_path is None:
        # use assembly_path as output path
        real_out_path = assembly_path.with_name(assembly_path.name + _MASH_SKETCH_EXT)
        out_path = assembly_path
        logger.warning(f' - mash sketch -o is not provided, output to {real_out_path}')
    elif out_path.suffix == _MASH_SKETCH_EXT:
        # remove .msh from out_path, or there will be two of them
        real_out_path = out_path
        out_path.with_suffix('')
    else:
        real_out_path = out_path.with_name(out_path.name + _MASH_SKETCH_EXT)
    # check if file exists
    file_to_write(real_out_path, overwrite)
    args += ['-o', out_path]

    logger.info(log_text)
    run_cmd(*args, stdin=stdin, raise_error=True)
    logger.info(f' - Mash sketch file saved as {real_out_path}')
    return real_out_path


def dist(ref_path: Path, query_path: Path | None=None, n_cpu: int=1) -> pd.DataFrame:
    """Run `mash dist {ref_path} {query_path}` and parse the TSV output as a pandas DataFrame. 
    Note: high memory usage when the number of assemblies in the sketch file is large. 
    
    Args:
        ref_path (Path): Path to the reference sketch file. It could include multiple sketches, merged with the 'mash paste' command. 
        query_path (Path | None, optional): Path to the query sketch. If None, query ref_path against itself. [None]
        n_cpu (int, optional): Number of processes to run in parallel. [1]

    Returns:
        pd.DataFrame: Tabular output of `mash dist`, each row is an assembly pair, 
            with columns ['ref', 'query', 'dist', 'pval', 'jaccard', 'shared', 'total']. 
    """
    if query_path is None:
        query_path = ref_path

    logger.info(' - Calculating Mash distances of assembly pairs...')
    cmd_out = run_cmd(
        'mash', 'dist', 
        '-p', str(n_cpu), 
        ref_path, 
        query_path
    )
    # parse stdout; pd.read_csv() does auto type conversion
    df = pd.read_csv(StringIO(cmd_out.stdout), sep='\t', header=None, names=_DIST_COL, index_col=False)
    df[['shared', 'total']] = df['jaccard'].str.split('/', expand=True).astype('int64')
    df['jaccard'] = df['shared'] / df['total']
    return df


def get_jaccard(ref_path: Path, query_path: Path | None=None, n_cpu: int=1, bufsize: int=1_000_000) -> Generator[float, None, None]:
    """Estimate the pairwise Jaccard index between (each assembly sketch in the the query) 
    and (each assembly sketch in the reference), with `mash dist`. Use a stream pipe to save memory. 
    
    Args:
        ref_path (Path): Path to the reference sketch file. It could include multiple sketches, merged with the 'mash paste' command. 
        query_path (Path | None, optional): Path to the query sketch. If None, query ref_path against itself. [None]
        n_cpu (int, optional): Number of processes to run in parallel. [1]
        bufsize (int, optional): `bufsize` option for `subprocess.Popen`. [1_000_000]

    Yields:
        float: Jaccard index of each assembly pair. 
    """
    if query_path is None:
        query_path = ref_path

    logger.info(' - Calculating Jaccard indexes of assembly pairs...')
    proc = subprocess.Popen(
        ('mash', 'dist', '-p', str(n_cpu), ref_path, query_path), 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True, 
        bufsize=bufsize
    )

    try:
        # get the jaccard from mash output
        for line in proc.stdout:
            *_, jaccard = line.strip().split('\t')
            shared, total = map(int, jaccard.split('/'))
            yield shared / total
    finally:
        # make sure these lines are always executed no matter what happens in the try block
        # e.g., an parsing error of mash output
        proc.terminate() # make sure mash stops writing to stdout
        proc.stdout.close()
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            log_and_raise(RuntimeError, f"'mash dist' exited with code {proc.returncode}:\n{stderr}")
