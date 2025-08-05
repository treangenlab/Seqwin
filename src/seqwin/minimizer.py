"""
Minimizer
=========

Generate minimizer sketches with `btllib<https://github.com/bcgsc/btllib>`__. 

Dependencies:
-------------
- numpy
- pandas
- btllib
- .utils

Functions:
----------
- indexlr
- indexlr_py
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import shutil
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from btllib import Indexlr, IndexlrFlag

from .utils import run_cmd

_INDEXLR_COL = dict( # Header and data types for Indexlr output
    hash='uint64', # hash values should be converted to unsigned int64, otherwise an overflow error will be raised
    pos='uint32', 
)
_STRAND_TYPE = CategoricalDtype(('+', '-'), ordered=False) # saves memory
_IDX_TYPE = np.uint16 # dtype for record_idx

KMER_DTYPE = np.dtype([ # for indexlr_py()
    ('hash', np.uint64), 
    ('pos', np.uint32), 
    ('record_idx', np.uint16), 
    ('assembly_idx', np.uint16), 
    ('is_target', np.bool_), 
])

if shutil.which('indexlr') is None:
    raise ImportError('btllib is not installed (`indexlr` is not found in your PATH).')


def indexlr(
    assembly_path: Path, 
    kmerlen: int=21, 
    windowsize: int=200, 
    get_strand: bool=False, 
    get_seq: bool=False
) -> tuple[list[pd.DataFrame], tuple[str]]:
    """Get the minimizers for each record in an assembly with the `indexlr` command from `btllib<https://github.com/bcgsc/btllib>`__. 
    - The number of threads to use is set to 1. 
    - Indexlr seems to skip sequence regions with ambiguous bases. So long runs of 'N's will be ignored and k-mer coordinates will seem discontinuous. 
    (e.g., NZ_QJGJ01000182.1 in assembly GCF_003200995.1). 
    - We are not using the Indexlr Python module, because it needs the start method to be set to 'spawn' in order to do multiprocessing. 
    Check `Python docs <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`__ 
    for more details. 

    Args:
        assembly_path (Path): Path to the assembly file in FASTA format. 
        kmerlen (int, optional): k-mer length. [21]
        windowsize (int, optional): Window size for minimizer sketch. [200]
        get_strand (bool, optional): If True, include minimizer strand in the output. [False]
        get_seq (bool, optional): If True, include minimizer sequence in the output. [False]

    Returns:
        tuple: A tuple containing
            1. list[pd.DataFrame]: A list of pandas DataFrames for each sequence record in the assembly. 
                Each row represents a minimizer in the sequence record, with columns, 
                - 'hash' (uint64): Hash value of the minimizer. 
                - 'pos' (int64): Position of the first base of the minimizer. 
                - 'record_idx' (int): 0-based index of the sequence records, in the same order as they appear in the FASTA file. 
                - 'strand' (category, optional): Which strand (+/-) has the smaller hash value. Use category type to save memory. 
                - 'seq' (str, optional): sequence of the minimizer (forward strand), available if `get_seq=True`. 
            2. tuple[str]: Record IDs of the assembly. 
    """
    # run indexlr
    args = [
        'indexlr', 
        '-k', str(kmerlen), 
        '-w', str(windowsize), 
        '-t', '1', # threads
        '--pos', # output minimizer position
        '--long' # optimize for sketching long sequence
    ]
    columns = _INDEXLR_COL.copy() # columns of df output
    if get_strand: # output minimizer sequence
        # output which strand has smaller hash value
        # the actual hash value of a k-mer is the sum of its forward hash and reverse hash
        # so that a k-mer and its reverse complement have the same hash value
        args.append('--strand')
        columns['strand'] = _STRAND_TYPE
    if get_seq: # output minimizer sequence
        # output sequence is always the forward k-mer, even if the reverse k-mer has smaller hash value
        args.append('--seq')
        columns['seq'] = 'str'
    cmd_out = run_cmd(*args, assembly_path)

    # parse stdout and convert to dataframe (efficient and saves memory)
    df_assembly: list[pd.DataFrame] = list()
    idx_to_id: list[str] = list()
    idx_next = 0 # record index
    for line in cmd_out.stdout.split('\n'):
        # each line is a single record in the assembly
        try:
            record_id, minimizers = line.split('\t')
            idx_to_id.append(record_id)
            idx = idx_next
            idx_next += 1
        except ValueError:
            # last empty line
            continue

        if minimizers == '':
            # current record has no minimizer (shorter than windowsize)
            continue

        df_record = pd.DataFrame(
            (m.split(':') for m in minimizers.split(' ')), 
            columns=columns
        ).astype(columns, copy=False, errors='raise')

        df_record['record_idx'] = _IDX_TYPE(idx)
        df_assembly.append(df_record)
    return df_assembly, tuple(idx_to_id)


def indexlr_py(
    assembly_path: Path, 
    kmerlen: int, 
    windowsize: int, 
    assembly_idx: int, 
    is_target: bool
) -> tuple[list[np.ndarray], tuple[str]]:
    """Get the minimizers for each record in an assembly with the Python module `Indexlr` from `btllib<https://github.com/bcgsc/btllib>`__. 
    - This function is specialized for Seqwin. 
    - The number of threads to use is set to 1. 
    - Indexlr seems to skip sequence regions with ambiguous bases. So long runs of 'N's will be ignored and k-mer coordinates will seem discontinuous. 
    (e.g., NZ_QJGJ01000182.1 in assembly GCF_003200995.1). 
    - To run this function with multiprocessing, set start method to `spawn` or `forkserver`:
    ```python
    with multiprocessing.get_context(method='spawn').Pool(processes=n_cpu) as pool:
        func_out = pool.starmap(indexlr_py, all_args)
    ```
    Check `Python docs <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`__ 
    for more details. 

    Args:
        assembly_path (Path): Path to the assembly file in FASTA format. 
        kmerlen (int): k-mer length. 
        windowsize (int): Window size for minimizer sketch. 
        assembly_idx (int): Assembly index. 
        is_target (bool): True if this is a target assembly. 

    Returns:
        tuple: A tuple containing
            1. list[np.ndarray]: A list of structured numpy arrays for each sequence record in the assembly. 
                Each row represents a minimizer in the sequence record, with columns, 
                - 'hash' (uint64): Hash value of the minimizer. 
                - 'pos' (uint32): Position of the first base of the minimizer. 
                - 'record_idx' (uint16): 0-based index of the sequence records, in the same order as they appear in the FASTA file. 
                - 'assembly_idx' (uint16): Assembly index. 
                - 'is_target' (bool): True for target assemblies. 
            2. tuple[str]: Record IDs of the assembly. 
    """
    kmers: list[np.ndarray] = list()
    idx_to_id: list[str] = list()
    with Indexlr(
        str(assembly_path), # must convert to string, or TypeError will be raised
        kmerlen, windowsize, 
        IndexlrFlag.LONG_MODE, # optimize for sketching long sequence
        1 # use 1 thread (max 5)
    ) as assembly:
        for idx, record in enumerate(assembly):
            idx_to_id.append(record.id)

            # record.minimizers is a list; empty if current record has no minimizer (shorter than windowsize)
            if record.minimizers:
                # kmers_record is a 1-D array with structured dtype (each element is a "tuple")
                kmers_record = np.array(
                    list(
                        (m.out_hash, m.pos, idx, assembly_idx, is_target) 
                        for m in record.minimizers
                    ), 
                    dtype=KMER_DTYPE
                )
                kmers.append(kmers_record)
    return kmers, tuple(idx_to_id)
