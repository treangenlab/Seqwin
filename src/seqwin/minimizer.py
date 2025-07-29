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

        # df_record['record_id'] = record_id
        df_record['record_idx'] = _IDX_TYPE(idx)
        df_assembly.append(df_record)
    return df_assembly, tuple(idx_to_id)


# NOTE: this function is outdated
def indexlr_py(
    assembly_path: Path, 
    kmerlen: int=21, 
    windowsize: int=200, 
    get_minimizer_seq: bool=False
) -> list[pd.DataFrame]:
    """Get the minimizers for each record in an assembly with the Python module `Indexlr` from `btllib<https://github.com/bcgsc/btllib>`__. 
    - The number of threads to use is set to 1. 
    - Indexlr seems to skip sequence regions with ambiguous bases. So long runs of 'N's will be ignored and k-mer coordinates will seem discontinuous. 
    (e.g., NZ_QJGJ01000182.1 in assembly GCF_003200995.1). 
    - To run this function with multiprocessing, set start method to `spawn`:
    ```python
    if __name__ == "__main__":
        multiprocessing.set_start_method('spawn')
    ```
    Check `Python docs <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`__ 
    for more details. 

    Args:
        assembly_path (Path): Path to the assembly file in FASTA format. 
        kmerlen (int, optional): k-mer length. [21]
        windowsize (int, optional): Window size for minimizer sketch. [200]
        get_minimizer_seq (bool, optional): Include minimizer sequence in the output if True. [False]

    Returns:
        list[pd.DataFrame]: A list of pandas DataFrames for each sequence record in the assembly. 
            Each row represents a minimizer in the sequence record, with columns, 
            1. 'hash' (uint64): Hash value of the minimizer. 
            2. 'pos' (int64): Position of the first base of the minimizer. 
            3. 'strand' (category): Which strand (+/-) has the smaller hash value. Use category type to save memory. 
            4. 'record_id' (str): ID of the sequence record. 
            5. 'seq' (str, optional): sequence of the minimizer (forward strand), available if `get_minimizer_seq=True`. 
    """
    if get_minimizer_seq:
        seq_flag = IndexlrFlag.SEQ
        columns = ('hash', 'pos', 'strand', 'seq')
        get_attr = lambda m: (m.out_hash, m.pos, '+' if m.forward else '-', m.seq)
    else:
        seq_flag = 0
        columns = ('hash', 'pos', 'strand')
        get_attr = lambda m: (m.out_hash, m.pos, '+' if m.forward else '-')

    df_assembly: list[pd.DataFrame] = list()
    with Indexlr(assembly_path, kmerlen, windowsize, seq_flag+IndexlrFlag.LONG_MODE, 1) as assembly:
        # Flag: return kmer sequence; optimize for sketching long sequence
        # use 1 thread (max 5)
        # for NCBI complete assemblies, first record is the chromosome
        for record in assembly:
            # m.forward: 
                # True if the forward k-mer has smaller hash value, compared to the reverse complement k-mer
                # the actual hash value (m.out_hash) of a k-mer is the sum of its forward hash and reverse hash
                # so that a k-mer and its reverse complement have the same hash value
            # m.seq
                # output sequence is always the forward k-mer, even if the reverse k-mer has smaller hash value
                # kind of useless since we already know the position of the k-mer
            df_record = pd.DataFrame(
                list(get_attr(m) for m in record.minimizers), 
                columns=columns
            )
            df_record['record_id'] = record.id
            df_assembly.append(df_record)
    return df_assembly
