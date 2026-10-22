"""
Evaluation
==========

Evaluate signature sequences with BLAST and calculate their metrics.

Classes:
--------
- SignatureMetrics

Functions:
----------
- eval_signatures
- eval_signatures_seqwin
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import csv, logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import time
from itertools import repeat
from dataclasses import dataclass, field, fields, replace

import pandas as pd

from .core._native import _METADATA_NAMES
from .ncbi import blast
from .utils import print_time_delta, log_and_raise, load_fasta, mp_wrapper
from .config import BLASTCONFIG

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SignatureMetrics:
    """
    Metrics of a signature, calculated from its BLAST alignments against target / non-target assemblies.
    Metrics default to zero when no BLAST hits are found.

    Attributes:
        conservation (float): Average fraction of identical bases between the signature and target assemblies.
        f_tar_hits (float): Fraction of target assemblies with a BLAST hit.
        divergence (float): Average fraction of mismatches and gaps between the signature and non-target assemblies.
        f_neg_hits (float): Fraction of non-target assemblies with a BLAST hit.
        avg_repeats_tar (float): Average number of repeats of this signature in target assemblies.
        avg_pident_tar (float): Average percentage of identical bases of all repeats in target assemblies.
        avg_repeats_neg (float): Average number of repeats of this signature in non-target assemblies.
        avg_pident_neg (float): Average percentage of identical bases of all repeats in non-target assemblies.
        blast (pd.DataFrame | None): BLAST alignments used to calculate the scalar metrics.
    """
    conservation: float = 0.0
    f_tar_hits: float = 0.0
    divergence: float = 0.0
    f_neg_hits: float = 0.0
    avg_repeats_tar: float = 0.0
    avg_pident_tar: float = 0.0
    avg_repeats_neg: float = 0.0
    avg_pident_neg: float = 0.0
    blast: pd.DataFrame | None = field(default=None, repr=False, compare=False)

# Scalar metrics (excluding 'blast')
_METRIC_NAMES = tuple(f.name for f in fields(SignatureMetrics) if f.name != 'blast')


def _get_avg_ident(blast_out: pd.DataFrame, query_len: int, n: int) -> float:
    """Given a list of BLAST hits, calculate the average sequence identity between the query and all subjects.
    The denominator (`n`) is the number of subject sequences that are expected to include the query sequence.
    Note that `n` might not be the same as `len(blast_out)`, since some subjects may have no hit.

    Args:
        blast_out (pd.DataFrame): Each row should be a BLAST hit of the query, with column
            'nident' (number of identical matches).
        query_len (int): Length of the query sequence.
        n (int): The number of subjects that are expected to include the query sequence.

    Returns:
        float: Conservation.
    """
    return sum(blast_out['nident']) / query_len / n


def _get_avg_dist(blast_out: pd.DataFrame, query_len: int, n: int) -> float:
    """Given a list of BLAST hits, calculate the average distance between the query and all subjects.
    The denominator (`n`) is the number of subject sequences that are expected to include the query sequence.
    Note that `n` might not be the same as `len(blast_out)`, since some subjects may have no hit.

    Args:
        blast_out (pd.DataFrame): Each row should be a BLAST hit of the query, with columns,
            1. 'mismatch': number of mismatches.
            2. 'gaps': total number of gaps in BOTH query and subject (might cause inaccuracy).
        query_len (int): Length of the query sequence.
        n (int): The number of subjects that are expected to include the query sequence.

    Returns:
        float: Divergence.
    """
    return sum(blast_out['mismatch'] + blast_out['gaps']) / query_len / n


def _get_metrics(
    blast_out: pd.DataFrame | None, marker_len: int, total_tar: int, total_neg: int
) -> SignatureMetrics:
    """Calculate the metrics of a marker based on its BLAST hits in all assemblies.
    - Conservation is calculated with `_get_avg_ident()` on target assemblies.
    - Divergence is calculated with `_get_avg_dist()` on non-target assemblies.

    Args:
        blast_out (pd.DataFrame | None): Each row is the best BLAST hit of the marker in an assembly.
            Required columns: `['is_target', 'nident', 'mismatch', 'gaps', 'n_hits', 'avg_nident']`.
            `None` if the marker has no blast hit in any assembly.
        marker_len (int): Marker length.
        total_tar (int): Number of target assemblies.
        total_neg (int): Number of non-target assemblies.

    Returns:
        SignatureMetrics: Signature metrics.
    """
    if blast_out is None: # no blast hit in any assembly
        return SignatureMetrics()

    metrics = {} # SignatureMetrics is frozen

    # calculate sensitivity
    df_tar = blast_out[blast_out['is_target'] == True]
    if len(df_tar) > 0:
        metrics['conservation'] = _get_avg_ident(df_tar, marker_len, total_tar)
        metrics['f_tar_hits'] = len(df_tar) / total_tar
        metrics['avg_repeats_tar'] = df_tar['n_hits'].mean()
        metrics['avg_pident_tar'] = df_tar['avg_nident'].mean() / marker_len

    # calculate specificity
    df_neg = blast_out[blast_out['is_target'] == False]
    if len(df_neg) > 0:
        metrics['divergence'] = _get_avg_dist(df_neg, marker_len, total_neg)
        metrics['f_neg_hits'] = len(df_neg) / total_neg
        metrics['avg_repeats_neg'] = df_neg['n_hits'].mean()
        metrics['avg_pident_neg'] = df_neg['avg_nident'].mean() / marker_len

    return SignatureMetrics(**metrics)


def eval_signatures(
    all_seqs: list[str], blastdb: Path, total_tar: int, total_neg: int, n_cpu: int=1
) -> list[SignatureMetrics]:
    """BLAST check each signature sequence against all / non-target assemblies, and calculate the metrics of each signature.

    Args:
        all_seqs (list[str]): A list of signature sequences.
        blastdb (Path): Path to a BLAST database generated by Seqwin (e.g., `seqwin-out/blastdb/`).
        total_tar (int): Number of target assemblies.
        total_neg (int): Number of non-target assemblies.
        n_cpu (int, optional): Number of threads to use. [1]

    Returns:
        list[SignatureMetrics]: Metrics and BLAST hits of each signature.
    """
    if blastdb.name == BLASTCONFIG.title_neg_only:
        neg_only = True
        logger.info(' - BLAST checking signatures against non-target assemblies...')
    elif blastdb.name == BLASTCONFIG.title_all:
        neg_only = False
        logger.info(' - BLAST checking signatures against all assemblies...')
    else:
        log_and_raise(
            ValueError,
            f'Invalid BLAST database title. Must be "{BLASTCONFIG.title_all}" or "{BLASTCONFIG.title_neg_only}"'
        )
    tik = time()
    n_seqs = len(all_seqs)

    # blast check all markers against all / non-target assemblies
    blast_out = blast(
        all_seqs,
        db=blastdb,
        task=BLASTCONFIG.task,
        columns=BLASTCONFIG.columns,
        n_cpu=n_cpu,
        batch_size=BLASTCONFIG.batch_size,
    )
    if len(blast_out) == 0:
        log_and_raise(RuntimeError, 'No BLAST hit found')
    # blast_out.to_pickle('blast_out.pkl')

    #---------- extract BLAST hits of each marker ----------#
    logger.info(' - Formatting BLAST output...')
    # get assembly id and record id, see Assemblies.makeblastdb()
    blast_out[['assembly_idx', 'is_target', 'record_id']] = blast_out['sseqid'].str.split(
        BLASTCONFIG.header_sep, expand=True
    )
    blast_out.drop(columns='sseqid', inplace=True)
    # unlike pd.read_csv() in blast(), df.str.split() does not do auto type conversion
    blast_out['assembly_idx'] = blast_out['assembly_idx'].astype(int)
    blast_out['is_target'] = blast_out['is_target'].map(BLASTCONFIG.str2bool)

    # keep the best alignment (highest bitscore) for each assembly
    blast_out.sort_values(
        by=['qseqid', 'assembly_idx', 'bitscore'],
        ascending=[True, True, False], inplace=True
    )
    blast_out = blast_out.groupby(
        by=['qseqid', 'assembly_idx'], as_index=True, sort=False
    )
    # also keep nident of other less optimal alignments
    nident = blast_out['nident'].agg(
        n_hits='count',
        avg_nident='mean'
    )
    blast_out = blast_out.head(1)
    nident.reset_index(drop=True, inplace=True)
    blast_out.reset_index(drop=True, inplace=True)
    blast_out = pd.concat([blast_out, nident], axis=1)

    # output a df for each query sequence (some markers might have no BLAST hit)
    all_blast = [None] * n_seqs
    for i, g in blast_out.groupby('qseqid', sort=False):
        g.drop(columns='qseqid', inplace=True)
        g.reset_index(drop=True, inplace=True)
        all_blast[i] = g
    #---------- extract BLAST hits of each marker ----------#

    if not neg_only: # check for markers with no BLAST hit
        for i, b in enumerate(all_blast):
            if b is None:
                logger.warning(f'Signature at index {i} (0-based) has no BLAST hit in any assembly ({all_seqs[i][:10]}...)')

    # calculate conservation and divergence for each marker based on its blast output
    logger.info(' - Evaluating each signature...')
    metrics_args = zip(
        all_blast,
        map(len, all_seqs),
        repeat(total_tar, n_seqs),
        repeat(total_neg, n_seqs),
    )
    metrics = mp_wrapper(
        _get_metrics, metrics_args, n_cpu, n_jobs=n_seqs
    )
    # add BLAST dataframes after to avoid multiprocessing round-trip
    metrics = list(
        replace(m, blast=blast_out)
        for m, blast_out in zip(metrics, all_blast, strict=True)
    )

    print_time_delta(time()-tik)
    return metrics


def eval_signatures_seqwin(
    fasta_path: Path,
    csv_path: Path,
    blastdb: Path,
    total_tar: int,
    total_neg: int,
    n_cpu: int = 1,
) -> list[SignatureMetrics]:
    """Evaluate Seqwin FASTA signatures and append scalar metrics to their CSV.

    FASTA records and CSV rows must match in count, header, length, and order.
    The CSV is replaced only after evaluation and serialization succeed.
    """
    records = load_fasta(fasta_path)
    with csv_path.open(encoding='utf-8', newline='') as source:
        reader = csv.reader(source)
        if tuple(next(reader, ())) != _METADATA_NAMES:
            raise ValueError('Expected the five mandatory signature CSV columns')
        rows = list(reader)
    if len(records) != len(rows):
        raise ValueError('FASTA and CSV signature counts do not match')
    for i, (record, row) in enumerate(zip(records, rows, strict=True)):
        if len(row) != len(_METADATA_NAMES):
            raise ValueError('Expected the five mandatory signature CSV columns')
        if record.id != row[0]:
            raise ValueError(f'FASTA/CSV header mismatch at index {i}')
        if len(record.sequence) != int(row[1]):
            raise ValueError(f'FASTA/CSV length mismatch at index {i}')

    metrics = eval_signatures(
        list(record.sequence for record in records),
        blastdb, total_tar, total_neg, n_cpu,
    )
    if len(metrics) != len(records):
        raise ValueError('Evaluation and signature counts do not match')

    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode='w', dir=csv_path.parent, encoding='utf-8', newline='', delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            writer = csv.writer(temporary, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow((*_METADATA_NAMES, *_METRIC_NAMES))
            for row, metric in zip(rows, metrics, strict=True):
                writer.writerow((
                    *row,
                    *(getattr(metric, name) for name in _METRIC_NAMES),
                ))
        temporary_path.replace(csv_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    logger.info(f' - Signature metrics added to {csv_path}')
    return metrics
