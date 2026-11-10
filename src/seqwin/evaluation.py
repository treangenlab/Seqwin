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
from dataclasses import dataclass, field, fields

import numpy as np
from numpy.typing import NDArray

from .core import read_fasta
from .core._native import _METADATA_NAMES
from .ncbi import blast
from .utils import print_time_delta, log_and_raise
from .config import BLASTCONFIG

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class SignatureMetrics:
    """
    Metrics of a signature, calculated from its BLAST alignments against target / non-target assemblies.
    Metrics default to zero for signatures with no BLAST hits.

    Attributes:
        conservation (float): Average fraction of identical bases between the signature and target assemblies.
        f_tar_hits (float): Fraction of target assemblies with a BLAST hit.
        divergence (float): Average fraction of mismatches and gaps between the signature and non-target assemblies.
        f_neg_hits (float): Fraction of non-target assemblies with a BLAST hit.
        avg_repeats_tar (float): Average number of repeats of this signature in target assemblies.
        avg_pident_tar (float): Average percentage of identical bases of all repeats in target assemblies.
        avg_repeats_neg (float): Average number of repeats of this signature in non-target assemblies.
        avg_pident_neg (float): Average percentage of identical bases of all repeats in non-target assemblies.
        blast (NDArray[np.void] | None): Best BLAST alignments in each assembly.
    """
    conservation: float = 0.0
    f_tar_hits: float = 0.0
    divergence: float = 0.0
    f_neg_hits: float = 0.0
    avg_repeats_tar: float = 0.0
    avg_pident_tar: float = 0.0
    avg_repeats_neg: float = 0.0
    avg_pident_neg: float = 0.0
    blast: NDArray[np.void] | None = field(default=None, repr=False, compare=False)

# Scalar metrics (excluding 'blast')
_METRIC_NAMES = tuple(f.name for f in fields(SignatureMetrics) if f.name != 'blast')


def _get_metrics(
    best: NDArray[np.void],
    indices: NDArray[np.int64],
    lengths: NDArray[np.int64],
    total_tar: int,
    total_neg: int,
) -> list[SignatureMetrics]:
    """Calculate the metrics of all signatures based on their BLAST hits.

    Args:
        best (NDArray[np.void]): Best alignment per signature/assembly pair, including repeat statistics.
        indices (NDArray[np.int64]): Signature index corresponding to each row of `best`, sorted in ascending order.
        lengths (NDArray[np.int64]): Length of each signature.
        total_tar (int): Number of target assemblies.
        total_neg (int): Number of non-target assemblies.

    Returns:
        list[SignatureMetrics]: Metrics and BLAST hits of each signature.
    """
    n_seqs = len(lengths)

    # calculate sensitivity
    mask = best['is_target']
    indices_mask = indices[mask]
    nident = np.bincount(indices_mask, weights=best['nident'][mask], minlength=n_seqs)
    counts = np.bincount(indices_mask, minlength=n_seqs)
    n_hits = np.bincount(indices_mask, weights=best['n_hits'][mask], minlength=n_seqs)
    avg_nident = np.bincount(indices_mask, weights=best['avg_nident'][mask], minlength=n_seqs)
    conservation = np.divide(
        nident, lengths * total_tar, out=np.zeros(n_seqs), where=(lengths != 0) & (total_tar != 0)
    )
    f_tar_hits = np.divide(
        counts, total_tar, out=np.zeros(n_seqs), where=total_tar != 0
    )
    avg_repeats_tar = np.divide(
        n_hits, counts, out=np.zeros(n_seqs), where=counts != 0
    )
    avg_pident_tar = np.divide(
        avg_nident, counts * lengths, out=np.zeros(n_seqs), where=(counts != 0) & (lengths != 0)
    )

    # calculate specificity
    mask = ~mask
    indices_mask = indices[mask]
    best_diff = best['mismatch'] + best['gaps']
    diff = np.bincount(indices_mask, weights=best_diff[mask], minlength=n_seqs)
    counts = np.bincount(indices_mask, minlength=n_seqs)
    n_hits = np.bincount(indices_mask, weights=best['n_hits'][mask], minlength=n_seqs)
    avg_nident = np.bincount(indices_mask, weights=best['avg_nident'][mask], minlength=n_seqs)
    divergence = np.divide(
        diff, lengths * total_neg, out=np.zeros(n_seqs), where=(lengths != 0) & (total_neg != 0)
    )
    f_neg_hits = np.divide(
        counts, total_neg, out=np.zeros(n_seqs), where=total_neg != 0
    )
    avg_repeats_neg = np.divide(
        n_hits, counts, out=np.zeros(n_seqs), where=counts != 0
    )
    avg_pident_neg = np.divide(
        avg_nident, counts * lengths, out=np.zeros(n_seqs), where=(counts != 0) & (lengths != 0)
    )

    offsets = np.searchsorted(indices, np.arange(n_seqs + 1))
    return list(
        SignatureMetrics(
            conservation=float(conservation[i]),
            f_tar_hits=float(f_tar_hits[i]),
            divergence=float(divergence[i]),
            f_neg_hits=float(f_neg_hits[i]),
            avg_repeats_tar=float(avg_repeats_tar[i]),
            avg_pident_tar=float(avg_pident_tar[i]),
            avg_repeats_neg=float(avg_repeats_neg[i]),
            avg_pident_neg=float(avg_pident_neg[i]),
            blast=best[offsets[i]:offsets[i+1]]
        )
        for i in range(n_seqs)
    )


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
        logger.info(' - BLAST checking signatures against non-target assemblies...')
    elif blastdb.name == BLASTCONFIG.title_all:
        logger.info(' - BLAST checking signatures against all assemblies...')
    else:
        log_and_raise(
            ValueError,
            f'Invalid BLAST database title. Must be "{BLASTCONFIG.title_all}" or "{BLASTCONFIG.title_neg_only}"'
        )
    tik = time()
    n_seqs = len(all_seqs)
    blast_out = blast(
        all_seqs,
        db=blastdb,
        task=BLASTCONFIG.task,
        columns=BLASTCONFIG.columns,
        n_cpu=n_cpu,
        batch_size=BLASTCONFIG.batch_size,
    )

    logger.info(' - Formatting BLAST output...')
    # get assembly id and record id, see Assemblies.makeblastdb()
    subject_dtype = [('assembly_idx', np.int64), ('is_target', bool), ('record_id', object)]
    subjects = np.fromiter((
        (int(assembly_idx), BLASTCONFIG.str2bool[is_target], record_id)
        for assembly_idx, is_target, record_id in (
            subject.split(BLASTCONFIG.header_sep) for subject in blast_out['sseqid']
        )
    ), dtype=subject_dtype, count=len(blast_out))

    # keep the best alignment (highest bitscore) for each assembly
    order = np.lexsort((-blast_out['bitscore'], subjects['assembly_idx'], blast_out['qseqid']))
    blast_out = blast_out[order]
    subjects = subjects[order]
    # group by signature and assembly
    starts = np.flatnonzero(np.r_[
        len(blast_out) > 0, # starts should be empty when blast_out is empty
        (blast_out['qseqid'][1:] != blast_out['qseqid'][:-1]) |
        (subjects['assembly_idx'][1:] != subjects['assembly_idx'][:-1])
    ])

    # construct the compacted table (best alignments only)
    kept_cols = [name for name in blast_out.dtype.names if name not in ('qseqid', 'sseqid')]
    best = np.empty(
        len(starts),
        dtype=[(name, blast_out.dtype[name]) for name in kept_cols] +
              subject_dtype +
              [('n_hits', np.int64), ('avg_nident', np.float64)]
    )
    np.take(blast_out[kept_cols], starts, out=best[kept_cols])
    np.take(subjects, starts, out=best[list(subjects.dtype.names)])
    # also keep nident of other less optimal alignments
    best['n_hits'] = np.diff(np.r_[starts, len(blast_out)])
    best['avg_nident'] = np.add.reduceat(blast_out['nident'], starts) / best['n_hits']

    logger.info(' - Evaluating each signature...')
    indices = blast_out['qseqid'][starts]
    lengths = np.fromiter(map(len, all_seqs), dtype=np.int64, count=n_seqs)
    metrics = _get_metrics(best, indices, lengths, total_tar, total_neg)

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
    records = read_fasta(fasta_path)
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
