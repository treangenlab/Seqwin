"""
NCBI
====

- Search and download genome assemblies with NCBI `Datasets <https://www.ncbi.nlm.nih.gov/datasets/>`__.
- Run NCBI `BLAST+ <https://www.ncbi.nlm.nih.gov/books/NBK131777/>`__.

Dependencies:
-------------
- numpy
- blast
- ncbi-datasets-cli
- .utils

Functions:
----------
- search_taxon
- get_assembly_paths
- download_taxon
- blast

Attributes:
-----------
- Format (str, Enum)
- Level (str, Enum)
- Source (str, Enum)
- Task (str, Enum)
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

import shutil, json, logging, zipfile, csv
from pathlib import Path
from enum import Enum
from io import StringIO
from collections.abc import Sequence

logger = logging.getLogger(__name__)

import numpy as np
from numpy.typing import NDArray

from .utils import log_and_raise, file_to_write, list_dir, run_cmd

_ZIP_EXT = '.zip' # File extension of NCBI genome package. ['.zip']
_BLAST_COLUMNS = { # Default columns to be included in the BLAST tsv output
    'qseqid': np.int64, # 1. query or source (e.g., gene) sequence id
    'sseqid': object, # 2. subject or target (e.g., reference genome) sequence id
    'length': np.int64, # 3. alignment length
    'pident': np.float64, # 4. percentage of identical matches (= nident / length)
    'nident': np.int64, # 5. number of identical matches (= length - mismatch - gaps)
    'mismatch': np.int64, # 6. number of mismatches
    'gapopen': np.int64, # 7. number of gap openings
    'gaps': np.int64, # 8. total number of gaps
    'qstart': np.int64, # 9. start of alignment in query
    'qend': np.int64, # 10. end of alignment in query
    'sstart': np.int64, # 11. start of alignment in subject
    'send': np.int64, # 12. end of alignment in subject
    'evalue': np.float64, # 13. expect value
    'bitscore': np.float64, # 14. bit score
    'qseq': object, # 15. aligned part of query sequence
    'sseq': object, # 16. aligned part of subject sequence
}
_MAX_REHYDRATE_WORKERS = 8 # Maximum number of CPUs for `datasets rehydrate` (avoid request bursts)
_MAX_HSPS = '1000' # Maximum number of HSPs per subject sequence to save for each query. ['1000']
_MAX_TARGET_SEQS = '50000' # Maximum number of aligned sequences to keep. ['50000']

class Format(str, Enum):
    """Genome file formats."""
    fasta = 'fasta'
    genbank = 'genbank'

class Level(str, Enum):
    """NCBI assembly levels."""
    contig = 'contig'
    scaffold = 'scaffold'
    chromosome = 'chromosome'
    complete = 'complete'

class Source(str, Enum):
    """NCBI download sources."""
    genbank = 'genbank'
    refseq = 'refseq'

class Task(str, Enum):
    """Preset BLAST tasks."""
    blastn = 'blastn'
    blastn_short = 'blastn-short'
    megablast = 'megablast'


def _add_api_key(args: list[str | Path], api_key: str | None) -> list[str | Path]:
    """Append `--api-key` for NCBI datasets commands when provided.
    """
    if api_key is not None:
        args += ['--api-key', api_key]
    return args


def search_taxon(taxon: str, api_key: str | None=None) -> tuple[str | None, str | None]:
    """Search a taxon on NCBI Taxonomy. Internet connection is needed.

    Args:
        taxon (str): Name or ID of the taxon (exact match).
        api_key (str | None, optional): NCBI API key. [None]

    Returns:
        tuple: A tuple containing
            1. str | None: NCBI Taxonomy ID of the taxon.
            2. str | None: Current scientific name of the taxon.
    """
    logger.info(f'Searching NCBI Taxonomy for "{taxon}"...')
    args = [
        'datasets', 'summary', 'taxonomy', 'taxon', str(taxon),
        '--as-json-lines', # output as json
        '--report', 'names', # do not output tax ids of children
    ]
    args = _add_api_key(args, api_key)
    summary = run_cmd(*args, raise_error=False)

    if summary.stdout == '':
        logger.error(summary.stderr)
        logger.error(f' - Unable to find taxon "{taxon}"')
        return None, None

    summary = json.loads(summary.stdout)
    tax_id = summary['taxonomy']['tax_id']
    tax_name = summary['taxonomy']['current_scientific_name']['name']
    logger.info(f' - Found NCBI Taxonomy ID: {tax_id}')
    logger.debug(f' - Found NCBI Taxon: {tax_name}')
    return tax_id, tax_name


def get_assembly_paths(package_dir: Path) -> list[Path]:
    """Get the file paths of all genome assemblies in a NCBI genome package.

    Args:
        package_dir (Path): Path of the genome package directory (e.g., ncbi_dataset).

    Returns:
        list[Path]: File paths of all genome assemblies in the package.
    """
    # sanity check
    if not package_dir.is_dir():
        log_and_raise(NotADirectoryError, f'Not a directory: {package_dir}')

    prefix = package_dir / 'ncbi_dataset' / 'data' # hard coded entries within package_dir
    assemblies = list_dir(prefix, mode='d')
    paths = list()
    for assembly_dir in assemblies:
        assembly_path = list_dir(assembly_dir, mode='f')
        if len(assembly_path) > 1:
            logger.warning(f' - Found more than one files under {assembly_dir}')
        elif len(assembly_path) == 0:
            log_and_raise(FileNotFoundError, f'No assembly file is found {assembly_dir}')
        paths.append(assembly_path[0])
    return paths


def download_taxon(
    taxon: str,
    prefix: Path = Path.cwd(),
    format: Format = Format.fasta,
    level: Level = Level.contig,
    source: Source = Source.genbank,
    annotated: bool = True,
    exclude_mag: bool = False,
    gzip: bool = True,
    api_key: str | None = None,
    overwrite: bool = False,
    n_cpu: int = 1
) -> list[Path] | None:
    """Download genome assemblies under a taxon from NCBI Taxonomy. Internet connection is needed.
    Atypical genomes and genomes from large multi-isolate projects are excluded.

    Args:
        taxon (str): Name or ID of the taxon (exact match).
        prefix (Path, optional): A directory where the data package is downloaded. [cwd]
        format (str, optional): Format of genome sequences, should be 'fasta' or 'genbank'. ['fasta']
        level (str, optional): Limit to genomes ≥ this assembly level ('contig' < 'scaffold' < 'chromosome' < 'complete'). ['contig']
        source (str, optional): Genome source, should be 'genbank' or 'refseq'. ['genbank']
        annotated (bool, optional): If True, limit to GenBank (submitter) or RefSeq annotated genomes, based on the selection of source. [True]
        exclude_mag (bool, optional): If True, exclude metagenome-assembled genomes (MAGs). [False]
        gzip (bool, optional): If True, download genome sequences in gzip format. [True]
        api_key (str | None, optional): NCBI API key. [None]
        overwrite (bool, optional): If True, overwrite the zipped genome package. [False]
        n_cpu (int, optional): Number of processes to run in parallel. [1]

    Returns:
        list[Path] | None: File paths of downloaded genome assemblies. Return None if the taxon is not found.
    """
    # sanity check
    if not prefix.is_dir():
        log_and_raise(NotADirectoryError, f'Cannot download genomes to this location, since it is not a directory: {prefix}')
    n_cpu = min(n_cpu, _MAX_REHYDRATE_WORKERS)

    # reuse existing genome package
    tax_dir = prefix / taxon.replace(' ', '-')
    if tax_dir.exists():
        logger.warning(f'Existing genome package is found {tax_dir}')
        try:
            assembly_paths = get_assembly_paths(tax_dir)
        except Exception as e:
            log_and_raise(
                RuntimeError,
                (f'Genome package might be incomplete {tax_dir}\n'
                'Consider deleting it and try again'),
                from_e=e
            )
        logger.info(f' - Found {len(assembly_paths)} genome assemblies.')
        return assembly_paths

    # search taxon id and name
    tax_id, tax_name = search_taxon(taxon, api_key=api_key)
    if tax_id is None:
        return None

    # path to the output genome package
    tax_dir = prefix / tax_name.replace(' ', '-') # avoid spaces in file paths

    # add .zip to tax_dir (don't use with_suffix since tax_name might contain '.')
    tax_zip = tax_dir.with_name(tax_dir.name + _ZIP_EXT)
    file_to_write(tax_zip, overwrite=overwrite)

    # arguments for the download command
    args = [
        'datasets', 'download', 'genome',
        'taxon', tax_id,
        '--filename', tax_zip,
        '--exclude-atypical', '--exclude-multi-isolate',
        '--no-progressbar', '--dehydrated',
    ]
    if format == Format.fasta:
        args += ['--include', 'genome']
    elif format == Format.genbank:
        args += ['--include', 'gbff']
    else:
        log_and_raise(ValueError, f'Invalid download format: {format}')

    if level == Level.contig:
        pass # include all levels by default
    elif level == Level.scaffold:
        args += ['--assembly-level', 'scaffold,chromosome,complete']
    elif level == Level.chromosome:
        args += ['--assembly-level', 'chromosome,complete']
    elif level == Level.complete:
        args += ['--assembly-level', 'complete']
    else:
        log_and_raise(ValueError, f'Invalid assembly level: {level}')

    if source == Source.genbank:
        args += ['--assembly-source', 'GenBank']
    elif source == Source.refseq:
        args += ['--assembly-source', 'RefSeq']
    else:
        log_and_raise(ValueError, f'Invalid download source: {source}')

    if annotated:
        args.append('--annotated')

    if exclude_mag:
        args += ['--mag', 'exclude']
    else:
        args += ['--mag', 'all']
    args = _add_api_key(args, api_key)

    logger.info(f'Downloading genome package for NCBI Taxonomy ID {tax_id}...')

    # download metadata
    logger.debug(' - Downloading dehydrated genome package (metadata only)...')
    download_log = run_cmd(*args, raise_error=False)
    if download_log.returncode != 0:
        logger.error(download_log.stderr)
        logger.error(f' - No genome assemblies were found for NCBI Taxonomy ID {tax_id}, try loosen the filters.')
        return None

    # unzip dehydrated package
    logger.debug(' - Unzipping dehydrated genome package...')
    try:
        with zipfile.ZipFile(tax_zip, 'r') as zf:
            zf.extractall(tax_dir)
    except Exception as e:
        shutil.rmtree(tax_dir) # remove incomplete dir
        log_and_raise(RuntimeError, f'Failed to unzip genome package for NCBI Taxonomy ID {tax_id}: {tax_zip}', from_e=e)

    # download actual sequences
    args = [
        'datasets', 'rehydrate', '--directory', tax_dir,
        '--max-workers', str(n_cpu), '--no-progressbar'
    ]
    if gzip:
        logger.debug(' - Rehydrating in gzip format...')
        args += ['--gzip']
    else:
        logger.debug(' - Rehydrating...')
    args = _add_api_key(args, api_key)
    try:
        run_cmd(*args, raise_error=True)
    except Exception as e:
        shutil.rmtree(tax_dir) # remove incomplete dir
        log_and_raise(
            RuntimeError,
            (f'Failed to rehydrate data package for taxon "{taxon}".\n'
            'NCBI might have blocked the request due to high usage. Try waiting for a while before running Seqwin again.\n'
            'Add --overwrite to the command so that downloaded taxon packages can be reused.'),
            from_e=e
        )

    # get the file path of each assembly
    assembly_paths = get_assembly_paths(tax_dir)
    logger.info(f' - Downloaded {len(assembly_paths)} genome assemblies for NCBI Taxonomy ID {tax_id}.')

    return assembly_paths


def _get_blast_dtype(columns: Sequence[str] | None) -> np.dtype:
    """Construct the structured dtype for the requested BLAST fields.
    """
    if columns is None:
        columns = tuple(_BLAST_COLUMNS)
    if len(columns) == 0 or len(set(columns)) != len(columns):
        log_and_raise(ValueError, 'BLAST columns must be non-empty and unique')
    if any(column not in _BLAST_COLUMNS for column in columns):
        log_and_raise(ValueError, f'Unsupported BLAST columns: {columns}')
    return np.dtype([(column, _BLAST_COLUMNS[column]) for column in columns])


def _get_blast_outfmt(columns: Sequence[str]) -> str:
    """Given the columns to be included in the BLAST TSV output, get the value to be provided to
    the -outfmt argument of the blastn command. See `blastn -help` for more info.
    """
    # we need a custom output format since some info (e.g., sequences) are not included in the tsv by default
    # do not surround the return str with double quotes (" ") even if there are spaces in it, `subprocess.run()` will do it for you
    # otherwise it will have double double quotes ("" "") and blastn will raise an error
    return f'6 {" ".join(columns)}'


def _blast_batch(
    seq_idx: Sequence[int],
    seq_list: Sequence[str],
    db: Path,
    task: str,
    columns: Sequence[str],
    outfmt: str,
    taxids: str | None,
    neg_taxids: str | None,
    n_cpu: int
) -> NDArray[np.void]:
    """Run BLAST on a list of sequences and return a structured array of the tabular output.

    Args:
        seq_idx (Sequence[int]): Indices of the input sequences.
        seq_list (Sequence[str]): A list of sequences for BLAST.
        db (Path): Path to the BLAST database.
        task (str): Preset BLAST tasks ('blastn', 'blastn-short', 'megablast').
        columns (Sequence[str]): Columns to be included in the structured array output.
        outfmt (str): Output of `_get_blast_outfmt()`.
        taxids (str | None): Only search for these taxonomy IDs and their descendants.
        neg_taxids (str | None): Do NOT search for these taxonomy IDs and their descendants.
        n_cpu (int): Number of processes to run in parallel.

    Returns:
        NDArray[np.void]: A structured array of the tabular output, empty when no hits are found.
    """
    # create input fasta for blast
    blast_in = ''.join(
        f'>{i}\n{seq}\n' for i, seq in zip(seq_idx, seq_list)
    )

    # prepare blastn args
    args = [
        'blastn',
        '-db', db,
        '-task', task,
        '-outfmt', outfmt, # output in tsv format with specified columns
        '-max_hsps', _MAX_HSPS, # maximum number of HSPs per subject sequence to save for each query
        '-max_target_seqs', _MAX_TARGET_SEQS, # maximum number of aligned sequences to keep
        '-num_threads', n_cpu
    ]
    if taxids is not None:
        args += ['-taxids', taxids]
    if neg_taxids is not None:
        args += ['-negative_taxids', neg_taxids]

    # run BLAST (output to stdout)
    blast_out = run_cmd(
        *args, stdin=blast_in # input fasta as stdin
    ).stdout

    dtype = _get_blast_dtype(columns)
    def records():
        for row in csv.reader(StringIO(blast_out), delimiter='\t'):
            if len(row) != len(dtype.names):
                log_and_raise(ValueError, f'Expected {len(dtype.names)} BLAST fields, got {len(row)}')
            yield tuple(row)
    return np.fromiter(records(), dtype=dtype)


def blast(
    seq_list: Sequence[str],
    db: Path,
    task: Task = Task.blastn,
    columns: Sequence[str] | None = None,
    taxids: Sequence[int] | None = None,
    neg_taxids: Sequence[int] | None = None,
    n_cpu: int = 1,
    batch_size: int = 1000
) -> NDArray[np.void]:
    """Run BLAST on a list of sequences and return a structured array of the tabular output.
    - -max_hsps is set to 1000.
    - -max_target_seqs is set to 50000.
    - To avoid having too many query sequences in a single BLAST run, each run only takes a batch
    of sequences with batch size determined by batch_size.

    Args:
        seq_list (Sequence[str]): A list of sequences for BLAST.
        db (Path): Path to the BLAST database.
        task (str): Preset BLAST tasks ('blastn', 'blastn-short', 'megablast'). ['blastn']
        columns (Sequence[str] | None): Columns to be included in the structured array output. None to use default columns. [None]
        taxids (Sequence[int] | None): Only search for these taxonomy IDs and their descendants. [None]
        neg_taxids (Sequence[int] | None): Do NOT search for these taxonomy IDs and their descendants. [None]
        n_cpu (int, optional): Number of processes to run in parallel. [1]
        batch_size (int, optional): Number of query sequences in a single BLAST run. [1000]

    Returns:
        NDArray[np.void]: A structured array of the tabular output, with default columns,
            1. 'qseqid': query or source (e.g., gene) sequence id.
            2. 'sseqid': subject or target (e.g., reference genome) sequence id.
            3. 'length': alignment length.
            4. 'pident': percentage of identical matches (= nident / length).
            5. 'nident': number of identical matches (= length - mismatch - gaps).
            6. 'mismatch': number of mismatches.
            7. 'gapopen': number of gap openings.
            8. 'gaps': total number of gaps in both query and subject.
            9. 'qstart': start of alignment in query.
            10. 'qend': end of alignment in query.
            11. 'sstart': start of alignment in subject.
            12. 'send': end of alignment in subject.
            13. 'evalue': expect value.
            14. 'bitscore': bit score.
            15. 'qseq': aligned part of query sequence.
            16. 'sseq': aligned part of subject sequence.
    """
    # check input
    tot_seq = len(seq_list)
    if tot_seq == 0:
        log_and_raise(ValueError, 'No input sequence provided for BLAST')
    seq_idx = list(range(tot_seq)) # indices have to be set in advance (before batches)

    # convert inputs to blastn CLI compatible formats
    columns = _get_blast_dtype(columns).names
    outfmt = _get_blast_outfmt(columns) # value for blastn -outfmt
    if taxids is not None:
        taxids = ','.join(map(str, taxids))
    if neg_taxids is not None:
        neg_taxids = ','.join(map(str, neg_taxids))
    n_cpu = str(n_cpu)

    # run blast on batches of sequences
    logger.info(f' - Running blastn on {len(seq_list)} sequences, with batch size of {batch_size} (threads={n_cpu})...')
    batch_start = 0
    blast_out = list()
    while batch_start < tot_seq:
        logger.info(f' - {batch_start}/{tot_seq}')
        batch_stop = batch_start + batch_size
        blast_out.append(_blast_batch(
            seq_idx[batch_start: batch_stop],
            seq_list[batch_start: batch_stop],
            db, task, columns, outfmt,
            taxids, neg_taxids, n_cpu
        ))
        batch_start = batch_stop

    if len(blast_out) == 1:
        return blast_out[0]
    else:
        return np.concatenate(blast_out)
