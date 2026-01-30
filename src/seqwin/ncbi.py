"""
NCBI
====

- Search and download genome assemblies with NCBI `Datasets <https://www.ncbi.nlm.nih.gov/datasets/>`__. 
- Run NCBI `BLAST+ <https://www.ncbi.nlm.nih.gov/books/NBK131777/>`__. 

Dependencies:
-------------
- pandas
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

import shutil, json, logging
from pathlib import Path
from time import time
from enum import Enum
from io import StringIO
from collections.abc import Sequence

logger = logging.getLogger(__name__)

import pandas as pd
from .utils import print_time_delta, log_and_raise, file_to_write, list_dir, run_cmd

_ZIP_EXT = '.zip' # File extension of NCBI genome package. ['.zip']
_BLAST_COL = ( # Default columns to be included in the BLAST tsv output
    'qseqid', # 1. query or source (e.g., gene) sequence id
    'sseqid', # 2. subject or target (e.g., reference genome) sequence id
    'length', # 3. alignment length
    'pident', # 4. percentage of identical matches (= nident / length)
    'nident', # 5. number of identical matches (= length - mismatch - gaps)
    'mismatch', # 6. number of mismatches
    'gapopen', # 7. number of gap openings
    'gaps', # 8. total number of gaps
    'qstart', # 9. start of alignment in query
    'qend', # 10. end of alignment in query
    'sstart', # 11. start of alignment in subject
    'send', # 12. end of alignment in subject
    'evalue', # 13. expect value
    'bitscore', # 14. bit score
    'qseq', # 15. aligned part of query sequence
    'sseq', # 16. aligned part of subject sequence
)
_MAX_REHYDRATE_WORKERS = 30 # Maximum number of CPUs for `datasets rehydrate`
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

if shutil.which('datasets') is None:
    raise ImportError('ncbi-datasets-cli is not installed (`datasets` is not found in your PATH).')
if shutil.which('blastn') is None:
    raise ImportError('BLAST+ is not installed (`blastn` is not found in your PATH).')


def search_taxon(taxon: str) -> tuple[str, str] | None:
    """Search a taxon on NCBI Taxonomy. Internet connection is needed. 

    Args:
        taxon (str): Name or ID of the taxon (exact match). 

    Returns:
        tuple | None: A tuple containing
            1. tax_id (str): NCBI Taxonomy ID of the taxon. 
            2. tax_name (str): Current scientific name of the taxon. 
        
        Return None if the taxon is not found. 
    """
    logger.info(f'Searching NCBI Taxonomy for "{taxon}"...')
    tik = time()
    summary = run_cmd(
        'datasets', 'summary', 'taxonomy', 'taxon', str(taxon), 
        '--as-json-lines', # output as json
        '--report', 'names' # do not output tax ids of children
    )

    if summary.stdout == '':
        logger.debug(summary.stderr)
        logger.error(f' - Unable to find taxon "{taxon}"')
        return None
    
    summary = json.loads(summary.stdout)
    tax_id = summary['taxonomy']['tax_id']
    tax_name = summary['taxonomy']['current_scientific_name']['name']
    logger.info(f' - Found NCBI Taxonomy ID: {tax_id}')
    logger.info(f' - Found NCBI Taxon: {tax_name}')
    print_time_delta(time()-tik)
    return tax_id, tax_name


def get_assembly_paths(package_dir: Path) -> list[Path]:
    """Get the file paths of all genome assemblies in a NCBI genome package. 

    Args:
        package_dir (Path): Path of the genome package directory (e.g., ncbi_dataset). 

    Returns:
        list[str]: File paths of all genome assemblies in the package. 
    """
    # sanity check
    if not package_dir.is_dir():
        log_and_raise(NotADirectoryError, f'Not a directory: {package_dir}')

    prefix = package_dir / 'ncbi_dataset' / 'data' # hard coded entries within package_dir
    assemblies = list_dir(prefix, mode='d')
    paths = list()
    for assembly_dir in assemblies:
        assembly_path = list_dir(assembly_dir, mode='f')
        if len(assembly_path) != 1:
            logger.warning(f' - Found more than one files under {assembly_dir}')
        paths.append(assembly_path[0])
    return paths


def download_taxon(
    taxon: str, 
    prefix: Path=Path.cwd(), 
    format: Format=Format.fasta, 
    level: Level=Level.contig, 
    source: Source=Source.genbank, 
    annotated: bool=True, 
    exclude_mag: bool=False, 
    gzip: bool=True, 
    n_cpu: int=1
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
        n_cpu (int, optional): Number of processes to run in parallel. [1]

    Returns:
        list[Path] | None: File paths of downloaded genome assemblies. Return None if the taxon is not found. 
    """
    # sanity check
    if not prefix.is_dir():
        log_and_raise(NotADirectoryError, f'Cannot download genomes to this location, since it is not a directory: {prefix}')
    n_cpu = min(n_cpu, _MAX_REHYDRATE_WORKERS) # --max-workers for rehydrate is 30

    # reuse existing genome package
    tax_dir = prefix / taxon.replace(' ', '-')
    if tax_dir.exists():
        logger.warning(f'Existing genome package is found {tax_dir}')
        assembly_paths = get_assembly_paths(tax_dir)
        logger.info(f' - Found {len(assembly_paths)} genome assemblies.')
        return assembly_paths

    # search taxon id and name
    try:
        tax_id, tax_name = search_taxon(taxon)
    except TypeError:
        return None

    # path to the output genome package
    tax_dir = prefix / tax_name.replace(' ', '-') # avoid spaces in file paths

    # add .zip to tax_dir (don't use with_suffix since tax_name might contain '.')
    tax_zip = tax_dir.with_name(tax_dir.name + _ZIP_EXT)
    file_to_write(tax_zip, overwrite=False)

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
    
    logger.info(f'Downloading genome package for NCBI Taxonomy ID {tax_id}...')
    tik = time()

    # download metadata
    logger.info(' - Downloading dehydrated genome package (metadata only)...')
    download_log = run_cmd(*args, raise_error=False)
    if download_log.returncode != 0:
        logger.debug(download_log.stderr)
        logger.error(f' - No genome assemblies were found for NCBI Taxonomy ID {tax_id}, try loosen the filters.')
        return None

    # unzip dehydrated package
    logger.info(' - Unzipping dehydrated genome package...')
    unzip_log = run_cmd('unzip', tax_zip, '-d', tax_dir, raise_error=False)
    if unzip_log.returncode != 0:
        logger.error(unzip_log.stderr)
        shutil.rmtree(tax_dir, ignore_errors=True) # remove incomplete dir
        log_and_raise(msg=f'Failed to unzip genome package for NCBI Taxonomy ID {tax_id}: {tax_zip}')
    
    # download actual sequences
    args = [
        'datasets', 'rehydrate', '--directory', tax_dir, 
        '--max-workers', str(n_cpu), '--no-progressbar'
    ]
    if gzip:
        logger.info(' - Rehydrating in gzip format...')
        args += ['--gzip']
    else:
        logger.info(' - Rehydrating...')
    rehydrate_log = run_cmd(*args, raise_error=False)
    if rehydrate_log.returncode != 0:
        logger.error(rehydrate_log.stderr)
        shutil.rmtree(tax_dir, ignore_errors=True) # remove incomplete dir
        log_and_raise(msg=f'Failed to rehydrate data package for taxon "{taxon}"')
    
    # get the file path of each assembly
    assembly_paths = get_assembly_paths(tax_dir)
    logger.info(f' - Downloaded {len(assembly_paths)} genome assemblies for NCBI Taxonomy ID {tax_id}.')

    print_time_delta(time()-tik)
    return assembly_paths


def _get_blast_outfmt(columns: Sequence[str]) -> str:
    """Given a tuple of columns to be included in the BLAST TSV output, get the value to be provided to 
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
) -> pd.DataFrame:
    """Run BLAST on a list of sequences and return a DataFrame of the tabular output. 

    Args:
        seq_idx (Sequence[int]): indices of the input sequences. 
        seq_list (Sequence[str]): A list of sequences for BLAST. 
        db (Path): Path to the BLAST database. 
        task (str): Preset BLAST tasks ('blastn', 'blastn-short', 'megablast'). 
        columns (Sequence[str]): Columns to be included in the DataFrame output. 
        outfmt (str): Output of `_get_blast_outfmt()`. 
        n_cpu (int): Number of processes to run in parallel. 

    Returns:
        pd.DataFrame: A DataFrame of the tabular output. 
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

    # convert BLAST output into a df. pd.read_csv() does auto type conversion (e.g., 'qseqid' as int)
    return pd.read_csv(StringIO(blast_out), sep='\t', header=None, names=columns, index_col=False)


def blast(
    seq_list: Sequence[str], 
    db: Path, 
    task: Task=Task.blastn, 
    columns: Sequence[str] | None=None, 
    taxids: Sequence[int] | None=None, 
    neg_taxids: Sequence[int] | None=None, 
    n_cpu: int=1, 
    batch_size: int=1000
) -> pd.DataFrame:
    """Run BLAST on a list of sequences and return a DataFrame of the tabular output. 
    - -max_hsps is set to 1000. 
    - -max_target_seqs is set to 50000. 
    - To avoid having too many query sequences in a single BLAST run, each run only takes a batch
    of sequences with batch size determined by batch_size. 

    Args:
        seq_list (Sequence[str]): A list of sequences for BLAST. 
        db (Path): Path to the BLAST database. 
        task (str): Preset BLAST tasks ('blastn', 'blastn-short', 'megablast'). ['blastn']
        columns (Sequence[str] | None): Columns to be included in the DataFrame output. None to use default columns. [None]
        n_cpu (int, optional): Number of processes to run in parallel. [1]
        batch_size (int, optional): Number of query sequences in a single BLAST run. [1000]

    Returns:
        pd.DataFrame: A DataFrame of the tabular output, with default columns, 
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
    if columns is None:
        # use default columns (including sequences)
        columns = _BLAST_COL
    outfmt = _get_blast_outfmt(columns) # value for blastn -outfmt
    if taxids is not None:
        taxids = ','.join(map(str, taxids))
    if neg_taxids is not None:
        neg_taxids = ','.join(map(str, neg_taxids))
    n_cpu = str(n_cpu)

    # run blast on batches of sequences
    logger.info(f' - Running blastn on {len(seq_list)} sequences, with batch size of {batch_size} (threads={n_cpu})...')
    batch_start = 0
    blast_out: list[pd.DataFrame] = list()

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

    # concat output dfs
    if len(blast_out) == 1:
        return blast_out[0]
    else:
        return pd.concat(blast_out, axis=0, ignore_index=True)
