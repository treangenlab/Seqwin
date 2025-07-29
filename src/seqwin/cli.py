"""
Seqwin CLI
==========

Dependencies:
-------------
- typer
- .core
- .ncbi
- .config
- ._version
"""

__author__ = 'Michael X. Wang'
__license__ = 'GPL 3.0'

from pathlib import Path

import typer

from .core import run
from .ncbi import Level, Source
from .config import Config
from ._version import __version__


app = typer.Typer(
    help=f'Seqwin: rapid and sensitive search of clade-specific markers', 
    add_completion=False, # do not show command completion options in help
    pretty_exceptions_show_locals=False # do not show huge blocks of local variables
)


def print_version(ctx: typer.Context, value: bool):
    if value:
        typer.echo(f'Seqwin v{__version__}')
        ctx.exit()


@app.command()
def main(
    # even if only one value is provided, typer will still provide a list with only one element
    tar_taxa: list[str] | None = typer.Option(
        None, '--tar-taxa', '-t', show_default=False, 
        help='Target NCBI taxonomy name / ID. Must be exact match. Repeat the option to pass multiple values (-t <tax1> -t <tax2> ...).'
    ), 
    neg_taxa: list[str] | None = typer.Option(
        None, '--neg-taxa', '-n', show_default=False, 
        help='Non-target NCBI taxonomy name / ID. Must be exact match. Repeat the option to pass multiple values (-n <tax1> -n <tax2> ...).'
    ), 
    tar_paths: Path | None = typer.Option(
        None, '--tar-paths', show_default=False, 
        help='A text file of paths to target assemblies in FASTA format (gzip supported), with one path per line.'
    ), 
    neg_paths: Path | None = typer.Option(
        None, '--neg-paths', show_default=False, 
        help='A text file of paths to non-target assemblies in FASTA format (gzip supported), with one path per line.'
    ), 
    prefix: Path = typer.Option(
        Path.cwd(), '--prefix', help='Path prefix for the output directory. Use the current directory by default.'
    ), 
    title: str = typer.Option(
        'seqwin-out', '--title', '-o', help='Name of the output directory.'
    ), 
    overwrite: bool = typer.Option(
        False, '--overwrite', is_flag=True, flag_value=True, 
        help='Overwrite existing output files.'
    ), 
    kmerlen: int = typer.Option(
        21, '--kmerlen', '-k', help='K-mer length.'
    ), 
    windowsize: int = typer.Option(
        200, '--windowsize', '-w', help='Window size for minimizer sketch.'
    ), 
    penalty_th: float | None = typer.Option(
        None, '--penalty-th', show_default=False, 
        help='Node penalty threshold, ranging between [0, 1]. Computed with Jaccard indexes if not provided.'
    ), 
    min_len: int = typer.Option(
        200, '--min-len', help='Min length of output markers.'
    ), 
    max_len: int | None = typer.Option(
        None, '--max-len', show_default=False, 
        help='Max length of output markers (estimated). No explicit limit if not provided.'
    ), 
    run_blast: bool = typer.Option(
        True, '--no-blast', is_flag=True, flag_value=False, show_default=False, 
        help='Do NOT evaluate (BLAST) marker sequences.'
    ), 
    # blast_neg_only: bool = typer.Option(
    #     False, '--fast-blast', is_flag=True, flag_value=True, show_default=False, 
    #     help='Only evaluate (BLAST) against non-target assemblies.'
    # ), 
    seed: int = typer.Option(
        42, '--seed', help='Random seed for reproducibility.'
    ), 
    n_cpu: int = typer.Option(
        1, '--threads', '-p', help='Number of threads to use.'
    ), 
    level: Level = typer.Option(
        Level.contig, '--level', metavar='[contig|scaffold|\nchromosome|complete]', # show choices in two lines
        help='NCBI download option. Limit to genomes ≥ this assembly level (contig < scaffold < chromosome < complete).'
    ), 
    source: Source = typer.Option(
        Source.genbank, '--source', 
        help="NCBI download option. Genome source ('genbank' or 'refseq')."
    ), 
    annotated: bool = typer.Option(
        False, '--annotated', is_flag=True, flag_value=True, show_default=False, 
        help='NCBI download option. Only include annotated genomes.'
    ), 
    exclude_mag: bool = typer.Option(
        False, '--exclude-mag', is_flag=True, flag_value=True, show_default=False, 
        help='NCBI download option. Exclude metagenome-assembled genomes (MAGs).'
    ), 
    gzip: bool = typer.Option(
        True, '--no-gzip', is_flag=True, flag_value=False, show_default=False, 
        help='NCBI download option. Do NOT download genomes as gzipped FASTA.'
    ), 
    download_only: bool = typer.Option(
        False, '--download-only', is_flag=True, flag_value=True, show_default=False, 
        help='Only download genome sequences without running Seqwin.'
    ), 
    version: bool = typer.Option(
        False, '--version', callback=print_version, is_eager=True, # run this before any other options
        is_flag=True, flag_value=True, show_default=False, 
        help='Show Seqwin version and exit.'
    )
):
    if not download_only:
        if (tar_paths is None) and (tar_taxa is None):
            raise typer.BadParameter('You must provide either --tar-paths or --tar-taxa')
        elif (neg_paths is None) and (neg_taxa is None):
            raise typer.BadParameter('You must provide either --neg-paths or --neg-taxa')
    elif (penalty_th is not None) and (penalty_th < 0 or penalty_th > 1):
            raise ValueError('--penalty-th must be between [0, 1]')
    elif (max_len is not None) and (max_len < min_len):
        raise typer.BadParameter('--max-len must be greater than --min-len')

    config = Config(
        tar_taxa=tar_taxa, 
        neg_taxa=neg_taxa, 
        tar_paths=tar_paths, 
        neg_paths=neg_paths, 
        prefix=prefix, 
        title=title, 
        overwrite=overwrite, 
        kmerlen=kmerlen, 
        windowsize=windowsize, 
        penalty_th=penalty_th, 
        min_len=min_len, 
        max_len=max_len, 
        run_blast=run_blast, 
        #blast_neg_only=blast_neg_only, 
        seed=seed, 
        n_cpu=n_cpu, 
        level=level, 
        source=source, 
        annotated=annotated, 
        exclude_mag=exclude_mag, 
        gzip=gzip, 
        download_only=download_only
    )
    _ = run(config)
