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
    help=f'Seqwin: Ultrafast identification of signature sequences', 
    add_completion=False, # do not show command completion options in help
    pretty_exceptions_show_locals=False, # do not show huge blocks of local variables
    add_help_option=False # disable built-in --help
)


def print_version(ctx: typer.Context, value: bool):
    if value:
        typer.echo(f'Seqwin v{__version__}')
        ctx.exit()


def print_help(ctx: typer.Context, value: bool):
    if value:
        typer.echo(ctx.get_help())
        ctx.exit()


@app.command()
def main(
    # even if only one value is provided, typer will still provide a list with only one element
    tar_taxa: list[str] | None = typer.Option(
        None, '--tar-taxa', '-t', show_default=False, 
        help='Target NCBI taxonomy name or ID. Must be an exact match. '
        'Repeat the option to pass multiple values (-t <tax1> -t <tax2> ...).', 
        rich_help_panel='Input selection'
    ), 
    neg_taxa: list[str] | None = typer.Option(
        None, '--neg-taxa', '-n', show_default=False, 
        help='Non-target NCBI taxonomy name or ID. Must be an exact match. '
        'Repeat the option to pass multiple values (-n <tax1> -n <tax2> ...).', 
        rich_help_panel='Input selection'
    ), 
    tar_paths: Path | None = typer.Option(
        None, '--tar-paths', show_default=False, 
        help='Text file containing paths to target genome FASTA files, one path per line. Gzipped FASTA is supported.', 
        rich_help_panel='Input selection'
    ), 
    neg_paths: Path | None = typer.Option(
        None, '--neg-paths', show_default=False, 
        help='Text file containing paths to non-target genome FASTA files, one path per line.', 
        rich_help_panel='Input selection'
    ), 
    tar_dir: Path | None = typer.Option(
        None, '--tar-dir', show_default=False,
        help='Directory containing target genome FASTA files.', 
        rich_help_panel='Input selection'
    ), 
    neg_dir: Path | None = typer.Option(
        None, '--neg-dir', show_default=False,
        help='Directory containing non-target genome FASTA files.', 
        rich_help_panel='Input selection'
    ), 
    prefix: Path = typer.Option(
        Path.cwd(), '--prefix', 
        help='Parent path where the output directory will be created. Use the current working directory by default.', 
        rich_help_panel='Output options'
    ), 
    title: str = typer.Option(
        'seqwin-out', '--title', '-o', 
        help='Name of the output directory created under --prefix.', 
        rich_help_panel='Output options'
    ), 
    overwrite: bool = typer.Option(
        False, '--overwrite', show_default=False, 
        help='Overwrite existing output files.', 
        rich_help_panel='Output options'
    ), 
    kmerlen: int = typer.Option(
        21, '--kmerlen', '-k', 
        help='K-mer length.', 
        rich_help_panel='Signature options'
    ), 
    windowsize: int = typer.Option(
        200, '--windowsize', '-w', 
        help='Window size for minimizer sketch.', 
        rich_help_panel='Signature options'
    ), 
    penalty_th: float | None = typer.Option(
        None, '--penalty-th', show_default=False, 
        help='Node penalty threshold, from 0 to 1. If not provided, Seqwin computes it automatically.', 
        rich_help_panel='Signature options'
    ), 
    # always default flags to False (can be reversed later in the function body)
    no_mash: bool = typer.Option(
        False, '--no-mash', show_default=False, 
        help='Do not run Mash to estimate node penalty threshold. Instead, use minimizer sketches. '
        'This is much faster but the estimation might be biased. '
        'Only used when --penalty-th is not provided.', 
        rich_help_panel='Signature options'
    ), 
    stringency: int = typer.Option(
        5, '--stringency', '-s', show_default=True, 
        help='Controls the sensitivity and specificity of output signatures (0-10). '
        'Increasing this value generally yields fewer and shorter signatures, while improving their sensitivity and specificity. '
        'Internally, Seqwin uses this setting to adjust the estimated node penalty threshold. '
        'Only used when `--penalty-th` is not provided.', 
        rich_help_panel='Signature options'
    ), 
    min_len: int = typer.Option(
        200, '--min-len', 
        help='Minimum length of output signatures.', 
        rich_help_panel='Signature options'
    ), 
    max_len: int | None = typer.Option(
        None, '--max-len', show_default=False, 
        help='Estimated maximum length of output signatures. If not provided, no explicit limit is applied.', 
        rich_help_panel='Signature options'
    ), 
    no_blast: bool = typer.Option(
        False, '--no-blast', show_default=False, 
        help='Do not evaluate signature sequences with BLAST.', 
        rich_help_panel='Signature options'
    ), 
    # blast_neg_only: bool = typer.Option(
    #     False, '--fast-blast', is_flag=True, flag_value=True, show_default=False, 
    #     help='Only evaluate (BLAST) against non-target assemblies.'
    # ), 
    level: Level = typer.Option(
        Level.contig, '--level', metavar='TEXT', # hide choices
        help="Limit downloads to genomes at or above this assembly level. "
        "Possible values follow this order: 'contig', 'scaffold', 'chromosome', 'complete'", 
        rich_help_panel='NCBI download options'
    ), 
    source: Source = typer.Option(
        Source.genbank, '--source', metavar='TEXT', # hide choices
        help="Genome source to download from. Supported values: 'genbank', 'refseq'", 
        rich_help_panel='NCBI download options'
    ), 
    annotated: bool = typer.Option(
        False, '--annotated', show_default=False, 
        help='Only include annotated genomes.', 
        rich_help_panel='NCBI download options'
    ), 
    exclude_mag: bool = typer.Option(
        False, '--exclude-mag', show_default=False, 
        help='Exclude metagenome-assembled genomes (MAGs).', 
        rich_help_panel='NCBI download options'
    ), 
    no_gzip: bool = typer.Option(
        False, '--no-gzip', show_default=False, 
        help='Do not download genomes as gzipped FASTA.', 
        rich_help_panel='NCBI download options'
    ), 
    download_only: bool = typer.Option(
        False, '--download-only', show_default=False, 
        help='Only download genome sequences without running Seqwin.', 
        rich_help_panel='NCBI download options'
    ), 
    seed: int = typer.Option(
        42, '--seed', 
        help='Random seed for reproducibility.', 
        rich_help_panel='Miscellaneous'
    ), 
    n_cpu: int = typer.Option(
        4, '--threads', '-p', 
        help='Number of parallel processes or threads to use.', 
        rich_help_panel='Miscellaneous'
    ), 
    version: bool = typer.Option(
        False, '--version', callback=print_version, show_default=False, expose_value=False, 
        is_eager=True, # run this before any other options
        help='Show Seqwin version and exit.', 
        rich_help_panel='Miscellaneous'
    ), 
    help_: bool = typer.Option(
        False, '--help', '-h', callback=print_help, show_default=False, expose_value=False, 
        is_eager=True, # run this before any other options
        help='Show this message and exit.', 
        rich_help_panel='Miscellaneous'
    )
):
    if not download_only:
        if (tar_paths is None) and (tar_taxa is None) and (tar_dir is None):
            raise typer.BadParameter('You must provide at least one target input: --tar-paths, --tar-taxa, or --tar-dir')
        elif (neg_paths is None) and (neg_taxa is None) and (neg_dir is None):
            raise typer.BadParameter('You must provide at least one non-target input: --neg-paths, --neg-taxa, or --neg-dir')

    config = Config(
        tar_taxa=tar_taxa, 
        neg_taxa=neg_taxa, 
        tar_paths=tar_paths, 
        neg_paths=neg_paths, 
        tar_dir=tar_dir,
        neg_dir=neg_dir,
        prefix=prefix, 
        title=title, 
        overwrite=overwrite, 
        kmerlen=kmerlen, 
        windowsize=windowsize, 
        penalty_th=penalty_th, 
        run_mash=not no_mash, 
        stringency=stringency, 
        min_len=min_len, 
        max_len=max_len, 
        run_blast=not no_blast, 
        #blast_neg_only=blast_neg_only, 
        seed=seed, 
        n_cpu=n_cpu, 
        level=level, 
        source=source, 
        annotated=annotated, 
        exclude_mag=exclude_mag, 
        gzip=not no_gzip, 
        download_only=download_only
    )
    _ = run(config)
