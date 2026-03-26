"""
Command-line interface for Agribound.

Provides the ``agribound`` CLI with commands for field boundary delineation,
GEE authentication, and listing available engines and sources.
"""

from __future__ import annotations

import logging
import sys

import click

from agribound._version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="agribound")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """Agribound: Unified agricultural field boundary delineation toolkit."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option("--study-area", required=True, help="Path to GeoJSON/Shapefile or GEE asset ID.")
@click.option("--source", default="sentinel2", help="Satellite source.")
@click.option("--year", default=2024, type=int, help="Target year.")
@click.option("--engine", default="delineate-anything", help="Delineation engine.")
@click.option("--output", "-o", default=None, help="Output file path.")
@click.option("--output-format", default="gpkg", type=click.Choice(["gpkg", "geojson", "parquet"]))
@click.option("--gee-project", default=None, help="GEE project ID.")
@click.option("--export-method", default="local", type=click.Choice(["local", "gdrive", "gcs"]))
@click.option("--gcs-bucket", default=None, help="GCS bucket name.")
@click.option(
    "--composite-method",
    default="median",
    type=click.Choice(["median", "greenest", "max_ndvi"]),
)
@click.option("--date-range", nargs=2, default=None, help="Date range: START END (YYYY-MM-DD).")
@click.option("--cloud-cover-max", default=20, type=int, help="Max cloud cover %%.")
@click.option("--local-tif", default=None, help="Path to local GeoTIFF (source=local).")
@click.option("--min-area", default=2500.0, type=float, help="Min field area in m².")
@click.option("--simplify", default=2.0, type=float, help="Simplification tolerance.")
@click.option("--device", default="auto", type=click.Choice(["auto", "cuda", "cpu", "mps"]))
@click.option("--n-workers", default=4, type=int, help="Parallel workers.")
@click.option("--reference", default=None, help="Reference boundaries for evaluation/fine-tuning.")
@click.option("--fine-tune", is_flag=True, help="Fine-tune engine on reference boundaries.")
@click.option("--config", "config_file", default=None, help="YAML config file.")
def delineate(
    study_area,
    source,
    year,
    engine,
    output,
    output_format,
    gee_project,
    export_method,
    gcs_bucket,
    composite_method,
    date_range,
    cloud_cover_max,
    local_tif,
    min_area,
    simplify,
    device,
    n_workers,
    reference,
    fine_tune,
    config_file,
):
    """Run field boundary delineation."""
    from agribound.config import AgriboundConfig
    from agribound.pipeline import delineate as run_delineate

    if config_file:
        config = AgriboundConfig.from_yaml(config_file)
    else:
        if output is None:
            output = f"fields_{source}_{year}.{output_format}"
        config = AgriboundConfig(
            study_area=study_area,
            source=source,
            year=year,
            engine=engine,
            output_path=output,
            output_format=output_format,
            gee_project=gee_project,
            export_method=export_method,
            gcs_bucket=gcs_bucket,
            composite_method=composite_method,
            date_range=tuple(date_range) if date_range else None,
            cloud_cover_max=cloud_cover_max,
            local_tif_path=local_tif,
            min_field_area_m2=min_area,
            simplify_tolerance=simplify,
            device=device,
            n_workers=n_workers,
            reference_boundaries=reference,
            fine_tune=fine_tune,
        )

    gdf = run_delineate(config=config, study_area=config.study_area)
    click.echo(f"Delineated {len(gdf)} field boundaries → {config.output_path}")


@main.command("list-engines")
def list_engines_cmd():
    """List available delineation engines."""
    from agribound.engines import list_engines

    engines = list_engines()
    click.echo("\nAvailable Engines:")
    click.echo("-" * 80)
    for name, info in engines.items():
        gpu = "GPU" if info["gpu_required"] else "CPU"
        click.echo(f"  {name:<25} {info['approach']:<45} [{gpu}]")
    click.echo()


@main.command("list-sources")
def list_sources_cmd():
    """List available satellite sources."""
    from agribound.composites import list_sources

    sources = list_sources()
    click.echo("\nAvailable Satellite Sources:")
    click.echo("-" * 80)
    for name, info in sources.items():
        res = f"{info['resolution_m']}m" if info["resolution_m"] else "varies"
        gee = "GEE" if info["requires_gee"] else "local"
        restricted = " (restricted)" if info.get("restricted") else ""
        click.echo(f"  {name:<20} {info['name']:<30} {res:<8} [{gee}]{restricted}")
    click.echo()


@main.command()
@click.option("--project", default=None, help="GEE project ID.")
@click.option("--service-account-key", default=None, help="Path to service account JSON key.")
def auth(project, service_account_key):
    """Authenticate with Google Earth Engine."""
    from agribound.auth import setup_gee

    try:
        setup_gee(project=project, service_account_key=service_account_key)
        click.echo("GEE authentication successful!")
    except Exception as exc:
        click.echo(f"Authentication failed: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
