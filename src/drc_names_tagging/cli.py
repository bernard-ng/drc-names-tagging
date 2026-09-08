from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from drc_names_tagging.config import (
    DEFAULT_DATASET_PATH,
    DEFAULT_EXPERIMENT_TEMPLATES_PATH,
    ExperimentSettings,
)
from drc_names_tagging.experiments import (
    ExperimentBuilder,
    TransitionBuilder,
    compare,
    compare_tokens,
    run_one,
    run_tokens,
)

app = typer.Typer(
    help="Run configuration-driven native/foreign token-tagging experiments.",
    no_args_is_help=True,
)
experiments_app = typer.Typer(help="Inspect configured tagging experiments.")
app.add_typer(experiments_app, name="experiments")
tokens_app = typer.Typer(help="Build reusable unique-token annotation datasets.")
app.add_typer(tokens_app, name="tokens")


@app.command("transitions")
def transitions_command(
    dataset: Annotated[Path, typer.Option(help="Annotated names.csv input.")] = DEFAULT_DATASET_PATH,
) -> None:
    """Generate native/foreign Markov transition matrices."""

    results = TransitionBuilder(dataset).run()
    for result in results:
        typer.echo(result.summary())


@app.command("tag")
def tag_command(
    name: Annotated[
        str, typer.Option("--name", help="Configured experiment name.")
    ] = "ollama_mistral_7b",
    experiment_type: Annotated[
        str, typer.Option("--type", help="Template section: baseline or advanced.")
    ] = "baseline",
    dataset: Annotated[Path, typer.Option(help="Shared names.csv input.")] = DEFAULT_DATASET_PATH,
    templates: Annotated[
        Path, typer.Option(help="Experiment template definitions.")
    ] = DEFAULT_EXPERIMENT_TEMPLATES_PATH,
    output: Annotated[Path | None, typer.Option(help="Optional CSV output path.")] = None,
) -> None:
    """Run one configured tagging experiment."""

    settings = ExperimentSettings(dataset_path=dataset, templates_path=templates)
    experiment = ExperimentBuilder(settings).build(name, experiment_type=experiment_type)
    run_one(experiment, settings, output=output)


@tokens_app.command("tag")
def tokens_tag_command(
    name: Annotated[
        str, typer.Option("--name", help="Configured experiment name.")
    ] = "ollama_mistral_7b",
    experiment_type: Annotated[
        str, typer.Option("--type", help="Template section: baseline or advanced.")
    ] = "baseline",
    dataset: Annotated[Path, typer.Option(help="Shared published dataset input.")] = DEFAULT_DATASET_PATH,
    templates: Annotated[
        Path, typer.Option(help="Experiment template definitions.")
    ] = DEFAULT_EXPERIMENT_TEMPLATES_PATH,
    output: Annotated[Path | None, typer.Option(help="Optional CSV output path.")] = None,
) -> None:
    """Tag every unique token once and save the lexical annotations."""

    settings = ExperimentSettings(dataset_path=dataset, templates_path=templates)
    experiment = ExperimentBuilder(settings).build(name, experiment_type=experiment_type)
    run_tokens(experiment, settings, output=output)


@tokens_app.command("compare")
def tokens_compare_command(
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Configured experiment name; repeat to select."),
    ] = None,
    experiment_type: Annotated[
        str, typer.Option("--type", help="Template section: baseline or advanced.")
    ] = "baseline",
    dataset: Annotated[Path, typer.Option(help="Shared published dataset input.")] = DEFAULT_DATASET_PATH,
    templates: Annotated[
        Path, typer.Option(help="Experiment template definitions.")
    ] = DEFAULT_EXPERIMENT_TEMPLATES_PATH,
    output_dir: Annotated[
        Path | None, typer.Option(help="Optional comparison output directory.")
    ] = None,
) -> None:
    """Compare unique-token annotations and save preferred training data."""

    settings = ExperimentSettings(dataset_path=dataset, templates_path=templates)
    builder = ExperimentBuilder(settings)
    selected = names or [
        str(template["name"]) for template in builder.templates(experiment_type)
    ]
    experiments = [
        builder.build(item, experiment_type=experiment_type) for item in selected
    ]
    compare_tokens(
        experiments,
        settings,
        reference_tagger=builder.token_reference(),
        output_dir=output_dir,
    )


@app.command("compare")
def compare_command(
    names: Annotated[
        list[str] | None,
        typer.Option("--name", help="Configured experiment name; repeat to select."),
    ] = None,
    experiment_type: Annotated[
        str, typer.Option("--type", help="Template section: baseline or advanced.")
    ] = "baseline",
    dataset: Annotated[Path, typer.Option(help="Shared names.csv input.")] = DEFAULT_DATASET_PATH,
    templates: Annotated[
        Path, typer.Option(help="Experiment template definitions.")
    ] = DEFAULT_EXPERIMENT_TEMPLATES_PATH,
    output_dir: Annotated[
        Path | None, typer.Option(help="Optional comparison output directory.")
    ] = None,
) -> None:
    """Compare selected configured experiments on identical names."""

    settings = ExperimentSettings(dataset_path=dataset, templates_path=templates)
    builder = ExperimentBuilder(settings)
    selected = names or [
        str(template["name"]) for template in builder.templates(experiment_type)
    ]
    experiments = [
        builder.build(item, experiment_type=experiment_type) for item in selected
    ]
    compare(experiments, settings, output_dir=output_dir)


@experiments_app.command("list")
def experiments_list(
    experiment_type: Annotated[
        str, typer.Option("--type", help="Template section: baseline or advanced.")
    ] = "baseline",
    templates: Annotated[
        Path, typer.Option(help="Experiment template definitions.")
    ] = DEFAULT_EXPERIMENT_TEMPLATES_PATH,
) -> None:
    """List configured tagging experiments."""

    settings = ExperimentSettings(templates_path=templates)
    builder = ExperimentBuilder(settings)
    values = [
        builder.build(str(item["name"]), experiment_type=experiment_type).to_dict()
        for item in builder.templates(experiment_type)
    ]
    typer.echo(json.dumps(values, indent=2, default=str))
