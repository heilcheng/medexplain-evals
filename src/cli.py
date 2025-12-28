"""Modern CLI for MEQ-Bench using Typer.

This module provides a rich command-line interface for running benchmarks,
evaluating models, and generating leaderboards.

Usage:
    meq-bench run --model gpt-4o --output results/
    meq-bench evaluate --input responses.json
    meq-bench leaderboard --input results/
"""

from __future__ import annotations

import asyncio
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table


app = typer.Typer(
    name="meq-bench",
    help="MEQ-Bench: Evaluate Audience-Adaptive Medical Explanations in LLMs",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


class ModelBackend(StrEnum):
    """Supported model backends."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"
    MLX = "mlx"
    LOCAL = "local"


class OutputFormat(StrEnum):
    """Output format options."""

    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        from src import __version__

        console.print(f"[bold blue]MEQ-Bench[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-V",
            help="Enable verbose output",
        ),
    ] = False,
) -> None:
    """MEQ-Bench: A Resource-Efficient Benchmark for Medical LLM Evaluation."""
    if verbose:
        from src.logging import configure_logging

        configure_logging(level="DEBUG", json_output=False)


@app.command()
def run(
    model: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model name or path (e.g., gpt-4o, claude-3-opus, meta-llama/Llama-3-8B)",
        ),
    ],
    backend: Annotated[
        ModelBackend,
        typer.Option(
            "--backend",
            "-b",
            help="Model backend to use",
        ),
    ] = ModelBackend.OPENAI,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output directory for results",
        ),
    ] = Path("results/"),
    dataset: Annotated[
        Optional[str],
        typer.Option(
            "--dataset",
            "-d",
            help="Dataset to use (medquad, medqa, icliniq, cochrane, healthsearchqa)",
        ),
    ] = None,
    samples: Annotated[
        Optional[int],
        typer.Option(
            "--samples",
            "-n",
            help="Number of samples to evaluate (default: all)",
        ),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size for parallel processing",
        ),
    ] = 10,
    async_mode: Annotated[
        bool,
        typer.Option(
            "--async/--sync",
            help="Use async API calls for better throughput",
        ),
    ] = True,
) -> None:
    """Run benchmark evaluation on a model.

    Examples:
        meq-bench run --model gpt-4o --backend openai
        meq-bench run --model claude-3-opus --backend anthropic --samples 100
        meq-bench run --model meta-llama/Llama-3-8B --backend huggingface
    """
    from src.logging import configure_logging, get_logger

    configure_logging(level="INFO")
    logger = get_logger(__name__)

    console.print(
        Panel.fit(
            f"[bold blue]Running MEQ-Bench Evaluation[/]\n"
            f"Model: [green]{model}[/]\n"
            f"Backend: [yellow]{backend}[/]\n"
            f"Output: [cyan]{output}[/]",
            title="Configuration",
        )
    )

    output.mkdir(parents=True, exist_ok=True)

    try:
        if async_mode:
            asyncio.run(_run_async_benchmark(model, backend, output, dataset, samples, batch_size))
        else:
            _run_sync_benchmark(model, backend, output, dataset, samples, batch_size)

        console.print("\n[bold green]✓ Benchmark completed successfully![/]")
        console.print(f"Results saved to: [cyan]{output}[/]")

    except Exception as e:
        logger.exception("Benchmark failed", error=str(e))
        console.print(f"\n[bold red]✗ Benchmark failed:[/] {e}")
        raise typer.Exit(code=1)


async def _run_async_benchmark(
    model: str,
    backend: ModelBackend,
    output: Path,
    dataset: str | None,
    samples: int | None,
    batch_size: int,
) -> None:
    """Run benchmark with async API calls."""
    from src.benchmark import MEQBench
    from src.evaluator import MEQBenchEvaluator
    from src.api_client import create_async_client

    async with create_async_client(backend, model) as client:
        bench = MEQBench()
        evaluator = MEQBenchEvaluator()

        items = bench.load_dataset(dataset) if dataset else bench.get_all_items()
        if samples:
            items = items[:samples]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Evaluating...", total=len(items))

            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                batch_results = await asyncio.gather(*[_evaluate_item_async(client, evaluator, item) for item in batch])
                results.extend(batch_results)
                progress.update(task, advance=len(batch))

        # Save results
        import json

        results_file = output / f"{model.replace('/', '_')}_results.json"
        with results_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)


async def _evaluate_item_async(client, evaluator, item):
    """Evaluate a single item asynchronously."""
    explanations = await client.generate_explanations(item.medical_content)
    scores = evaluator.evaluate_all_audiences(item.medical_content, explanations)
    return {
        "item_id": item.id,
        "explanations": explanations,
        "scores": scores,
    }


def _run_sync_benchmark(
    model: str,
    backend: ModelBackend,
    output: Path,
    dataset: str | None,
    samples: int | None,
    batch_size: int,
) -> None:
    """Run benchmark synchronously."""
    from src.benchmark import MEQBench
    from src.evaluator import MEQBenchEvaluator

    bench = MEQBench()
    evaluator = MEQBenchEvaluator()

    items = bench.load_dataset(dataset) if dataset else bench.get_all_items()
    if samples:
        items = items[:samples]

    console.print(f"Loaded {len(items)} items for evaluation")

    # Placeholder for sync implementation
    console.print("[yellow]Sync mode evaluation...[/]")


@app.command()
def evaluate(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input file with model responses (JSON format)",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file for evaluation results",
        ),
    ] = Path("evaluation_results.json"),
    judge_model: Annotated[
        str,
        typer.Option(
            "--judge",
            "-j",
            help="Model to use as LLM judge",
        ),
    ] = "gpt-4-turbo",
) -> None:
    """Evaluate model responses from a file.

    Examples:
        meq-bench evaluate responses.json
        meq-bench evaluate responses.json --output scores.json --judge gpt-4o
    """
    import json

    from src.evaluator import MEQBenchEvaluator
    from src.logging import configure_logging

    configure_logging(level="INFO")

    if not input_file.exists():
        console.print(f"[bold red]Error:[/] Input file not found: {input_file}")
        raise typer.Exit(code=1)

    console.print(f"Loading responses from [cyan]{input_file}[/]...")

    with input_file.open() as f:
        responses = json.load(f)

    evaluator = MEQBenchEvaluator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating responses...", total=len(responses))

        results = []
        for response in responses:
            scores = evaluator.evaluate_all_audiences(response.get("medical_content", ""), response.get("explanations", {}))
            results.append({"item_id": response.get("id"), "scores": scores})
            progress.update(task, advance=1)

    with output.open("w") as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[bold green]✓ Evaluation complete![/] Results saved to [cyan]{output}[/]")


@app.command()
def leaderboard(
    input_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory containing evaluation results",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output file for leaderboard",
        ),
    ] = Path("leaderboard.html"),
    format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            "-f",
            help="Output format",
        ),
    ] = OutputFormat.HTML,
    title: Annotated[
        str,
        typer.Option(
            "--title",
            "-t",
            help="Leaderboard title",
        ),
    ] = "MEQ-Bench Leaderboard",
) -> None:
    """Generate a leaderboard from evaluation results.

    Examples:
        meq-bench leaderboard results/
        meq-bench leaderboard results/ --format markdown --output README.md
    """
    import json

    from src.leaderboard import LeaderboardGenerator

    if not input_dir.exists():
        console.print(f"[bold red]Error:[/] Input directory not found: {input_dir}")
        raise typer.Exit(code=1)

    # Load all result files
    result_files = list(input_dir.glob("*_results.json"))
    if not result_files:
        console.print(f"[bold yellow]Warning:[/] No result files found in {input_dir}")
        raise typer.Exit(code=1)

    console.print(f"Found {len(result_files)} result file(s)")

    all_results = []
    for result_file in result_files:
        with result_file.open() as f:
            data = json.load(f)
            model_name = result_file.stem.replace("_results", "")
            all_results.append({"model": model_name, "results": data})

    generator = LeaderboardGenerator(title=title)

    if format == OutputFormat.HTML:
        generator.generate_html(all_results, output)
    elif format == OutputFormat.MARKDOWN:
        generator.generate_markdown(all_results, output)
    elif format == OutputFormat.CSV:
        generator.generate_csv(all_results, output)
    else:
        generator.generate_json(all_results, output)

    console.print(f"\n[bold green]✓ Leaderboard generated![/] Saved to [cyan]{output}[/]")


@app.command()
def info() -> None:
    """Show configuration and environment information."""
    from src.settings import get_settings

    settings = get_settings()

    table = Table(title="MEQ-Bench Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("App Name", settings.app.name)
    table.add_row("Version", settings.app.version)
    table.add_row("Log Level", settings.app.log_level)
    table.add_row("Data Path", str(settings.app.data_path))
    table.add_row("Output Path", str(settings.app.output_path))
    table.add_row("Audiences", ", ".join(a.value for a in settings.audiences))
    table.add_row("Default Judge Model", settings.llm_judge.default_model)
    table.add_row("Batch Size", str(settings.performance.batch_size))
    table.add_row("Cache Enabled", str(settings.performance.cache_enabled))

    # Check API keys
    table.add_row("OpenAI API Key", "✓ Set" if settings.openai_api_key else "✗ Not set")
    table.add_row("Anthropic API Key", "✓ Set" if settings.anthropic_api_key else "✗ Not set")
    table.add_row("Google API Key", "✓ Set" if settings.google_api_key else "✗ Not set")

    console.print(table)


@app.command()
def validate(
    config_file: Annotated[
        Path,
        typer.Argument(
            help="Configuration file to validate",
        ),
    ] = Path("config.yaml"),
) -> None:
    """Validate a configuration file.

    Examples:
        meq-bench validate config.yaml
        meq-bench validate custom_config.yaml
    """
    from pydantic import ValidationError

    from src.settings import Settings

    if not config_file.exists():
        console.print(f"[bold red]Error:[/] Configuration file not found: {config_file}")
        raise typer.Exit(code=1)

    try:
        settings = Settings.from_yaml(config_file)
        console.print(f"[bold green]✓ Configuration is valid![/]")
        console.print(f"  Loaded {len(settings.audiences)} audiences")
        console.print(f"  Default judge model: {settings.llm_judge.default_model}")
    except ValidationError as e:
        console.print(f"[bold red]✗ Configuration validation failed:[/]")
        for error in e.errors():
            loc = " → ".join(str(l) for l in error["loc"])
            console.print(f"  • {loc}: {error['msg']}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
