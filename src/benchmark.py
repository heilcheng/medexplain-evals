"""Main MedExplain-Evals benchmark implementation.

This module contains the core benchmark classes and functionality for MedExplain-Evals.
It provides tools for loading medical datasets, generating audience-adaptive
explanations, and running comprehensive evaluations.

Key classes:
    MedExplainItem: Represents a single benchmark item with medical content
    MedExplain: Main benchmark class for running evaluations
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable

try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
from dataclasses import dataclass
from pathlib import Path

from .config import config
from .prompt_templates import AudienceAdaptivePrompt
from .evaluator import MedExplainEvaluator

logger = logging.getLogger("medexplain.benchmark")


# TypedDict definitions for structured data
class EvaluationResultDict(TypedDict):
    """Type definition for evaluation results."""

    model_name: str
    total_items: int
    audience_scores: Dict[str, List[float]]
    complexity_scores: Dict[str, List[float]]
    detailed_results: List[Dict[str, Any]]
    summary: Dict[str, float]


class ItemResultDict(TypedDict):
    """Type definition for individual item results."""

    item_id: str
    complexity_level: str
    source_dataset: str
    explanations: Dict[str, str]
    scores: Dict[str, Dict[str, float]]


class BenchmarkStatsDict(TypedDict):
    """Type definition for benchmark statistics."""

    total_items: int
    complexity_distribution: Dict[str, int]
    source_distribution: Dict[str, int]


@dataclass
class MedExplainItem:
    """Represents a single benchmark item for evaluation.

    A benchmark item contains medical content to be explained, along with metadata
    about its complexity level and source dataset. Optionally includes reference
    explanations for different audiences.

    Attributes:
        id: Unique identifier for the benchmark item.
        medical_content: The medical information to be adapted for different audiences.
        complexity_level: Difficulty level of the content ("basic", "intermediate", "advanced").
        source_dataset: Name of the dataset this item was sourced from.
        reference_explanations: Optional reference explanations for each audience,
            mapping audience names to explanation text.
    """

    id: str
    medical_content: str
    complexity_level: str  # "basic", "intermediate", "advanced"
    source_dataset: str
    reference_explanations: Optional[Dict[str, str]] = None


class MedExplain:
    """Main benchmark class for MedExplain-Evals evaluation.

    This class provides the core functionality for running MedExplain-Evals evaluations,
    including loading benchmark data, generating audience-adaptive explanations,
    and evaluating model performance across different audiences and complexity levels.

    The class manages benchmark items, interfaces with evaluation components,
    and provides comprehensive evaluation results with detailed statistics.

    Attributes:
        data_path: Path to the benchmark data directory.
        evaluator: MedExplainEvaluator instance for scoring explanations.
        prompt_template: AudienceAdaptivePrompt instance for generating prompts.
        benchmark_items: List of loaded MedExplainItem objects.

    Example:
        ```python
        # Initialize benchmark
        bench = MedExplain(data_path="/path/to/data")

        # Generate explanations for a model
        explanations = bench.generate_explanations(medical_content, model_func)

        # Run full evaluation
        results = bench.evaluate_model(model_func, max_items=100)
        ```
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        """Initialize MedExplain-Evals instance.

        Sets up the benchmark with the specified data directory and initializes
        the evaluator and prompt template components. Automatically loads benchmark
        data if the data directory exists.

        Args:
            data_path: Path to benchmark data directory. If None, uses default
                'data' directory relative to the package root.
        """
        self.data_path = self._resolve_data_path(data_path)
        # Initialize evaluator with graceful fallback for missing dependencies
        try:
            self.evaluator: MedExplainEvaluator = MedExplainEvaluator()
        except Exception as e:
            logger.warning(f"Could not initialize full evaluator: {e}")
            logger.info("Some evaluation features may be limited due to missing dependencies or configuration")
        self.prompt_template: AudienceAdaptivePrompt = AudienceAdaptivePrompt()
        self.benchmark_items: List[MedExplainItem] = []

        # Load benchmark data if available
        self._load_benchmark_data()

    def _resolve_data_path(self, data_path: Optional[str] = None) -> Path:
        """Resolve data directory path with fallback options.

        Args:
            data_path: Optional custom data path

        Returns:
            Resolved Path object for data directory
        """
        if data_path:
            # Use provided path (can be relative or absolute)
            resolved_path = Path(data_path).resolve()
        else:
            # Try multiple fallback locations
            possible_paths = [
                # Relative to package directory
                Path(__file__).parent.parent / "data",
                # Current working directory
                Path.cwd() / "data",
                # Environment variable if set
                Path(os.environ.get("MEQ_BENCH_DATA_PATH", "")) if os.environ.get("MEQ_BENCH_DATA_PATH") else None,
                # Config-based path
                Path(config.get_data_path()) if hasattr(config, "get_data_path") else None,
            ]

            # Find first existing path or use first option as default
            resolved_path = None
            for path in possible_paths:
                if path and path.exists():
                    resolved_path = path.resolve()
                    break

            if not resolved_path:
                # Default to package relative path
                resolved_path = (Path(__file__).parent.parent / "data").resolve()

        # Ensure directory exists
        if resolved_path is not None:
            resolved_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using data directory: {resolved_path}")
            return resolved_path
        else:
            raise ValueError("Could not resolve data directory path")

    def _load_benchmark_data(self) -> None:
        """Load benchmark data from JSON files with error handling.

        Loads benchmark items from benchmark_items.json in the data directory.
        Each item is converted to a MedExplainItem object and added to the
        benchmark_items list. Includes comprehensive error handling for missing
        files, invalid JSON, and malformed data.

        The JSON file should contain a list of dictionaries with the following keys:
        - id: Unique identifier for the item
        - medical_content: Medical information to be explained
        - complexity_level: Difficulty level ("basic", "intermediate", "advanced")
        - source_dataset: Name of the source dataset
        - reference_explanations: Optional reference explanations dictionary
        """
        try:
            benchmark_file = self.data_path / "benchmark_items.json"

            if not benchmark_file.exists():
                logger.warning(f"Benchmark data file not found: {benchmark_file}")
                logger.info("Use create_sample_dataset() to generate sample data")
                return

            with open(benchmark_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Benchmark data must be a list of items")

            for i, item_data in enumerate(data):
                try:
                    # Validate required fields
                    required_fields = ["id", "medical_content", "complexity_level", "source_dataset"]
                    for field in required_fields:
                        if field not in item_data:
                            raise KeyError(f"Missing required field '{field}' in item {i}")

                    item = MedExplainItem(
                        id=item_data["id"],
                        medical_content=item_data["medical_content"],
                        complexity_level=item_data["complexity_level"],
                        source_dataset=item_data["source_dataset"],
                        reference_explanations=item_data.get("reference_explanations"),
                    )
                    self.benchmark_items.append(item)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error loading benchmark item {i}: {e}")
                    continue

            logger.info(f"Loaded {len(self.benchmark_items)} benchmark items from {benchmark_file}")

        except FileNotFoundError:
            logger.warning(f"Benchmark data directory not found: {self.data_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in benchmark data file: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading benchmark data: {e}")

    def add_benchmark_item(self, item: MedExplainItem) -> None:
        """Add a new benchmark item to the evaluation set.

        Validates the item data and adds it to the benchmark items list.
        Includes checks for data integrity and duplicate IDs.

        Args:
            item: MedExplainItem object to add to the benchmark.

        Raises:
            TypeError: If item is not an instance of MedExplainItem.
            ValueError: If item data is invalid or ID already exists.
        """
        if not isinstance(item, MedExplainItem):
            raise TypeError("item must be an instance of MedExplainItem")

        # Validate item data
        if not item.id or not isinstance(item.id, str):
            raise ValueError("item.id must be a non-empty string")

        if not item.medical_content or not isinstance(item.medical_content, str):
            raise ValueError("item.medical_content must be a non-empty string")

        if item.complexity_level not in ["basic", "intermediate", "advanced"]:
            raise ValueError("item.complexity_level must be 'basic', 'intermediate', or 'advanced'")

        # Check for duplicate IDs
        if any(existing_item.id == item.id for existing_item in self.benchmark_items):
            raise ValueError(f"Item with ID '{item.id}' already exists")

        self.benchmark_items.append(item)
        logger.debug(f"Added benchmark item: {item.id}")

    def generate_explanations(self, medical_content: str, model_func: Callable[[str], str]) -> Dict[str, str]:
        """Generate audience-adaptive explanations using a model.

        Uses the configured prompt template to generate explanations tailored
        for different healthcare audiences (physicians, nurses, patients, caregivers).

        Args:
            medical_content: Medical information to be adapted for different audiences.
            model_func: Callable that takes a prompt string and returns the model's
                response as a string.

        Returns:
            Dictionary mapping audience names to their respective explanations.
            Keys are audience names (e.g., 'physician', 'nurse', 'patient', 'caregiver').

        Raises:
            ValueError: If medical_content is empty or invalid
            TypeError: If model_func is not callable

        Example:
            ```python
            def my_model(prompt):
                return openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content

            explanations = bench.generate_explanations(
                "Hypertension is high blood pressure",
                my_model
            )
            ```
        """
        # Input validation
        if not medical_content or not isinstance(medical_content, str):
            raise ValueError("medical_content must be a non-empty string")

        if medical_content.strip() == "":
            raise ValueError("medical_content cannot be empty or contain only whitespace")

        if len(medical_content.strip()) < 10:
            raise ValueError("medical_content must be at least 10 characters long")

        if not callable(model_func):
            raise TypeError("model_func must be a callable function")

        # Additional content validation
        if len(medical_content) > 10000:  # Reasonable upper limit
            logger.warning(f"Medical content is very long ({len(medical_content)} chars). Consider splitting.")

        # Sanitize content - remove excessive whitespace
        sanitized_content = " ".join(medical_content.split())

        try:
            prompt = self.prompt_template.format_prompt(sanitized_content)
            logger.debug(f"Generated prompt with {len(prompt)} characters")

            response = model_func(prompt)

            # Validate model response
            if not response or not isinstance(response, str):
                raise ValueError("Model function returned empty or invalid response")

            if response.strip() == "":
                raise ValueError("Model function returned empty response")

            explanations = self.prompt_template.parse_response(response)

            # Validate parsed explanations
            if not explanations:
                raise ValueError("Failed to parse any explanations from model response")

            # Log successful generation
            logger.info(f"Generated explanations for {len(explanations)} audiences")

            return explanations

        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            if isinstance(e, (ValueError, TypeError)):
                raise
            else:
                raise RuntimeError(f"Unexpected error during explanation generation: {e}") from e

    def evaluate_model(self, model_func: Callable[[str], str], max_items: Optional[int] = None) -> EvaluationResultDict:
        """Evaluate a model on the full benchmark.

        Runs comprehensive evaluation of a model's performance across all benchmark
        items and audiences. Generates explanations for each item and evaluates them
        using the configured evaluator.

        Args:
            model_func: Callable that takes a prompt string and returns the model's
                response as a string.
            max_items: Maximum number of benchmark items to evaluate. If None,
                evaluates all available items. Useful for testing with smaller subsets.

        Returns:
            Dictionary containing comprehensive evaluation results with the following keys:
            - model_name: Name of the evaluated model
            - total_items: Number of items evaluated
            - audience_scores: Scores grouped by audience type
            - complexity_scores: Scores grouped by complexity level
            - detailed_results: Per-item detailed evaluation results
            - summary: Summary statistics including means, standard deviations, etc.

        Example:
            ```python
            results = bench.evaluate_model(my_model_func, max_items=50)
            print(f"Overall mean score: {results['summary']['overall_mean']}")
            ```
        """
        results: EvaluationResultDict = {
            "model_name": getattr(model_func, "__name__", "unknown"),
            "total_items": len(self.benchmark_items[:max_items]) if max_items else len(self.benchmark_items),
            "audience_scores": {"physician": [], "nurse": [], "patient": [], "caregiver": []},
            "complexity_scores": {"basic": [], "intermediate": [], "advanced": []},
            "detailed_results": [],
            "summary": {},
        }

        items_to_evaluate = self.benchmark_items[:max_items] if max_items else self.benchmark_items

        for item in items_to_evaluate:
            # Generate explanations
            explanations = self.generate_explanations(item.medical_content, model_func)

            # Evaluate explanations
            evaluation_results = self.evaluator.evaluate_all_audiences(item.medical_content, explanations)

            # Store detailed results
            item_result = {
                "item_id": item.id,
                "complexity_level": item.complexity_level,
                "source_dataset": item.source_dataset,
                "explanations": explanations,
                "scores": {
                    audience: {
                        "readability": score.readability,
                        "terminology": score.terminology,
                        "safety": score.safety,
                        "coverage": score.coverage,
                        "quality": score.quality,
                        "overall": score.overall,
                    }
                    for audience, score in evaluation_results.items()
                },
            }
            results["detailed_results"].append(item_result)

            # Aggregate scores by audience
            for audience, score in evaluation_results.items():
                if audience in results["audience_scores"]:
                    results["audience_scores"][audience].append(score.overall)

            # Aggregate scores by complexity
            avg_overall = sum(score.overall for score in evaluation_results.values()) / len(evaluation_results)
            results["complexity_scores"][item.complexity_level].append(avg_overall)

        # Calculate summary statistics
        results["summary"] = self._calculate_summary_stats(dict(results))

        return results

    def _calculate_summary_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results.

        Computes descriptive statistics for audience-level and complexity-level
        scores, including means, standard deviations, minimums, and maximums.

        Args:
            results: Dictionary containing evaluation results with audience_scores
                and complexity_scores keys.

        Returns:
            Dictionary containing summary statistics with keys like:
            - {audience}_mean: Mean score for each audience
            - {audience}_std: Standard deviation for each audience
            - {audience}_min: Minimum score for each audience
            - {audience}_max: Maximum score for each audience
            - {complexity}_mean: Mean score for each complexity level
            - {complexity}_std: Standard deviation for each complexity level
            - overall_mean: Overall mean score across all evaluations
            - overall_std: Overall standard deviation
        """
        summary = {}

        # Audience-level statistics
        for audience, scores in results["audience_scores"].items():
            if scores:
                summary[f"{audience}_mean"] = sum(scores) / len(scores)
                summary[f"{audience}_std"] = (sum((x - summary[f"{audience}_mean"]) ** 2 for x in scores) / len(scores)) ** 0.5
                summary[f"{audience}_min"] = min(scores)
                summary[f"{audience}_max"] = max(scores)

        # Complexity-level statistics
        for complexity, scores in results["complexity_scores"].items():
            if scores:
                summary[f"{complexity}_mean"] = sum(scores) / len(scores)
                summary[f"{complexity}_std"] = (
                    sum((x - summary[f"{complexity}_mean"]) ** 2 for x in scores) / len(scores)
                ) ** 0.5

        # Overall statistics
        all_scores = []
        for audience_scores in results["audience_scores"].values():
            all_scores.extend(audience_scores)

        if all_scores:
            summary["overall_mean"] = sum(all_scores) / len(all_scores)
            summary["overall_std"] = (sum((x - summary["overall_mean"]) ** 2 for x in all_scores) / len(all_scores)) ** 0.5

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save evaluation results to JSON file.

        Serializes the evaluation results dictionary to a JSON file with
        proper formatting for readability. Includes proper path handling
        and error handling.

        Args:
            results: Dictionary containing evaluation results from evaluate_model().
            output_path: File path where results should be saved.

        Raises:
            Exception: If file writing fails.

        Example:
            ```python
            results = bench.evaluate_model(model_func)
            bench.save_results(results, "results/model_evaluation.json")
            ```
        """
        output_file = Path(output_path).resolve()

        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")
            raise

    def create_sample_dataset(self, output_path: Optional[str] = None) -> List[MedExplainItem]:
        """Create a sample dataset for testing.

        Generates a small set of sample medical content items with different
        complexity levels for testing and demonstration purposes.

        Args:
            output_path: Optional path to save the sample dataset as JSON.
                If provided, saves the dataset to this file.

        Returns:
            List of MedExplainItem objects containing sample medical content
            with basic, intermediate, and advanced complexity levels.

        Example:
            ```python
            # Create sample data
            sample_items = bench.create_sample_dataset()

            # Create and save sample data
            sample_items = bench.create_sample_dataset("data/sample_dataset.json")
            ```
        """
        sample_items = [
            MedExplainItem(
                id="sample_001",
                medical_content=(
                    "Hypertension, also known as high blood pressure, is a condition where the force of blood "
                    "against artery walls is consistently too high. It can lead to heart disease, stroke, and "
                    "kidney problems if left untreated. Treatment typically involves lifestyle changes and medication."
                ),
                complexity_level="basic",
                source_dataset="sample",
            ),
            MedExplainItem(
                id="sample_002",
                medical_content=(
                    "Myocardial infarction occurs when blood flow to a part of the heart muscle is blocked, usually "
                    "by a blood clot in a coronary artery. This results in damage or death of heart muscle cells. "
                    "Immediate treatment with medications to dissolve clots or procedures to restore blood flow is critical."
                ),
                complexity_level="intermediate",
                source_dataset="sample",
            ),
            MedExplainItem(
                id="sample_003",
                medical_content=(
                    "Diabetic ketoacidosis (DKA) is a serious complication of diabetes mellitus characterized by "
                    "hyperglycemia, ketosis, and metabolic acidosis. It typically occurs in type 1 diabetes due to "
                    "absolute insulin deficiency. Treatment involves IV fluids, insulin therapy, and electrolyte "
                    "replacement while addressing underlying precipitating factors."
                ),
                complexity_level="advanced",
                source_dataset="sample",
            ),
        ]

        if output_path:
            # Save sample dataset with proper path handling
            output_file = Path(output_path).resolve()
            output_file.parent.mkdir(parents=True, exist_ok=True)

            data_to_save = []
            for item in sample_items:
                data_to_save.append(
                    {
                        "id": item.id,
                        "medical_content": item.medical_content,
                        "complexity_level": item.complexity_level,
                        "source_dataset": item.source_dataset,
                        "reference_explanations": item.reference_explanations,
                    }
                )

            try:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                logger.info(f"Sample dataset saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save sample dataset to {output_file}: {e}")
                raise

        return sample_items

    def get_benchmark_stats(self) -> Union[BenchmarkStatsDict, Dict[str, Union[int, str]]]:
        """Get statistics about the benchmark dataset.

        Provides summary statistics about the loaded benchmark items,
        including total counts and distributions by complexity level
        and source dataset.

        Returns:
            Dictionary containing benchmark statistics with keys:
            - total_items: Total number of benchmark items
            - complexity_distribution: Count of items by complexity level
            - source_distribution: Count of items by source dataset
            - message: Informational message if no items are loaded

        Example:
            ```python
            stats = bench.get_benchmark_stats()
            print(f"Total items: {stats['total_items']}")
            print(f"Complexity distribution: {stats['complexity_distribution']}")
            ```
        """
        if not self.benchmark_items:
            return {"total_items": 0, "message": "No benchmark items loaded"}

        stats: BenchmarkStatsDict = {
            "total_items": len(self.benchmark_items),
            "complexity_distribution": {},
            "source_distribution": {},
        }

        for item in self.benchmark_items:
            # Count by complexity
            if item.complexity_level not in stats["complexity_distribution"]:
                stats["complexity_distribution"][item.complexity_level] = 0
            stats["complexity_distribution"][item.complexity_level] += 1

            # Count by source
            if item.source_dataset not in stats["source_distribution"]:
                stats["source_distribution"][item.source_dataset] = 0
            stats["source_distribution"][item.source_dataset] += 1

        return stats

    def validate_benchmark(self) -> Dict[str, Any]:
        """Validate the benchmark dataset and return validation report.

        Returns:
            Dictionary containing validation results and any issues found
        """
        validation_report: Dict[str, Any] = {
            "valid": True,
            "total_items": len(self.benchmark_items),
            "issues": [],
            "warnings": [],
            "statistics": {},
        }

        if not self.benchmark_items:
            validation_report["valid"] = False
            validation_report["issues"].append("No benchmark items loaded")
            return validation_report

        # Check for duplicate IDs
        ids = [item.id for item in self.benchmark_items]
        duplicate_ids = set([id for id in ids if ids.count(id) > 1])
        if duplicate_ids:
            validation_report["valid"] = False
            validation_report["issues"].append(f"Duplicate item IDs found: {duplicate_ids}")

        # Validate complexity level distribution
        complexity_counts: Dict[str, int] = {}
        for item in self.benchmark_items:
            complexity = item.complexity_level
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1

        validation_report["statistics"]["complexity_distribution"] = complexity_counts

        # Check for balanced distribution
        if len(complexity_counts) < 3:
            validation_report["warnings"].append("Not all complexity levels represented")

        # Validate content length
        content_lengths = [len(item.medical_content) for item in self.benchmark_items]
        avg_length = sum(content_lengths) / len(content_lengths)
        validation_report["statistics"]["average_content_length"] = avg_length

        if avg_length < 50:
            validation_report["warnings"].append("Average content length is quite short")
        elif avg_length > 2000:
            validation_report["warnings"].append("Average content length is quite long")

        # Check for empty or very short content
        short_content_items = [item.id for item in self.benchmark_items if len(item.medical_content.strip()) < 20]
        if short_content_items:
            validation_report["valid"] = False
            validation_report["issues"].append(f"Items with very short content: {short_content_items}")

        logger.info(f"Benchmark validation completed. Valid: {validation_report['valid']}")
        return validation_report
