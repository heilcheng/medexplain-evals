#!/usr/bin/env python3
"""Main evaluation orchestrator for MedExplain-Evals.

This script runs the full evaluation pipeline:
1. Load benchmark items
2. Generate explanations for each model
3. Compute evaluation scores
4. Generate comparison reports

Features:
    - Multi-model evaluation
    - Parallel model processing
    - Checkpoint/resume
    - Cost tracking
    - Progress reporting

Usage:
    python scripts/run_evaluation.py \
        --benchmark data/benchmark_v2/test.json \
        --output results/ \
        --models gpt-5.1 claude-opus-4.5 \
        --max-items 100
"""

import argparse
import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Default models for baseline evaluation
BASELINE_MODELS = [
    "gpt-5.1",
    "gpt-4o",
    "claude-opus-4.5",
    "claude-sonnet-4.5",
    "gemini-3-pro",
    "deepseek-v3",
    "qwen3-max",
]

# Model tiers for cost estimation
MODEL_TIERS = {
    "gpt-5.1": "flagship",
    "gpt-4o": "advanced",
    "claude-opus-4.5": "flagship",
    "claude-sonnet-4.5": "efficient",
    "gemini-3-pro": "standard",
    "deepseek-v3": "frontier",
    "qwen3-max": "frontier",
    "llama-4-scout": "open",
}


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    benchmark_path: str
    output_dir: str
    models: List[str]
    max_items: Optional[int] = None
    batch_size: int = 10
    parallel_models: int = 1
    resume: bool = True
    skip_generation: bool = False
    skip_scoring: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelEvaluationResult:
    """Result for one model's evaluation."""
    model: str
    status: str  # success, failed, partial
    items_evaluated: int
    mean_overall_score: float
    scores_by_audience: Dict[str, float]
    scores_by_dimension: Dict[str, float]
    safety_pass_rate: float
    total_cost: float
    total_time_seconds: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Summary of full evaluation run."""
    config: Dict[str, Any]
    models_evaluated: int
    total_items: int
    results: List[Dict[str, Any]]
    model_rankings: List[Dict[str, Any]]
    started_at: str
    completed_at: str
    total_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationRunner:
    """Orchestrate the full evaluation pipeline."""
    
    def __init__(self, config: EvaluationConfig):
        """Initialize runner.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_items = []
        self.results: List[ModelEvaluationResult] = []
    
    def load_benchmark(self) -> int:
        """Load benchmark items.
        
        Returns:
            Number of items loaded
        """
        path = Path(self.config.benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            self.benchmark_items = json.load(f)
        
        if self.config.max_items:
            self.benchmark_items = self.benchmark_items[:self.config.max_items]
        
        logger.info(f"Loaded {len(self.benchmark_items)} benchmark items")
        return len(self.benchmark_items)
    
    def run_model_evaluation(self, model: str) -> ModelEvaluationResult:
        """Run evaluation for a single model.
        
        Args:
            model: Model name
            
        Returns:
            Evaluation result for the model
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating model: {model}")
        logger.info(f"{'='*50}")
        
        start_time = datetime.now()
        
        model_output_dir = self.output_dir / model
        explanations_dir = model_output_dir / "explanations"
        scores_dir = model_output_dir / "scores"
        
        total_cost = 0.0
        
        try:
            # Step 1: Generate explanations
            if not self.config.skip_generation:
                logger.info(f"[{model}] Generating explanations...")
                
                # Save benchmark items for this run
                items_path = model_output_dir / "items.json"
                model_output_dir.mkdir(parents=True, exist_ok=True)
                with open(items_path, "w") as f:
                    json.dump(self.benchmark_items, f, indent=2)
                
                # Call generate_explanations script
                gen_result = self._run_generation(
                    model=model,
                    items_path=items_path,
                    output_dir=explanations_dir,
                )
                
                if gen_result.get("success"):
                    total_cost += gen_result.get("cost", 0.0)
                else:
                    raise RuntimeError(f"Generation failed: {gen_result.get('error')}")
            
            # Step 2: Compute scores
            if not self.config.skip_scoring:
                logger.info(f"[{model}] Computing scores...")
                
                score_result = self._run_scoring(
                    model=model,
                    explanations_dir=explanations_dir,
                    benchmark_path=self.config.benchmark_path,
                    output_dir=scores_dir,
                )
                
                if not score_result.get("success"):
                    raise RuntimeError(f"Scoring failed: {score_result.get('error')}")
            
            # Step 3: Load aggregated scores
            aggregated = self._load_aggregated_scores(scores_dir)
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            return ModelEvaluationResult(
                model=model,
                status="success",
                items_evaluated=aggregated.get("total_items", len(self.benchmark_items)),
                mean_overall_score=aggregated.get("mean_overall", 0.0),
                scores_by_audience=aggregated.get("mean_by_audience", {}),
                scores_by_dimension=aggregated.get("mean_by_dimension", {}),
                safety_pass_rate=aggregated.get("safety_pass_rate", 0.0),
                total_cost=total_cost,
                total_time_seconds=elapsed,
            )
            
        except Exception as e:
            logger.error(f"[{model}] Evaluation failed: {e}")
            return ModelEvaluationResult(
                model=model,
                status="failed",
                items_evaluated=0,
                mean_overall_score=0.0,
                scores_by_audience={},
                scores_by_dimension={},
                safety_pass_rate=0.0,
                total_cost=total_cost,
                total_time_seconds=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
    
    def _run_generation(
        self,
        model: str,
        items_path: Path,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run explanation generation (internal or subprocess)."""
        try:
            from scripts.generate_explanations import ExplanationGenerator
            
            with open(items_path, "r") as f:
                items = json.load(f)
            
            generator = ExplanationGenerator(
                model=model,
                temperature=0.3,
                rate_limit_delay=0.5,
            )
            
            checkpoint_path = output_dir / "checkpoint.json" if self.config.resume else None
            
            results = generator.generate_batch(
                items=items,
                output_dir=output_dir,
                checkpoint_path=checkpoint_path,
            )
            
            total_cost = sum(r.cost for r in results)
            
            return {
                "success": True,
                "items": len(results),
                "cost": total_cost,
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _run_scoring(
        self,
        model: str,
        explanations_dir: Path,
        benchmark_path: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Run score computation."""
        try:
            from scripts.compute_scores import ScoreComputer
            
            with open(benchmark_path, "r") as f:
                benchmark_items = json.load(f)
            
            scorer = ScoreComputer(
                use_ensemble=True,
                use_safety=True,
                use_grounding=False,
            )
            
            scorer.compute_all_scores(
                explanations_dir=explanations_dir,
                benchmark_items=benchmark_items,
                output_dir=output_dir,
            )
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Scoring error: {e}")
            return {
                "success": False,
                "error": str(e),
            }
    
    def _load_aggregated_scores(self, scores_dir: Path) -> Dict[str, Any]:
        """Load aggregated scores from scores directory."""
        agg_path = scores_dir / "aggregated_scores.json"
        
        if agg_path.exists():
            with open(agg_path, "r") as f:
                return json.load(f)
        
        return {}
    
    def run_all(self) -> EvaluationSummary:
        """Run evaluation for all configured models.
        
        Returns:
            Evaluation summary
        """
        start_time = datetime.now()
        
        # Load benchmark
        self.load_benchmark()
        
        # Evaluate each model
        if self.config.parallel_models > 1:
            # Parallel evaluation
            with ThreadPoolExecutor(max_workers=self.config.parallel_models) as executor:
                futures = {
                    executor.submit(self.run_model_evaluation, model): model
                    for model in self.config.models
                }
                
                for future in as_completed(futures):
                    model = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"Model {model} failed: {e}")
        else:
            # Sequential evaluation
            for model in self.config.models:
                result = self.run_model_evaluation(model)
                self.results.append(result)
        
        # Generate rankings
        rankings = self._generate_rankings()
        
        # Create summary
        summary = EvaluationSummary(
            config=self.config.to_dict(),
            models_evaluated=len(self.results),
            total_items=len(self.benchmark_items),
            results=[r.to_dict() for r in self.results],
            model_rankings=rankings,
            started_at=start_time.isoformat(),
            completed_at=datetime.now().isoformat(),
            total_cost=sum(r.total_cost for r in self.results),
        )
        
        # Save summary
        summary_path = self.output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info(f"\nEvaluation complete! Summary saved to {summary_path}")
        
        return summary
    
    def _generate_rankings(self) -> List[Dict[str, Any]]:
        """Generate model rankings by overall score."""
        successful = [r for r in self.results if r.status == "success"]
        
        sorted_results = sorted(
            successful,
            key=lambda x: x.mean_overall_score,
            reverse=True,
        )
        
        rankings = []
        for rank, result in enumerate(sorted_results, 1):
            rankings.append({
                "rank": rank,
                "model": result.model,
                "mean_overall_score": result.mean_overall_score,
                "safety_pass_rate": result.safety_pass_rate,
            })
        
        return rankings


def print_summary(summary: EvaluationSummary) -> None:
    """Print evaluation summary to console."""
    print("\n" + "="*60)
    print("MEQ-BENCH 2.0 EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nModels evaluated: {summary.models_evaluated}")
    print(f"Total items: {summary.total_items}")
    print(f"Total cost: ${summary.total_cost:.2f}")
    
    print("\n" + "-"*60)
    print("MODEL RANKINGS (by overall score)")
    print("-"*60)
    
    print(f"{'Rank':<6}{'Model':<25}{'Score':<10}{'Safety':<10}")
    print("-"*60)
    
    for ranking in summary.model_rankings:
        print(f"{ranking['rank']:<6}{ranking['model']:<25}"
              f"{ranking['mean_overall_score']:<10.3f}"
              f"{ranking['safety_pass_rate']*100:<10.1f}%")
    
    print("\n" + "-"*60)
    print("DETAILED RESULTS")
    print("-"*60)
    
    for result in summary.results:
        print(f"\n{result['model']}:")
        print(f"  Status: {result['status']}")
        print(f"  Overall: {result['mean_overall_score']:.3f}")
        
        if result.get('scores_by_audience'):
            print("  By Audience:")
            for aud, score in result['scores_by_audience'].items():
                print(f"    {aud}: {score:.3f}")
        
        if result.get('scores_by_dimension'):
            print("  By Dimension:")
            for dim, score in result['scores_by_dimension'].items():
                print(f"    {dim}: {score:.3f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Run MedExplain-Evals evaluation"
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="data/benchmark_v2/test.json",
        help="Benchmark items JSON file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results/",
        help="Output directory"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        default=None,
        help="Models to evaluate (default: all baseline models)"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum items to evaluate (for testing)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of models to evaluate in parallel"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoints"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation (use existing explanations)"
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring (use existing scores)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available baseline models:")
        for model in BASELINE_MODELS:
            tier = MODEL_TIERS.get(model, "unknown")
            print(f"  {model:<25} [{tier}]")
        return 0
    
    # Determine models
    models = args.models if args.models else BASELINE_MODELS
    
    # Create config
    config = EvaluationConfig(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        models=models,
        max_items=args.max_items,
        batch_size=args.batch_size,
        parallel_models=args.parallel,
        resume=not args.no_resume,
        skip_generation=args.skip_generation,
        skip_scoring=args.skip_scoring,
    )
    
    logger.info("MedExplain-Evals Evaluation Runner")
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Output: {args.output}")
    
    # Run evaluation
    runner = EvaluationRunner(config)
    
    try:
        summary = runner.run_all()
        print_summary(summary)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

