#!/usr/bin/env python3
"""API cost estimator for MEQ-Bench evaluation.

This script estimates API costs before running evaluations to help
researchers budget their experiments.

Usage:
    python scripts/estimate_cost.py --models gpt-5.1 claude-opus-4.5 --items 1500
    python scripts/estimate_cost.py --full-benchmark
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Pricing as of December 2025 (hypothetical, update with actual prices)
# Format: (input_per_1m_tokens, output_per_1m_tokens)
MODEL_PRICING = {
    # OpenAI
    "gpt-5.2": (15.00, 60.00),
    "gpt-5.1": (10.00, 40.00),
    "gpt-5": (5.00, 20.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-oss-120b": (0.50, 1.00),
    
    # Anthropic
    "claude-opus-4.5": (15.00, 75.00),
    "claude-sonnet-4.5": (3.00, 15.00),
    "claude-haiku-4.5": (0.25, 1.25),
    
    # Google
    "gemini-3-ultra": (10.00, 30.00),
    "gemini-3-pro": (1.25, 5.00),
    "gemini-3-flash": (0.075, 0.30),
    
    # Meta (via vLLM - estimate compute cost)
    "llama-4-behemoth": (0.00, 0.00),  # Self-hosted
    "llama-4-maverick": (0.00, 0.00),
    "llama-4-scout": (0.00, 0.00),
    
    # DeepSeek
    "deepseek-v3": (0.27, 1.10),
    
    # Alibaba
    "qwen3-max": (0.40, 1.20),
    
    # Amazon (Bedrock)
    "nova-pro": (0.80, 3.20),
    "nova-omni": (1.20, 4.80),
}

# Default judge models used in ensemble
JUDGE_MODELS = ["gpt-5.1", "claude-opus-4.5", "gemini-3-pro"]

# Token estimates per task
TOKENS_PER_ITEM = {
    "input_medical_content": 500,      # Average input tokens
    "output_explanation": 400,         # Average explanation tokens
    "input_judge_prompt": 1200,        # Judge prompt with explanation
    "output_judge_response": 300,      # Judge reasoning + scores
}


@dataclass
class CostEstimate:
    """Cost estimate for a model or operation."""
    model: str
    role: str  # "subject" or "judge"
    num_items: int
    num_audiences: int
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "role": self.role,
            "items": self.num_items,
            "audiences": self.num_audiences,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": f"${self.input_cost:.2f}",
            "output_cost": f"${self.output_cost:.2f}",
            "total_cost": f"${self.total_cost:.2f}",
        }


class CostEstimator:
    """Estimate API costs for MEQ-Bench evaluation."""
    
    def __init__(
        self,
        pricing: Optional[Dict[str, tuple]] = None,
        judge_models: Optional[List[str]] = None,
    ):
        """Initialize estimator.
        
        Args:
            pricing: Custom pricing dict
            judge_models: Judge model list
        """
        self.pricing = pricing or MODEL_PRICING
        self.judge_models = judge_models or JUDGE_MODELS
    
    def estimate_subject_cost(
        self,
        model: str,
        num_items: int,
        num_audiences: int = 4,
    ) -> CostEstimate:
        """Estimate cost for generating explanations.
        
        Args:
            model: Model to estimate
            num_items: Number of benchmark items
            num_audiences: Number of audience types
            
        Returns:
            CostEstimate object
        """
        if model not in self.pricing:
            raise ValueError(f"Unknown model: {model}")
        
        input_price, output_price = self.pricing[model]
        
        # Total generations = items * audiences
        total_generations = num_items * num_audiences
        
        # Token estimates
        input_tokens = total_generations * TOKENS_PER_ITEM["input_medical_content"]
        output_tokens = total_generations * TOKENS_PER_ITEM["output_explanation"]
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        
        return CostEstimate(
            model=model,
            role="subject",
            num_items=num_items,
            num_audiences=num_audiences,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
        )
    
    def estimate_judge_cost(
        self,
        subject_model: str,
        num_items: int,
        num_audiences: int = 4,
    ) -> List[CostEstimate]:
        """Estimate cost for judging explanations.
        
        Args:
            subject_model: Model being judged
            num_items: Number of benchmark items
            num_audiences: Number of audience types
            
        Returns:
            List of CostEstimate objects (one per judge)
        """
        estimates = []
        
        total_evaluations = num_items * num_audiences
        
        for judge_model in self.judge_models:
            if judge_model not in self.pricing:
                continue
            
            input_price, output_price = self.pricing[judge_model]
            
            input_tokens = total_evaluations * TOKENS_PER_ITEM["input_judge_prompt"]
            output_tokens = total_evaluations * TOKENS_PER_ITEM["output_judge_response"]
            
            input_cost = (input_tokens / 1_000_000) * input_price
            output_cost = (output_tokens / 1_000_000) * output_price
            
            estimates.append(CostEstimate(
                model=judge_model,
                role=f"judge (for {subject_model})",
                num_items=num_items,
                num_audiences=num_audiences,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=input_cost + output_cost,
            ))
        
        return estimates
    
    def estimate_full_evaluation(
        self,
        models: List[str],
        num_items: int,
        num_audiences: int = 4,
    ) -> Dict[str, Any]:
        """Estimate total cost for full evaluation.
        
        Args:
            models: Models to evaluate
            num_items: Number of benchmark items
            num_audiences: Number of audience types
            
        Returns:
            Dict with all estimates and totals
        """
        estimates = {
            "subject_costs": [],
            "judge_costs": [],
            "total_subject_cost": 0.0,
            "total_judge_cost": 0.0,
            "grand_total": 0.0,
        }
        
        for model in models:
            # Subject model cost
            subject_est = self.estimate_subject_cost(model, num_items, num_audiences)
            estimates["subject_costs"].append(subject_est.to_dict())
            estimates["total_subject_cost"] += subject_est.total_cost
            
            # Judge costs for this model
            judge_ests = self.estimate_judge_cost(model, num_items, num_audiences)
            for judge_est in judge_ests:
                estimates["judge_costs"].append(judge_est.to_dict())
                estimates["total_judge_cost"] += judge_est.total_cost
        
        estimates["grand_total"] = estimates["total_subject_cost"] + estimates["total_judge_cost"]
        
        return estimates


def print_estimates(estimates: Dict[str, Any]) -> None:
    """Pretty print cost estimates."""
    print("\n" + "="*70)
    print("MEQ-BENCH 2.0 - API COST ESTIMATE")
    print("="*70)
    
    # Subject model costs
    print("\n📝 SUBJECT MODEL COSTS (Explanation Generation)")
    print("-"*70)
    print(f"{'Model':<25} {'Items':>8} {'Input ($)':>12} {'Output ($)':>12} {'Total':>10}")
    print("-"*70)
    
    for est in estimates["subject_costs"]:
        print(f"{est['model']:<25} {est['items']:>8} {est['input_cost']:>12} "
              f"{est['output_cost']:>12} {est['total_cost']:>10}")
    
    print("-"*70)
    print(f"{'SUBTOTAL':<25} {'':>8} {'':>12} {'':>12} ${estimates['total_subject_cost']:>9.2f}")
    
    # Judge costs
    print("\n⚖️  JUDGE COSTS (Ensemble Evaluation)")
    print("-"*70)
    print(f"{'Judge Model':<25} {'Role':<20} {'Input ($)':>12} {'Output ($)':>12} {'Total':>10}")
    print("-"*70)
    
    # Group judge costs
    for est in estimates["judge_costs"]:
        role = est['role'][:20]
        print(f"{est['model']:<25} {role:<20} {est['input_cost']:>12} "
              f"{est['output_cost']:>12} {est['total_cost']:>10}")
    
    print("-"*70)
    print(f"{'SUBTOTAL':<25} {'':>20} {'':>12} {'':>12} ${estimates['total_judge_cost']:>9.2f}")
    
    # Grand total
    print("\n" + "="*70)
    print(f"{'GRAND TOTAL':<45} {'':>12} ${estimates['grand_total']:>12.2f}")
    print("="*70)
    
    # Warnings
    print("\n⚠️  NOTES:")
    print("  • Prices are estimates and may vary. Check provider pricing pages.")
    print("  • Actual costs depend on prompt lengths and response verbosity.")
    print("  • Self-hosted models (Llama 4) show $0 but require compute resources.")
    print("  • Consider using --max-items for initial testing.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Estimate API costs for MEQ-Bench evaluation"
    )
    parser.add_argument(
        "--models", "-m",
        type=str,
        nargs="+",
        help="Models to estimate costs for"
    )
    parser.add_argument(
        "--items", "-n",
        type=int,
        default=1500,
        help="Number of benchmark items (default: 1500)"
    )
    parser.add_argument(
        "--audiences",
        type=int,
        default=4,
        help="Number of audience types (default: 4)"
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Estimate for all baseline models"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models with pricing"
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Models and Pricing (per 1M tokens):")
        print("-"*60)
        print(f"{'Model':<25} {'Input':>15} {'Output':>15}")
        print("-"*60)
        for model, (input_p, output_p) in sorted(MODEL_PRICING.items()):
            print(f"{model:<25} ${input_p:>14.2f} ${output_p:>14.2f}")
        return 0
    
    # Determine models
    if args.full_benchmark:
        models = [
            "gpt-5.1", "gpt-4o",
            "claude-opus-4.5", "claude-sonnet-4.5",
            "gemini-3-pro",
            "deepseek-v3",
            "qwen3-max",
        ]
    elif args.models:
        models = args.models
    else:
        print("Error: Specify --models or use --full-benchmark")
        return 1
    
    # Validate models
    invalid_models = [m for m in models if m not in MODEL_PRICING]
    if invalid_models:
        print(f"Warning: Unknown models will be skipped: {invalid_models}")
        models = [m for m in models if m in MODEL_PRICING]
    
    if not models:
        print("Error: No valid models to estimate")
        return 1
    
    # Estimate costs
    estimator = CostEstimator()
    estimates = estimator.estimate_full_evaluation(
        models=models,
        num_items=args.items,
        num_audiences=args.audiences,
    )
    
    if args.json:
        import json
        print(json.dumps(estimates, indent=2))
    else:
        print_estimates(estimates)
    
    return 0


if __name__ == "__main__":
    exit(main())

