#!/usr/bin/env python3
"""Batch explanation generation for MEQ-Bench evaluation.

This script generates audience-adaptive medical explanations from 
benchmark items using specified LLM models.

Features:
    - Batch processing with rate limiting
    - Checkpoint/resume for long runs
    - Multi-audience generation
    - Cost tracking
    - Progress logging

Usage:
    python scripts/generate_explanations.py \
        --input data/benchmark_v2/test.json \
        --output results/explanations/gpt-5.1/ \
        --model gpt-5.1 \
        --batch-size 10
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Prompt template for audience-adaptive explanation generation
EXPLANATION_PROMPT = """You are a medical communication expert. Your task is to explain medical information for a specific audience, adapting your language, terminology, and detail level appropriately.

## Target Audience: {audience}

{audience_description}

## Medical Information to Explain:

{medical_content}

## Instructions:

1. Explain this medical information for the {audience} audience
2. Use appropriate terminology and reading level
3. Include relevant actionable guidance
4. Add appropriate safety warnings if needed
5. Use an appropriate tone for this audience

Write your explanation below:"""

AUDIENCE_DESCRIPTIONS = {
    "physician": "A medical doctor who expects clinical precision, evidence-based information, and appropriate medical terminology. Use clinical language, include relevant differentials, and provide guideline-concordant recommendations.",
    
    "nurse": "A healthcare professional who needs practical, actionable information with appropriate medical terminology. Focus on monitoring parameters, nursing interventions, patient education points, and when to escalate.",
    
    "patient": "A person without medical training who needs clear, simple explanations without jargon. Use plain language (grade 6-10 reading level), be empathetic and reassuring, include clear action steps, and explain when to seek help.",
    
    "caregiver": "A family member or professional caregiver who needs practical guidance on caring for someone with a medical condition. Focus on daily care tasks, warning signs to watch for, medication reminders, and when to call for help.",
}


@dataclass
class GenerationResult:
    """Result from generating explanations for one item."""
    item_id: str
    model: str
    explanations: Dict[str, str]  # audience -> explanation
    generation_time_ms: float
    cost: float
    success: bool
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationCheckpoint:
    """Checkpoint for resumable generation."""
    model: str
    total_items: int
    completed_items: int
    completed_ids: List[str]
    total_cost: float
    total_time_ms: float
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "GenerationCheckpoint":
        return cls(**data)


class ExplanationGenerator:
    """Generate audience-adaptive explanations using LLMs."""
    
    def __init__(
        self,
        model: str,
        audiences: List[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
        rate_limit_delay: float = 0.5,
    ):
        """Initialize generator.
        
        Args:
            model: Model name to use
            audiences: Target audiences (default: all 4)
            temperature: Generation temperature
            max_tokens: Max tokens per explanation
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.model = model
        self.audiences = audiences or ["physician", "nurse", "patient", "caregiver"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rate_limit_delay = rate_limit_delay
        
        self.client = None
        self._init_client()
    
    def _init_client(self):
        """Initialize model client."""
        try:
            from src.model_clients import UnifiedModelClient
            self.client = UnifiedModelClient()
            logger.info(f"Initialized client for model: {self.model}")
        except ImportError as e:
            logger.warning(f"Could not import model client: {e}")
            logger.warning("Running in mock mode")
            self.client = None
    
    def generate_for_item(
        self,
        item: Dict[str, Any],
    ) -> GenerationResult:
        """Generate explanations for all audiences for one item.
        
        Args:
            item: Benchmark item
            
        Returns:
            GenerationResult with explanations for all audiences
        """
        item_id = item.get("id", "unknown")
        medical_content = item.get("medical_content", "")
        
        explanations = {}
        total_cost = 0.0
        start_time = time.time()
        
        try:
            for audience in self.audiences:
                prompt = EXPLANATION_PROMPT.format(
                    audience=audience.upper(),
                    audience_description=AUDIENCE_DESCRIPTIONS.get(audience, ""),
                    medical_content=medical_content,
                )
                
                if self.client:
                    response = self.client.generate(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                    explanations[audience] = response.content
                    total_cost += response.cost
                else:
                    # Mock response for testing
                    explanations[audience] = f"[Mock {audience} explanation for: {medical_content[:100]}...]"
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            generation_time = (time.time() - start_time) * 1000
            
            return GenerationResult(
                item_id=item_id,
                model=self.model,
                explanations=explanations,
                generation_time_ms=generation_time,
                cost=total_cost,
                success=True,
            )
            
        except Exception as e:
            logger.error(f"Error generating for {item_id}: {e}")
            return GenerationResult(
                item_id=item_id,
                model=self.model,
                explanations={},
                generation_time_ms=(time.time() - start_time) * 1000,
                cost=total_cost,
                success=False,
                error=str(e),
            )
    
    def generate_batch(
        self,
        items: List[Dict[str, Any]],
        output_dir: Path,
        checkpoint_path: Optional[Path] = None,
    ) -> List[GenerationResult]:
        """Generate explanations for a batch of items with checkpointing.
        
        Args:
            items: List of benchmark items
            output_dir: Directory to save results
            checkpoint_path: Path to checkpoint file
            
        Returns:
            List of generation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create checkpoint
        checkpoint = None
        completed_ids = set()
        
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r") as f:
                checkpoint = GenerationCheckpoint.from_dict(json.load(f))
            completed_ids = set(checkpoint.completed_ids)
            logger.info(f"Resuming from checkpoint: {len(completed_ids)} items completed")
        else:
            checkpoint = GenerationCheckpoint(
                model=self.model,
                total_items=len(items),
                completed_items=0,
                completed_ids=[],
                total_cost=0.0,
                total_time_ms=0.0,
            )
        
        results = []
        
        for i, item in enumerate(items):
            item_id = item.get("id", f"item_{i}")
            
            # Skip if already completed
            if item_id in completed_ids:
                continue
            
            logger.info(f"Processing {i+1}/{len(items)}: {item_id}")
            
            result = self.generate_for_item(item)
            results.append(result)
            
            # Save individual result
            result_path = output_dir / f"{item_id}.json"
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            
            # Update checkpoint
            checkpoint.completed_items += 1
            checkpoint.completed_ids.append(item_id)
            checkpoint.total_cost += result.cost
            checkpoint.total_time_ms += result.generation_time_ms
            checkpoint.last_updated = datetime.now().isoformat()
            
            if checkpoint_path:
                with open(checkpoint_path, "w") as f:
                    json.dump(checkpoint.to_dict(), f, indent=2)
            
            # Progress log
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {checkpoint.completed_items}/{len(items)}, "
                           f"Cost: ${checkpoint.total_cost:.4f}")
        
        logger.info(f"Generation complete: {len(results)} items, "
                   f"Total cost: ${checkpoint.total_cost:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate explanations for MEQ-Bench evaluation"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input benchmark items JSON"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for explanations"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o",
        help="Model to use for generation"
    )
    parser.add_argument(
        "--audiences",
        type=str,
        nargs="+",
        default=["physician", "nurse", "patient", "caregiver"],
        help="Target audiences"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Generation temperature"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between API calls (seconds)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum items to process (for testing)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    if args.max_items:
        items = items[:args.max_items]
    
    logger.info(f"Loaded {len(items)} items from {input_path}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Audiences: {args.audiences}")
    
    # Initialize generator
    generator = ExplanationGenerator(
        model=args.model,
        audiences=args.audiences,
        temperature=args.temperature,
        rate_limit_delay=args.rate_limit,
    )
    
    # Set up paths
    output_dir = Path(args.output)
    checkpoint_path = output_dir / "checkpoint.json" if args.resume else None
    
    # Generate explanations
    results = generator.generate_batch(
        items,
        output_dir,
        checkpoint_path=checkpoint_path,
    )
    
    # Save summary
    summary = {
        "model": args.model,
        "total_items": len(items),
        "successful": sum(1 for r in results if r.success),
        "failed": sum(1 for r in results if not r.success),
        "total_cost": sum(r.cost for r in results),
        "total_time_ms": sum(r.generation_time_ms for r in results),
        "completed_at": datetime.now().isoformat(),
    }
    
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Successful: {summary['successful']}, Failed: {summary['failed']}")
    
    return 0


if __name__ == "__main__":
    exit(main())

