#!/usr/bin/env python3
"""Generate train/dev/test splits for MedExplain-Evals dataset.

This script creates stratified splits of the benchmark dataset,
ensuring balanced representation across specialties and complexity levels.

Usage:
    python scripts/generate_splits.py --input data/benchmark_v2/full_dataset.json \
        --output-dir data/benchmark_v2/ \
        --train-ratio 0.70 --dev-ratio 0.15 --test-ratio 0.15
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def stratified_split(
    items: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_keys: List[str] = None,
    random_seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create stratified train/dev/test splits.
    
    Args:
        items: List of dataset items
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        test_ratio: Proportion for test set
        stratify_keys: Keys to stratify on (default: specialty, complexity_level)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train, dev, test) item lists
    """
    if stratify_keys is None:
        stratify_keys = ["specialty", "complexity_level"]
    
    # Set random seed
    random.seed(random_seed)
    
    # Validate ratios
    total_ratio = train_ratio + dev_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Group items by stratification key
    def get_strat_key(item: Dict) -> str:
        return "|".join(str(item.get(k, "unknown")) for k in stratify_keys)
    
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        key = get_strat_key(item)
        groups[key].append(item)
    
    train, dev, test = [], [], []
    
    # Split each group proportionally
    for group_key, group_items in groups.items():
        random.shuffle(group_items)
        n = len(group_items)
        
        train_n = int(n * train_ratio)
        dev_n = int(n * dev_ratio)
        # Remaining goes to test
        
        train.extend(group_items[:train_n])
        dev.extend(group_items[train_n:train_n + dev_n])
        test.extend(group_items[train_n + dev_n:])
    
    # Shuffle each split
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    
    return train, dev, test


def verify_splits(
    train: List[Dict],
    dev: List[Dict],
    test: List[Dict],
) -> Dict[str, Any]:
    """Verify split quality and generate statistics.
    
    Args:
        train: Training split
        dev: Development split
        test: Test split
        
    Returns:
        Verification statistics
    """
    def get_distribution(items: List[Dict], key: str) -> Dict[str, float]:
        from collections import Counter
        counts = Counter(item.get(key, "unknown") for item in items)
        total = len(items)
        return {k: v / total for k, v in counts.items()}
    
    stats = {
        "sizes": {
            "train": len(train),
            "dev": len(dev),
            "test": len(test),
            "total": len(train) + len(dev) + len(test),
        },
        "ratios": {
            "train": len(train) / (len(train) + len(dev) + len(test)),
            "dev": len(dev) / (len(train) + len(dev) + len(test)),
            "test": len(test) / (len(train) + len(dev) + len(test)),
        },
        "distributions": {},
    }
    
    # Check distributions for key fields
    for key in ["specialty", "complexity_level", "source_dataset"]:
        stats["distributions"][key] = {
            "train": get_distribution(train, key),
            "dev": get_distribution(dev, key),
            "test": get_distribution(test, key),
        }
    
    # Check for data leakage (same IDs in multiple splits)
    train_ids = set(item.get("id") for item in train)
    dev_ids = set(item.get("id") for item in dev)
    test_ids = set(item.get("id") for item in test)
    
    overlap_train_dev = train_ids & dev_ids
    overlap_train_test = train_ids & test_ids
    overlap_dev_test = dev_ids & test_ids
    
    stats["leakage"] = {
        "train_dev": len(overlap_train_dev),
        "train_test": len(overlap_train_test),
        "dev_test": len(overlap_dev_test),
        "clean": len(overlap_train_dev) == 0 and len(overlap_train_test) == 0 and len(overlap_dev_test) == 0,
    }
    
    return stats


def save_splits(
    train: List[Dict],
    dev: List[Dict],
    test: List[Dict],
    output_dir: Path,
    stats: Dict[str, Any],
) -> None:
    """Save splits to JSON files.
    
    Args:
        train: Training split
        dev: Development split
        test: Test split
        output_dir: Output directory
        stats: Split statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    splits = {
        "train.json": train,
        "dev.json": dev,
        "test.json": test,
    }
    
    for filename, data in splits.items():
        filepath = output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(data)} items to {filepath}")
    
    # Save metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "statistics": stats,
        "version": "2.0",
        "splits": {
            "train": "train.json",
            "dev": "dev.json",
            "test": "test.json",
        },
    }
    
    metadata_path = output_dir / "splits_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/dev/test splits for MedExplain-Evals"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input dataset JSON"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        required=True,
        help="Output directory for splits"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training set ratio (default: 0.70)"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Development set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stratify",
        type=str,
        nargs="+",
        default=["specialty", "complexity_level"],
        help="Keys to stratify on"
    )
    
    args = parser.parse_args()
    
    # Load input data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    logger.info(f"Loading dataset from {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    
    logger.info(f"Loaded {len(items)} items")
    
    # Create splits
    logger.info(f"Creating splits with ratios: train={args.train_ratio}, dev={args.dev_ratio}, test={args.test_ratio}")
    logger.info(f"Stratifying on: {args.stratify}")
    
    train, dev, test = stratified_split(
        items,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        stratify_keys=args.stratify,
        random_seed=args.seed,
    )
    
    # Verify splits
    logger.info("Verifying splits...")
    stats = verify_splits(train, dev, test)
    
    logger.info(f"Split sizes: train={len(train)}, dev={len(dev)}, test={len(test)}")
    logger.info(f"Ratios: train={stats['ratios']['train']:.2%}, dev={stats['ratios']['dev']:.2%}, test={stats['ratios']['test']:.2%}")
    
    if not stats["leakage"]["clean"]:
        logger.warning("Data leakage detected between splits!")
        logger.warning(f"  train/dev overlap: {stats['leakage']['train_dev']}")
        logger.warning(f"  train/test overlap: {stats['leakage']['train_test']}")
        logger.warning(f"  dev/test overlap: {stats['leakage']['dev_test']}")
    else:
        logger.info("No data leakage detected ✓")
    
    # Save splits
    save_splits(train, dev, test, args.output_dir, stats)
    
    logger.info("Split generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())

