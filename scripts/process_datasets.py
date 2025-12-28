#!/usr/bin/env python3
"""
Script to process and combine medical datasets into MedExplain-Evals format.

This script loads data from MedQA-USMLE, iCliniq, and Cochrane Reviews datasets,
applies complexity stratification using Flesch-Kincaid scores, and creates a
1,000-item benchmark dataset saved as data/benchmark_items.json.

Usage:
    python scripts/process_datasets.py --medqa data/medqa_usmle.json \
                                      --icliniq data/icliniq.json \
                                      --cochrane data/cochrane.json \
                                      --output data/benchmark_items.json \
                                      --max-items 1000

Author: MedExplain-Evals Team
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loaders import (
    load_medqa_usmle,
    load_icliniq, 
    load_cochrane_reviews,
    save_benchmark_items,
    calculate_complexity_level
)
from benchmark import MedExplainItem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('process_datasets')


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process medical datasets for MedExplain-Evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all datasets with equal distribution
    python scripts/process_datasets.py --medqa data/medqa_usmle.json \\
                                      --icliniq data/icliniq.json \\
                                      --cochrane data/cochrane.json \\
                                      --output data/benchmark_items.json

    # Process with custom item limits per dataset
    python scripts/process_datasets.py --medqa data/medqa_usmle.json \\
                                      --icliniq data/icliniq.json \\
                                      --cochrane data/cochrane.json \\
                                      --output data/benchmark_items.json \\
                                      --max-items 1500 \\
                                      --medqa-items 600 \\
                                      --icliniq-items 500 \\
                                      --cochrane-items 400
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/benchmark_items.json',
        help='Output path for the combined benchmark dataset (default: data/benchmark_items.json)'
    )
    
    # Dataset file arguments
    parser.add_argument(
        '--medqa',
        type=str,
        help='Path to MedQA-USMLE dataset JSON file'
    )
    
    parser.add_argument(
        '--icliniq',
        type=str,
        help='Path to iCliniq dataset JSON file'
    )
    
    parser.add_argument(
        '--cochrane',
        type=str,
        help='Path to Cochrane Reviews dataset JSON file'
    )
    
    # Control arguments
    parser.add_argument(
        '--max-items',
        type=int,
        default=1000,
        help='Maximum total number of items in final benchmark (default: 1000)'
    )
    
    parser.add_argument(
        '--medqa-items',
        type=int,
        help='Maximum items from MedQA-USMLE (default: auto-calculated)'
    )
    
    parser.add_argument(
        '--icliniq-items',
        type=int,
        help='Maximum items from iCliniq (default: auto-calculated)'
    )
    
    parser.add_argument(
        '--cochrane-items',
        type=int,
        help='Maximum items from Cochrane Reviews (default: auto-calculated)'
    )
    
    parser.add_argument(
        '--auto-complexity',
        action='store_true',
        default=True,
        help='Automatically calculate complexity levels using Flesch-Kincaid scores (default: True)'
    )
    
    parser.add_argument(
        '--no-auto-complexity',
        action='store_false',
        dest='auto_complexity',
        help='Disable automatic complexity calculation'
    )
    
    parser.add_argument(
        '--balance-complexity',
        action='store_true',
        default=True,
        help='Balance the dataset across complexity levels (default: True)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible dataset creation (default: 42)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the final dataset after creation'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show detailed statistics about the created dataset'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def calculate_dataset_limits(total_items: int, num_datasets: int) -> Dict[str, int]:
    """Calculate balanced item limits for each dataset."""
    base_items = total_items // num_datasets
    remainder = total_items % num_datasets
    
    limits = {}
    dataset_names = ['medqa', 'icliniq', 'cochrane']
    
    for i, name in enumerate(dataset_names[:num_datasets]):
        limits[name] = base_items + (1 if i < remainder else 0)
    
    return limits


def balance_complexity_distribution(items: List[MedExplainItem], target_distribution: Optional[Dict[str, float]] = None) -> List[MedExplainItem]:
    """Balance the complexity distribution of items."""
    if target_distribution is None:
        # Default: roughly equal distribution
        target_distribution = {'basic': 0.33, 'intermediate': 0.34, 'advanced': 0.33}
    
    # Group items by complexity
    complexity_groups = {'basic': [], 'intermediate': [], 'advanced': []}
    for item in items:
        if item.complexity_level in complexity_groups:
            complexity_groups[item.complexity_level].append(item)
    
    # Calculate target counts
    total_items = len(items)
    target_counts = {
        level: int(total_items * ratio) 
        for level, ratio in target_distribution.items()
    }
    
    # Adjust for rounding differences
    actual_total = sum(target_counts.values())
    if actual_total < total_items:
        target_counts['intermediate'] += total_items - actual_total
    
    # Sample items for balanced distribution
    balanced_items = []
    import random
    
    for level, target_count in target_counts.items():
        available_items = complexity_groups[level]
        if len(available_items) >= target_count:
            # Randomly sample target_count items
            sampled_items = random.sample(available_items, target_count)
        else:
            # Use all available items
            sampled_items = available_items
            logger.warning(f"Only {len(available_items)} {level} items available, target was {target_count}")
        
        balanced_items.extend(sampled_items)
    
    logger.info(f"Balanced dataset: {len(balanced_items)} total items")
    for level in ['basic', 'intermediate', 'advanced']:
        count = sum(1 for item in balanced_items if item.complexity_level == level)
        percentage = (count / len(balanced_items)) * 100 if balanced_items else 0
        logger.info(f"  {level}: {count} items ({percentage:.1f}%)")
    
    return balanced_items


def validate_dataset(items: List[MedExplainItem]) -> Dict[str, Any]:
    """Validate the created dataset and return validation report."""
    validation_report = {
        'valid': True,
        'total_items': len(items),
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    if not items:
        validation_report['valid'] = False
        validation_report['issues'].append("Dataset is empty")
        return validation_report
    
    # Check for duplicate IDs
    ids = [item.id for item in items]
    duplicate_ids = set([id for id in ids if ids.count(id) > 1])
    if duplicate_ids:
        validation_report['valid'] = False
        validation_report['issues'].append(f"Duplicate item IDs found: {duplicate_ids}")
    
    # Complexity distribution
    complexity_counts = {}
    for item in items:
        complexity = item.complexity_level
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    validation_report['statistics']['complexity_distribution'] = complexity_counts
    
    # Source distribution
    source_counts = {}
    for item in items:
        source = item.source_dataset
        source_counts[source] = source_counts.get(source, 0) + 1
    
    validation_report['statistics']['source_distribution'] = source_counts
    
    # Check for balanced distribution
    if len(complexity_counts) < 3:
        validation_report['warnings'].append("Not all complexity levels represented")
    
    # Content length statistics
    content_lengths = [len(item.medical_content) for item in items]
    avg_length = sum(content_lengths) / len(content_lengths)
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    
    validation_report['statistics']['content_length'] = {
        'average': avg_length,
        'minimum': min_length,
        'maximum': max_length
    }
    
    if avg_length < 50:
        validation_report['warnings'].append("Average content length is quite short")
    elif avg_length > 2000:
        validation_report['warnings'].append("Average content length is quite long")
    
    # Check for very short content
    short_content_items = [
        item.id for item in items 
        if len(item.medical_content.strip()) < 20
    ]
    if short_content_items:
        validation_report['valid'] = False
        validation_report['issues'].append(f"Items with very short content: {short_content_items}")
    
    return validation_report


def print_dataset_statistics(items: List[MedExplainItem]) -> None:
    """Print detailed statistics about the dataset."""
    if not items:
        print("Dataset is empty")
        return
    
    print(f"\n📊 Dataset Statistics")
    print(f"{'='*50}")
    print(f"Total items: {len(items)}")
    
    # Complexity distribution
    complexity_counts = {}
    for item in items:
        complexity = item.complexity_level
        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
    
    print(f"\n🎯 Complexity Distribution:")
    for level in ['basic', 'intermediate', 'advanced']:
        count = complexity_counts.get(level, 0)
        percentage = (count / len(items)) * 100 if items else 0
        print(f"  {level.capitalize():<12}: {count:>4} items ({percentage:>5.1f}%)")
    
    # Source distribution
    source_counts = {}
    for item in items:
        source = item.source_dataset
        source_counts[source] = source_counts.get(source, 0) + 1
    
    print(f"\n📚 Source Distribution:")
    for source, count in sorted(source_counts.items()):
        percentage = (count / len(items)) * 100 if items else 0
        print(f"  {source:<15}: {count:>4} items ({percentage:>5.1f}%)")
    
    # Content length statistics
    content_lengths = [len(item.medical_content) for item in items]
    avg_length = sum(content_lengths) / len(content_lengths)
    min_length = min(content_lengths)
    max_length = max(content_lengths)
    
    print(f"\n📏 Content Length Statistics:")
    print(f"  Average: {avg_length:>6.1f} characters")
    print(f"  Minimum: {min_length:>6} characters")
    print(f"  Maximum: {max_length:>6} characters")


def main():
    """Main processing function."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    
    logger.info("Starting MedExplain-Evals dataset processing")
    logger.info(f"Target total items: {args.max_items}")
    
    # Check which datasets are provided
    datasets_to_load = []
    if args.medqa:
        datasets_to_load.append('medqa')
    if args.icliniq:
        datasets_to_load.append('icliniq')
    if args.cochrane:
        datasets_to_load.append('cochrane')
    
    if not datasets_to_load:
        logger.error("No dataset files provided. Use --medqa, --icliniq, or --cochrane to specify input files.")
        sys.exit(1)
    
    # Calculate item limits per dataset
    if args.medqa_items or args.icliniq_items or args.cochrane_items:
        # Use custom limits
        dataset_limits = {}
        if args.medqa_items:
            dataset_limits['medqa'] = args.medqa_items
        if args.icliniq_items:
            dataset_limits['icliniq'] = args.icliniq_items
        if args.cochrane_items:
            dataset_limits['cochrane'] = args.cochrane_items
    else:
        # Calculate balanced limits
        dataset_limits = calculate_dataset_limits(args.max_items, len(datasets_to_load))
    
    logger.info(f"Dataset limits: {dataset_limits}")
    
    # Load datasets
    all_items = []
    
    if 'medqa' in datasets_to_load and args.medqa:
        logger.info(f"Loading MedQA-USMLE from: {args.medqa}")
        try:
            medqa_items = load_medqa_usmle(
                args.medqa, 
                max_items=dataset_limits.get('medqa'),
                auto_complexity=args.auto_complexity
            )
            all_items.extend(medqa_items)
            logger.info(f"Loaded {len(medqa_items)} MedQA-USMLE items")
        except Exception as e:
            logger.error(f"Failed to load MedQA-USMLE: {e}")
            if Path(args.medqa).exists():
                logger.error("File exists but failed to load. Check file format.")
            else:
                logger.error("File not found. Check the file path.")
    
    if 'icliniq' in datasets_to_load and args.icliniq:
        logger.info(f"Loading iCliniq from: {args.icliniq}")
        try:
            icliniq_items = load_icliniq(
                args.icliniq,
                max_items=dataset_limits.get('icliniq'),
                auto_complexity=args.auto_complexity
            )
            all_items.extend(icliniq_items)
            logger.info(f"Loaded {len(icliniq_items)} iCliniq items")
        except Exception as e:
            logger.error(f"Failed to load iCliniq: {e}")
            if Path(args.icliniq).exists():
                logger.error("File exists but failed to load. Check file format.")
            else:
                logger.error("File not found. Check the file path.")
    
    if 'cochrane' in datasets_to_load and args.cochrane:
        logger.info(f"Loading Cochrane Reviews from: {args.cochrane}")
        try:
            cochrane_items = load_cochrane_reviews(
                args.cochrane,
                max_items=dataset_limits.get('cochrane'),
                auto_complexity=args.auto_complexity
            )
            all_items.extend(cochrane_items)
            logger.info(f"Loaded {len(cochrane_items)} Cochrane Reviews items")
        except Exception as e:
            logger.error(f"Failed to load Cochrane Reviews: {e}")
            if Path(args.cochrane).exists():
                logger.error("File exists but failed to load. Check file format.")
            else:
                logger.error("File not found. Check the file path.")
    
    if not all_items:
        logger.error("No items were successfully loaded from any dataset")
        sys.exit(1)
    
    logger.info(f"Total items loaded: {len(all_items)}")
    
    # Balance complexity distribution if requested
    if args.balance_complexity and len(all_items) > 10:
        logger.info("Balancing complexity distribution...")
        all_items = balance_complexity_distribution(all_items)
    
    # Limit to max_items if necessary
    if len(all_items) > args.max_items:
        logger.info(f"Limiting dataset to {args.max_items} items")
        import random
        all_items = random.sample(all_items, args.max_items)
    
    # Validate dataset if requested
    if args.validate:
        logger.info("Validating dataset...")
        validation_report = validate_dataset(all_items)
        
        if validation_report['valid']:
            logger.info("✅ Dataset validation passed")
        else:
            logger.error("❌ Dataset validation failed")
            for issue in validation_report['issues']:
                logger.error(f"  Issue: {issue}")
        
        for warning in validation_report['warnings']:
            logger.warning(f"  Warning: {warning}")
    
    # Save the combined dataset
    logger.info(f"Saving combined dataset to: {args.output}")
    try:
        save_benchmark_items(all_items, args.output, pretty_print=True)
        logger.info(f"✅ Successfully saved {len(all_items)} items to {args.output}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        print_dataset_statistics(all_items)
    
    # Summary
    logger.info(f"\n🎉 Dataset processing completed successfully!")
    logger.info(f"Final dataset: {len(all_items)} items saved to {args.output}")


if __name__ == "__main__":
    main()