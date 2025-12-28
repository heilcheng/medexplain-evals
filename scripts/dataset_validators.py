"""Dataset validation utilities for MEQ-Bench 2.0.

This module provides quality checks and validation functions for
benchmark dataset curation.

Features:
    - Content length validation
    - Duplicate detection
    - PII detection (basic patterns)
    - Medical content verification
    - Distribution validation
"""

import re
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from collections import Counter

logger = logging.getLogger("meq_bench.validators")


@dataclass
class ValidationResult:
    """Result from validation check."""
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # error, warning, info


@dataclass
class DatasetValidationReport:
    """Comprehensive validation report for a dataset."""
    total_items: int
    valid_items: int
    removed_items: int
    validation_results: List[ValidationResult]
    passed: bool
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items": self.total_items,
            "valid_items": self.valid_items,
            "removed_items": self.removed_items,
            "passed": self.passed,
            "summary": self.summary,
            "results": [
                {"passed": r.passed, "message": r.message, "severity": r.severity}
                for r in self.validation_results
            ],
        }


class ContentValidator:
    """Validate individual item content quality."""
    
    # Minimum and maximum content lengths
    MIN_CONTENT_LENGTH = 50  # characters
    MAX_CONTENT_LENGTH = 10000
    
    # PII patterns (basic detection)
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{9}\b",  # SSN without dashes
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone
        r"\b\d{5}(-\d{4})?\b",  # ZIP code
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",  # Full dates
        r"\bMRN[:\s]*\d+\b",  # Medical Record Number
        r"\bPatient\s+ID[:\s]*\d+\b",  # Patient ID
    ]
    
    # Medical content indicators
    MEDICAL_INDICATORS = [
        r"\b(diagnosis|treatment|medication|symptoms?|patient|disease|condition|therapy)\b",
        r"\b(mg|ml|dose|dosage|prescription|hospital|clinic|physician|doctor|nurse)\b",
        r"\b(blood|heart|lung|liver|kidney|brain|cancer|diabetes|hypertension)\b",
    ]
    
    def __init__(self):
        self.pii_compiled = [re.compile(p, re.IGNORECASE) for p in self.PII_PATTERNS]
        self.medical_compiled = [re.compile(p, re.IGNORECASE) for p in self.MEDICAL_INDICATORS]
    
    def validate_content_length(self, content: str) -> ValidationResult:
        """Check content length is within acceptable range."""
        length = len(content)
        
        if length < self.MIN_CONTENT_LENGTH:
            return ValidationResult(
                passed=False,
                message=f"Content too short: {length} chars (min: {self.MIN_CONTENT_LENGTH})",
                details={"length": length, "min": self.MIN_CONTENT_LENGTH},
                severity="error",
            )
        
        if length > self.MAX_CONTENT_LENGTH:
            return ValidationResult(
                passed=False,
                message=f"Content too long: {length} chars (max: {self.MAX_CONTENT_LENGTH})",
                details={"length": length, "max": self.MAX_CONTENT_LENGTH},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message=f"Content length OK: {length} chars",
            details={"length": length},
        )
    
    def check_pii(self, content: str) -> ValidationResult:
        """Check for potential PII in content."""
        pii_found = []
        
        for i, pattern in enumerate(self.pii_compiled):
            matches = pattern.findall(content)
            if matches:
                pii_found.append({
                    "pattern_index": i,
                    "count": len(matches),
                })
        
        if pii_found:
            return ValidationResult(
                passed=False,
                message=f"Potential PII detected: {len(pii_found)} pattern types matched",
                details={"pii_matches": pii_found},
                severity="error",
            )
        
        return ValidationResult(
            passed=True,
            message="No PII detected",
        )
    
    def verify_medical_content(self, content: str) -> ValidationResult:
        """Verify content appears to be medical in nature."""
        matches = 0
        
        for pattern in self.medical_compiled:
            if pattern.search(content):
                matches += 1
        
        if matches == 0:
            return ValidationResult(
                passed=False,
                message="Content does not appear to be medical",
                details={"medical_indicators": 0},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message=f"Medical content verified: {matches} indicator patterns",
            details={"medical_indicators": matches},
        )
    
    def check_language_quality(self, content: str) -> ValidationResult:
        """Basic check for language quality."""
        # Check for excessive special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s.,!?;:\-\'"()]', content)) / max(len(content), 1)
        
        if special_ratio > 0.15:
            return ValidationResult(
                passed=False,
                message=f"Excessive special characters: {special_ratio:.1%}",
                details={"special_char_ratio": special_ratio},
                severity="warning",
            )
        
        # Check for reasonable word count
        words = content.split()
        if len(words) < 10:
            return ValidationResult(
                passed=False,
                message=f"Too few words: {len(words)}",
                details={"word_count": len(words)},
                severity="error",
            )
        
        return ValidationResult(
            passed=True,
            message="Language quality OK",
            details={"word_count": len(words), "special_char_ratio": special_ratio},
        )
    
    def validate_item(self, item: Dict[str, Any]) -> Tuple[bool, List[ValidationResult]]:
        """Run all validations on an item.
        
        Args:
            item: Dictionary with 'medical_content' key
            
        Returns:
            Tuple of (passed, list of validation results)
        """
        content = item.get("medical_content", "")
        results = []
        
        # Run all checks
        results.append(self.validate_content_length(content))
        results.append(self.check_pii(content))
        results.append(self.verify_medical_content(content))
        results.append(self.check_language_quality(content))
        
        # Check for required fields
        required_fields = ["id", "medical_content", "complexity_level", "source_dataset"]
        for field in required_fields:
            if field not in item or not item[field]:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Missing required field: {field}",
                    severity="error",
                ))
        
        # Overall pass if no errors (warnings OK)
        errors = [r for r in results if not r.passed and r.severity == "error"]
        passed = len(errors) == 0
        
        return passed, results


class DuplicateDetector:
    """Detect and remove duplicate items."""
    
    def __init__(self, similarity_threshold: float = 0.9):
        """Initialize duplicate detector.
        
        Args:
            similarity_threshold: Jaccard similarity threshold for near-duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.seen_content: Dict[str, str] = {}  # hash -> id
    
    def get_content_hash(self, content: str) -> str:
        """Get hash of normalized content."""
        # Normalize: lowercase, remove extra whitespace
        normalized = " ".join(content.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_content_signature(self, content: str) -> Set[str]:
        """Get word set signature for similarity comparison."""
        words = set(content.lower().split())
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "and", "or", "but", "of", "to", "for", "in", "on", "at", "with"}
        return words - stopwords
    
    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def is_duplicate(self, item: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if item is a duplicate.
        
        Args:
            item: Item to check
            
        Returns:
            Tuple of (is_duplicate, duplicate_of_id)
        """
        content = item.get("medical_content", "")
        item_id = item.get("id", "unknown")
        
        # Check exact hash match
        content_hash = self.get_content_hash(content)
        if content_hash in self.seen_hashes:
            return True, self.seen_content.get(content_hash)
        
        # Store for future comparison
        self.seen_hashes.add(content_hash)
        self.seen_content[content_hash] = item_id
        
        return False, None
    
    def reset(self) -> None:
        """Reset duplicate tracking state."""
        self.seen_hashes.clear()
        self.seen_content.clear()


class DistributionValidator:
    """Validate dataset distribution across key dimensions."""
    
    # Target distributions
    TARGET_COMPLEXITY = {
        "basic": 0.25,
        "intermediate": 0.35,
        "advanced": 0.30,
        "expert": 0.10,
    }
    
    TARGET_SPECIALTY_MIN = 0.05  # At least 5% per specialty
    TARGET_SPECIALTY_MAX = 0.15  # At most 15% per specialty
    
    def __init__(self, tolerance: float = 0.10):
        """Initialize distribution validator.
        
        Args:
            tolerance: Allowed deviation from target distribution
        """
        self.tolerance = tolerance
    
    def validate_complexity_distribution(
        self,
        items: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate complexity level distribution."""
        complexity_counts = Counter(item.get("complexity_level") for item in items)
        total = len(items)
        
        if total == 0:
            return ValidationResult(
                passed=False,
                message="No items to validate",
                severity="error",
            )
        
        actual = {k: v / total for k, v in complexity_counts.items()}
        deviations = {}
        
        for level, target in self.TARGET_COMPLEXITY.items():
            actual_pct = actual.get(level, 0)
            deviation = abs(actual_pct - target)
            deviations[level] = {
                "target": target,
                "actual": actual_pct,
                "deviation": deviation,
            }
        
        max_deviation = max(d["deviation"] for d in deviations.values())
        
        if max_deviation > self.tolerance:
            return ValidationResult(
                passed=False,
                message=f"Complexity distribution deviation too high: {max_deviation:.1%}",
                details={"distributions": deviations},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message="Complexity distribution OK",
            details={"distributions": deviations},
        )
    
    def validate_specialty_distribution(
        self,
        items: List[Dict[str, Any]]
    ) -> ValidationResult:
        """Validate medical specialty distribution."""
        specialty_counts = Counter(item.get("specialty") for item in items)
        total = len(items)
        
        if total == 0:
            return ValidationResult(
                passed=False,
                message="No items to validate",
                severity="error",
            )
        
        issues = []
        distributions = {}
        
        for specialty, count in specialty_counts.items():
            pct = count / total
            distributions[specialty] = pct
            
            if pct < self.TARGET_SPECIALTY_MIN:
                issues.append(f"{specialty}: {pct:.1%} (below {self.TARGET_SPECIALTY_MIN:.0%})")
            elif pct > self.TARGET_SPECIALTY_MAX:
                issues.append(f"{specialty}: {pct:.1%} (above {self.TARGET_SPECIALTY_MAX:.0%})")
        
        if issues:
            return ValidationResult(
                passed=False,
                message=f"Specialty distribution issues: {len(issues)}",
                details={"distributions": distributions, "issues": issues},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message=f"Specialty distribution OK ({len(specialty_counts)} specialties)",
            details={"distributions": distributions},
        )
    
    def validate_source_diversity(
        self,
        items: List[Dict[str, Any]],
        min_sources: int = 3
    ) -> ValidationResult:
        """Validate dataset source diversity."""
        source_counts = Counter(item.get("source_dataset") for item in items)
        
        if len(source_counts) < min_sources:
            return ValidationResult(
                passed=False,
                message=f"Insufficient source diversity: {len(source_counts)} (min: {min_sources})",
                details={"sources": dict(source_counts)},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message=f"Source diversity OK: {len(source_counts)} sources",
            details={"sources": dict(source_counts)},
        )
    
    def validate_safety_coverage(
        self,
        items: List[Dict[str, Any]],
        target_safety_ratio: float = 0.30
    ) -> ValidationResult:
        """Validate coverage of safety-critical items."""
        safety_count = sum(1 for item in items if item.get("safety_critical", False))
        total = len(items)
        
        if total == 0:
            return ValidationResult(
                passed=False,
                message="No items to validate",
                severity="error",
            )
        
        ratio = safety_count / total
        
        if ratio < target_safety_ratio * 0.5:
            return ValidationResult(
                passed=False,
                message=f"Insufficient safety-critical items: {ratio:.1%} (target: {target_safety_ratio:.0%})",
                details={"safety_count": safety_count, "ratio": ratio},
                severity="warning",
            )
        
        return ValidationResult(
            passed=True,
            message=f"Safety coverage OK: {ratio:.1%} ({safety_count} items)",
            details={"safety_count": safety_count, "ratio": ratio},
        )


class DatasetValidator:
    """Main validator orchestrating all validation checks."""
    
    def __init__(self):
        self.content_validator = ContentValidator()
        self.duplicate_detector = DuplicateDetector()
        self.distribution_validator = DistributionValidator()
    
    def validate_dataset(
        self,
        items: List[Dict[str, Any]],
        remove_invalid: bool = True
    ) -> Tuple[List[Dict[str, Any]], DatasetValidationReport]:
        """Validate entire dataset and optionally remove invalid items.
        
        Args:
            items: List of benchmark items
            remove_invalid: Whether to remove invalid items
            
        Returns:
            Tuple of (validated items, validation report)
        """
        logger.info(f"Validating {len(items)} items...")
        
        all_results = []
        valid_items = []
        removed_count = 0
        
        # Reset duplicate detector
        self.duplicate_detector.reset()
        
        # Validate each item
        for item in items:
            # Check for duplicates
            is_dup, dup_of = self.duplicate_detector.is_duplicate(item)
            if is_dup:
                all_results.append(ValidationResult(
                    passed=False,
                    message=f"Duplicate of {dup_of}",
                    details={"item_id": item.get("id"), "duplicate_of": dup_of},
                    severity="error",
                ))
                if remove_invalid:
                    removed_count += 1
                    continue
            
            # Validate content
            passed, item_results = self.content_validator.validate_item(item)
            all_results.extend(item_results)
            
            if passed or not remove_invalid:
                valid_items.append(item)
            else:
                removed_count += 1
        
        # Validate distributions (on remaining items)
        if valid_items:
            all_results.append(
                self.distribution_validator.validate_complexity_distribution(valid_items)
            )
            all_results.append(
                self.distribution_validator.validate_specialty_distribution(valid_items)
            )
            all_results.append(
                self.distribution_validator.validate_source_diversity(valid_items)
            )
            all_results.append(
                self.distribution_validator.validate_safety_coverage(valid_items)
            )
        
        # Generate summary
        errors = [r for r in all_results if not r.passed and r.severity == "error"]
        warnings = [r for r in all_results if not r.passed and r.severity == "warning"]
        
        summary = {
            "errors": len(errors),
            "warnings": len(warnings),
            "duplicates_removed": sum(1 for r in all_results if "Duplicate" in r.message),
            "pii_detected": sum(1 for r in all_results if "PII" in r.message and not r.passed),
        }
        
        report = DatasetValidationReport(
            total_items=len(items),
            valid_items=len(valid_items),
            removed_items=removed_count,
            validation_results=all_results,
            passed=len(errors) == 0,
            summary=summary,
        )
        
        logger.info(f"Validation complete: {len(valid_items)}/{len(items)} valid items")
        logger.info(f"Removed: {removed_count}, Errors: {len(errors)}, Warnings: {len(warnings)}")
        
        return valid_items, report

