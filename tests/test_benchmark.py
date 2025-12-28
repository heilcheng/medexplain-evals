"""
Unit tests for MedExplain class
"""

import pytest
import json
from pathlib import Path

from src.benchmark import MedExplain, MedExplainItem


class TestMedExplainItem:
    """Test MedExplainItem dataclass"""

    def test_creation(self):
        """Test basic item creation"""
        item = MedExplainItem(id="test_001", medical_content="Test content", complexity_level="basic", source_dataset="test")

        assert item.id == "test_001"
        assert item.medical_content == "Test content"
        assert item.complexity_level == "basic"
        assert item.source_dataset == "test"
        assert item.reference_explanations is None

    def test_creation_with_references(self):
        """Test item creation with reference explanations"""
        references = {"physician": "Technical explanation", "patient": "Simple explanation"}

        item = MedExplainItem(
            id="test_002",
            medical_content="Test content",
            complexity_level="intermediate",
            source_dataset="test",
            reference_explanations=references,
        )

        assert item.reference_explanations == references


class TestMedExplain:
    """Test MedExplain class"""

    def test_initialization(self):
        """Test benchmark initialization"""
        bench = MedExplain()
        assert bench.benchmark_items == []
        assert bench.evaluator is not None
        assert bench.prompt_template is not None

    def test_add_benchmark_item(self, sample_benchmark_item):
        """Test adding benchmark items"""
        bench = MedExplain()
        bench.add_benchmark_item(sample_benchmark_item)

        assert len(bench.benchmark_items) == 1
        assert bench.benchmark_items[0] == sample_benchmark_item

    def test_generate_explanations(self, sample_medical_content, dummy_model_function):
        """Test explanation generation"""
        bench = MedExplain()
        explanations = bench.generate_explanations(sample_medical_content, dummy_model_function)

        assert isinstance(explanations, dict)
        # Should contain at least some audience explanations
        assert len(explanations) > 0

    def test_create_sample_dataset(self, tmp_path):
        """Test sample dataset creation"""
        bench = MedExplain()
        output_path = tmp_path / "sample_dataset.json"

        sample_items = bench.create_sample_dataset(str(output_path))

        # Check items were created
        assert len(sample_items) > 0
        assert all(isinstance(item, MedExplainItem) for item in sample_items)

        # Check file was saved
        assert output_path.exists()

        # Check file contents
        with open(output_path, "r") as f:
            data = json.load(f)

        assert len(data) == len(sample_items)
        assert all("id" in item for item in data)
        assert all("medical_content" in item for item in data)

    def test_get_benchmark_stats_empty(self):
        """Test stats for empty benchmark"""
        bench = MedExplain()
        stats = bench.get_benchmark_stats()

        assert stats["total_items"] == 0
        assert "message" in stats

    def test_get_benchmark_stats_with_items(self, sample_benchmark_item):
        """Test stats with benchmark items"""
        bench = MedExplain()
        bench.add_benchmark_item(sample_benchmark_item)

        stats = bench.get_benchmark_stats()

        assert stats["total_items"] == 1
        assert "complexity_distribution" in stats
        assert "source_distribution" in stats
        assert stats["complexity_distribution"]["basic"] == 1
        assert stats["source_distribution"]["test"] == 1

    def test_evaluate_model_basic(self, sample_benchmark_item, dummy_model_function):
        """Test basic model evaluation"""
        bench = MedExplain()
        bench.add_benchmark_item(sample_benchmark_item)

        results = bench.evaluate_model(dummy_model_function, max_items=1)

        assert "total_items" in results
        assert "audience_scores" in results
        assert "complexity_scores" in results
        assert "detailed_results" in results
        assert "summary" in results

        assert results["total_items"] == 1
        assert len(results["detailed_results"]) == 1

    def test_save_results(self, tmp_path):
        """Test results saving"""
        bench = MedExplain()
        output_path = tmp_path / "test_results.json"

        test_results = {"total_items": 1, "test_data": "test_value"}

        bench.save_results(test_results, str(output_path))

        assert output_path.exists()

        with open(output_path, "r") as f:
            loaded_results = json.load(f)

        assert loaded_results == test_results

    def test_add_duplicate_benchmark_item(self, sample_benchmark_item):
        """Test that adding a benchmark item with a duplicate ID raises a ValueError"""
        bench = MedExplain()

        # Add the first item
        bench.add_benchmark_item(sample_benchmark_item)
        assert len(bench.benchmark_items) == 1

        # Try to add another item with the same ID
        duplicate_item = MedExplainItem(
            id=sample_benchmark_item.id,  # Same ID as the first item
            medical_content="Different content",
            complexity_level="intermediate",
            source_dataset="different_source",
        )

        # Should raise ValueError due to duplicate ID
        with pytest.raises(ValueError, match=f"Item with ID '{sample_benchmark_item.id}' already exists"):
            bench.add_benchmark_item(duplicate_item)

        # Verify the original item count remains the same
        assert len(bench.benchmark_items) == 1

    def test_generate_explanations_empty_content(self, dummy_model_function):
        """Test that generate_explanations raises a ValueError when medical_content is empty"""
        bench = MedExplain()

        # Test with completely empty string
        with pytest.raises(ValueError, match="medical_content cannot be empty or contain only whitespace"):
            bench.generate_explanations("", dummy_model_function)

        # Test with whitespace-only string
        with pytest.raises(ValueError, match="medical_content cannot be empty or contain only whitespace"):
            bench.generate_explanations("   \n\t  ", dummy_model_function)

        # Test with very short content (less than 10 characters)
        with pytest.raises(ValueError, match="medical_content must be at least 10 characters long"):
            bench.generate_explanations("short", dummy_model_function)

    def test_evaluate_model_no_items(self, dummy_model_function):
        """Test that evaluate_model returns appropriate result when there are no benchmark items"""
        bench = MedExplain()

        # Ensure no items are loaded
        assert len(bench.benchmark_items) == 0

        # Evaluate model with no items
        results = bench.evaluate_model(dummy_model_function)

        # Should return a valid results structure but with zero items
        assert isinstance(results, dict)
        assert results["total_items"] == 0
        assert "audience_scores" in results
        assert "complexity_scores" in results
        assert "detailed_results" in results
        assert "summary" in results

        # All audience scores should be empty lists
        for audience in ["physician", "nurse", "patient", "caregiver"]:
            assert results["audience_scores"][audience] == []

        # All complexity scores should be empty lists
        for complexity in ["basic", "intermediate", "advanced"]:
            assert results["complexity_scores"][complexity] == []

        # Detailed results should be empty
        assert results["detailed_results"] == []

        # Summary should handle empty data gracefully
        summary = results["summary"]
        assert isinstance(summary, dict)
        # Most summary stats should be absent or 0 for empty data
        if "overall_mean" in summary:
            # If present, should be a reasonable default or empty value
            assert summary["overall_mean"] is None or isinstance(summary["overall_mean"], (int, float))
