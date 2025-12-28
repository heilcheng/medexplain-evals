"""
Unit tests for the data processing script
"""

import pytest
import json
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the script modules for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from process_datasets import (
        calculate_dataset_limits,
        balance_complexity_distribution,
        validate_dataset,
        print_dataset_statistics,
        setup_argument_parser,
    )
except ImportError:
    # If direct import fails, we'll test through subprocess calls
    process_datasets = None

from src.benchmark import MedExplainItem


class TestDatasetLimitsCalculation:
    """Test dataset limits calculation"""

    def test_equal_distribution_three_datasets(self):
        """Test equal distribution across three datasets"""
        limits = calculate_dataset_limits(1000, 3)

        assert len(limits) == 3
        assert "medqa" in limits
        assert "icliniq" in limits
        assert "cochrane" in limits

        # Should distribute as evenly as possible
        total = sum(limits.values())
        assert total == 1000

        # Each should get approximately 333-334 items
        for limit in limits.values():
            assert 333 <= limit <= 334

    def test_equal_distribution_two_datasets(self):
        """Test equal distribution across two datasets"""
        limits = calculate_dataset_limits(1000, 2)

        assert len(limits) == 2
        total = sum(limits.values())
        assert total == 1000

        # Each should get 500 items
        for limit in limits.values():
            assert limit == 500

    def test_uneven_distribution_handling(self):
        """Test handling of uneven divisions"""
        limits = calculate_dataset_limits(1001, 3)

        total = sum(limits.values())
        assert total == 1001

        # Should handle remainder correctly
        limit_values = list(limits.values())
        assert max(limit_values) - min(limit_values) <= 1  # Difference should be at most 1

    def test_small_total_items(self):
        """Test handling of small total item counts"""
        limits = calculate_dataset_limits(5, 3)

        total = sum(limits.values())
        assert total == 5

        # Some datasets might get 1 item, others 2
        for limit in limits.values():
            assert limit >= 1


class TestComplexityBalancing:
    """Test complexity distribution balancing"""

    @pytest.fixture
    def sample_items(self):
        """Create sample items with different complexity levels"""
        items = []

        # Create 10 basic items
        for i in range(10):
            items.append(
                MedExplainItem(
                    id=f"basic_{i}",
                    medical_content=f"Basic medical content {i}",
                    complexity_level="basic",
                    source_dataset="test",
                )
            )

        # Create 5 intermediate items
        for i in range(5):
            items.append(
                MedExplainItem(
                    id=f"intermediate_{i}",
                    medical_content=f"Intermediate medical content {i}",
                    complexity_level="intermediate",
                    source_dataset="test",
                )
            )

        # Create 2 advanced items
        for i in range(2):
            items.append(
                MedExplainItem(
                    id=f"advanced_{i}",
                    medical_content=f"Advanced medical content {i}",
                    complexity_level="advanced",
                    source_dataset="test",
                )
            )

        return items

    @patch("process_datasets.random.sample")
    @patch("process_datasets.random.seed")
    def test_balance_complexity_distribution(self, mock_seed, mock_sample, sample_items):
        """Test complexity balancing functionality"""

        # Mock random.sample to return predictable results
        def side_effect(population, k):
            return population[:k]  # Return first k items

        mock_sample.side_effect = side_effect

        balanced_items = balance_complexity_distribution(sample_items)

        # Should have roughly equal distribution
        complexity_counts = {}
        for item in balanced_items:
            complexity_counts[item.complexity_level] = complexity_counts.get(item.complexity_level, 0) + 1

        # Check that balancing was attempted
        assert len(balanced_items) <= len(sample_items)
        assert "basic" in complexity_counts
        assert "intermediate" in complexity_counts
        assert "advanced" in complexity_counts

    def test_balance_empty_items(self):
        """Test balancing with empty item list"""
        balanced_items = balance_complexity_distribution([])
        assert balanced_items == []

    def test_balance_single_complexity_level(self):
        """Test balancing when only one complexity level exists"""
        items = [
            MedExplainItem(
                id=f"basic_{i}", medical_content=f"Basic content {i}", complexity_level="basic", source_dataset="test"
            )
            for i in range(5)
        ]

        balanced_items = balance_complexity_distribution(items)

        # Should still return items even if balancing isn't possible
        assert len(balanced_items) > 0
        assert all(item.complexity_level == "basic" for item in balanced_items)


class TestDatasetValidation:
    """Test dataset validation functionality"""

    def test_validate_empty_dataset(self):
        """Test validation of empty dataset"""
        report = validate_dataset([])

        assert report["valid"] is False
        assert report["total_items"] == 0
        assert "Dataset is empty" in report["issues"]

    def test_validate_valid_dataset(self):
        """Test validation of valid dataset"""
        items = [
            MedExplainItem(
                id="test_001",
                medical_content="This is valid medical content for testing purposes and is long enough.",
                complexity_level="basic",
                source_dataset="test",
            ),
            MedExplainItem(
                id="test_002",
                medical_content="This is another valid medical content item for comprehensive testing.",
                complexity_level="intermediate",
                source_dataset="test",
            ),
        ]

        report = validate_dataset(items)

        assert report["valid"] is True
        assert report["total_items"] == 2
        assert len(report["issues"]) == 0
        assert "complexity_distribution" in report["statistics"]
        assert "source_distribution" in report["statistics"]

    def test_validate_duplicate_ids(self):
        """Test detection of duplicate IDs"""
        items = [
            MedExplainItem(
                id="duplicate_id",
                medical_content="First item with duplicate ID and sufficient content length.",
                complexity_level="basic",
                source_dataset="test",
            ),
            MedExplainItem(
                id="duplicate_id",  # Same ID
                medical_content="Second item with duplicate ID and sufficient content length.",
                complexity_level="intermediate",
                source_dataset="test",
            ),
        ]

        report = validate_dataset(items)

        assert report["valid"] is False
        assert any("Duplicate item IDs" in issue for issue in report["issues"])

    def test_validate_short_content(self):
        """Test detection of very short content"""
        items = [
            MedExplainItem(id="test_001", medical_content="Short", complexity_level="basic", source_dataset="test")  # Too short
        ]

        report = validate_dataset(items)

        assert report["valid"] is False
        assert any("very short content" in issue for issue in report["issues"])

    def test_validate_missing_complexity_levels(self):
        """Test warning for missing complexity levels"""
        items = [
            MedExplainItem(
                id="test_001",
                medical_content="This is valid medical content with sufficient length for testing.",
                complexity_level="basic",  # Only basic level
                source_dataset="test",
            )
        ]

        report = validate_dataset(items)

        assert report["valid"] is True  # Still valid, just warning
        assert any("Not all complexity levels represented" in warning for warning in report["warnings"])

    def test_validate_content_length_statistics(self):
        """Test content length statistics calculation"""
        items = [
            MedExplainItem(
                id="test_001",
                medical_content="Short but valid medical content for testing purposes.",  # ~50 chars
                complexity_level="basic",
                source_dataset="test",
            ),
            MedExplainItem(
                id="test_002",
                medical_content="This is a much longer medical content item that contains significantly more text to test the average content length calculation functionality."
                * 3,  # ~400+ chars
                complexity_level="intermediate",
                source_dataset="test",
            ),
        ]

        report = validate_dataset(items)

        assert "content_length" in report["statistics"]
        assert "average" in report["statistics"]["content_length"]
        assert "minimum" in report["statistics"]["content_length"]
        assert "maximum" in report["statistics"]["content_length"]

        # Average should be reasonable
        avg_length = report["statistics"]["content_length"]["average"]
        assert 50 < avg_length < 500


class TestStatisticsPrinting:
    """Test statistics printing functionality"""

    def test_print_empty_dataset_statistics(self, capsys):
        """Test printing statistics for empty dataset"""
        print_dataset_statistics([])

        captured = capsys.readouterr()
        assert "Dataset is empty" in captured.out

    def test_print_valid_dataset_statistics(self, capsys):
        """Test printing statistics for valid dataset"""
        items = [
            MedExplainItem(
                id="test_001",
                medical_content="Valid medical content for testing statistics display functionality.",
                complexity_level="basic",
                source_dataset="MedQA-USMLE",
            ),
            MedExplainItem(
                id="test_002",
                medical_content="Another valid medical content item for comprehensive statistics testing.",
                complexity_level="intermediate",
                source_dataset="iCliniq",
            ),
            MedExplainItem(
                id="test_003",
                medical_content="Third valid medical content item for advanced complexity testing and statistics.",
                complexity_level="advanced",
                source_dataset="Cochrane Reviews",
            ),
        ]

        print_dataset_statistics(items)

        captured = capsys.readouterr()
        output = captured.out

        # Check that all expected sections are present
        assert "Dataset Statistics" in output
        assert "Complexity Distribution" in output
        assert "Source Distribution" in output
        assert "Content Length Statistics" in output

        # Check that complexity levels are shown
        assert "Basic" in output
        assert "Intermediate" in output
        assert "Advanced" in output

        # Check that sources are shown
        assert "MedQA-USMLE" in output
        assert "iCliniq" in output
        assert "Cochrane Reviews" in output

        # Check that statistics are shown
        assert "Total items: 3" in output
        assert "Average:" in output
        assert "Minimum:" in output
        assert "Maximum:" in output


class TestArgumentParser:
    """Test command line argument parser"""

    def test_argument_parser_setup(self):
        """Test that argument parser is set up correctly"""
        parser = setup_argument_parser()

        # Test that parser exists and has expected arguments
        args = parser.parse_args(
            ["--medqa", "medqa.json", "--icliniq", "icliniq.json", "--cochrane", "cochrane.json", "--output", "output.json"]
        )

        assert args.medqa == "medqa.json"
        assert args.icliniq == "icliniq.json"
        assert args.cochrane == "cochrane.json"
        assert args.output == "output.json"
        assert args.max_items == 1000  # Default value

    def test_argument_parser_defaults(self):
        """Test default argument values"""
        parser = setup_argument_parser()

        # Test with minimal arguments
        args = parser.parse_args(["--medqa", "test.json"])

        assert args.output == "data/benchmark_items.json"  # Default output
        assert args.max_items == 1000  # Default max items
        assert args.auto_complexity is True  # Default auto complexity
        assert args.balance_complexity is True  # Default balance
        assert args.seed == 42  # Default seed

    def test_argument_parser_custom_values(self):
        """Test custom argument values"""
        parser = setup_argument_parser()

        args = parser.parse_args(
            [
                "--medqa",
                "medqa.json",
                "--max-items",
                "500",
                "--medqa-items",
                "200",
                "--no-auto-complexity",
                "--seed",
                "123",
                "--validate",
                "--stats",
                "--verbose",
            ]
        )

        assert args.max_items == 500
        assert args.medqa_items == 200
        assert args.auto_complexity is False
        assert args.seed == 123
        assert args.validate is True
        assert args.stats is True
        assert args.verbose is True


class TestScriptIntegration:
    """Integration tests for the complete script"""

    @pytest.mark.skipif(process_datasets is None, reason="process_datasets module not available")
    def test_script_with_sample_data(self):
        """Test the complete script with sample data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample dataset files
            medqa_data = [
                {
                    "id": "medqa_001",
                    "question": "What is hypertension?",
                    "answer": "A",
                    "explanation": "High blood pressure condition.",
                }
            ]

            icliniq_data = [
                {
                    "id": "icliniq_001",
                    "patient_question": "I have chest pain",
                    "doctor_answer": "Consult a cardiologist",
                    "speciality": "Cardiology",
                }
            ]

            cochrane_data = [
                {"id": "cochrane_001", "title": "Statin effectiveness", "abstract": "Systematic review of statins"}
            ]

            # Save sample files
            medqa_file = temp_path / "medqa.json"
            icliniq_file = temp_path / "icliniq.json"
            cochrane_file = temp_path / "cochrane.json"
            output_file = temp_path / "output.json"

            with open(medqa_file, "w") as f:
                json.dump(medqa_data, f)
            with open(icliniq_file, "w") as f:
                json.dump(icliniq_data, f)
            with open(cochrane_file, "w") as f:
                json.dump(cochrane_data, f)

            # Run the script through subprocess to test CLI
            script_path = Path(__file__).parent.parent / "scripts" / "process_datasets.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--medqa",
                    str(medqa_file),
                    "--icliniq",
                    str(icliniq_file),
                    "--cochrane",
                    str(cochrane_file),
                    "--output",
                    str(output_file),
                    "--max-items",
                    "10",
                    "--no-auto-complexity",
                    "--validate",
                    "--stats",
                ],
                capture_output=True,
                text=True,
            )

            # Check that script ran successfully
            assert result.returncode == 0, f"Script failed with error: {result.stderr}"

            # Check that output file was created
            assert output_file.exists()

            # Verify output content
            with open(output_file, "r") as f:
                output_data = json.load(f)

            assert len(output_data) == 3  # Should have loaded all 3 items
            assert any(item["source_dataset"] == "MedQA-USMLE" for item in output_data)
            assert any(item["source_dataset"] == "iCliniq" for item in output_data)
            assert any(item["source_dataset"] == "Cochrane Reviews" for item in output_data)

    def test_script_error_handling(self):
        """Test script error handling with invalid inputs"""
        script_path = Path(__file__).parent.parent / "scripts" / "process_datasets.py"

        # Test with non-existent file
        result = subprocess.run(
            [sys.executable, str(script_path), "--medqa", "/nonexistent/file.json", "--output", "output.json"],
            capture_output=True,
            text=True,
        )

        # Should exit with error code
        assert result.returncode != 0

    def test_script_help_output(self):
        """Test that script provides help output"""
        script_path = Path(__file__).parent.parent / "scripts" / "process_datasets.py"

        result = subprocess.run([sys.executable, str(script_path), "--help"], capture_output=True, text=True)

        assert result.returncode == 0
        assert "Process medical datasets for MedExplain-Evals" in result.stdout
        assert "--medqa" in result.stdout
        assert "--icliniq" in result.stdout
        assert "--cochrane" in result.stdout


class TestScriptPerformance:
    """Test script performance characteristics"""

    @pytest.mark.skipif(process_datasets is None, reason="process_datasets module not available")
    def test_script_performance_with_large_dataset(self):
        """Test script performance with larger dataset"""
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create larger sample datasets
            large_medqa_data = [
                {
                    "id": f"medqa_{i:03d}",
                    "question": f"Medical question {i}",
                    "answer": "A",
                    "explanation": f"Medical explanation {i}",
                }
                for i in range(200)
            ]

            large_icliniq_data = [
                {
                    "id": f"icliniq_{i:03d}",
                    "patient_question": f"Patient question {i}",
                    "doctor_answer": f"Doctor answer {i}",
                    "speciality": "General",
                }
                for i in range(200)
            ]

            # Save large files
            medqa_file = temp_path / "large_medqa.json"
            icliniq_file = temp_path / "large_icliniq.json"
            output_file = temp_path / "large_output.json"

            with open(medqa_file, "w") as f:
                json.dump(large_medqa_data, f)
            with open(icliniq_file, "w") as f:
                json.dump(large_icliniq_data, f)

            # Time the script execution
            start_time = time.time()

            script_path = Path(__file__).parent.parent / "scripts" / "process_datasets.py"

            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--medqa",
                    str(medqa_file),
                    "--icliniq",
                    str(icliniq_file),
                    "--output",
                    str(output_file),
                    "--max-items",
                    "100",
                    "--no-auto-complexity",  # Skip complexity calculation for speed
                ],
                capture_output=True,
                text=True,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within reasonable time (30 seconds for large dataset)
            assert execution_time < 30.0, f"Script took {execution_time:.2f}s"
            assert result.returncode == 0
            assert output_file.exists()

            # Verify output quality
            with open(output_file, "r") as f:
                output_data = json.load(f)
            assert len(output_data) == 100  # Should respect max_items limit
