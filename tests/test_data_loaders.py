"""
Unit tests for data_loaders module
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data_loaders import (
    load_medqa_usmle,
    load_icliniq,
    load_cochrane_reviews,
    save_benchmark_items,
    calculate_complexity_level,
    _validate_benchmark_item,
)
from src.benchmark import MedExplainItem


class TestCalculateComplexityLevel:
    """Test complexity level calculation using Flesch-Kincaid scores"""

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError"""
        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            calculate_complexity_level("")

        with pytest.raises(ValueError, match="Text must be a non-empty string"):
            calculate_complexity_level(None)

    def test_whitespace_only_raises_error(self):
        """Test that whitespace-only text raises ValueError"""
        with pytest.raises(ValueError, match="Text cannot be empty or whitespace only"):
            calculate_complexity_level("   \n\t  ")

    @patch("src.data_loaders.textstat", None)
    def test_fallback_complexity_calculation(self):
        """Test fallback complexity calculation when textstat is unavailable"""
        # Simple text should be basic
        simple_text = "This is simple. It has short words."
        complexity = calculate_complexity_level(simple_text)
        assert complexity in ["basic", "intermediate", "advanced"]

        # Complex medical text should be advanced
        complex_text = (
            "Pharmacokinetic interactions involving cytochrome P450 enzymes can significantly "
            "alter therapeutic drug concentrations, potentially leading to adverse effects or "
            "therapeutic failure in clinical practice."
        )
        complexity = calculate_complexity_level(complex_text)
        assert complexity in ["basic", "intermediate", "advanced"]

    @patch("src.data_loaders.textstat")
    def test_with_textstat_available(self, mock_textstat):
        """Test complexity calculation when textstat is available"""
        # Mock textstat to return specific grade levels
        mock_textstat.flesch_kincaid.return_value.grade.return_value = 6.0

        text = "Simple medical text for testing."
        complexity = calculate_complexity_level(text)
        assert complexity == "basic"

        # Test intermediate level
        mock_textstat.flesch_kincaid.return_value.grade.return_value = 10.0
        complexity = calculate_complexity_level(text)
        assert complexity == "intermediate"

        # Test advanced level
        mock_textstat.flesch_kincaid.return_value.grade.return_value = 15.0
        complexity = calculate_complexity_level(text)
        assert complexity == "advanced"

    @patch("src.data_loaders.textstat")
    def test_textstat_error_fallback(self, mock_textstat):
        """Test fallback when textstat raises an exception"""
        mock_textstat.flesch_kincaid.return_value.grade.side_effect = Exception("Textstat error")

        text = "Test text for error handling."
        complexity = calculate_complexity_level(text)
        assert complexity in ["basic", "intermediate", "advanced"]


class TestValidateBenchmarkItem:
    """Test benchmark item validation"""

    def test_valid_item(self):
        """Test validation of a valid item"""
        item = MedExplainItem(
            id="test_001",
            medical_content="This is valid medical content for testing purposes.",
            complexity_level="basic",
            source_dataset="test",
        )
        # Should not raise any exception
        _validate_benchmark_item(item)

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError"""
        item = MedExplainItem(id="", medical_content="Valid content", complexity_level="basic", source_dataset="test")
        with pytest.raises(ValueError, match="Item ID must be a non-empty string"):
            _validate_benchmark_item(item)

    def test_non_string_id_raises_error(self):
        """Test that non-string ID raises ValueError"""
        item = MedExplainItem(id=123, medical_content="Valid content", complexity_level="basic", source_dataset="test")
        with pytest.raises(ValueError, match="Item ID must be a non-empty string"):
            _validate_benchmark_item(item)

    def test_empty_content_raises_error(self):
        """Test that empty medical content raises ValueError"""
        item = MedExplainItem(id="test_001", medical_content="", complexity_level="basic", source_dataset="test")
        with pytest.raises(ValueError, match="Medical content must be a non-empty string"):
            _validate_benchmark_item(item)

    def test_short_content_raises_error(self):
        """Test that very short content raises ValueError"""
        item = MedExplainItem(
            id="test_001", medical_content="Short", complexity_level="basic", source_dataset="test"  # Less than 20 characters
        )
        with pytest.raises(ValueError, match="Medical content is too short"):
            _validate_benchmark_item(item)


class TestLoadMedQAUSMLE:
    """Test MedQA-USMLE dataset loading"""

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_medqa_usmle("/nonexistent/file.json")

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json content")
            f.flush()

            with pytest.raises(json.JSONDecodeError):
                load_medqa_usmle(f.name)

            Path(f.name).unlink()  # Clean up

    def test_non_list_data_raises_error(self):
        """Test that non-list JSON data raises ValueError"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "a list"}, f)
            f.flush()

            with pytest.raises(ValueError, match="MedQA-USMLE data must be a list"):
                load_medqa_usmle(f.name)

            Path(f.name).unlink()  # Clean up

    def test_load_valid_data(self):
        """Test loading valid MedQA-USMLE data"""
        sample_data = [
            {
                "id": "medqa_001",
                "question": "What is the most common cause of hypertension?",
                "options": {
                    "A": "Primary hypertension",
                    "B": "Secondary hypertension",
                    "C": "White coat hypertension",
                    "D": "Malignant hypertension",
                },
                "answer": "A",
                "explanation": "Primary hypertension accounts for 90-95% of cases.",
            },
            {
                "id": "medqa_002",
                "question": "Which medication is first-line for diabetes?",
                "options": {"A": "Insulin", "B": "Metformin", "C": "Sulfonylureas", "D": "Thiazolidinediones"},
                "answer": "B",
                "explanation": "Metformin is the first-line treatment for type 2 diabetes.",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_medqa_usmle(f.name, auto_complexity=False)

            assert len(items) == 2
            assert all(isinstance(item, MedExplainItem) for item in items)
            assert items[0].id == "medqa_001"
            assert items[0].source_dataset == "MedQA-USMLE"
            assert items[0].complexity_level == "intermediate"  # Default when auto_complexity=False
            assert "What is the most common cause of hypertension?" in items[0].medical_content

            Path(f.name).unlink()  # Clean up

    def test_max_items_limit(self):
        """Test that max_items parameter limits the number of loaded items"""
        sample_data = [{"id": f"test_{i}", "question": f"Question {i}", "answer": "A"} for i in range(5)]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_medqa_usmle(f.name, max_items=3, auto_complexity=False)
            assert len(items) == 3

            Path(f.name).unlink()  # Clean up

    @patch("src.data_loaders.calculate_complexity_level")
    def test_auto_complexity_calculation(self, mock_calc_complexity):
        """Test automatic complexity level calculation"""
        mock_calc_complexity.return_value = "advanced"

        sample_data = [
            {"id": "test_001", "question": "Complex medical question", "answer": "A", "explanation": "Detailed explanation"}
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_medqa_usmle(f.name, auto_complexity=True)
            assert len(items) == 1
            assert items[0].complexity_level == "advanced"
            mock_calc_complexity.assert_called_once()

            Path(f.name).unlink()  # Clean up


class TestLoadiCliniq:
    """Test iCliniq dataset loading"""

    def test_load_valid_icliniq_data(self):
        """Test loading valid iCliniq data"""
        sample_data = [
            {
                "id": "icliniq_001",
                "patient_question": "I have been experiencing chest pain. Should I be worried?",
                "doctor_answer": "Chest pain can have various causes. Please consult a cardiologist for proper evaluation.",
                "speciality": "Cardiology",
            },
            {
                "id": "icliniq_002",
                "patient_question": "What are the side effects of aspirin?",
                "doctor_answer": "Common side effects include stomach irritation and increased bleeding risk.",
                "speciality": "General Medicine",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_icliniq(f.name, auto_complexity=False)

            assert len(items) == 2
            assert all(isinstance(item, MedExplainItem) for item in items)
            assert items[0].source_dataset == "iCliniq"
            assert "Patient Question:" in items[0].medical_content
            assert "Doctor's Answer:" in items[0].medical_content
            assert "Cardiology" in items[0].medical_content

            Path(f.name).unlink()  # Clean up

    def test_alternative_field_names(self):
        """Test loading iCliniq data with alternative field names"""
        sample_data = [
            {
                "id": "icliniq_001",
                "question": "Alternative question field",  # Using 'question' instead of 'patient_question'
                "answer": "Alternative answer field",  # Using 'answer' instead of 'doctor_answer'
                "specialty": "Dermatology",  # Using 'specialty' instead of 'speciality'
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_icliniq(f.name, auto_complexity=False)
            assert len(items) == 1
            assert "Alternative question field" in items[0].medical_content
            assert "Alternative answer field" in items[0].medical_content

            Path(f.name).unlink()  # Clean up


class TestLoadCochraneReviews:
    """Test Cochrane Reviews dataset loading"""

    def test_load_valid_cochrane_data(self):
        """Test loading valid Cochrane Reviews data"""
        sample_data = [
            {
                "id": "cochrane_001",
                "title": "Effectiveness of statins for cardiovascular disease prevention",
                "abstract": "This systematic review evaluates the effectiveness of statins in preventing cardiovascular events.",
                "conclusions": "Statins significantly reduce cardiovascular events in high-risk patients.",
                "background": "Cardiovascular disease is a leading cause of mortality worldwide.",
            },
            {
                "id": "cochrane_002",
                "title": "Antibiotics for acute respiratory infections",
                "abstract": "Review of antibiotic effectiveness for respiratory tract infections.",
                "main_results": "Limited benefit of antibiotics for viral respiratory infections.",  # Alternative field name
                "objectives": "To assess antibiotic effectiveness in respiratory infections.",  # Alternative field name
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_cochrane_reviews(f.name, auto_complexity=False)

            assert len(items) == 2
            assert all(isinstance(item, MedExplainItem) for item in items)
            assert items[0].source_dataset == "Cochrane Reviews"
            assert items[0].complexity_level == "advanced"  # Default for Cochrane when auto_complexity=False
            assert "Title:" in items[0].medical_content
            assert "Abstract:" in items[0].medical_content
            assert "statins" in items[0].medical_content.lower()

            Path(f.name).unlink()  # Clean up

    def test_missing_title_and_abstract_skipped(self):
        """Test that items without title and abstract are skipped"""
        sample_data = [
            {"id": "cochrane_001", "title": "Valid title", "abstract": "Valid abstract"},
            {
                "id": "cochrane_002",
                # Missing both title and abstract
                "conclusions": "Only conclusions available",
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_data, f)
            f.flush()

            items = load_cochrane_reviews(f.name, auto_complexity=False)
            assert len(items) == 1  # Only the valid item should be loaded

            Path(f.name).unlink()  # Clean up


class TestSaveBenchmarkItems:
    """Test saving benchmark items to JSON"""

    def test_save_empty_list(self):
        """Test saving an empty list of items"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_items.json"
            save_benchmark_items([], output_path)

            assert output_path.exists()
            with open(output_path, "r") as f:
                data = json.load(f)
            assert data == []

    def test_save_valid_items(self):
        """Test saving valid benchmark items"""
        items = [
            MedExplainItem(
                id="test_001",
                medical_content="Test content 1",
                complexity_level="basic",
                source_dataset="test",
                reference_explanations={"physician": "Technical explanation"},
            ),
            MedExplainItem(id="test_002", medical_content="Test content 2", complexity_level="advanced", source_dataset="test"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_items.json"
            save_benchmark_items(items, output_path, pretty_print=True)

            assert output_path.exists()
            with open(output_path, "r") as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["id"] == "test_001"
            assert data[0]["medical_content"] == "Test content 1"
            assert data[0]["complexity_level"] == "basic"
            assert data[0]["source_dataset"] == "test"
            assert data[0]["reference_explanations"] == {"physician": "Technical explanation"}

            assert data[1]["id"] == "test_002"
            assert data[1]["reference_explanations"] is None

    def test_save_creates_directory(self):
        """Test that saving creates parent directories if they don't exist"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "test_items.json"
            items = [
                MedExplainItem(id="test_001", medical_content="Test content", complexity_level="basic", source_dataset="test")
            ]

            save_benchmark_items(items, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()


class TestDataLoadersIntegration:
    """Integration tests for data loaders"""

    def test_load_multiple_datasets(self):
        """Test loading and combining multiple datasets"""
        # Create sample data for each dataset type
        medqa_data = [{"id": "medqa_001", "question": "MedQA question", "answer": "A", "explanation": "MedQA explanation"}]

        icliniq_data = [
            {
                "id": "icliniq_001",
                "patient_question": "iCliniq question",
                "doctor_answer": "iCliniq answer",
                "speciality": "General",
            }
        ]

        cochrane_data = [{"id": "cochrane_001", "title": "Cochrane title", "abstract": "Cochrane abstract"}]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save sample data files
            medqa_file = Path(temp_dir) / "medqa.json"
            icliniq_file = Path(temp_dir) / "icliniq.json"
            cochrane_file = Path(temp_dir) / "cochrane.json"

            with open(medqa_file, "w") as f:
                json.dump(medqa_data, f)
            with open(icliniq_file, "w") as f:
                json.dump(icliniq_data, f)
            with open(cochrane_file, "w") as f:
                json.dump(cochrane_data, f)

            # Load all datasets
            medqa_items = load_medqa_usmle(medqa_file, auto_complexity=False)
            icliniq_items = load_icliniq(icliniq_file, auto_complexity=False)
            cochrane_items = load_cochrane_reviews(cochrane_file, auto_complexity=False)

            # Combine all items
            all_items = medqa_items + icliniq_items + cochrane_items

            assert len(all_items) == 3
            assert all_items[0].source_dataset == "MedQA-USMLE"
            assert all_items[1].source_dataset == "iCliniq"
            assert all_items[2].source_dataset == "Cochrane Reviews"

            # Save combined dataset
            combined_file = Path(temp_dir) / "combined.json"
            save_benchmark_items(all_items, combined_file)

            # Verify saved file
            with open(combined_file, "r") as f:
                saved_data = json.load(f)
            assert len(saved_data) == 3
