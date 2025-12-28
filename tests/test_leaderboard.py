"""
Unit tests for leaderboard module
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.leaderboard import LeaderboardGenerator


class TestLeaderboardGenerator:
    """Test LeaderboardGenerator class"""

    @pytest.fixture
    def sample_results_data(self):
        """Sample evaluation results data for testing"""
        return [
            {
                "model_name": "GPT-4",
                "total_items": 100,
                "audience_scores": {
                    "physician": [0.9, 0.8, 0.85],
                    "nurse": [0.8, 0.75, 0.8],
                    "patient": [0.7, 0.65, 0.7],
                    "caregiver": [0.75, 0.7, 0.75],
                },
                "complexity_scores": {"basic": [0.8, 0.75], "intermediate": [0.7, 0.8], "advanced": [0.6, 0.65]},
                "summary": {
                    "overall_mean": 0.75,
                    "physician_mean": 0.85,
                    "nurse_mean": 0.78,
                    "patient_mean": 0.68,
                    "caregiver_mean": 0.73,
                },
            },
            {
                "model_name": "Claude-3",
                "total_items": 100,
                "audience_scores": {
                    "physician": [0.85, 0.9, 0.88],
                    "nurse": [0.82, 0.85, 0.83],
                    "patient": [0.75, 0.8, 0.78],
                    "caregiver": [0.8, 0.82, 0.81],
                },
                "complexity_scores": {"basic": [0.85, 0.9], "intermediate": [0.8, 0.85], "advanced": [0.75, 0.8]},
                "summary": {
                    "overall_mean": 0.82,
                    "physician_mean": 0.88,
                    "nurse_mean": 0.83,
                    "patient_mean": 0.78,
                    "caregiver_mean": 0.81,
                },
            },
            {
                "model_name": "LLaMA-2",
                "total_items": 100,
                "audience_scores": {
                    "physician": [0.7, 0.75, 0.72],
                    "nurse": [0.68, 0.7, 0.69],
                    "patient": [0.6, 0.65, 0.62],
                    "caregiver": [0.65, 0.68, 0.66],
                },
                "complexity_scores": {"basic": [0.7, 0.75], "intermediate": [0.65, 0.7], "advanced": [0.55, 0.6]},
                "summary": {
                    "overall_mean": 0.67,
                    "physician_mean": 0.72,
                    "nurse_mean": 0.69,
                    "patient_mean": 0.62,
                    "caregiver_mean": 0.66,
                },
            },
        ]

    @pytest.fixture
    def leaderboard_generator(self):
        """Create LeaderboardGenerator instance for testing"""
        return LeaderboardGenerator()

    def test_initialization(self, leaderboard_generator):
        """Test proper initialization of LeaderboardGenerator"""
        assert hasattr(leaderboard_generator, "results_data")
        assert hasattr(leaderboard_generator, "benchmark_stats")
        assert leaderboard_generator.results_data == []
        assert leaderboard_generator.benchmark_stats == {}

    def test_load_results_file_not_found(self, leaderboard_generator):
        """Test error handling when results directory doesn't exist"""
        with pytest.raises(FileNotFoundError):
            leaderboard_generator.load_results(Path("/nonexistent/directory"))

    def test_load_results_no_json_files(self, leaderboard_generator):
        """Test error handling when no JSON files found"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory with no JSON files
            Path(temp_dir, "not_json.txt").write_text("not json")

            with pytest.raises(ValueError, match="No JSON result files found"):
                leaderboard_generator.load_results(Path(temp_dir))

    def test_load_results_invalid_json(self, leaderboard_generator):
        """Test handling of invalid JSON files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create invalid JSON file
            invalid_file = Path(temp_dir) / "invalid.json"
            invalid_file.write_text("invalid json content")

            # Should log error but not crash
            leaderboard_generator.load_results(Path(temp_dir))
            assert len(leaderboard_generator.results_data) == 0

    def test_load_results_missing_required_fields(self, leaderboard_generator):
        """Test handling of JSON files missing required fields"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create JSON file missing required fields
            incomplete_data = {"model_name": "TestModel"}  # Missing other required fields
            incomplete_file = Path(temp_dir) / "incomplete.json"
            with open(incomplete_file, "w") as f:
                json.dump(incomplete_data, f)

            # Should skip invalid files
            leaderboard_generator.load_results(Path(temp_dir))
            assert len(leaderboard_generator.results_data) == 0

    def test_load_results_valid_data(self, leaderboard_generator, sample_results_data):
        """Test loading valid results data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid JSON files
            for i, result in enumerate(sample_results_data):
                result_file = Path(temp_dir) / f"result_{i}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f)

            leaderboard_generator.load_results(Path(temp_dir))

            assert len(leaderboard_generator.results_data) == 3
            assert leaderboard_generator.results_data[0]["model_name"] == "GPT-4"
            assert leaderboard_generator.results_data[1]["model_name"] == "Claude-3"
            assert leaderboard_generator.results_data[2]["model_name"] == "LLaMA-2"

    def test_calculate_leaderboard_stats_empty_data(self, leaderboard_generator):
        """Test stats calculation with empty data"""
        stats = leaderboard_generator.calculate_leaderboard_stats()
        assert stats == {}

    def test_calculate_leaderboard_stats_valid_data(self, leaderboard_generator, sample_results_data):
        """Test stats calculation with valid data"""
        leaderboard_generator.results_data = sample_results_data
        stats = leaderboard_generator.calculate_leaderboard_stats()

        assert stats["total_models"] == 3
        assert stats["total_evaluations"] == 300  # 3 models * 100 items each
        assert set(stats["audiences"]) == {"physician", "nurse", "patient", "caregiver"}
        assert set(stats["complexity_levels"]) == {"basic", "intermediate", "advanced"}
        assert stats["best_score"] == 0.82  # Claude-3's score
        assert stats["worst_score"] == 0.67  # LLaMA-2's score
        assert 0.6 < stats["average_score"] < 0.8
        assert "last_updated" in stats

    def test_rank_models(self, leaderboard_generator, sample_results_data):
        """Test model ranking functionality"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()

        # Should be ranked by overall_mean in descending order
        assert len(ranked_models) == 3
        assert ranked_models[0]["model_name"] == "Claude-3"  # Highest score (0.82)
        assert ranked_models[0]["rank"] == 1
        assert ranked_models[0]["overall_score"] == 0.82

        assert ranked_models[1]["model_name"] == "GPT-4"  # Middle score (0.75)
        assert ranked_models[1]["rank"] == 2
        assert ranked_models[1]["overall_score"] == 0.75

        assert ranked_models[2]["model_name"] == "LLaMA-2"  # Lowest score (0.67)
        assert ranked_models[2]["rank"] == 3
        assert ranked_models[2]["overall_score"] == 0.67

    def test_generate_audience_breakdown(self, leaderboard_generator, sample_results_data):
        """Test audience-specific performance breakdown"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()
        audience_breakdown = leaderboard_generator.generate_audience_breakdown(ranked_models)

        # Should have breakdown for each audience
        assert set(audience_breakdown.keys()) == {"physician", "nurse", "patient", "caregiver"}

        # Check physician audience breakdown
        physician_breakdown = audience_breakdown["physician"]
        assert len(physician_breakdown) == 3

        # Models should be ranked by their physician-specific scores
        physician_scores = [model["score"] for model in physician_breakdown]
        assert physician_scores == sorted(physician_scores, reverse=True)

        # Check that rankings are assigned correctly
        for i, model in enumerate(physician_breakdown):
            assert model["rank"] == i + 1
            assert "model_name" in model
            assert "score" in model
            assert "num_items" in model

    def test_generate_complexity_breakdown(self, leaderboard_generator, sample_results_data):
        """Test complexity-specific performance breakdown"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()
        complexity_breakdown = leaderboard_generator.generate_complexity_breakdown(ranked_models)

        # Should have breakdown for each complexity level
        assert set(complexity_breakdown.keys()) == {"basic", "intermediate", "advanced"}

        # Check basic complexity breakdown
        basic_breakdown = complexity_breakdown["basic"]
        assert len(basic_breakdown) == 3

        # Models should be ranked by their basic-specific scores
        basic_scores = [model["score"] for model in basic_breakdown]
        assert basic_scores == sorted(basic_scores, reverse=True)

        # Check that rankings are assigned correctly
        for i, model in enumerate(basic_breakdown):
            assert model["rank"] == i + 1

    def test_generate_html_no_data_raises_error(self, leaderboard_generator):
        """Test that HTML generation raises error with no data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.html"

            with pytest.raises(ValueError, match="No results data loaded"):
                leaderboard_generator.generate_html(output_path)

    def test_generate_html_creates_file(self, leaderboard_generator, sample_results_data):
        """Test that HTML generation creates file"""
        leaderboard_generator.results_data = sample_results_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "leaderboard.html"
            leaderboard_generator.generate_html(output_path)

            assert output_path.exists()
            html_content = output_path.read_text()

            # Check that HTML contains expected elements
            assert "<!DOCTYPE html>" in html_content
            assert "MedExplain-Evals Leaderboard" in html_content
            assert "Claude-3" in html_content  # Top-ranked model
            assert "GPT-4" in html_content
            assert "LLaMA-2" in html_content
            assert "physician" in html_content
            assert "patient" in html_content

    def test_generate_html_creates_parent_directory(self, leaderboard_generator, sample_results_data):
        """Test that HTML generation creates parent directories"""
        leaderboard_generator.results_data = sample_results_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "leaderboard.html"
            leaderboard_generator.generate_html(output_path)

            assert output_path.exists()
            assert output_path.parent.exists()

    def test_html_template_structure(self, leaderboard_generator, sample_results_data):
        """Test that generated HTML has proper structure"""
        leaderboard_generator.results_data = sample_results_data

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test.html"
            leaderboard_generator.generate_html(output_path)

            html_content = output_path.read_text()

            # Check HTML structure
            assert "<html" in html_content
            assert "<head>" in html_content
            assert "<body>" in html_content
            assert "<title>" in html_content
            assert "</html>" in html_content

            # Check CSS inclusion
            assert "<style>" in html_content
            assert "font-family" in html_content

            # Check JavaScript inclusion
            assert "<script>" in html_content
            assert "function showTab" in html_content

            # Check Chart.js inclusion
            assert "chart.js" in html_content

    def test_overall_rankings_table_generation(self, leaderboard_generator, sample_results_data):
        """Test generation of overall rankings table"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()

        table_html = leaderboard_generator._generate_overall_rankings_table(ranked_models)

        # Check table structure
        assert "<table>" in table_html
        assert "<thead>" in table_html
        assert "<tbody>" in table_html
        assert "</table>" in table_html

        # Check table headers
        assert "Rank" in table_html
        assert "Model" in table_html
        assert "Overall Score" in table_html
        assert "Physician" in table_html
        assert "Patient" in table_html

        # Check model data
        assert "Claude-3" in table_html
        assert "#1" in table_html  # First rank
        assert "rank-1" in table_html  # CSS class for first place

    def test_audience_breakdown_section_generation(self, leaderboard_generator, sample_results_data):
        """Test generation of audience breakdown section"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()
        audience_breakdown = leaderboard_generator.generate_audience_breakdown(ranked_models)

        section_html = leaderboard_generator._generate_audience_breakdown_section(audience_breakdown)

        # Check section structure
        assert "audience-section" in section_html
        assert "Physician Audience Rankings" in section_html
        assert "Patient Audience Rankings" in section_html

        # Check that all models appear in sections
        assert "Claude-3" in section_html
        assert "GPT-4" in section_html
        assert "LLaMA-2" in section_html

    def test_complexity_breakdown_section_generation(self, leaderboard_generator, sample_results_data):
        """Test generation of complexity breakdown section"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()
        complexity_breakdown = leaderboard_generator.generate_complexity_breakdown(ranked_models)

        section_html = leaderboard_generator._generate_complexity_breakdown_section(complexity_breakdown)

        # Check section structure
        assert "complexity-section" in section_html
        assert "Basic Complexity Level Rankings" in section_html
        assert "Advanced Complexity Level Rankings" in section_html

        # Check that all models appear in sections
        assert "Claude-3" in section_html
        assert "GPT-4" in section_html
        assert "LLaMA-2" in section_html

    def test_javascript_generation(self, leaderboard_generator, sample_results_data):
        """Test JavaScript generation for interactive features"""
        leaderboard_generator.results_data = sample_results_data
        ranked_models = leaderboard_generator.rank_models()
        audience_breakdown = leaderboard_generator.generate_audience_breakdown(ranked_models)
        stats = leaderboard_generator.calculate_leaderboard_stats()

        js_content = leaderboard_generator._generate_javascript(ranked_models, audience_breakdown, stats)

        # Check JavaScript structure
        assert "function showTab" in js_content
        assert "function initCharts" in js_content
        assert "Chart(" in js_content

        # Check data inclusion
        assert "Claude-3" in js_content or "Claude" in js_content  # Model names
        assert "physician" in js_content
        assert "patient" in js_content

        # Check chart types
        assert "'bar'" in js_content
        assert "'radar'" in js_content

    def test_css_styles_generation(self, leaderboard_generator):
        """Test CSS styles generation"""
        css_content = leaderboard_generator._get_css_styles()

        # Check CSS structure and important styles
        assert "body {" in css_content
        assert "font-family" in css_content
        assert ".container" in css_content
        assert ".tab-button" in css_content
        assert ".rank-1" in css_content
        assert ".rank-2" in css_content
        assert ".rank-3" in css_content

        # Check responsive design
        assert "@media" in css_content
        assert "max-width" in css_content


class TestLeaderboardIntegration:
    """Integration tests for leaderboard functionality"""

    def test_end_to_end_leaderboard_generation(self, sample_results_data):
        """Test complete end-to-end leaderboard generation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample result files
            results_dir = Path(temp_dir) / "results"
            results_dir.mkdir()

            for i, result in enumerate(sample_results_data):
                result_file = results_dir / f"model_{i}_results.json"
                with open(result_file, "w") as f:
                    json.dump(result, f)

            # Generate leaderboard
            generator = LeaderboardGenerator()
            generator.load_results(results_dir)

            output_path = Path(temp_dir) / "leaderboard.html"
            generator.generate_html(output_path)

            # Verify the complete leaderboard
            assert output_path.exists()
            html_content = output_path.read_text()

            # Check that all major components are present
            assert "MedExplain-Evals Leaderboard" in html_content
            assert "Overall Rankings" in html_content
            assert "By Audience" in html_content
            assert "By Complexity" in html_content
            assert "Analytics" in html_content

            # Check that all models are represented
            for result in sample_results_data:
                assert result["model_name"] in html_content

            # Check that audience types are represented
            for audience in ["physician", "nurse", "patient", "caregiver"]:
                assert audience in html_content

            # Check that complexity levels are represented
            for complexity in ["basic", "intermediate", "advanced"]:
                assert complexity in html_content

    def test_leaderboard_with_mixed_data_quality(self):
        """Test leaderboard generation with mixed quality data"""
        mixed_results = [
            # Complete data
            {
                "model_name": "CompleteModel",
                "total_items": 100,
                "audience_scores": {"physician": [0.8, 0.9], "patient": [0.7, 0.8]},
                "complexity_scores": {"basic": [0.8], "advanced": [0.7]},
                "summary": {"overall_mean": 0.8},
            },
            # Missing complexity scores
            {
                "model_name": "PartialModel",
                "total_items": 50,
                "audience_scores": {"physician": [0.7], "patient": [0.6]},
                "summary": {"overall_mean": 0.65},
            },
            # Minimal data
            {
                "model_name": "MinimalModel",
                "total_items": 25,
                "audience_scores": {"patient": [0.5]},
                "summary": {"overall_mean": 0.5},
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "results"
            results_dir.mkdir()

            for i, result in enumerate(mixed_results):
                result_file = results_dir / f"result_{i}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f)

            # Should handle mixed data gracefully
            generator = LeaderboardGenerator()
            generator.load_results(results_dir)

            output_path = Path(temp_dir) / "leaderboard.html"
            generator.generate_html(output_path)

            assert output_path.exists()
            html_content = output_path.read_text()

            # All models should still appear
            assert "CompleteModel" in html_content
            assert "PartialModel" in html_content
            assert "MinimalModel" in html_content

    def test_leaderboard_performance_with_large_dataset(self):
        """Test leaderboard performance with larger dataset"""
        import time

        # Create larger dataset
        large_results = []
        for i in range(20):  # 20 models
            result = {
                "model_name": f"Model_{i:02d}",
                "total_items": 1000,
                "audience_scores": {
                    "physician": [0.5 + (i * 0.02)] * 100,
                    "nurse": [0.6 + (i * 0.015)] * 100,
                    "patient": [0.4 + (i * 0.025)] * 100,
                    "caregiver": [0.55 + (i * 0.02)] * 100,
                },
                "complexity_scores": {
                    "basic": [0.6 + (i * 0.02)] * 100,
                    "intermediate": [0.5 + (i * 0.02)] * 100,
                    "advanced": [0.4 + (i * 0.02)] * 100,
                },
                "summary": {"overall_mean": 0.5 + (i * 0.02)},
            }
            large_results.append(result)

        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "results"
            results_dir.mkdir()

            for i, result in enumerate(large_results):
                result_file = results_dir / f"result_{i}.json"
                with open(result_file, "w") as f:
                    json.dump(result, f)

            # Time the leaderboard generation
            start_time = time.time()

            generator = LeaderboardGenerator()
            generator.load_results(results_dir)

            output_path = Path(temp_dir) / "leaderboard.html"
            generator.generate_html(output_path)

            end_time = time.time()
            generation_time = end_time - start_time

            # Should complete within reasonable time (30 seconds for large dataset)
            assert generation_time < 30.0, f"Leaderboard generation took {generation_time:.2f}s"

            # Verify output quality
            assert output_path.exists()
            html_content = output_path.read_text()
            assert len(html_content) > 10000  # Should be substantial HTML content
            assert "Model_19" in html_content  # Highest ranked model should appear
