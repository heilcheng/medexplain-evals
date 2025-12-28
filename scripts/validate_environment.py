#!/usr/bin/env python3
"""Environment validation for MEQ-Bench.

This script checks that all required dependencies, configurations,
and API credentials are properly set up.

Usage:
    python scripts/validate_environment.py
    python scripts/validate_environment.py --verbose
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class Check:
    """Represents a single validation check."""
    
    def __init__(
        self,
        name: str,
        required: bool = True,
        category: str = "general",
    ):
        self.name = name
        self.required = required
        self.category = category
        self.passed = False
        self.message = ""
        self.details: List[str] = []
    
    def pass_(self, message: str = "OK"):
        self.passed = True
        self.message = message
    
    def fail(self, message: str):
        self.passed = False
        self.message = message
    
    def add_detail(self, detail: str):
        self.details.append(detail)


class EnvironmentValidator:
    """Validate MEQ-Bench environment setup."""
    
    REQUIRED_PACKAGES = [
        ("numpy", None),
        ("pandas", None),
        ("yaml", "pyyaml"),
        ("requests", None),
    ]
    
    OPTIONAL_PACKAGES = [
        ("matplotlib", None),
        ("plotly", None),
        ("scipy", None),
        ("torch", None),
        ("transformers", None),
        ("openai", None),
        ("anthropic", None),
    ]
    
    API_KEYS = [
        ("OPENAI_API_KEY", "OpenAI API"),
        ("ANTHROPIC_API_KEY", "Anthropic API"),
        ("GOOGLE_API_KEY", "Google AI API"),
        ("DEEPSEEK_API_KEY", "DeepSeek API"),
    ]
    
    REQUIRED_DIRS = [
        "src",
        "scripts",
        "data",
        "analysis",
    ]
    
    REQUIRED_FILES = [
        "config.yaml",
        "requirements.txt",
        "src/__init__.py",
    ]
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks: List[Check] = []
        self.project_root = Path(__file__).parent.parent
    
    def add_check(self, check: Check):
        self.checks.append(check)
    
    def check_python_version(self) -> Check:
        """Check Python version."""
        check = Check("Python Version", required=True, category="python")
        
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version.major >= 3 and version.minor >= 10:
            check.pass_(f"Python {version_str}")
        else:
            check.fail(f"Python {version_str} (requires 3.10+)")
        
        self.add_check(check)
        return check
    
    def check_package(self, package_name: str, pip_name: Optional[str] = None) -> Check:
        """Check if a package is installed."""
        check = Check(f"Package: {package_name}", category="packages")
        pip_name = pip_name or package_name
        
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            check.pass_(f"v{version}")
        except ImportError:
            check.fail(f"Not installed (pip install {pip_name})")
        
        self.add_check(check)
        return check
    
    def check_required_packages(self):
        """Check all required packages."""
        for package, pip_name in self.REQUIRED_PACKAGES:
            check = self.check_package(package, pip_name)
            check.required = True
    
    def check_optional_packages(self):
        """Check optional packages."""
        for package, pip_name in self.OPTIONAL_PACKAGES:
            check = self.check_package(package, pip_name)
            check.required = False
    
    def check_api_keys(self) -> List[Check]:
        """Check API key environment variables."""
        checks = []
        
        for env_var, service_name in self.API_KEYS:
            check = Check(f"API Key: {service_name}", required=False, category="api")
            
            value = os.environ.get(env_var)
            if value:
                # Mask the key
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                check.pass_(f"Set ({masked})")
            else:
                check.fail(f"Not set (export {env_var}=...)")
            
            self.add_check(check)
            checks.append(check)
        
        return checks
    
    def check_directory_structure(self):
        """Check required directories exist."""
        for dir_name in self.REQUIRED_DIRS:
            check = Check(f"Directory: {dir_name}", required=True, category="structure")
            
            dir_path = self.project_root / dir_name
            if dir_path.exists() and dir_path.is_dir():
                check.pass_("Exists")
            else:
                check.fail(f"Missing (mkdir {dir_name})")
            
            self.add_check(check)
    
    def check_required_files(self):
        """Check required files exist."""
        for file_name in self.REQUIRED_FILES:
            check = Check(f"File: {file_name}", required=True, category="structure")
            
            file_path = self.project_root / file_name
            if file_path.exists():
                check.pass_("Exists")
            else:
                check.fail("Missing")
            
            self.add_check(check)
    
    def check_config_file(self) -> Check:
        """Check config.yaml is valid."""
        check = Check("Config: config.yaml", required=True, category="config")
        
        config_path = self.project_root / "config.yaml"
        
        if not config_path.exists():
            check.fail("File not found")
            self.add_check(check)
            return check
        
        try:
            import yaml
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            required_sections = ["app_settings", "target_audiences", "evaluation_metrics"]
            missing = [s for s in required_sections if s not in config]
            
            if missing:
                check.fail(f"Missing sections: {missing}")
            else:
                check.pass_("Valid YAML with required sections")
                check.add_detail(f"Sections: {list(config.keys())}")
        except Exception as e:
            check.fail(f"Parse error: {e}")
        
        self.add_check(check)
        return check
    
    def check_data_availability(self) -> Check:
        """Check for benchmark data."""
        check = Check("Benchmark Data", required=False, category="data")
        
        data_paths = [
            "data/benchmark_v2/full_dataset.json",
            "data/benchmark_v2/test.json",
            "data/sample_dataset.json",
        ]
        
        found = []
        for path in data_paths:
            full_path = self.project_root / path
            if full_path.exists():
                found.append(path)
        
        if found:
            check.pass_(f"Found {len(found)} dataset(s)")
            for path in found:
                check.add_detail(path)
        else:
            check.fail("No datasets found (run dataset curation)")
        
        self.add_check(check)
        return check
    
    def check_meqbench_imports(self) -> Check:
        """Check MEQ-Bench modules can be imported."""
        check = Check("MEQ-Bench Modules", required=True, category="meqbench")
        
        modules_to_check = [
            "src.data_loaders",
            "src.evaluator",
            "src.config",
        ]
        
        failed = []
        for module in modules_to_check:
            try:
                importlib.import_module(module)
            except ImportError as e:
                failed.append(f"{module}: {e}")
        
        if failed:
            check.fail(f"{len(failed)} module(s) failed to import")
            for f in failed:
                check.add_detail(f)
        else:
            check.pass_(f"All {len(modules_to_check)} modules importable")
        
        self.add_check(check)
        return check
    
    def run_all_checks(self):
        """Run all validation checks."""
        logger.info("="*60)
        logger.info("MEQ-BENCH 2.0 ENVIRONMENT VALIDATION")
        logger.info("="*60)
        
        # Python version
        self.check_python_version()
        
        # Required packages
        self.check_required_packages()
        
        # Optional packages
        self.check_optional_packages()
        
        # API keys
        self.check_api_keys()
        
        # Directory structure
        self.check_directory_structure()
        
        # Required files
        self.check_required_files()
        
        # Config file
        self.check_config_file()
        
        # Data availability
        self.check_data_availability()
        
        # MEQ-Bench imports
        self.check_meqbench_imports()
    
    def print_results(self):
        """Print validation results."""
        # Group by category
        categories = {}
        for check in self.checks:
            if check.category not in categories:
                categories[check.category] = []
            categories[check.category].append(check)
        
        total_passed = 0
        total_failed = 0
        required_failed = 0
        
        for category, checks in categories.items():
            logger.info(f"\n📋 {category.upper()}")
            logger.info("-"*50)
            
            for check in checks:
                status = "✅" if check.passed else ("❌" if check.required else "⚠️")
                req = "" if check.required else " (optional)"
                
                logger.info(f"  {status} {check.name}: {check.message}{req}")
                
                if self.verbose and check.details:
                    for detail in check.details:
                        logger.info(f"      └─ {detail}")
                
                if check.passed:
                    total_passed += 1
                else:
                    total_failed += 1
                    if check.required:
                        required_failed += 1
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"  Total checks: {len(self.checks)}")
        logger.info(f"  Passed: {total_passed}")
        logger.info(f"  Failed: {total_failed}")
        
        if required_failed > 0:
            logger.error(f"\n❌ VALIDATION FAILED: {required_failed} required check(s) failed")
            return False
        else:
            logger.info("\n✅ VALIDATION PASSED: Environment is ready!")
            return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate MEQ-Bench environment"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed check results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Only run required checks"
    )
    
    args = parser.parse_args()
    
    validator = EnvironmentValidator(verbose=args.verbose)
    validator.run_all_checks()
    success = validator.print_results()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())

