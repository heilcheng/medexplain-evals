#!/usr/bin/env python3
"""
Release validation script for MedExplain-Evals.

This script validates that a release is ready by checking:
- Package can be imported
- Basic functionality works
- All expected modules are present
- Documentation is accessible

Usage:
    python scripts/validate_release.py
    python scripts/validate_release.py --package-path dist/medexplain-1.1.0-py3-none-any.whl
"""

import argparse
import importlib
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReleaseValidator:
    """Validates MedExplain-Evals releases"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.validation_results = {}
        
    def validate_imports(self) -> bool:
        """Test that all main modules can be imported"""
        logger.info("🔍 Validating module imports...")
        
        required_modules = [
            'src',
            'src.benchmark',
            'src.evaluator', 
            'src.data_loaders',
            'src.leaderboard',
            'src.strategies',
            'src.config'
        ]
        
        failed_imports = []
        
        for module in required_modules:
            try:
                importlib.import_module(module)
                logger.info(f"✅ {module}")
            except ImportError as e:
                logger.error(f"❌ {module}: {e}")
                failed_imports.append(module)
        
        success = len(failed_imports) == 0
        self.validation_results['imports'] = {
            'success': success,
            'failed_modules': failed_imports
        }
        
        return success
    
    def validate_basic_functionality(self) -> bool:
        """Test basic functionality works"""
        logger.info("🔍 Validating basic functionality...")
        
        try:
            # Test benchmark creation
            from src.benchmark import MedExplain
            bench = MedExplain()
            logger.info("✅ MedExplain initialization")
            
            # Test sample dataset creation
            sample_items = bench.create_sample_dataset()
            assert len(sample_items) > 0
            logger.info("✅ Sample dataset creation")
            
            # Test evaluator
            from src.evaluator import MedExplainEvaluator
            evaluator = MedExplainEvaluator()
            logger.info("✅ MedExplainEvaluator initialization")
            
            # Test data loader
            from src.data_loaders import load_medquad
            logger.info("✅ Data loader imports")
            
            # Test leaderboard
            from src.leaderboard import LeaderboardGenerator
            leaderboard = LeaderboardGenerator()
            logger.info("✅ LeaderboardGenerator initialization")
            
            success = True
            
        except Exception as e:
            logger.error(f"❌ Basic functionality test failed: {e}")
            success = False
        
        self.validation_results['functionality'] = {'success': success}
        return success
    
    def validate_configuration(self) -> bool:
        """Validate configuration system works"""
        logger.info("🔍 Validating configuration...")
        
        try:
            from src.config import config
            
            # Test config loading
            config_data = config.get_config()
            assert isinstance(config_data, dict)
            logger.info("✅ Configuration loading")
            
            # Test required config sections
            required_sections = ['audiences', 'complexity_levels', 'evaluation']
            for section in required_sections:
                assert section in config_data
            logger.info("✅ Required configuration sections present")
            
            success = True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            success = False
        
        self.validation_results['configuration'] = {'success': success}
        return success
    
    def validate_version_consistency(self) -> bool:
        """Check version consistency across files"""
        logger.info("🔍 Validating version consistency...")
        
        versions = {}
        
        # Check setup.py
        setup_py = self.project_root / "setup.py"
        if setup_py.exists():
            with open(setup_py, 'r') as f:
                content = f.read()
                import re
                match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    versions['setup.py'] = match.group(1)
        
        # Check src/__init__.py
        init_py = self.project_root / "src" / "__init__.py"
        if init_py.exists():
            try:
                from src import __version__
                versions['src/__init__.py'] = __version__
            except ImportError:
                pass
        
        # Check docs/conf.py
        conf_py = self.project_root / "docs" / "conf.py"
        if conf_py.exists():
            with open(conf_py, 'r') as f:
                content = f.read()
                import re
                match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
                if match:
                    versions['docs/conf.py'] = match.group(1)
        
        # Check consistency
        unique_versions = set(versions.values())
        success = len(unique_versions) <= 1
        
        if success:
            logger.info(f"✅ Version consistency: {list(unique_versions)[0] if unique_versions else 'No version found'}")
        else:
            logger.error(f"❌ Version inconsistency: {versions}")
        
        self.validation_results['version_consistency'] = {
            'success': success,
            'versions': versions
        }
        
        return success
    
    def validate_examples(self) -> bool:
        """Test that examples can run without errors"""
        logger.info("🔍 Validating examples...")
        
        examples_dir = self.project_root / "examples"
        if not examples_dir.exists():
            logger.warning("⚠️ Examples directory not found")
            return True
        
        # Test basic usage example (syntax check only)
        basic_usage = examples_dir / "basic_usage.py"
        if basic_usage.exists():
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(basic_usage)],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    logger.info("✅ basic_usage.py syntax check")
                    success = True
                else:
                    logger.error(f"❌ basic_usage.py syntax error: {result.stderr}")
                    success = False
                    
            except Exception as e:
                logger.error(f"❌ Error checking basic_usage.py: {e}")
                success = False
        else:
            logger.warning("⚠️ basic_usage.py not found")
            success = True
        
        self.validation_results['examples'] = {'success': success}
        return success
    
    def validate_package_installability(self, package_path: str = None) -> bool:
        """Test package can be installed and imported in fresh environment"""
        logger.info("🔍 Validating package installability...")
        
        if not package_path:
            # Look for built packages
            dist_dir = self.project_root / "dist"
            if dist_dir.exists():
                wheels = list(dist_dir.glob("*.whl"))
                if wheels:
                    package_path = str(wheels[0])
        
        if not package_path:
            logger.warning("⚠️ No package path provided and no wheel found")
            return True
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create virtual environment
                venv_dir = Path(temp_dir) / "test_venv"
                result = subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_dir)],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to create virtual environment: {result.stderr}")
                    return False
                
                # Determine pip path
                if sys.platform == "win32":
                    pip_path = venv_dir / "Scripts" / "pip.exe"
                    python_path = venv_dir / "Scripts" / "python.exe"
                else:
                    pip_path = venv_dir / "bin" / "pip"
                    python_path = venv_dir / "bin" / "python"
                
                # Install package
                result = subprocess.run(
                    [str(pip_path), "install", package_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode != 0:
                    logger.error(f"❌ Failed to install package: {result.stderr}")
                    return False
                
                # Test import
                result = subprocess.run(
                    [str(python_path), "-c", "import src; print('Import successful')"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    logger.info("✅ Package installation and import")
                    success = True
                else:
                    logger.error(f"❌ Package import failed: {result.stderr}")
                    success = False
                    
        except Exception as e:
            logger.error(f"❌ Package installability test failed: {e}")
            success = False
        
        self.validation_results['installability'] = {'success': success}
        return success
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_checks = len(self.validation_results)
        passed_checks = sum(1 for result in self.validation_results.values() if result['success'])
        
        report = {
            'overall_success': passed_checks == total_checks,
            'passed_checks': passed_checks,
            'total_checks': total_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'detailed_results': self.validation_results,
            'summary': f"{passed_checks}/{total_checks} checks passed"
        }
        
        return report
    
    def run_all_validations(self, package_path: str = None) -> bool:
        """Run all validation checks"""
        logger.info("🚀 Starting MedExplain-Evals release validation...")
        
        validations = [
            self.validate_imports,
            self.validate_basic_functionality,
            self.validate_configuration,
            self.validate_version_consistency,
            self.validate_examples,
            lambda: self.validate_package_installability(package_path)
        ]
        
        all_passed = True
        
        for validation in validations:
            try:
                result = validation()
                if not result:
                    all_passed = False
            except Exception as e:
                logger.error(f"❌ Validation error: {e}")
                all_passed = False
        
        # Generate report
        report = self.generate_validation_report()
        
        logger.info("\n" + "="*60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Overall Status: {'✅ PASS' if report['overall_success'] else '❌ FAIL'}")
        logger.info(f"Checks Passed: {report['summary']}")
        logger.info(f"Success Rate: {report['success_rate']:.1%}")
        
        if not report['overall_success']:
            logger.info("\n❌ Failed Checks:")
            for check, result in report['detailed_results'].items():
                if not result['success']:
                    logger.info(f"  - {check}")
        
        logger.info("="*60)
        
        return all_passed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Validate MedExplain-Evals release",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--package-path",
        help="Path to built package (.whl file) to test installation"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Run validation
    validator = ReleaseValidator(project_root)
    success = validator.run_all_validations(args.package_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()