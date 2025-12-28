#!/usr/bin/env python3
"""
Release preparation script for MedExplain-Evals.

This script automates the release preparation process including:
- Version validation and updating
- Documentation generation
- Test execution
- Changelog validation
- Package building
- Pre-release checks

Usage:
    python scripts/prepare_release.py --version 1.1.0 --type minor
    python scripts/prepare_release.py --version 1.0.1 --type patch --dry-run
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReleasePreparation:
    """Handles the complete release preparation process"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.version_files = [
            "setup.py",
            "src/__init__.py",
            "docs/conf.py"
        ]
        self.current_version = None
        self.new_version = None
        
    def get_current_version(self) -> str:
        """Extract current version from setup.py"""
        setup_py = self.project_root / "setup.py"
        
        if not setup_py.exists():
            raise FileNotFoundError("setup.py not found")
            
        with open(setup_py, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Look for version in setup.py
        version_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
        if not version_match:
            raise ValueError("Could not find version in setup.py")
            
        return version_match.group(1)
    
    def validate_version_format(self, version: str) -> bool:
        """Validate semantic version format"""
        pattern = r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?$"
        return bool(re.match(pattern, version))
    
    def compare_versions(self, version1: str, version2: str) -> int:
        """Compare two semantic versions. Returns 1 if v1 > v2, -1 if v1 < v2, 0 if equal"""
        def parse_version(v):
            parts = v.split('-')[0].split('.')  # Remove pre-release suffix
            return tuple(int(x) for x in parts)
        
        v1_parts = parse_version(version1)
        v2_parts = parse_version(version2)
        
        if v1_parts > v2_parts:
            return 1
        elif v1_parts < v2_parts:
            return -1
        else:
            return 0
    
    def update_version_files(self, new_version: str, dry_run: bool = False) -> List[str]:
        """Update version in all relevant files"""
        updated_files = []
        
        for file_path in self.version_files:
            full_path = self.project_root / file_path
            
            if not full_path.exists():
                logger.warning(f"Version file not found: {file_path}")
                continue
                
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Update version based on file type
            if file_path == "setup.py":
                new_content = re.sub(
                    r"version\s*=\s*['\"][^'\"]+['\"]",
                    f'version="{new_version}"',
                    content
                )
            elif file_path == "src/__init__.py":
                new_content = re.sub(
                    r"__version__\s*=\s*['\"][^'\"]+['\"]",
                    f'__version__ = "{new_version}"',
                    content
                )
                # Add __version__ if it doesn't exist
                if "__version__" not in content:
                    new_content = f'__version__ = "{new_version}"\n' + content
            elif file_path == "docs/conf.py":
                new_content = re.sub(
                    r"version\s*=\s*['\"][^'\"]+['\"]",
                    f'version = "{new_version}"',
                    content
                )
                new_content = re.sub(
                    r"release\s*=\s*['\"][^'\"]+['\"]",
                    f'release = "{new_version}"',
                    new_content
                )
            else:
                continue
            
            if new_content != content:
                if not dry_run:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                updated_files.append(file_path)
                logger.info(f"{'Would update' if dry_run else 'Updated'} version in {file_path}")
        
        return updated_files
    
    def update_changelog(self, new_version: str, release_type: str, dry_run: bool = False) -> bool:
        """Update CHANGELOG.md with release information"""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        if not changelog_path.exists():
            logger.error("CHANGELOG.md not found")
            return False
        
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if there's an [Unreleased] section with content
        unreleased_match = re.search(r'## \[Unreleased\](.*?)(?=## \[|$)', content, re.DOTALL)
        if not unreleased_match:
            logger.error("No [Unreleased] section found in CHANGELOG.md")
            return False
        
        unreleased_content = unreleased_match.group(1).strip()
        if not unreleased_content or len(unreleased_content.split('\n')) < 3:
            logger.error("No meaningful content in [Unreleased] section")
            return False
        
        # Create new release section
        release_date = datetime.now().strftime('%Y-%m-%d')
        release_section = f"## [{new_version}] - {release_date}"
        
        # Replace [Unreleased] with new release and add empty [Unreleased]
        new_unreleased = "## [Unreleased]\n\n### Added\n- \n\n### Changed\n- \n\n### Fixed\n- \n\n"
        new_content = content.replace(
            "## [Unreleased]",
            new_unreleased + release_section
        )
        
        if not dry_run:
            with open(changelog_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
        
        logger.info(f"{'Would update' if dry_run else 'Updated'} CHANGELOG.md")
        return True
    
    def run_tests(self) -> bool:
        """Run the test suite"""
        logger.info("Running test suite...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ All tests passed")
                return True
            else:
                logger.error("❌ Tests failed:")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Tests timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error running tests: {e}")
            return False
    
    def run_linting(self) -> bool:
        """Run linting checks"""
        logger.info("Running linting checks...")
        
        lint_commands = [
            ["python", "-m", "flake8", "src/", "tests/", "--max-line-length=120"],
            ["python", "-m", "mypy", "src/"],
            ["python", "-m", "bandit", "-r", "src/", "-f", "json"]
        ]
        
        all_passed = True
        
        for cmd in lint_commands:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    logger.info(f"✅ {cmd[1]} passed")
                else:
                    logger.warning(f"⚠️ {cmd[1]} issues found:")
                    logger.warning(result.stdout)
                    # Don't fail release for linting issues, just warn
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not run {cmd[1]}: {e}")
        
        return True  # Don't block release on linting issues
    
    def build_package(self, dry_run: bool = False) -> bool:
        """Build the package"""
        logger.info("Building package...")
        
        if dry_run:
            logger.info("Would build package (dry run)")
            return True
        
        try:
            # Clean previous builds
            dist_dir = self.project_root / "dist"
            if dist_dir.exists():
                import shutil
                shutil.rmtree(dist_dir)
            
            # Build package
            result = subprocess.run(
                [sys.executable, "setup.py", "sdist", "bdist_wheel"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Package built successfully")
                return True
            else:
                logger.error("❌ Package build failed:")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"❌ Error building package: {e}")
            return False
    
    def validate_package(self) -> bool:
        """Validate the built package"""
        logger.info("Validating package...")
        
        try:
            # Check if package can be imported
            result = subprocess.run(
                [sys.executable, "-c", "import src; print('Package import successful')"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ Package validation successful")
                return True
            else:
                logger.error("❌ Package validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error validating package: {e}")
            return False
    
    def generate_release_notes(self, new_version: str) -> str:
        """Generate release notes from changelog"""
        changelog_path = self.project_root / "CHANGELOG.md"
        
        if not changelog_path.exists():
            return f"Release {new_version}\n\nPlease see CHANGELOG.md for details."
        
        with open(changelog_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the section for this version
        version_pattern = rf"## \[{re.escape(new_version)}\].*?\n(.*?)(?=## \[|$)"
        match = re.search(version_pattern, content, re.DOTALL)
        
        if match:
            return f"# MedExplain-Evals {new_version}\n\n{match.group(1).strip()}"
        else:
            return f"# MedExplain-Evals {new_version}\n\nRelease notes not found in CHANGELOG.md"
    
    def perform_release_checks(self) -> Dict[str, bool]:
        """Perform all pre-release checks"""
        checks = {}
        
        logger.info("🔍 Performing pre-release checks...")
        
        # Check git status
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            checks["clean_git"] = len(result.stdout.strip()) == 0
            if not checks["clean_git"]:
                logger.warning("⚠️ Git working directory is not clean")
        except:
            checks["clean_git"] = False
        
        # Check if we're on main/master branch
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            current_branch = result.stdout.strip()
            checks["main_branch"] = current_branch in ["main", "master"]
            if not checks["main_branch"]:
                logger.warning(f"⚠️ Not on main/master branch (current: {current_branch})")
        except:
            checks["main_branch"] = False
        
        # Run tests
        checks["tests_pass"] = self.run_tests()
        
        # Run linting
        checks["linting_pass"] = self.run_linting()
        
        # Validate package
        checks["package_valid"] = self.validate_package()
        
        return checks
    
    def prepare_release(
        self, 
        new_version: str, 
        release_type: str, 
        dry_run: bool = False,
        skip_tests: bool = False
    ) -> bool:
        """Main release preparation workflow"""
        logger.info(f"🚀 Preparing release {new_version} ({release_type})")
        
        # Get current version
        try:
            self.current_version = self.get_current_version()
            logger.info(f"Current version: {self.current_version}")
        except Exception as e:
            logger.error(f"❌ Failed to get current version: {e}")
            return False
        
        # Validate new version
        if not self.validate_version_format(new_version):
            logger.error(f"❌ Invalid version format: {new_version}")
            return False
        
        # Check version increment
        if self.compare_versions(new_version, self.current_version) <= 0:
            logger.error(f"❌ New version {new_version} must be greater than current {self.current_version}")
            return False
        
        self.new_version = new_version
        
        # Perform pre-release checks
        if not skip_tests:
            checks = self.perform_release_checks()
            failed_checks = [check for check, passed in checks.items() if not passed]
            
            if failed_checks:
                logger.warning(f"⚠️ Some checks failed: {', '.join(failed_checks)}")
                if not dry_run:
                    response = input("Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        logger.info("Release preparation cancelled")
                        return False
        
        # Update version files
        updated_files = self.update_version_files(new_version, dry_run)
        if not updated_files:
            logger.error("❌ No version files were updated")
            return False
        
        # Update changelog
        if not self.update_changelog(new_version, release_type, dry_run):
            logger.error("❌ Failed to update changelog")
            return False
        
        # Build package
        if not self.build_package(dry_run):
            logger.error("❌ Failed to build package")
            return False
        
        # Generate release notes
        release_notes = self.generate_release_notes(new_version)
        
        if not dry_run:
            notes_file = self.project_root / f"release_notes_{new_version}.md"
            with open(notes_file, 'w', encoding='utf-8') as f:
                f.write(release_notes)
            logger.info(f"📝 Release notes written to {notes_file}")
        
        logger.info("✅ Release preparation completed successfully!")
        
        if not dry_run:
            logger.info("\n📋 Next steps:")
            logger.info("1. Review the changes made")
            logger.info("2. Commit the version updates:")
            logger.info(f"   git add {' '.join(updated_files)} CHANGELOG.md")
            logger.info(f"   git commit -m 'Prepare release {new_version}'")
            logger.info(f"3. Create and push a tag:")
            logger.info(f"   git tag v{new_version}")
            logger.info(f"   git push origin v{new_version}")
            logger.info("4. Push the changes:")
            logger.info("   git push origin main")
            logger.info("5. Create a GitHub release using the generated release notes")
            logger.info("6. Upload the built package to PyPI if desired")
        
        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Prepare MedExplain-Evals for release",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Prepare a minor release
    python scripts/prepare_release.py --version 1.1.0 --type minor
    
    # Prepare a patch release with dry run
    python scripts/prepare_release.py --version 1.0.1 --type patch --dry-run
    
    # Skip tests (for faster preparation)
    python scripts/prepare_release.py --version 1.0.1 --type patch --skip-tests
        """
    )
    
    parser.add_argument(
        "--version",
        required=True,
        help="New version number (e.g., 1.1.0)"
    )
    
    parser.add_argument(
        "--type",
        choices=["major", "minor", "patch"],
        required=True,
        help="Type of release"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests (faster but less safe)"
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
    
    # Initialize release preparation
    release_prep = ReleasePreparation(project_root)
    
    # Perform release preparation
    success = release_prep.prepare_release(
        new_version=args.version,
        release_type=args.type,
        dry_run=args.dry_run,
        skip_tests=args.skip_tests
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()