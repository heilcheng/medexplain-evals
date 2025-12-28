"""
Setup script for MedExplain-Evals: A Resource-Efficient Benchmark for Evaluating 
Audience-Adaptive Explanation Quality in Medical Large Language Models.
"""

import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure Python version compatibility
if sys.version_info < (3, 8):
    raise RuntimeError("MedExplain-Evals requires Python 3.8 or higher")

# Package metadata
PACKAGE_NAME = "medexplain-evals"
VERSION = "1.0.0"
AUTHOR = "MedExplain-Evals Team"
AUTHOR_EMAIL = "contact@medexplain-evals.org"
URL = "https://github.com/heilcheng/MedExplain-Evals"
DESCRIPTION = "A Resource-Efficient Benchmark for Evaluating Audience-Adaptive Explanation Quality in Medical Large Language Models"

# Read long description from README
def read_file(filename: str) -> str:
    """Read content from a file."""
    file_path = Path(__file__).parent / filename
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return ""

# Parse requirements from requirements.txt
def parse_requirements(filename: str) -> list:
    """Parse requirements from requirements file."""
    file_path = Path(__file__).parent / filename
    if not file_path.exists():
        # Return core dependencies if requirements.txt doesn't exist
        return [
            "numpy>=1.21.0",
            "pandas>=1.5.0", 
            "PyYAML>=6.0.0",
            "requests>=2.28.0",
            "textstat>=0.7.0",
        ]
    
    requirements = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith("#"):
            # Handle -e git+https:// lines
            if line.startswith("-e "):
                line = line[3:]
            requirements.append(line)
    return requirements

# Package classifiers
CLASSIFIERS = [
    # Development Status
    "Development Status :: 4 - Beta",
    
    # Intended Audience
    "Intended Audience :: Science/Research",
    "Intended Audience :: Healthcare Industry",
    "Intended Audience :: Developers",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Operating System
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    
    # Programming Language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3 :: Only",
    
    # Topic
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Text Processing :: Linguistic",
]

# Extra dependencies organized by use case
EXTRAS_REQUIRE = {
    # Development dependencies  
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "black>=23.0.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "isort>=5.12.0",
        "bandit>=1.7.0",
        "pre-commit>=3.0.0",
    ],
    
    # Testing dependencies
    "test": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "responses>=0.23.0",
    ],
    
    # Documentation dependencies
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=0.18.0",
        "sphinx-autodoc-typehints>=1.19.0",
        "sphinx-autobuild>=2021.3.14",
    ],
    
    # Machine learning dependencies
    "ml": [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.10.0",
        "scikit-learn>=1.3.0",
        "sentence-transformers>=2.2.0",
        "bert-score>=0.3.13",
        "spacy>=3.6.0",
        "scispacy>=0.5.0",
        "nltk>=3.8.0",
        "textstat>=0.7.0",
    ],
    
    # LLM API dependencies
    "llm": [
        "openai>=1.0.0",
        "anthropic>=0.3.0",
    ],
    
    # Apple Silicon optimizations
    "apple": [
        "mlx>=0.5.0",
        "mlx-lm>=0.5.0",
    ],
    
    # Analysis and visualization dependencies
    "analysis": [
        "jupyter>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.13.0",
        "pandas>=1.5.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
}

# Convenience dependency groups
EXTRAS_REQUIRE.update({
    # Full installation matching main branch structure
    "full": (
        EXTRAS_REQUIRE["ml"] + 
        EXTRAS_REQUIRE["llm"] + 
        EXTRAS_REQUIRE["apple"] + 
        EXTRAS_REQUIRE["analysis"]
    ),
    
    # All dependencies
    "all": [
        dep for deps in EXTRAS_REQUIRE.values() 
        for dep in deps
    ],
    
    # Complete development environment
    "dev-full": (
        EXTRAS_REQUIRE["dev"] + 
        EXTRAS_REQUIRE["test"] + 
        EXTRAS_REQUIRE["docs"]
    ),
})

# Entry points for command-line interface
ENTRY_POINTS = {
    "console_scripts": [
        "medexplain-evals=src.benchmark:main",
        "meq-evaluate=src.evaluator:main", 
    ],
}

# Package data to include
PACKAGE_DATA = {
    "medexplain-evals": [
        "data/*.json",
        "data/*.yaml", 
        "docs/*.md",
        "configs/*.yaml",
        "templates/*.txt",
    ],
}

# Run setup
if __name__ == "__main__":
    setup(
        # Basic package information
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=read_file("README.md"),
        long_description_content_type="text/markdown",
        url=URL,
        
        # Package discovery and requirements
        packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
        python_requires=">=3.8",
        install_requires=parse_requirements("requirements.txt"),
        extras_require=EXTRAS_REQUIRE,
        
        # Package metadata
        classifiers=CLASSIFIERS,
        keywords="medical nlp benchmark evaluation llm ai healthcare",
        project_urls={
            "Bug Reports": f"{URL}/issues",
            "Source": URL,
            "Documentation": f"{URL}#readme",
            "Changelog": f"{URL}/blob/main/CHANGELOG.md",
        },
        
        # Entry points and package data
        entry_points=ENTRY_POINTS,
        include_package_data=True,
        package_data=PACKAGE_DATA,
        zip_safe=False,
        
        # Additional metadata
        platforms=["any"],
        license="MIT",
    )
