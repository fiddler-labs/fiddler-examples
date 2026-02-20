"""Setup script for fiddler_utils package.

Admin automation library that provides high-level abstractions over the
Fiddler Python client for common administrative tasks.

Designed for field engineers and customers to reduce code duplication
across utility scripts and notebooks.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "fiddler_utils" / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="fiddler-utils",
    version="0.1.2",
    description="Admin automation library for Fiddler administrative tasks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fiddler AI",
    author_email="support@fiddler.ai",
    url="https://github.com/fiddler-labs/fiddler-examples",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "fiddler-client>=3.10.0",
        "tqdm>=4.65.0",  # Progress bars for bulk operations
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "mypy>=1.0.0",
            "black>=23.0.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    package_data={
        "fiddler_utils": ["py.typed"],
    },
    include_package_data=True,
)
