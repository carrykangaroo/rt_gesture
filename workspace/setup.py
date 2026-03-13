import re
from pathlib import Path

from setuptools import find_packages, setup


def _read_version() -> str:
    version_file = Path(__file__).resolve().parent / "rt_gesture" / "__init__.py"
    content = version_file.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*[\"\']([^\"\']+)[\"\']', content, re.MULTILINE)
    if match is None:
        raise RuntimeError("Unable to determine package version.")
    return match.group(1)


setup(
    name="rt_gesture",
    version=_read_version(),
    description="Real-time discrete gesture recognition workspace package",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
)
