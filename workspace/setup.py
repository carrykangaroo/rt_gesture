from pathlib import Path

from setuptools import find_packages, setup


def _read_version() -> str:
    namespace: dict[str, str] = {}
    version_file = Path(__file__).resolve().parent / "rt_gesture" / "__init__.py"
    exec(version_file.read_text(encoding="utf-8"), namespace)
    return namespace["__version__"]


setup(
    name="rt_gesture",
    version=_read_version(),
    description="Real-time discrete gesture recognition workspace package",
    packages=find_packages(),
    python_requires=">=3.9,<3.13",
)
