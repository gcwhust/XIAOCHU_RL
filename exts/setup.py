"""Installation script for the 'bipedal_locomotion' python package."""
# ...existing code...
import itertools
import os
# 优先使用 Python 3.11+ 的内置 tomllib，回退到第三方 toml（如果可用）
try:
    import tomllib as _toml_lib  # Python 3.11+
    def _load_toml(path):
        with open(path, "rb") as f:
            return _toml_lib.load(f)
except Exception:
    try:
        import toml as _toml  # type: ignore
        def _load_toml(path):
            with open(path, "r", encoding="utf-8") as f:
                return _toml.load(f)
    except Exception:
        def _load_toml(path):
            raise RuntimeError(
                    "Missing 'toml' package and no tomllib available. "
                    "Install the 'toml' package or use Python >= 3.11."
                )

from setuptools import setup
 
 # Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
 # Read the extension.toml file
#EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))
EXTENSION_TOML_DATA = _load_toml(os.path.join(EXTENSION_PATH, "config", "extension.toml"))
# ...existing code...

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "numpy",
    # "torch==2.4.0",
    # "torchvision>=0.14.1",  # ensure compatibility with torch 1.13.1
    # 5.26.0 introduced a breaking change, so we restricted it for now.
    # See issue https://github.com/tensorflow/tensorboard/issues/6808 for details.
    "protobuf>=3.20.2, < 5.0.0",
    # data collection
    "h5py",
    # basic logger
    "tensorboard",
    # video recording
    "moviepy",
]

PYTORCH_INDEX_URL = ["https://download.pytorch.org/whl/cu118"]

current_dir = os.path.dirname(os.path.abspath(__file__))
rsl_rl_path = os.path.join(current_dir, "..", "rsl_rl")

# Extra dependencies for RL agents
EXTRAS_REQUIRE = {
    "rsl-rl": [rsl_rl_path],
}

# Add the names with hyphens as aliases for convenience
EXTRAS_REQUIRE["rsl_rl"] = EXTRAS_REQUIRE["rsl-rl"]

# Cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))

# Remove duplicates in the all list to avoid double installations
EXTRAS_REQUIRE["all"] = list(set(EXTRAS_REQUIRE["all"]))

# Installation operation
setup(
    name="bipedal_locomotion",
    packages=["bipedal_locomotion"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Isaac Sim :: 4.5.0",
    ],
    zip_safe=False,
)
