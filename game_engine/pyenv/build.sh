#!/usr/bin/env bash
# Build the halite_pyenv pybind11 module (CPU step env) for RL self-play.
# Run inside WSL Ubuntu. CPU-only: no CUDA toolkit required.
set -euo pipefail

PY="${HALITE_PYENV_PYTHON:-/home/hp/halite-train-venv/bin/python}"
CMAKE="${HALITE_PYENV_CMAKE:-/home/hp/halite-train-venv/bin/cmake}"
ENGINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${ENGINE_DIR}/build_pyenv"

PYBIND_DIR="$("$PY" -m pybind11 --cmakedir)"

echo "engine dir : $ENGINE_DIR"
echo "python     : $PY"
echo "pybind11   : $PYBIND_DIR"

"$CMAKE" -S "$ENGINE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DHALITE_BUILD_PYENV=ON \
    -Dpybind11_DIR="$PYBIND_DIR" \
    -DPython_EXECUTABLE="$PY" \
    -DPYTHON_EXECUTABLE="$PY"

"$CMAKE" --build "$BUILD_DIR" --target halite_pyenv -j"$(nproc)"

echo "built module:"
find "$BUILD_DIR" -name "halite_pyenv*.so"
