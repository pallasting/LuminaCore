#!/bin/bash
set -e
export CARGO_HOME=$(pwd)/lumina_kernel/.cargo_home
mkdir -p $CARGO_HOME
cd lumina_kernel
../venv/bin/maturin build --release -i ../venv/bin/python
cd ..
./venv/bin/python -m build
