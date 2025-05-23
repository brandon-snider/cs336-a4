#!/usr/bin/env bash
set -euo pipefail

# uv run pytest -v ./tests --junitxml=test_results.xml || true
echo "Done running tests"

# Set the name of the output tar.gz file
output_file="cs336-spring2025-assignment-4-submission.zip"
rm "$output_file" || true

# Compress all files in the current directory into a single zip file
zip -r "$output_file" . \
    -x '*egg-info*' \
    -x '*mypy_cache*' \
    -x '*pytest_cache*' \
    -x '*build*' \
    -x '*ipynb_checkpoints*' \
    -x '*__pycache__*' \
    -x '*.pkl' \
    -x '*.pickle' \
    -x '*.txt' \
    -x '*.log' \
    -x '*.json' \
    -x '*.out' \
    -x '*.err' \
    -x '.git*' \
    -x '.venv/*' \
    -x '*.bin' \
    -x '*.pt' \
    -x '*.pth' \
    -x 'cs336-basics/.venv/*' \
    -x 'cs336-basics/wandb/*' \
    -x 'cs336-basics/outputs/*' \
    -x 'cs336-basics/output/*' \
    -x '*.ipynb' \
    -x '*.pdf' \
    -x 'a*.typ' \
    -x '_unused*' \
    -x 'out/*'

echo "All files have been compressed into $output_file"