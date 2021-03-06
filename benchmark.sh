#!/usr/bin/env bash

set -euo pipefail

TIMEOUT=40m

usage() {
    echo "$(basename "$0") <RESULT_FILE> <DIRECTORY>" 1>&2
}

build() {
    export RUSTFLAGS="-C target-cpu=native"
    cargo build --release
}

main() {
    build
    local result_file="${1:?expected result file path}"
    local bench_dir="${2:?expected benchmark data directory}"
    local command
    cargo run --release --quiet -- header >> "$result_file"
    for graph in "$bench_dir/"*.gr; do
        echo '================================================================================'
        for step in stats exact approx1 approx2s approx2p; do
            echo '--------------------------------------------------------------------------------'
            command="timeout $TIMEOUT cargo run --release --quiet -- ${step} ${graph}"
            echo "executing '${command}'..."
            ($command || echo "-1, -1, ") >> "$result_file"
        done
    done
}

main "$@"
