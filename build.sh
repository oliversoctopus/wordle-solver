#!/bin/sh
# Builds the Rust core and the native binaries.
#
#     ./build.sh              release, into target/release
#     ./build.sh --debug      unoptimized, with debug assertions
#     ./build.sh --test       release, then the unit tests
#
# Nothing here that `cargo build --release` does not do.  It exists because
# rustup was installed without touching PATH, so cargo is not on it by default.

set -e
cd "$(dirname "$0")"
export PATH="$HOME/.cargo/bin:$PATH"

command -v cargo >/dev/null 2>&1 || {
    echo "cargo is not on PATH and is not in ~/.cargo/bin" >&2
    echo "install it from https://rustup.rs, or add it to PATH yourself" >&2
    exit 1
}

case "$1" in
    --debug) cargo build ;;
    --test)  cargo build --release && cargo test --release ;;
    *)       cargo build --release ;;
esac
