"""The wall-clock figures README quotes for the Rust port.

A sweep of every answer in Rust against the Python, what keeping games in
flight is worth, what the answer-major dispatch is worth over a sweep, and
what a whole `train` costs each way.  NEEDS AN IDLE MACHINE.  Each Rust sweep
runs twice and the faster is kept; the Python sweep runs in a fresh process.
`--train` adds the trainer probes, which overwrite the estimator cache and
restore it.  Backs README *Fitting in Rust* and *Keeping the port in step*.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import common
import rust
import rust_train

ROOT = common.ROOT
PY = sys.executable

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train", action="store_true",
                    help="also time the trainer; overwrites and restores the cache")
parser.add_argument("--reps", type=int, default=2,
                    help="runs per Rust probe, the fastest kept")
args = parser.parse_args()


def swept(flags):
    """Seconds the native sweep reports, best of the reps."""
    best = None
    for _ in range(args.reps):
        done = subprocess.run([rust.binary("every_answer"), ROOT] + flags,
                              capture_output=True, text=True, check=True)
        seconds = float(re.search(r"swept in\s+([\d.]+) s", done.stdout).group(1))
        best = seconds if best is None else min(best, seconds)
    return best


def python_sweep(mode):
    """A cold whole-answer sweep in Python, in its own process."""
    code = ("import os, sys, time; os.environ['WORDLE_NO_RUST'] = '1'; "
            "sys.path[:0] = [%r, %r]; import common; "
            "from solver import WordleSolver; "
            "s = WordleSolver(mode=%r); s.best_guess(s.all_candidates()); "
            "t = time.perf_counter(); common.every_answer(s); "
            "print(time.perf_counter() - t)"
            % (ROOT, common.HERE, mode))
    done = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                          check=True, cwd=ROOT)
    return float(done.stdout.strip().splitlines()[-1])


common.header("a sweep of every answer, the shipped prior")
print(f"  {'':12s} {'chunked':>9s} {'serial':>9s} {'fused':>9s} {'python':>9s}")
for mode in ("expected", "entropy"):
    chunked = swept(["--mode", mode])
    serial = swept(["--mode", mode, "--serial"])
    fused = swept(["--mode", mode, "--fused"])
    python = python_sweep(mode)
    print(f"  {mode:12s} {chunked:8.2f}s {serial:8.2f}s {fused:8.2f}s {python:8.2f}s")
    common.record(f"sweep.{mode}.rust", chunked)
    common.record(f"sweep.{mode}.python", python)
    common.record(f"sweep.{mode}.python_over_rust", python / chunked)
    common.record(f"sweep.{mode}.chunking", serial / chunked)
    common.record(f"sweep.{mode}.fused", chunked / fused)
print("  chunking is serial / chunked; fused is chunked / fused")

common.header("the fused band, one scan each")
done = subprocess.run([rust.binary("bench_position"), ROOT], capture_output=True,
                      text=True, check=True)
for m, ratio in re.findall(r"(\d+) candidates\s+shipped.*?fused.*?([\d.]+)x",
                           done.stdout):
    print(f"  {int(m):>5} candidates   {float(ratio):.2f}x")
    common.record(f"scan.at{m}", float(ratio))

common.header("scoring live against reading the table, one scan each")
for label, size, table_ms, live_ms in re.findall(
        r"(median|worst) bucket \((\d+) candidates\):\s+table\s+([\d.]+) ms"
        r"\s+live\s+([\d.]+) ms", done.stdout):
    print(f"  {label:>6} bucket, {int(size):>4} candidates   table {float(table_ms):.1f} ms"
          f"   live {float(live_ms):.1f} ms   {float(live_ms) / float(table_ms):.1f}x")
    common.record(f"live.{label}.candidates", int(size))
    common.record(f"live.{label}.table_ms", float(table_ms))
    common.record(f"live.{label}.live_ms", float(live_ms))
    common.record(f"live.{label}.ratio", float(live_ms) / float(table_ms))

if args.train:
    trainer = rust_train.binary()
    if trainer is None:
        raise SystemExit("the native trainer is not built; run ./build.sh")

    def rounds(flags):
        """Mean seconds per round over the five priors, from the trainer's log."""
        done = subprocess.run([trainer, ROOT] + flags, capture_output=True,
                              text=True, check=True)
        seconds = [float(m) for m in re.findall(r"\(([\d.]+)s\)", done.stderr)]
        return sum(seconds) / len(seconds)

    def loop(flags):
        """Seconds the whole loop took, from the trainer's log."""
        done = subprocess.run([trainer, ROOT] + flags, capture_output=True,
                              text=True, check=True)
        return sum(float(m) for m in re.findall(r"rounds in ([\d.]+)s", done.stderr))

    common.header("a training round, by games in flight")
    for chunks in (1, 2, 3):
        seconds = rounds(["--chunks", str(chunks)])
        print(f"  {chunks} chunk(s)   {seconds:.3f} s a round")
        common.record(f"train.round.chunks{chunks}", seconds)
    shipped = "repeat1_curation1_r-binned_c-wordlebot"
    one = loop(["--prior", shipped, "--chunks", "1", "--exact-max", "60"])
    three = loop(["--prior", shipped, "--chunks", "3", "--exact-max", "60"])
    print(f"  with --exact-max 60 on the shipped prior: {one:.1f} s serially, "
          f"{three:.1f} s at three chunks")
    common.record("train.exact60.chunks1", one)
    common.record("train.exact60.chunks3", three)
    common.record("train.exact60.gain", one / three)

    common.header("a whole `train`, both ways, best of two")
    cache = os.path.join(ROOT, "estimator_cache")
    keep = tempfile.mkdtemp()
    shutil.copytree(cache, os.path.join(keep, "estimator_cache"))
    try:
        times = {}
        for label, extra in (("rust", []), ("python", ["--no-rust"])):
            best = None
            for _ in range(2):
                start = time.perf_counter()
                subprocess.run([PY, os.path.join(ROOT, "wordle_solver.py"), "train"]
                               + extra, capture_output=True, check=True, cwd=ROOT)
                elapsed = time.perf_counter() - start
                best = elapsed if best is None else min(best, elapsed)
            times[label] = best
            print(f"  {label:8s} {best:.1f} s")
    finally:
        shutil.rmtree(cache)
        shutil.copytree(os.path.join(keep, "estimator_cache"), cache)
        shutil.rmtree(keep)
        print("  the estimator cache is restored")
    common.record("train.rust", times["rust"])
    common.record("train.python", times["python"])
    common.record("train.gain", times["python"] / times["rust"])
