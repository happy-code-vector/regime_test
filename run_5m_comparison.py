"""Run all 5m train scripts in 'both' mode and save comparison to log file.

Usage:
    python run_5m_comparison.py
    python run_5m_comparison.py --csv training_5m_new.csv
    python run_5m_comparison.py --trials 50
    python run_5m_comparison.py --scripts 5m 5m_features
"""

import sys
import os
import time
import logging
import importlib
from datetime import datetime
from pathlib import Path

SCRIPTS = [
    ("train_5m", "5m baseline"),
    ("train_5m_features", "5m + engineered features"),
]

LOG_DIR = Path("logs")


class Tee:
    """Write to multiple file-like objects simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(log_path):
    """Configure logging: console + file, and redirect stdout via Tee."""
    log_file = open(log_path, "w", encoding="utf-8")

    # Tee: print() goes to both original stdout and log file
    sys.stdout = Tee(sys.__stdout__, log_file)

    # Logger for the runner's own messages (goes to same Tee'd stdout)
    logger = logging.getLogger("comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger, log_file


def run_script(module_name, csv, trials, logger):
    """Import and run a train script's main()."""
    logger.info(f"\n{'#' * 60}")
    logger.info(f"# {SCRIPTS_DICT[module_name]}")
    logger.info(f"{'#' * 60}\n")

    mod = importlib.import_module(module_name)
    mod.main(csv_path=csv, mode="both", n_trials=trials)


def extract_comparison(log_path):
    """Parse static/optuna accuracy and macro-F1 from log file."""
    static_acc = optuna_acc = static_f1 = optuna_f1 = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "Static  accuracy:" in line:
                parts = line.split("|")
                static_acc = float(parts[0].split(":")[-1].strip())
                if len(parts) > 1:
                    static_f1 = float(parts[1].split(":")[-1].strip())
            if "Optuna  accuracy:" in line:
                parts = line.split("|")
                optuna_acc = float(parts[0].split(":")[-1].strip())
                if len(parts) > 1:
                    optuna_f1 = float(parts[1].split(":")[-1].strip())
    return static_acc, optuna_acc, static_f1, optuna_f1


SCRIPTS_DICT = {m: d for m, d in SCRIPTS}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all 5m train scripts and log results")
    parser.add_argument("--csv", default="training_5m.csv")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--scripts", nargs="+", default=None,
                        help="Subset of scripts to run (e.g. 5m 5m_features)")
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"comparison_5m_{timestamp}.log"

    logger, log_file = setup_logging(log_path)

    scripts_to_run = SCRIPTS
    if args.scripts:
        scripts_to_run = [
            (m, d) for m, d in SCRIPTS
            if any(tag in m for tag in args.scripts)
        ]

    logger.info(f"Log file: {log_path.resolve()}")
    logger.info(f"Running {len(scripts_to_run)} scripts with {args.trials} trials each\n")
    logger.info(f"5m Model Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"CSV: {args.csv}  |  Trials: {args.trials}\n")

    results = []

    for module_name, description in scripts_to_run:
        start = time.time()
        try:
            run_script(module_name, args.csv, args.trials, logger)
            rc = 0
        except Exception as e:
            logger.error(f"FAILED: {e}")
            rc = 1
        elapsed = time.time() - start

        static_acc, optuna_acc, static_f1, optuna_f1 = extract_comparison(log_path)
        results.append({
            "description": description,
            "static_acc": static_acc,
            "optuna_acc": optuna_acc,
            "static_f1": static_f1,
            "optuna_f1": optuna_f1,
            "exit_code": rc,
            "elapsed_s": elapsed,
        })

    # Summary table
    w = 90
    summary = f"\n{'=' * w}\nSUMMARY\n{'=' * w}\n\n"
    summary += f"{'Script':<30} {'Static':>8} {'Optuna':>8} {'Δ Acc':>8} {'Static F1':>10} {'Optuna F1':>10} {'Δ F1':>8} {'Time':>6}\n"
    summary += "-" * w + "\n"

    for r in results:
        sa = f"{r['static_acc']:.4f}" if r["static_acc"] is not None else "n/a"
        oa = f"{r['optuna_acc']:.4f}" if r["optuna_acc"] is not None else "n/a"
        sf = f"{r['static_f1']:.4f}" if r["static_f1"] is not None else "n/a"
        of_ = f"{r['optuna_f1']:.4f}" if r["optuna_f1"] is not None else "n/a"
        da = f"{r['optuna_acc'] - r['static_acc']:+.4f}" if r["static_acc"] and r["optuna_acc"] else "n/a"
        df = f"{r['optuna_f1'] - r['static_f1']:+.4f}" if r["static_f1"] and r["optuna_f1"] else "n/a"
        t = f"{r['elapsed_s']:.0f}s"
        status = "OK" if r["exit_code"] == 0 else f"ERR({r['exit_code']})"
        summary += f"{r['description']:<30} {sa:>8} {oa:>8} {da:>8} {sf:>10} {of_:>10} {df:>8} {t:>6}  {status}\n"

    summary += "\n"
    logger.info(summary)

    log_file.close()
    sys.stdout = sys.__stdout__
    print(f"\nLog saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
