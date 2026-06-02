"""Run all 5m train scripts in 'both' mode and save comparison to log file.

Usage:
    python run_5m_comparison.py
    python run_5m_comparison.py --csv training_5m_new.csv
    python run_5m_comparison.py --trials 50
    python run_5m_comparison.py --scripts 5m 5m_features
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS = [
    ("train_5m.py", "5m baseline"),
    ("train_5m_features.py", "5m + engineered features"),
]

LOG_DIR = Path("logs")


def run_script(script_path, csv, trials, log_file):
    """Run a train script, tee output to console + log file."""
    cmd = [
        sys.executable, script_path,
        "--csv", csv,
        "--mode", "both",
        "--trials", str(trials),
    ]

    log_file.write(f"\n{'=' * 70}\n")
    log_file.write(f"RUNNING: {' '.join(cmd)}\n")
    log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"{'=' * 70}\n\n")
    log_file.flush()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    for line in process.stdout:
        print(line, end="")          # console
        log_file.write(line)         # file
        log_file.flush()
        lines.append(line)

    process.wait()
    return process.returncode, lines


def extract_comparison(output_lines):
    """Extract static/optuna accuracy and macro-F1 from output."""
    static_acc = None
    optuna_acc = None
    static_f1 = None
    optuna_f1 = None
    for line in output_lines:
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

    # Filter scripts if --scripts specified
    scripts_to_run = SCRIPTS
    if args.scripts:
        scripts_to_run = [
            (s, d) for s, d in SCRIPTS
            if any(tag in s for tag in args.scripts)
        ]

    print(f"Log file: {log_path.resolve()}")
    print(f"Running {len(scripts_to_run)} scripts with {args.trials} trials each\n")

    results = []

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"5m Model Comparison — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"CSV: {args.csv}  |  Trials: {args.trials}\n")

        for script, description in scripts_to_run:
            print(f"\n{'#' * 60}")
            print(f"# {description}")
            print(f"{'#' * 60}\n")

            start = time.time()
            rc, lines = run_script(script, args.csv, args.trials, log_file)
            elapsed = time.time() - start

            static_acc, optuna_acc, static_f1, optuna_f1 = extract_comparison(lines)
            results.append({
                "script": script,
                "description": description,
                "static_acc": static_acc,
                "optuna_acc": optuna_acc,
                "static_f1": static_f1,
                "optuna_f1": optuna_f1,
                "exit_code": rc,
                "elapsed_s": elapsed,
            })

        # Write summary table
        summary = "\n" + "=" * 90 + "\n"
        summary += "SUMMARY\n"
        summary += "=" * 90 + "\n\n"
        summary += f"{'Script':<30} {'Static':>8} {'Optuna':>8} {'Δ Acc':>8} {'Static F1':>10} {'Optuna F1':>10} {'Δ F1':>8} {'Time':>6}\n"
        summary += "-" * 90 + "\n"

        for r in results:
            sa = f"{r['static_acc']:.4f}" if r["static_acc"] is not None else "n/a"
            oa = f"{r['optuna_acc']:.4f}" if r["optuna_acc"] is not None else "n/a"
            sf = f"{r['static_f1']:.4f}" if r["static_f1"] is not None else "n/a"
            of_ = f"{r['optuna_f1']:.4f}" if r["optuna_f1"] is not None else "n/a"
            if r["static_acc"] is not None and r["optuna_acc"] is not None:
                da = f"{r['optuna_acc'] - r['static_acc']:+.4f}"
            else:
                da = "n/a"
            if r["static_f1"] is not None and r["optuna_f1"] is not None:
                df = f"{r['optuna_f1'] - r['static_f1']:+.4f}"
            else:
                df = "n/a"
            t = f"{r['elapsed_s']:.0f}s"
            status = "OK" if r["exit_code"] == 0 else f"ERR({r['exit_code']})"
            summary += f"{r['description']:<30} {sa:>8} {oa:>8} {da:>8} {sf:>10} {of_:>10} {df:>8} {t:>6}  {status}\n"

        summary += "\n"
        log_file.write(summary)
        print(summary)

    print(f"\nLog saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
