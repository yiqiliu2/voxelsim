#!/usr/bin/env python3
"""
Sequential Test Runner for 3D-Stack Simulator
Runs all test modes from run_all_tests.py sequentially

Usage:
    python3 run_all_modes.py [options]

Options:
    --prefill          Run in prefill mode
    --no-parallel       Disable parallel execution within modes
    --modes 1,2,3       Run specific modes only (comma-separated)
    --run-all           Force re-run all tests (ignore existing outputs)
"""

import sys
import os
import time
import argparse
from datetime import datetime, timedelta
import subprocess
from pathlib import Path

# Mode descriptions
MODE_DESCRIPTIONS = {
    1: "Dense Sweep (NoC topology and bandwidth)",
    2: "Individual Parameter Sweeps",
    3: "Paired Parameter Sweeps",
    5: "SPMD Compiler and DRAM Mapping",
    7: "Dataflow Paradigm",
    8: "Uniform DRAM Mapping vs Bandwidth",
    9: "Default Configuration Run",
    11: "DRAM tRP Sweep (row-conflict overhead)",
}

# ANSI color codes
class Colors:
    """ANSI escape codes for coloured terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'=' * 60}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.YELLOW}➜ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def _progress_bar(done, total, width=24):
    if total <= 0:
        return "[" + (" " * width) + "]"
    filled = round(width * done / total)
    return f"[{('█' * filled).ljust(width, '░')}]"


def _extract_dry_run_summary(text):
    latest = None
    current = {}

    def flush_current():
        nonlocal latest, current
        if current:
            latest = current.copy()
            current = {}

    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "DRY RUN SUMMARY":
            flush_current()
            continue
        if stripped.startswith("Total expected outputs:"):
            current["total"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Existing outputs:"):
            current["existing"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Good outputs (Overall Util present):"):
            current["good"] = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("Missing outputs:"):
            current["missing"] = stripped.split(":", 1)[1].strip()
            continue
    flush_current()
    return latest or {}

def check_dependencies():
    """Check if required Python packages are installed"""
    print_info("Checking dependencies...")
    try:
        import numpy
        import matplotlib
        import sklearn
        import ujson
        import scipy
        print_success("All dependencies available")
        return True
    except ImportError as e:
        print_error(f"Missing dependency: {e}")
        print_warning("Please run: python3 -m pip install -r requirements.txt --user")
        return False

def run_mode(mode, prefill=False, parallel=True, run_all=False, dry_run=False, log_dir=None, decode_parallel_limit=3, reverse_order=False, model_filter="", parallel_all=False, extreme_parallel=False, sweep_params=None, cg_list_vals=None, exclude_models=None):
    """
    Run a single test mode by modifying and executing run_all_tests.py

    Args:
        mode: Mode number (1-9)
        prefill: Enable prefill mode
        parallel: Enable parallel execution
        run_all: Force re-run all tests
        log_dir: Directory for log files
        decode_parallel_limit: Max concurrent decode processes (default 3)
        reverse_order: Reverse iteration order for modes and configs

    Returns:
        tuple: (success: bool, duration: float, error_msg: str)
    """
    start_time = time.time()

    if not dry_run:
        print_header(f"Mode {mode}: {MODE_DESCRIPTIONS[mode]}")

    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None if dry_run else log_dir / f"mode_{mode}_{timestamp}.log"

    if not dry_run:
        if log_file is not None:
            print_info(f"Log file: {log_file}")
        else:
            print_info("Log file: disabled in dry-run (terminal only)")

    # Create a modified version of run_all_tests.py for this mode
    script_dir = Path(__file__).parent
    benchmark_dir = script_dir / "benchmark_scripts"
    original_script = benchmark_dir / "run_all_tests.py"

    if not original_script.exists():
        return False, 0.0, f"run_all_tests.py not found at {original_script}"

    # Read the original script
    with open(original_script, 'r') as f:
        content = f.read()

    # Modify the mode settings
    # Find and replace the MODE, PARALLEL, PREFILL, and RUN_ALL settings
    import re

    # Replace MODE
    content = re.sub(r'MODE = \d+', f'MODE = {mode}', content)

    # Replace PARALLEL
    parallel_str = "True" if parallel else "False"
    content = re.sub(r'PARALLEL = (True|False)', f'PARALLEL = {parallel_str}', content)

    # Replace PREFILL
    prefill_str = "True" if prefill else "False"
    content = re.sub(r'PREFILL = (True|False)', f'PREFILL = {prefill_str}', content)

    # Replace RUN_ALL
    run_all_str = "True" if run_all else "False"
    content = re.sub(r'RUN_ALL\s*=\s*(True|False)', f'RUN_ALL = {run_all_str}', content)

    # Replace DRY_RUN
    dry_run_str = "True" if dry_run else "False"
    if 'DRY_RUN' in content:
        content = re.sub(r'DRY_RUN\s*=\s*(True|False)', f'DRY_RUN = {dry_run_str}', content)
    else:
        content = re.sub(
            r'(RUN_ALL\s*=\s*(True|False))',
            f'\\1\nDRY_RUN = {dry_run_str}',
            content,
        )

    # Replace MAX_DECODE_WORKERS (add if not exists)
    if 'MAX_DECODE_WORKERS' in content:
        content = re.sub(r'MAX_DECODE_WORKERS\s*=\s*\d+', f'MAX_DECODE_WORKERS = {decode_parallel_limit}', content)
    else:
        # Insert after GB_PER_INFERENCE_LAUNCH
        content = re.sub(
            r'(GB_PER_INFERENCE_LAUNCH\s*=\s*\d+)',
            f'\\1\nMAX_DECODE_WORKERS = {decode_parallel_limit}',
            content
        )

    # Replace REVERSE_ORDER
    reverse_str = "True" if reverse_order else "False"
    content = re.sub(r'REVERSE_ORDER\s*=\s*(True|False)', f'REVERSE_ORDER = {reverse_str}', content)

    # Replace MODEL_FILTER
    content = re.sub(r'MODEL_FILTER\s*=\s*"[^"]*"', f'MODEL_FILTER = "{model_filter}"', content)

    # Replace PARALLEL_ALL
    parallel_all_str = "True" if parallel_all else "False"
    content = re.sub(r'PARALLEL_ALL\s*=\s*(True|False)', f'PARALLEL_ALL = {parallel_all_str}', content)

    # Replace EXTREME_PARALLEL
    extreme_str = "True" if extreme_parallel else "False"
    content = re.sub(r'EXTREME_PARALLEL\s*=\s*(True|False)', f'EXTREME_PARALLEL = {extreme_str}', content)

    # Replace SWEEP_PARAMS if provided (mode 2)
    if sweep_params is not None:
        content = re.sub(r'SWEEP_PARAMS\s*=\s*\[.*?\]', f'SWEEP_PARAMS = {sweep_params}', content)

    # Replace cg_list if provided (mode 3)
    if cg_list_vals is not None:
        content = re.sub(r'cg_list\s*=\s*\[.*?\]', f'cg_list = {cg_list_vals}', content)

    # Replace SKIP_MODELS: inject variable + unconditional filter after all_list.
    if exclude_models is not None:
        skip_repr = repr(set(exclude_models))
        if 'SKIP_MODELS' in content:
            content = re.sub(r'SKIP_MODELS\s*=\s*set\(\).*', f'SKIP_MODELS = {skip_repr}', content)
        else:
            content = re.sub(
                r'(MODEL_FILTER\s*=\s*"[^"]*")',
                f'\\1\nSKIP_MODELS = {skip_repr}',
                content,
            )
        if 'if SKIP_MODELS:' not in content:
            content = re.sub(
                r'(if MODEL_FILTER:\s*\n\s*all_list\s*=\s*\[\(m.*?if m == MODEL_FILTER\])',
                f'\\1\nif SKIP_MODELS:\n    all_list = [(m, l, sf, fmha) for m, l, sf, fmha in all_list if m not in SKIP_MODELS]',
                content,
            )

    if not dry_run:
        print_info(f"Configuration: MODE={mode}, PARALLEL={parallel}, PREFILL={prefill}, RUN_ALL={run_all}, DRY_RUN={dry_run}, DECODE_LIMIT={decode_parallel_limit}, REVERSE={reverse_order}, MODEL_FILTER={model_filter!r}, PARALLEL_ALL={parallel_all}, EXTREME_PARALLEL={extreme_parallel}")

    # Execute the modified script
    try:
        if dry_run:
            process = subprocess.Popen(
                [sys.executable, '-c', content],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            captured_output = process.communicate()[0]
            summary = _extract_dry_run_summary(captured_output)
            process.wait()

            if process.returncode == 0:
                duration = time.time() - start_time
                return True, duration, summary

            duration = time.time() - start_time
            error_msg = f"Mode {mode} exited with code {process.returncode}"
            if not dry_run:
                print_error(error_msg)
            return False, duration, {"error": error_msg}

        if log_file is None:
            process = subprocess.Popen(
                [sys.executable, '-c', content],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in process.stdout:
                print(line, end='')

            process.wait()

            if process.returncode == 0:
                duration = time.time() - start_time
                print_success(f"Mode {mode} completed successfully")
                return True, duration, None

            duration = time.time() - start_time
            error_msg = f"Mode {mode} exited with code {process.returncode}"
            print_error(error_msg)
            return False, duration, error_msg

        with open(log_file, 'w') as log_f:
            log_f.write(f"=== Mode {mode}: {MODE_DESCRIPTIONS[mode]} ===\n")
            log_f.write(f"Started at: {datetime.now()}\n")
            log_f.write(f"Configuration: PARALLEL={parallel}, PREFILL={prefill}, RUN_ALL={run_all}\n\n")
            log_f.flush()

            # Execute using subprocess
            process = subprocess.Popen(
                [sys.executable, '-c', content],
                cwd=str(script_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log_f.write(line)
                log_f.flush()

            process.wait()

            if process.returncode == 0:
                duration = time.time() - start_time
                print_success(f"Mode {mode} completed successfully")
                return True, duration, None
            else:
                duration = time.time() - start_time
                error_msg = f"Mode {mode} exited with code {process.returncode}"
                print_error(error_msg)
                return False, duration, error_msg

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Exception during mode {mode}: {str(e)}"
        print_error(error_msg)
        return False, duration, error_msg

def run_prefill_and_decode(mode, run_all=False, dry_run=False, log_dir=None, decode_parallel_limit=3, reverse_order=False, prefill_modes=None, sweep_params=None, cg_list_vals=None, parallel_all=False, extreme_parallel=False, exclude_models=None, model_filter=""):
    """
    Run both prefill and decode modes sequentially.

    Args:
        mode: Mode number (1-9)
        run_all: Force re-run all tests
        log_dir: Directory for log files
        decode_parallel_limit: Max concurrent decode processes (default 3)
        reverse_order: Reverse iteration order for modes and configs
        prefill_modes: Set of mode numbers that should run prefill.
                       If None, all modes run prefill.  Modes not in this
                       set get a zeroed prefill summary (for dry-run).
        parallel_all: If True, parallelize all runs (ignore last-model guard)
        extreme_parallel: If True, skip _wait_all() between one_pass calls; single barrier at end

    Returns:
        tuple: (combined_success, combined_duration, combined_error,
                prefill_summary, decode_summary)
               where prefill_summary/decode_summary are dicts (dry_run) or None
    """
    if prefill_modes is None:
        prefill_modes = set()
    skip_prefill = prefill_modes and mode not in prefill_modes

    if not dry_run:
        print_header(f"Running Both Prefill and Decode for Mode {mode}")

    # ── Prefill ──
    prefill_ok = True
    prefill_duration = 0.0
    prefill_summary = None
    prefill_error = None

    if skip_prefill:
        if dry_run:
            prefill_summary = {"total": 0, "existing": 0, "good": 0, "missing": 0}
        elif not dry_run:
            print_info("Prefill skipped (not in prefill_modes)")
    else:
        if not dry_run:
            print_info("Step 1/2: Running prefill mode...")
        prefill_result = run_mode(
            mode=mode,
            prefill=True,
            parallel=False,
            run_all=run_all,
            dry_run=dry_run,
            log_dir=log_dir,
            decode_parallel_limit=decode_parallel_limit,
            reverse_order=reverse_order,
            sweep_params=sweep_params,
            cg_list_vals=cg_list_vals,
            exclude_models=exclude_models,
            model_filter=model_filter,
        )
        prefill_ok = prefill_result[0]
        prefill_duration = prefill_result[1]
        prefill_summary = prefill_result[2] if dry_run else None
        prefill_error = prefill_result[2] if not dry_run and not prefill_ok else None

    if not prefill_ok and not skip_prefill:
        if not dry_run:
            print_error("Prefill failed, skipping decode mode")
        return (False, prefill_duration, f"Prefill: {prefill_error}", prefill_summary, None)

    if not dry_run and not skip_prefill:
        print_success("Prefill completed successfully")
        print()

    # ── Decode ──
    if not dry_run:
        print_info(f"Step 2/2: Running decode mode (max {decode_parallel_limit} parallel processes)...")
    decode_result = run_mode(
        mode=mode,
        prefill=False,
        parallel=True,
        run_all=run_all,
        dry_run=dry_run,
        log_dir=log_dir,
        decode_parallel_limit=decode_parallel_limit,
        reverse_order=reverse_order,
        sweep_params=sweep_params,
        cg_list_vals=cg_list_vals,
        parallel_all=parallel_all,
        extreme_parallel=extreme_parallel,
        exclude_models=exclude_models,
        model_filter=model_filter,
    )

    decode_ok = decode_result[0]
    decode_duration = decode_result[1]
    decode_summary = decode_result[2] if dry_run else None
    decode_error = decode_result[2] if not dry_run and not decode_ok else None

    combined_success = prefill_ok and decode_ok
    combined_duration = prefill_duration + decode_duration
    if not combined_success:
        error_parts = []
        if not prefill_ok and not skip_prefill:
            error_parts.append(f"Prefill: {prefill_error}")
        if not decode_ok:
            error_parts.append(f"Decode: {decode_error}")
        combined_error = "; ".join(error_parts) if error_parts else None
    else:
        combined_error = None

    return (combined_success, combined_duration, combined_error, prefill_summary, decode_summary)


def format_duration(seconds):
    """Format duration in seconds to human-readable format"""
    duration = timedelta(seconds=int(seconds))
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def generate_summary(results, total_start, total_end, log_dir, dry_run=False, dry_run_raw=False):
    """Generate and print a tabular execution summary to the terminal.

    Args:
        results:     Dict mapping mode number to result tuples.
                     Single-mode:  (success, duration, error_or_summary)
                     Run-both:     (success, duration, error, prefill_summary, decode_summary)
        total_start: Epoch timestamp when the overall run started.
        total_end:   Epoch timestamp when the overall run finished.
        log_dir:     Path to the directory containing per-mode log files.

    Returns:
        Exit code: 0 if all modes succeeded, 1 otherwise.
    """
    success_count = 0
    fail_count = 0

    if not dry_run:
        print_header("Test Execution Summary")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Completed at: {timestamp}\n")

        print(f"{'Mode':<6} {'Description':<45} {'Status':<10} {'Duration':<15}")
        print("-" * 80)

        for mode, result in results.items():
            success, duration, error = result[0], result[1], result[2]
            if success:
                success_count += 1
            else:
                fail_count += 1
                if error:
                    print(f"Mode {mode} error: {error}")

            status = "SUCCESS" if success else "FAILED"
            status_colored = f"{Colors.GREEN}{status}{Colors.END}" if success else f"{Colors.RED}{status}{Colors.END}"
            duration_str = format_duration(duration)
            print(f"{mode:<6} {MODE_DESCRIPTIONS[mode]:<45} {status_colored:<20} {duration_str:<15}")
            if not success and error:
                print(f"       Error: {error}")

        print("-" * 80)
    else:
        # ── Dry-run ──
        if dry_run_raw:
            # Machine-parseable output for shell script to aggregate
            total_p_expected = 0
            total_p_existing = 0
            total_d_expected = 0
            total_d_existing = 0
            for mode, result in results.items():
                if len(result) >= 5:
                    success = result[0]
                    prefill_summary = result[3] if isinstance(result[3], dict) else {}
                    decode_summary = result[4] if isinstance(result[4], dict) else {}
                else:
                    success = result[0]
                    summary = result[2] if isinstance(result[2], dict) else {}
                    prefill_summary = summary
                    decode_summary = {}

                if success:
                    success_count += 1
                else:
                    fail_count += 1

                pt = int(prefill_summary.get("total", 0))
                pe = int(prefill_summary.get("existing", 0))
                dt = int(decode_summary.get("total", 0))
                de = int(decode_summary.get("existing", 0))
                total_p_expected += pt; total_p_existing += pe
                total_d_expected += dt; total_d_existing += de

                print(f"M{mode} P {pe} {pt}")
                print(f"M{mode} D {de} {dt}")
            print(f"ALL P {total_p_existing} {total_p_expected}")
            print(f"ALL D {total_d_existing} {total_d_expected}")
            # skip summary header + overall statistics in raw mode
            return 0 if fail_count == 0 else 1
        else:
            # Pretty progress-bar output
            print_header("Dry-Run Output Summary")

            bar_width = 20
            DASH = "──"

            def _bar_or_dash(done, total, width):
                if total <= 0:
                    return f" {DASH.center(width, ' ')} "
                filled = round(width * done / total)
                return f"[{('█' * filled).ljust(width, '░')}]"

            def _psum(s):
                return (int(s.get("total", 0)), int(s.get("existing", 0)))

            total_p_expected = 0
            total_p_existing = 0
            total_d_expected = 0
            total_d_existing = 0

            for mode, result in results.items():
                if len(result) >= 5:
                    success = result[0]
                    prefill_summary = result[3] if isinstance(result[3], dict) else {}
                    decode_summary = result[4] if isinstance(result[4], dict) else {}
                else:
                    success = result[0]
                    summary = result[2] if isinstance(result[2], dict) else {}
                    prefill_summary = summary
                    decode_summary = {}

                if success:
                    success_count += 1
                else:
                    fail_count += 1

                pt, pe = _psum(prefill_summary)
                dt, de = _psum(decode_summary)

                total_p_expected += pt
                total_p_existing += pe
                total_d_expected += dt
                total_d_existing += de

                def _num(done, total):
                    if total <= 0:
                        return "  ─ / ─  "
                    return f"{done:>4}/{total:<4}"

                print(f"M{mode:<2} P {_bar_or_dash(pe, pt, bar_width)} {_num(pe, pt)}")
                print(f"    D {_bar_or_dash(de, dt, bar_width)} {_num(de, dt)}")

            print("-" * 48)
            print(f"ALL P {_bar_or_dash(total_p_existing, total_p_expected, bar_width)} {_num(total_p_existing, total_p_expected)}")
            print(f"    D {_bar_or_dash(total_d_existing, total_d_expected, bar_width)} {_num(total_d_existing, total_d_expected)}")

    # Overall statistics
    print_header("Overall Statistics")

    total_duration = total_end - total_start
    total_modes = success_count + fail_count

    print(f"Total Modes Run: {total_modes}")
    print_success(f"Successful: {success_count}")

    if fail_count > 0:
        print_error(f"Failed: {fail_count}")
    else:
        print(f"Failed: {fail_count}")

    print(f"Total Duration: {format_duration(total_duration)}\n")

    if not dry_run:
        # Output locations
        print_header("Output Locations")
        script_dir = Path(__file__).parent
        print(f"Individual Mode Logs: {log_dir}/")
        print(f"Test Results: {script_dir}/results/logs/")
        print(f"Pickled Data: {script_dir}/results/pickles/\n")

    # Final status
    if fail_count == 0:
        print_success("All tests completed successfully!")
        return 0
    else:
        print_error(f"{fail_count} test(s) failed. Check individual logs for details.")
        return 1

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Sequential Test Runner for 3D-Stack Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available modes:
  1: Dense sweep (NoC topology and bandwidth)
  2: Individual parameter sweeps
  3: Paired parameter sweeps
  5: SPMD compiler and uniform DRAM mapping
  7: Dataflow paradigm
  8: Uniform DRAM mapping vs DRAM bandwidth
  9: Default configuration run
  11: DRAM tRP sweep (row-conflict overhead vs tRP)

Examples:
  # Run decode mode only (default)
  python3 run_all_modes.py --modes 9

  # Run both prefill and decode with limited parallelism
  python3 run_all_modes.py --modes 9 --run-both --decode-parallel-limit 3

  # Run in reverse order (modes 9,8,7... and configs reversed)
  python3 run_all_modes.py --modes 1,2,3 --reverse

  # Run prefill only
  python3 run_all_modes.py --modes 9 --prefill
        """
    )

    parser.add_argument('--prefill', '--training', dest='prefill', action='store_true',
                       help='Run in prefill mode (prefill only)')
    parser.add_argument('--parallel-mode', type=str, default='default',
                       choices=['none', 'default', 'extreme'],
                       help='Parallelism level: none=sequential, default=guarded parallel, extreme=bypass all limits (default: default)')
    parser.add_argument('--modes', type=str, default='1,2,3,5,7,8,9',
                       help='Comma-separated list of modes to run (default: all)')
    parser.add_argument('--run-all', action='store_true',
                       help='Force re-run all tests (ignore existing outputs)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Only inspect output files and print counts; never launch simulations')
    parser.add_argument('--dry-run-raw', action='store_true',
                       help='With --dry-run: output machine-parseable counts only (no progress bars)')
    parser.add_argument('--run-both', action='store_true',
                       help='Run both prefill and decode for each mode')
    parser.add_argument('--decode-parallel-limit', type=int, default=3,
                       help='Max concurrent decode processes to prevent OOM (default: 3)')
    parser.add_argument('--reverse', action='store_true',
                       help='Run in reverse order (modes and configs reversed)')
    parser.add_argument('--model', type=str, default='',
                       help='Restrict simulation to a single model name (e.g. llama2-13)')
    parser.add_argument('--prefill-modes', type=str, default='',
                        help='Comma-separated modes that should run prefill (others skip prefill, default: all)')
    parser.add_argument('--sweep-params', type=str, default='',
                        help='Override SWEEP_PARAMS for mode 2 (comma-separated, e.g. "noc_bw,dram_bw,sa,num_cores,sram_kb,noc_topo")')
    parser.add_argument('--cg-list', type=str, default='',
                        help='Override cg_list for mode 3 (comma-separated, e.g. "1,2,4,8")')
    parser.add_argument('--exclude-models', type=str, default='',
                        help='Comma-separated model names to exclude from all_list (e.g. "dit-xl")')

    args = parser.parse_args()

    # --dry-run-raw implies --dry-run
    if args.dry_run_raw:
        args.dry_run = True

    # Validate mutually exclusive options
    if args.run_both and args.prefill:
        print_error("--run-both and --prefill are mutually exclusive")
        print_info("Use --run-both to run both prefill and decode, or --prefill for prefill only")
        return 1

    # Prefill uses much more RAM per process.
    # Force sequential execution to avoid OOM / system hang.
    if args.prefill and args.parallel_mode != 'none':
        args.parallel_mode = 'none'
        print_warning("Training mode detected: forcing --parallel-mode none to avoid OOM")

    # Parse modes
    try:
        modes_to_run = [int(m.strip()) for m in args.modes.split(',')]
    except ValueError:
        print_error("Invalid mode specification. Use comma-separated numbers (e.g., 1,2,3)")
        return 1

    # Validate modes
    for mode in modes_to_run:
        if mode not in MODE_DESCRIPTIONS:
            print_error(f"Invalid mode: {mode}")
            return 1

    # Setup
    script_dir = Path(__file__).parent
    log_dir = script_dir / "test_logs"
    log_dir.mkdir(exist_ok=True)

    # Reverse mode order if requested
    if args.reverse:
        modes_to_run = list(reversed(modes_to_run))

    # Parse prefill_modes filter (only meaningful with --run-both)
    prefill_modes_set = None
    if args.run_both and args.prefill_modes:
        try:
            prefill_modes_set = {int(m.strip()) for m in args.prefill_modes.split(',')}
        except ValueError:
            print_error("Invalid --prefill-modes format (use comma-separated numbers)")
            return 1
        if not args.dry_run:
            print_info(f"Prefill only for modes: {sorted(prefill_modes_set)}")

    # Parse --sweep-params and --cg-list
    sweep_params = None
    if args.sweep_params:
        sweep_params = [s.strip() for s in args.sweep_params.split(',')]
    cg_list_vals = None
    if args.cg_list:
        cg_list_vals = [int(s.strip()) for s in args.cg_list.split(',')]
    exclude_models = None
    if args.exclude_models:
        exclude_models = [s.strip() for s in args.exclude_models.split(',')]

    # Banner
    print_header("3D-Stack Sequential Test Runner")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at: {timestamp}")
    print(f"Modes to run: {', '.join(map(str, modes_to_run))}")
    if args.run_both:
        print(f"Mode: Both prefill and decode")
        print(f"Decode parallel limit: {args.decode_parallel_limit}")
    else:
        print(f"Prefill mode: {args.prefill}")
        print(f"Parallel mode: {args.parallel_mode}")
        if not args.prefill:
            print(f"Decode parallel limit: {args.decode_parallel_limit}")
    print(f"Reverse order: {args.reverse}")
    print(f"Force re-run: {args.run_all}\n")

    # Check dependencies
    if not check_dependencies():
        return 1

    # Run modes sequentially
    results = {}
    total_start = time.time()

    for mode in modes_to_run:
        if args.run_both:
            # Run both prefill and decode — returns 5-tuple:
            # (combined_success, combined_duration, combined_error,
            #  prefill_summary, decode_summary)
            result = run_prefill_and_decode(
                mode=mode,
                run_all=args.run_all,
                dry_run=args.dry_run,
                log_dir=log_dir,
                decode_parallel_limit=args.decode_parallel_limit,
                reverse_order=args.reverse,
                prefill_modes=prefill_modes_set,
                sweep_params=sweep_params,
                cg_list_vals=cg_list_vals,
                parallel_all=args.parallel_mode in ('default', 'extreme'),
                extreme_parallel=args.parallel_mode == 'extreme',
                exclude_models=exclude_models,
                model_filter=args.model,
            )
            results[mode] = result
        else:
            # Run single mode (prefill or decode)
            success, duration, error = run_mode(
                mode=mode,
                prefill=args.prefill,
                parallel=args.parallel_mode != 'none',
                run_all=args.run_all,
                dry_run=args.dry_run,
                log_dir=log_dir,
                decode_parallel_limit=args.decode_parallel_limit,
                reverse_order=args.reverse,
                model_filter=args.model,
                parallel_all=args.parallel_mode in ('default', 'extreme'),
                extreme_parallel=args.parallel_mode == 'extreme',
                sweep_params=sweep_params,
                cg_list_vals=cg_list_vals,
                exclude_models=exclude_models,
            )
            results[mode] = (success, duration, error)
        if not args.dry_run:
            print()  # Extra newline between modes

    total_end = time.time()

    # Generate summary
    exit_code = generate_summary(results, total_start, total_end, log_dir, dry_run=args.dry_run, dry_run_raw=args.dry_run_raw)

    return exit_code

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
