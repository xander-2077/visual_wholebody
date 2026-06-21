#!/usr/bin/env python3
"""Download a remote low-level training run into the matching local log path."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath


REMOTE_INSPECT = r"""
import glob
import json
import os
import re
import sys

run_dir, model_glob, latest_by = sys.argv[1:4]
required = ["b1z1_config.py", "manip_loco.py"]

missing = [name for name in required if not os.path.isfile(os.path.join(run_dir, name))]
models = [path for path in glob.glob(os.path.join(run_dir, model_glob)) if os.path.isfile(path)]

if not os.path.isdir(run_dir):
    print(json.dumps({"error": "remote run directory does not exist", "run_dir": run_dir}))
    sys.exit(2)
if missing:
    print(json.dumps({"error": "required files are missing", "missing": missing}))
    sys.exit(3)
if not models:
    print(json.dumps({"error": "no model files matched", "model_glob": model_glob}))
    sys.exit(4)

def model_step(path):
    name = os.path.basename(path)
    match = re.search(r"model_(\d+)", name)
    return int(match.group(1)) if match else -1

def sort_key(path):
    stat = os.stat(path)
    step = model_step(path)
    name = os.path.basename(path)
    if latest_by == "mtime":
        return (stat.st_mtime_ns, step, name)
    return (step, stat.st_mtime_ns, name)

latest = os.path.basename(max(models, key=sort_key))
print(json.dumps({"files": required + [latest], "latest_model": latest}))
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download b1z1_config.py, manip_loco.py, and the latest model_* "
            "from a remote run directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python low-level/legged_gym/scripts/download_remote_run.py "
            "HostName "
            "/home/zdj/Code/visual_wholebody/low-level/logs/b2z1-low/"
            "260620_203621-b2z1_test\n\n"
            "This creates:\n"
            "  low-level/logs/b2z1-low/260620_203621-b2z1_test-HostName/"
        ),
    )
    parser.add_argument("host", help="SSH hostname or host alias. user@host is also accepted.")
    parser.add_argument("remote_run", help="Remote run directory to download from.")
    parser.add_argument("--user", help="SSH user. Ignored if host already contains '@'.")
    parser.add_argument("--port", type=int, help="SSH port.")
    parser.add_argument("--identity-file", help="SSH private key path.")
    parser.add_argument(
        "--suffix",
        help="Suffix appended to the run directory. Defaults to the sanitized hostname.",
    )
    parser.add_argument(
        "--local-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Local repository root. Defaults to this visual_whole_body checkout.",
    )
    parser.add_argument(
        "--anchor",
        default="low-level",
        help="Remote path segment used to map into the local repo root.",
    )
    parser.add_argument(
        "--model-glob",
        default="model_*",
        help="Remote model glob. Use 'model_*.py' if that is the actual file pattern.",
    )
    parser.add_argument(
        "--latest-by",
        choices=("step", "mtime"),
        default="step",
        help="How to choose the latest model. 'step' uses the number in model_N.",
    )
    parser.add_argument(
        "--remote-python",
        default="python3",
        help="Python executable on the remote host.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned download without creating files.",
    )
    return parser.parse_args()


def ssh_destination(host, user):
    if user and "@" not in host:
        return f"{user}@{host}"
    return host


def ssh_options(args):
    opts = []
    if args.port:
        opts.extend(["-p", str(args.port)])
    if args.identity_file:
        opts.extend(["-i", os.path.expanduser(args.identity_file)])
    return opts


def ssh_command(args, remote_command):
    return ["ssh", *ssh_options(args), ssh_destination(args.host, args.user), remote_command]


def sanitize_suffix(host, suffix):
    raw = suffix or host.rsplit("@", 1)[-1]
    raw = raw.split(":", 1)[0]
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    if not sanitized:
        raise ValueError("hostname suffix is empty after sanitizing")
    return sanitized


def local_destination(remote_run, local_root, anchor, suffix):
    remote_path = PurePosixPath(remote_run.rstrip("/"))
    parts = list(remote_path.parts)
    if anchor not in parts:
        raise ValueError(
            f"remote path must contain '{anchor}' so it can be mapped into the local repo"
        )

    anchor_index = len(parts) - 1 - parts[::-1].index(anchor)
    rel_parts = parts[anchor_index:]
    run_name = rel_parts[-1]
    rel_parts[-1] = f"{run_name}-{suffix}"
    return Path(local_root).expanduser().resolve().joinpath(*rel_parts)


def quote_remote_args(*values):
    return " ".join(shlex.quote(str(value)) for value in values)


def inspect_remote_run(args):
    remote_command = "{} -c {} {}".format(
        shlex.quote(args.remote_python),
        shlex.quote(REMOTE_INSPECT),
        quote_remote_args(args.remote_run, args.model_glob, args.latest_by),
    )
    result = subprocess.run(
        ssh_command(args, remote_command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stdout.strip() or result.stderr.strip()
        raise RuntimeError(f"remote inspection failed: {detail}")
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload_text = stdout_lines[-1] if stdout_lines else ""
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"remote inspection returned invalid JSON: {result.stdout}") from exc
    if "error" in payload:
        raise RuntimeError(json.dumps(payload, ensure_ascii=False))
    return payload


def download_with_tar(args, files, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    remote_command = "cd {} && tar -cf - -- {}".format(
        shlex.quote(args.remote_run),
        quote_remote_args(*files),
    )
    ssh_proc = subprocess.Popen(
        ssh_command(args, remote_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tar_proc = subprocess.run(
        ["tar", "-xf", "-", "-C", str(dest_dir)],
        check=False,
        stdin=ssh_proc.stdout,
        stderr=subprocess.PIPE,
    )
    if ssh_proc.stdout is not None:
        ssh_proc.stdout.close()
    ssh_stderr = ssh_proc.stderr.read().decode("utf-8", errors="replace") if ssh_proc.stderr else ""
    ssh_returncode = ssh_proc.wait()
    tar_stderr = tar_proc.stderr.decode("utf-8", errors="replace")

    if ssh_returncode != 0 or tar_proc.returncode != 0:
        raise RuntimeError(
            "download failed\n"
            f"ssh return code: {ssh_returncode}\n"
            f"tar return code: {tar_proc.returncode}\n"
            f"ssh stderr: {ssh_stderr.strip()}\n"
            f"tar stderr: {tar_stderr.strip()}"
        )


def main():
    args = parse_args()
    suffix = sanitize_suffix(args.host, args.suffix)
    dest_dir = local_destination(args.remote_run, args.local_root, args.anchor, suffix)

    payload = inspect_remote_run(args)
    files = payload["files"]

    print(f"remote: {ssh_destination(args.host, args.user)}:{args.remote_run}")
    print(f"local:  {dest_dir}")
    print(f"files:  {', '.join(files)}")
    print(f"latest: {payload['latest_model']}")

    if args.dry_run:
        return 0

    download_with_tar(args, files, dest_dir)
    print("done")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
