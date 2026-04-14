"""
08_parallel_camoufox.py
-----------------------
Launch 3 Camoufox workers in parallel and continue from existing progress state.

State priority:
1) state/camoufox_rr3_state.json
2) state/scrapling_rr3_state.json
3) state/rr3_workers_state.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CRAWLER_FILE = SCRIPT_DIR / "08_camoufox_crawl_by_org.py"

CAMOUFOX_RR_STATE_FILE = SCRIPT_DIR / "state" / "camoufox_rr3_state.json"
SCRAPLING_RR_STATE_FILE = SCRIPT_DIR / "state" / "scrapling_rr3_state.json"
LEGACY_RR3_STATE_FILE = SCRIPT_DIR / "state" / "rr3_workers_state.json"

DEFAULT_STATE_DIR = SCRIPT_DIR / "state"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_PROFILES_ROOT = DEFAULT_STATE_DIR / "camoufox_profiles"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs" / "camoufox_parallel"


@dataclass
class Task:
    org_id: int
    start_page: int
    end_page: int


WORKER_CONFIGS: dict[str, list[Task]] = {
    "w1": [Task(1, 1, 50), Task(1, 101, 150)],
    "w2": [Task(1, 51, 100), Task(1, 151, 200)],
    "w3": [
        Task(3, 1, 999), Task(12, 1, 999), Task(6, 1, 999),
        Task(19, 1, 999), Task(22, 1, 999), Task(23, 1, 999),
        Task(26, 1, 999), Task(33, 1, 999), Task(95, 1, 999),
        Task(97, 1, 999), Task(98, 1, 999), Task(104, 1, 999),
    ],
}
WORKERS = ["w1", "w2", "w3"]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _normalize_worker_state(worker_id: str, raw: dict | None) -> dict:
    tasks = WORKER_CONFIGS[worker_id]
    task_count = len(tasks)
    first = tasks[0]

    if not isinstance(raw, dict):
        return {
            "task_index": 0,
            "next_page": first.start_page,
            "completed": False,
            "updated_at": None,
        }

    def to_int(name: str, default: int) -> int:
        try:
            return int(raw.get(name, default))
        except Exception:
            return default

    task_index = max(0, min(to_int("task_index", 0), task_count))
    completed = bool(raw.get("completed", False)) or task_index >= task_count
    updated_at = raw.get("updated_at")

    if completed:
        next_page = tasks[-1].end_page + 1
        return {
            "task_index": task_index,
            "next_page": next_page,
            "completed": True,
            "updated_at": updated_at,
        }

    task = tasks[task_index]
    next_page = to_int("next_page", task.start_page)
    next_page = max(task.start_page, min(next_page, task.end_page + 1))
    return {
        "task_index": task_index,
        "next_page": next_page,
        "completed": False,
        "updated_at": updated_at,
    }


def load_resume_state() -> tuple[dict[str, dict], str]:
    source = "fresh_init"
    raw: dict = {}

    if CAMOUFOX_RR_STATE_FILE.exists():
        source = CAMOUFOX_RR_STATE_FILE.name
        try:
            raw = json.loads(CAMOUFOX_RR_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    elif SCRAPLING_RR_STATE_FILE.exists():
        source = SCRAPLING_RR_STATE_FILE.name
        try:
            raw = json.loads(SCRAPLING_RR_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
    elif LEGACY_RR3_STATE_FILE.exists():
        source = LEGACY_RR3_STATE_FILE.name
        try:
            raw = json.loads(LEGACY_RR3_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    workers_raw = raw.get("workers", {}) if isinstance(raw, dict) else {}
    norm: dict[str, dict] = {}
    for wid in WORKERS:
        norm[wid] = _normalize_worker_state(wid, workers_raw.get(wid))
    return norm, source


def worker_pending_tasks(worker_id: str, worker_state: dict) -> list[Task]:
    tasks = WORKER_CONFIGS[worker_id]
    if worker_state.get("completed"):
        return []

    idx = int(worker_state.get("task_index", 0))
    if idx >= len(tasks):
        return []

    pending: list[Task] = []
    for i in range(idx, len(tasks)):
        t = tasks[i]
        start = t.start_page
        if i == idx:
            start = max(t.start_page, min(int(worker_state.get("next_page", t.start_page)), t.end_page + 1))
        if start <= t.end_page:
            pending.append(Task(t.org_id, start, t.end_page))
    return pending


def build_orgs_ranges_for_worker(tasks: list[Task]) -> tuple[str, str]:
    if not tasks:
        return "", ""

    org_ids = {t.org_id for t in tasks}
    ranges = {(t.start_page, t.end_page) for t in tasks}

    # Case A: same org, many ranges (w1/w2)
    if len(org_ids) == 1:
        orgs = str(tasks[0].org_id)
        ranges_text = ",".join(f"{t.start_page}-{t.end_page}" for t in tasks)
        return orgs, ranges_text

    # Case B: many orgs, one common range (w3 layout)
    if len(ranges) == 1:
        orgs = ",".join(str(t.org_id) for t in tasks)
        r = tasks[0]
        ranges_text = f"{r.start_page}-{r.end_page}"
        return orgs, ranges_text

    # Case C: many orgs with different start pages (resume in the middle of first org).
    # Use the broadest common range to avoid skipping remaining orgs.
    seen: set[int] = set()
    orgs_list: list[str] = []
    min_start = min(t.start_page for t in tasks)
    max_end = max(t.end_page for t in tasks)
    for t in tasks:
        if t.org_id in seen:
            continue
        seen.add(t.org_id)
        orgs_list.append(str(t.org_id))
    orgs = ",".join(orgs_list)
    ranges_text = f"{min_start}-{max_end}"
    return orgs, ranges_text


def launch_worker(
    *,
    worker_id: str,
    orgs: str,
    ranges_text: str,
    python_bin: str,
    delay: float,
    cf_manual_wait: int,
    captcha_manual_wait: int,
    captcha_retries: int,
    navigation_retries: int,
    proxy: str,
    state_dir: Path,
    output_dir: Path,
    profiles_root: Path,
    headless: bool,
    log_dir: Path,
) -> subprocess.Popen | None:
    if not orgs or not ranges_text:
        return None

    profiles_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = profiles_root / worker_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{worker_id}.log"

    cmd = [
        python_bin,
        str(CRAWLER_FILE),
        "--worker", worker_id,
        "--orgs", orgs,
        "--ranges", ranges_text,
        "--delay", str(delay),
        "--profile-dir", str(profile_dir),
        "--state-dir", str(state_dir),
        "--output-dir", str(output_dir),
        "--cf-manual-wait", str(cf_manual_wait),
        "--captcha-manual-wait", str(captcha_manual_wait),
        "--captcha-retries", str(captcha_retries),
        "--navigation-retries", str(navigation_retries),
    ]
    if proxy.strip():
        cmd.extend(["--proxy", proxy.strip()])
    if headless:
        cmd.append("--headless")

    with log_file.open("a", encoding="utf-8") as f:
        f.write(
            f"[{now_iso()}] START worker={worker_id} orgs={orgs} ranges={ranges_text} "
            f"delay={delay} proxy={'on' if proxy.strip() else 'off'}\n"
        )

    stdout = log_file.open("a", encoding="utf-8")
    stderr = subprocess.STDOUT
    return subprocess.Popen(cmd, cwd=str(SCRIPT_DIR), stdout=stdout, stderr=stderr, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 3 Camoufox workers in parallel")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--delay", type=float, default=15.0, help="Base delay per worker (seconds)")
    parser.add_argument("--cf-manual-wait", type=int, default=30, help="Manual Cloudflare wait seconds")
    parser.add_argument("--captcha-manual-wait", type=int, default=30, help="Manual captcha wait seconds")
    parser.add_argument("--captcha-retries", type=int, default=8, help="Captcha OCR retries")
    parser.add_argument("--navigation-retries", type=int, default=4, help="Navigation retries")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR), help="State directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--profiles-root", type=str, default=str(DEFAULT_PROFILES_ROOT), help="Profiles root")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR), help="Parallel log directory")
    parser.add_argument("--proxy-w1", type=str, default="", help="Proxy for worker w1")
    parser.add_argument("--proxy-w2", type=str, default="", help="Proxy for worker w2")
    parser.add_argument("--proxy-w3", type=str, default="", help="Proxy for worker w3")
    args = parser.parse_args()

    state_dir = Path(args.state_dir)
    output_dir = Path(args.output_dir)
    profiles_root = Path(args.profiles_root)
    log_dir = Path(args.log_dir)

    workers_state, source = load_resume_state()
    print(f"Resume source: {source}", flush=True)

    proxies = {"w1": args.proxy_w1, "w2": args.proxy_w2, "w3": args.proxy_w3}
    procs: list[tuple[str, subprocess.Popen]] = []

    for wid in WORKERS:
        pending = worker_pending_tasks(wid, workers_state[wid])
        if not pending:
            print(f"[{wid}] completed in state, skip launch.", flush=True)
            continue

        orgs, ranges_text = build_orgs_ranges_for_worker(pending)
        print(f"[{wid}] launch orgs={orgs} ranges={ranges_text}", flush=True)
        proc = launch_worker(
            worker_id=wid,
            orgs=orgs,
            ranges_text=ranges_text,
            python_bin=args.python_bin,
            delay=args.delay,
            cf_manual_wait=args.cf_manual_wait,
            captcha_manual_wait=args.captcha_manual_wait,
            captcha_retries=args.captcha_retries,
            navigation_retries=args.navigation_retries,
            proxy=proxies[wid],
            state_dir=state_dir,
            output_dir=output_dir,
            profiles_root=profiles_root,
            headless=bool(args.headless),
            log_dir=log_dir,
        )
        if proc is not None:
            procs.append((wid, proc))

    if not procs:
        print("No worker launched. Everything appears completed from current state.")
        return 0

    rc = 0
    for wid, proc in procs:
        code = proc.wait()
        print(f"[{wid}] exited rc={code}", flush=True)
        if code != 0 and rc == 0:
            rc = code

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
