"""
08_round_robin_camoufox.py
--------------------------
Sequential round-robin runner for 3 Camoufox workers.

State migration priority:
1) state/camoufox_rr3_state.json
2) state/scrapling_rr3_state.json
3) state/rr3_workers_state.json
4) fresh init
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CRAWLER_FILE = SCRIPT_DIR / "08_camoufox_crawl_by_org.py"

DEFAULT_STATE_FILE = SCRIPT_DIR / "state" / "camoufox_rr3_state.json"
LEGACY_SCRAPLING_STATE_FILE = SCRIPT_DIR / "state" / "scrapling_rr3_state.json"
LEGACY_RR3_STATE_FILE = SCRIPT_DIR / "state" / "rr3_workers_state.json"

DEFAULT_STATE_DIR = SCRIPT_DIR / "state"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_PROFILES_ROOT = DEFAULT_STATE_DIR / "camoufox_profiles"
DEFAULT_LOG_DIR = SCRIPT_DIR / "logs" / "camoufox_rr"

STATE_LAYOUT = "camoufox_rr3_v1"


@dataclass
class Task:
    org_id: int
    start_page: int
    end_page: int


WORKER_CONFIGS = {
    "w1": {"tasks": [Task(1, 1, 50), Task(1, 101, 150)]},
    "w2": {"tasks": [Task(1, 51, 100), Task(1, 151, 200)]},
    "w3": {"tasks": [
        Task(3, 1, 999), Task(12, 1, 999), Task(6, 1, 999),
        Task(19, 1, 999), Task(22, 1, 999), Task(23, 1, 999),
        Task(26, 1, 999), Task(33, 1, 999), Task(95, 1, 999),
        Task(97, 1, 999), Task(98, 1, 999), Task(104, 1, 999),
    ]},
}
WORKER_IDS = ["w1", "w2", "w3"]

DONE_RE = re.compile(
    r"Done\.\s+completed=(True|False),\s+items=(\d+),\s+links=(\d+),\s+requests=(\d+),",
    re.IGNORECASE,
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def build_worker_order(start_from_worker: int) -> list[str]:
    if not 1 <= start_from_worker <= len(WORKER_IDS):
        raise ValueError(f"--start-from-worker must be in 1..{len(WORKER_IDS)}")
    idx = start_from_worker - 1
    return WORKER_IDS[idx:] + WORKER_IDS[:idx]


def default_worker_state(worker_id: str) -> dict:
    first = WORKER_CONFIGS[worker_id]["tasks"][0]
    return {
        "task_index": 0,
        "next_page": first.start_page,
        "completed": False,
        "total_items": 0,
        "turn_count": 0,
        "empty_streak": 0,
        "empty_task_key": "",
        "updated_at": None,
    }


def normalize_worker_state(worker_id: str, raw: dict | None) -> dict:
    ws = default_worker_state(worker_id)
    if not isinstance(raw, dict):
        return ws

    tasks = WORKER_CONFIGS[worker_id]["tasks"]
    task_count = len(tasks)

    def to_int(name: str, default: int) -> int:
        try:
            return int(raw.get(name, default))
        except Exception:
            return default

    task_index = max(0, min(to_int("task_index", 0), task_count))
    ws["task_index"] = task_index
    ws["completed"] = bool(raw.get("completed", False)) or task_index >= task_count

    total_items = to_int("total_items", to_int("total_docs", 0))
    ws["total_items"] = max(0, total_items)
    ws["turn_count"] = max(0, to_int("turn_count", 0))
    ws["empty_streak"] = max(0, to_int("empty_streak", 0))
    ws["empty_task_key"] = str(raw.get("empty_task_key", "") or "")
    ws["updated_at"] = raw.get("updated_at")

    if ws["completed"]:
        ws["next_page"] = tasks[-1].end_page + 1
        ws["empty_streak"] = 0
        ws["empty_task_key"] = ""
        return ws

    task = tasks[task_index]
    next_page = to_int("next_page", task.start_page)
    next_page = max(task.start_page, min(next_page, task.end_page + 1))
    ws["next_page"] = next_page
    return ws


def migrate_state_from(raw: dict | None, source: str) -> dict:
    workers = {}
    if isinstance(raw, dict):
        workers = raw.get("workers", {}) or {}

    state = {
        "layout": STATE_LAYOUT,
        "updated_at": now_iso(),
        "migrated_from": source,
        "workers": {},
    }
    for wid in WORKER_IDS:
        state["workers"][wid] = normalize_worker_state(wid, workers.get(wid))
    return state


def load_or_init_state(state_file: Path) -> dict:
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if state_file.exists():
        try:
            raw = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        state = migrate_state_from(raw, "camoufox_current")
        save_state(state_file, state)
        return state

    if LEGACY_SCRAPLING_STATE_FILE.exists():
        try:
            raw = json.loads(LEGACY_SCRAPLING_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        state = migrate_state_from(raw, "scrapling_rr3_state")
        save_state(state_file, state)
        return state

    if LEGACY_RR3_STATE_FILE.exists():
        try:
            raw = json.loads(LEGACY_RR3_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        state = migrate_state_from(raw, "rr3_workers_state")
        save_state(state_file, state)
        return state

    state = {
        "layout": STATE_LAYOUT,
        "updated_at": now_iso(),
        "migrated_from": "fresh_init",
        "workers": {wid: default_worker_state(wid) for wid in WORKER_IDS},
    }
    save_state(state_file, state)
    return state


def save_state(state_file: Path, state: dict) -> None:
    state["updated_at"] = now_iso()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_task(worker_id: str, worker_state: dict) -> Task | None:
    idx = int(worker_state["task_index"])
    tasks = WORKER_CONFIGS[worker_id]["tasks"]
    if idx >= len(tasks):
        return None
    return tasks[idx]


def advance_task(worker_id: str, worker_state: dict) -> None:
    worker_state["task_index"] = int(worker_state["task_index"]) + 1
    worker_state["empty_streak"] = 0
    worker_state["empty_task_key"] = ""
    nxt = get_task(worker_id, worker_state)
    if nxt is None:
        worker_state["completed"] = True
    else:
        worker_state["completed"] = False
        worker_state["next_page"] = nxt.start_page


def parse_done_metrics(output_text: str) -> tuple[int, int, int]:
    m = DONE_RE.search(output_text)
    if not m:
        return -1, -1, -1
    return int(m.group(2)), int(m.group(3)), int(m.group(4))


def append_worker_log(log_dir: Path, worker_id: str, text: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / f"{worker_id}.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_one_page(
    *,
    python_bin: str,
    worker_id: str,
    task: Task,
    page: int,
    delay: float,
    cf_manual_wait: int,
    captcha_manual_wait: int,
    captcha_retries: int,
    navigation_retries: int,
    proxy: str,
    profile_dir: Path,
    output_dir: Path,
    state_dir: Path,
    headless: bool,
) -> tuple[int, int, int, int, str]:
    cmd = [
        python_bin,
        str(CRAWLER_FILE),
        "--worker", worker_id,
        "--orgs", str(task.org_id),
        "--ranges", f"{task.start_page}-{task.end_page}",
        "--single-page", str(page),
        "--delay", str(delay),
        "--cf-manual-wait", str(cf_manual_wait),
        "--captcha-manual-wait", str(captcha_manual_wait),
        "--captcha-retries", str(captcha_retries),
        "--navigation-retries", str(navigation_retries),
        "--profile-dir", str(profile_dir),
        "--output-dir", str(output_dir),
        "--state-dir", str(state_dir),
    ]

    if proxy.strip():
        cmd.extend(["--proxy", proxy.strip()])
    if headless:
        cmd.append("--headless")

    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    items, links, requests = parse_done_metrics(out)
    return proc.returncode, items, links, requests, out


def run_worker_turn(
    *,
    worker_id: str,
    worker_state: dict,
    pages_per_turn: int,
    python_bin: str,
    delay: float,
    cf_manual_wait: int,
    captcha_manual_wait: int,
    captcha_retries: int,
    navigation_retries: int,
    proxy: str,
    profile_dir: Path,
    output_dir: Path,
    state_dir: Path,
    log_dir: Path,
    headless: bool,
) -> None:
    if worker_state.get("completed"):
        print(f"[{worker_id}] completed, skip.")
        return

    pages_done = 0
    items_done = 0

    while pages_done < pages_per_turn and not worker_state.get("completed"):
        task = get_task(worker_id, worker_state)
        if task is None:
            worker_state["completed"] = True
            break

        page = int(worker_state["next_page"])
        if page > task.end_page:
            print(f"[{worker_id}] finish task org={task.org_id} range={task.start_page}-{task.end_page}")
            advance_task(worker_id, worker_state)
            continue

        print(
            f"[{worker_id}] crawl org={task.org_id} page={page} "
            f"(task {worker_state['task_index'] + 1}/{len(WORKER_CONFIGS[worker_id]['tasks'])})"
        )
        rc, items, links, requests, out = run_one_page(
            python_bin=python_bin,
            worker_id=worker_id,
            task=task,
            page=page,
            delay=delay,
            cf_manual_wait=cf_manual_wait,
            captcha_manual_wait=captcha_manual_wait,
            captcha_retries=captcha_retries,
            navigation_retries=navigation_retries,
            proxy=proxy,
            profile_dir=profile_dir,
            output_dir=output_dir,
            state_dir=state_dir,
            headless=headless,
        )

        log_text = (
            f"[{now_iso()}] worker={worker_id} org={task.org_id} page={page} "
            f"rc={rc} items={items} links={links} requests={requests}\n"
        )
        append_worker_log(log_dir, worker_id, log_text)

        if rc != 0:
            print(f"[{worker_id}] page={page} failed (rc={rc}), keep page for retry next turn.")
            append_worker_log(log_dir, worker_id, out[-4000:])
            break

        worker_state["next_page"] = page + 1
        pages_done += 1
        if items > 0:
            items_done += items

        task_key = f"{task.org_id}:{task.start_page}:{task.end_page}"
        if links == 0:
            if worker_state.get("empty_task_key") != task_key:
                worker_state["empty_task_key"] = task_key
                worker_state["empty_streak"] = 0
            worker_state["empty_streak"] = int(worker_state.get("empty_streak", 0)) + 1
            if int(worker_state["empty_streak"]) >= 2:
                print(f"[{worker_id}] org={task.org_id} got 2 empty pages, move to next task.")
                advance_task(worker_id, worker_state)
                continue
        else:
            worker_state["empty_streak"] = 0
            worker_state["empty_task_key"] = ""

    worker_state["turn_count"] = int(worker_state.get("turn_count", 0)) + 1
    worker_state["total_items"] = int(worker_state.get("total_items", 0)) + items_done
    worker_state["updated_at"] = now_iso()
    print(f"[{worker_id}] turn done: pages={pages_done}, items={items_done}")


def all_completed(state: dict) -> bool:
    return all(state["workers"][wid].get("completed") for wid in WORKER_IDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sequential round-robin runner for Camoufox workers")
    parser.add_argument("--pages-per-turn", type=int, default=10, help="Pages per worker per turn")
    parser.add_argument("--delay-between-workers", type=float, default=2.0, help="Sleep between workers (seconds)")
    parser.add_argument("--delay", type=float, default=18.0, help="Base delay passed to crawler")
    parser.add_argument("--start-from-worker", type=int, default=1, help="1..3")
    parser.add_argument("--once", action="store_true", help="Run only one full cycle then stop")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter path")
    parser.add_argument("--cf-manual-wait", type=int, default=30, help="Manual CF wait seconds")
    parser.add_argument("--captcha-manual-wait", type=int, default=30, help="Manual captcha wait seconds")
    parser.add_argument("--captcha-retries", type=int, default=8, help="Captcha OCR retries")
    parser.add_argument("--navigation-retries", type=int, default=4, help="Navigation retries")
    parser.add_argument("--headless", action="store_true", help="Run Camoufox workers headless")
    parser.add_argument("--state-file", type=str, default=str(DEFAULT_STATE_FILE), help="Round-robin state file")
    parser.add_argument("--state-dir", type=str, default=str(DEFAULT_STATE_DIR), help="Crawler state directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Crawler output directory")
    parser.add_argument("--profiles-root", type=str, default=str(DEFAULT_PROFILES_ROOT), help="Camoufox profiles root")
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR), help="Worker log directory")
    parser.add_argument("--proxy-w1", type=str, default="", help="Optional proxy for worker w1")
    parser.add_argument("--proxy-w2", type=str, default="", help="Optional proxy for worker w2")
    parser.add_argument("--proxy-w3", type=str, default="", help="Optional proxy for worker w3")
    args = parser.parse_args()

    if args.pages_per_turn <= 0:
        print("--pages-per-turn must be > 0")
        return 2

    try:
        order = build_worker_order(args.start_from_worker)
    except ValueError as exc:
        print(exc)
        return 2

    state_file = Path(args.state_file)
    state_dir = Path(args.state_dir)
    output_dir = Path(args.output_dir)
    profiles_root = Path(args.profiles_root)
    log_dir = Path(args.log_dir)

    state = load_or_init_state(state_file)
    save_state(state_file, state)

    proxy_map = {"w1": args.proxy_w1, "w2": args.proxy_w2, "w3": args.proxy_w3}

    print(
        "START RR CAMOUFOX: order={} pages_per_turn={} delay={} cf_wait={}".format(
            "->".join(order), args.pages_per_turn, args.delay, args.cf_manual_wait
        ),
        flush=True,
    )
    print(f"State file: {state_file}", flush=True)

    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== Cycle {cycle} ===", flush=True)

        for wid in order:
            ws = state["workers"][wid]
            run_worker_turn(
                worker_id=wid,
                worker_state=ws,
                pages_per_turn=args.pages_per_turn,
                python_bin=args.python_bin,
                delay=args.delay,
                cf_manual_wait=args.cf_manual_wait,
                captcha_manual_wait=args.captcha_manual_wait,
                captcha_retries=args.captcha_retries,
                navigation_retries=args.navigation_retries,
                proxy=proxy_map[wid],
                profile_dir=profiles_root / wid,
                output_dir=output_dir,
                state_dir=state_dir,
                log_dir=log_dir,
                headless=bool(args.headless),
            )
            save_state(state_file, state)
            time.sleep(max(0.0, args.delay_between_workers))

        if args.once:
            print("Stop after one cycle (--once).", flush=True)
            break
        if all_completed(state):
            print("All workers completed. Exit.", flush=True)
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
