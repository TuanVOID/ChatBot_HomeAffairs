"""
07_round_robin_scrapling.py
---------------------------
Run Scrapling workers sequentially in round-robin mode (one worker at a time).

Why:
- Reduce Cloudflare pressure compared with running 3 workers in parallel.
- Keep per-worker progress in a dedicated RR state file.

Each turn:
- Worker runs N listing pages (default 10) then yields to next worker.
- Worker state is saved after every turn (resume-safe).
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
SCRAPER_FILE = SCRIPT_DIR / "07_scrapling_crawl_by_org.py"
RR_STATE_FILE = SCRIPT_DIR / "state" / "scrapling_rr3_state.json"
LOG_DIR = SCRIPT_DIR / "logs" / "scrapling_rr"


@dataclass
class Task:
    org_id: int
    start_page: int
    end_page: int


WORKER_CONFIGS = {
    "w1": {"port": 9222, "tasks": [Task(1, 1, 50), Task(1, 101, 150)]},
    "w2": {"port": 9223, "tasks": [Task(1, 51, 100), Task(1, 151, 200)]},
    "w3": {"port": 9224, "tasks": [
        Task(3, 1, 999), Task(12, 1, 999), Task(6, 1, 999),
        Task(19, 1, 999), Task(22, 1, 999), Task(23, 1, 999),
        Task(26, 1, 999), Task(33, 1, 999), Task(95, 1, 999),
        Task(97, 1, 999), Task(98, 1, 999), Task(104, 1, 999),
    ]},
}
WORKER_IDS = ["w1", "w2", "w3"]

DONE_RE = re.compile(
    r"Done\.\s+completed=(True|False),\s+paused=(True|False),\s+items=(\d+),\s+requests=(\d+),",
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
    ws["total_items"] = max(0, to_int("total_items", 0))
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


def load_state() -> dict:
    RR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RR_STATE_FILE.exists():
        try:
            data = json.loads(RR_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    workers = data.get("workers", {}) if isinstance(data, dict) else {}
    state = {
        "layout": "scrapling_rr3_v1",
        "updated_at": now_iso(),
        "workers": {},
    }
    for wid in WORKER_IDS:
        state["workers"][wid] = normalize_worker_state(wid, workers.get(wid))
    return state


def save_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    RR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RR_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


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


def parse_done_metrics(output_text: str) -> tuple[int, int]:
    """
    Returns (items, requests). If not found, returns (-1, -1).
    """
    m = DONE_RE.search(output_text)
    if not m:
        return -1, -1
    return int(m.group(3)), int(m.group(4))


def run_one_page(
    *,
    python_bin: str,
    worker_id: str,
    port: int,
    org_id: int,
    page: int,
    delay: float,
    cf_manual_wait: int,
    max_blocked_retries: int,
    dynamic: bool,
    force_proxy_context: bool,
    proxy: str,
) -> tuple[int, int, int]:
    cmd = [
        python_bin,
        str(SCRAPER_FILE),
        "--worker", worker_id,
        "--port", str(port),
        "--orgs", str(org_id),
        "--ranges", f"{page}-{page}",
        "--delay", str(delay),
        "--cf-manual-wait", str(cf_manual_wait),
        "--max-blocked-retries", str(max_blocked_retries),
    ]
    if dynamic:
        cmd.append("--dynamic")
    if force_proxy_context:
        cmd.append("--force-proxy-context")
    if proxy.strip():
        cmd.extend(["--proxy", proxy.strip()])

    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    items, requests = parse_done_metrics(out)
    return proc.returncode, items, requests


def append_worker_log(worker_id: str, text: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / f"{worker_id}.log"
    with p.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def run_worker_turn(
    *,
    worker_id: str,
    worker_state: dict,
    pages_per_turn: int,
    python_bin: str,
    delay: float,
    cf_manual_wait: int,
    max_blocked_retries: int,
    dynamic: bool,
    force_proxy_context: bool,
    proxy: str,
) -> None:
    if worker_state.get("completed"):
        print(f"[{worker_id}] completed, skip.")
        return

    config = WORKER_CONFIGS[worker_id]
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
            f"(task {worker_state['task_index'] + 1}/{len(config['tasks'])})"
        )
        rc, items, requests = run_one_page(
            python_bin=python_bin,
            worker_id=worker_id,
            port=config["port"],
            org_id=task.org_id,
            page=page,
            delay=delay,
            cf_manual_wait=cf_manual_wait,
            max_blocked_retries=max_blocked_retries,
            dynamic=dynamic,
            force_proxy_context=force_proxy_context,
            proxy=proxy,
        )

        log_text = (
            f"[{now_iso()}] worker={worker_id} org={task.org_id} page={page} "
            f"rc={rc} items={items} requests={requests}\n"
        )
        append_worker_log(worker_id, log_text)

        if rc != 0:
            print(f"[{worker_id}] page={page} failed (rc={rc}), keep page for retry next turn.")
            break

        # Success run: advance to next page
        worker_state["next_page"] = page + 1
        pages_done += 1
        if items > 0:
            items_done += items

        # Empty-page heuristic: if 2 consecutive empty pages in same task, move on.
        task_key = f"{task.org_id}:{task.start_page}:{task.end_page}"
        if items == 0:
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
    parser = argparse.ArgumentParser(description="Sequential round-robin runner for Scrapling workers")
    parser.add_argument("--pages-per-turn", type=int, default=10, help="Pages per worker per turn")
    parser.add_argument("--delay-between-workers", type=float, default=2.0, help="Sleep between workers (seconds)")
    parser.add_argument("--delay", type=float, default=18.0, help="Base delay passed to scraper")
    parser.add_argument("--cf-manual-wait", type=int, default=30, help="Manual CF wait seconds")
    parser.add_argument("--max-blocked-retries", type=int, default=8, help="Blocked retry count")
    parser.add_argument("--start-from-worker", type=int, default=1, help="1..3")
    parser.add_argument("--once", action="store_true", help="Run only one full cycle then stop")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python interpreter path")
    parser.add_argument("--dynamic", action="store_true", default=True, help="Run scraper in dynamic mode")
    parser.add_argument("--force-proxy-context", action="store_true", help="Force proxy context in scraper")
    parser.add_argument("--proxy-w1", type=str, default="", help="Optional proxy for worker w1")
    parser.add_argument("--proxy-w2", type=str, default="", help="Optional proxy for worker w2")
    parser.add_argument("--proxy-w3", type=str, default="", help="Optional proxy for worker w3")
    args = parser.parse_args()

    if args.pages_per_turn <= 0:
        print("--pages-per-turn must be > 0")
        return 2

    order = build_worker_order(args.start_from_worker)
    state = load_state()
    save_state(state)

    proxy_map = {"w1": args.proxy_w1, "w2": args.proxy_w2, "w3": args.proxy_w3}

    print(
        "START RR SCRAPLING: order={} pages_per_turn={} delay={} cf_wait={}".format(
            "->".join(order), args.pages_per_turn, args.delay, args.cf_manual_wait
        ),
        flush=True,
    )

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
                max_blocked_retries=args.max_blocked_retries,
                dynamic=args.dynamic,
                force_proxy_context=args.force_proxy_context,
                proxy=proxy_map[wid],
            )
            save_state(state)
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

