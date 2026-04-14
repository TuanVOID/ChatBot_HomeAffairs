"""
06_round_robin_crawl.py - Crawl luan phien 3 workers de giam captcha.

Layout moi:
- Worker 1 + Worker 2: om toan bo Co quan TW (chia 4 range)
- Worker 3: om 12 co quan con lai

Co ho tro migrate state cu (6 workers) de tai su dung tien do w1/w2.
"""

import argparse
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CRAWLER_FILE = SCRIPT_DIR / "05_crawl_by_org.py"
RR_STATE_FILE = SCRIPT_DIR / "state" / "rr3_workers_state.json"
LEGACY_RR_STATE_FILE = SCRIPT_DIR / "state" / "rr_workers_state.json"
RR_STATE_LAYOUT = "3workers_v1"


spec = importlib.util.spec_from_file_location("crawl_by_org_mod", str(CRAWLER_FILE))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Khong the nap module: {CRAWLER_FILE}")

crawl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crawl_mod)


@dataclass
class Task:
    org_id: int
    start_page: int
    end_page: int


WORKER_CONFIGS = {
    # Tai su dung tien do cu cua worker 1 (p001) roi tiep tuc phan 101-150
    "w1": {"port": 9222, "tasks": [Task(1, 1, 50), Task(1, 101, 150)]},
    # Tai su dung tien do cu cua worker 2 (p051) roi tiep tuc phan 151-200
    "w2": {"port": 9223, "tasks": [Task(1, 51, 100), Task(1, 151, 200)]},
    # Gom toan bo 12 co quan con lai
    "w3": {"port": 9224, "tasks": [
        Task(3, 1, 999), Task(12, 1, 999), Task(6, 1, 999),
        Task(19, 1, 999), Task(22, 1, 999), Task(23, 1, 999),
        Task(26, 1, 999), Task(33, 1, 999), Task(95, 1, 999),
        Task(97, 1, 999), Task(98, 1, 999), Task(104, 1, 999),
    ]},
}

WORKER_IDS = ["w1", "w2", "w3"]


def build_worker_order(start_from_worker: int) -> list[str]:
    """Tra ve thu tu worker vong tron, bat dau tu worker duoc chi dinh."""
    if not 1 <= start_from_worker <= len(WORKER_IDS):
        raise ValueError(f"--start-from-worker phai trong khoang 1..{len(WORKER_IDS)}")

    start_idx = start_from_worker - 1
    return WORKER_IDS[start_idx:] + WORKER_IDS[:start_idx]


def _default_worker_state(worker_id: str) -> dict:
    first_task = WORKER_CONFIGS[worker_id]["tasks"][0]
    return {
        "task_index": 0,
        "next_page": first_task.start_page,
        "completed": False,
        "total_docs": 0,
        "empty_streak": 0,
        "empty_task_key": "",
        "updated_at": None,
    }


def _normalize_worker_state(worker_id: str, raw_state: dict | None) -> dict:
    ws = _default_worker_state(worker_id)
    if not isinstance(raw_state, dict):
        return ws

    task_count = len(WORKER_CONFIGS[worker_id]["tasks"])

    try:
        task_index = int(raw_state.get("task_index", ws["task_index"]))
    except (TypeError, ValueError):
        task_index = ws["task_index"]
    task_index = max(0, min(task_index, task_count))

    try:
        total_docs = int(raw_state.get("total_docs", 0))
    except (TypeError, ValueError):
        total_docs = 0
    total_docs = max(0, total_docs)

    try:
        empty_streak = int(raw_state.get("empty_streak", 0))
    except (TypeError, ValueError):
        empty_streak = 0
    empty_streak = max(0, empty_streak)

    ws["task_index"] = task_index
    ws["total_docs"] = total_docs
    ws["empty_streak"] = empty_streak
    ws["empty_task_key"] = str(raw_state.get("empty_task_key", "") or "")
    ws["updated_at"] = raw_state.get("updated_at")

    if task_index >= task_count:
        ws["completed"] = True
        ws["next_page"] = WORKER_CONFIGS[worker_id]["tasks"][-1].end_page + 1
        ws["empty_streak"] = 0
        ws["empty_task_key"] = ""
        return ws

    ws["completed"] = False
    task = WORKER_CONFIGS[worker_id]["tasks"][task_index]

    try:
        next_page = int(raw_state.get("next_page", task.start_page))
    except (TypeError, ValueError):
        next_page = task.start_page

    if next_page < task.start_page:
        next_page = task.start_page
    if next_page > task.end_page + 1:
        next_page = task.end_page + 1

    ws["next_page"] = next_page
    return ws


def _migrate_from_legacy_state(legacy_state: dict | None) -> dict:
    """Migrate tu state 6-worker cu, chi giu tien do w1/w2; reset w3 moi."""
    legacy_workers = {}
    if isinstance(legacy_state, dict):
        legacy_workers = legacy_state.get("workers", {}) or {}

    migrated = {
        "layout": RR_STATE_LAYOUT,
        "migrated_from": "legacy_6workers",
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "workers": {},
    }

    migrated["workers"]["w1"] = _normalize_worker_state("w1", legacy_workers.get("w1", {}))
    migrated["workers"]["w2"] = _normalize_worker_state("w2", legacy_workers.get("w2", {}))
    migrated["workers"]["w3"] = _default_worker_state("w3")
    return migrated


def save_rr_state(state: dict):
    RR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RR_STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def ensure_rr_state() -> dict:
    RR_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if RR_STATE_FILE.exists():
        with RR_STATE_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)

        state = {"layout": RR_STATE_LAYOUT, "workers": {}}
        workers = loaded.get("workers", {}) if isinstance(loaded, dict) else {}
        for worker_id in WORKER_IDS:
            state["workers"][worker_id] = _normalize_worker_state(worker_id, workers.get(worker_id, {}))
        save_rr_state(state)
        return state

    if LEGACY_RR_STATE_FILE.exists():
        with LEGACY_RR_STATE_FILE.open("r", encoding="utf-8") as f:
            legacy = json.load(f)
        state = _migrate_from_legacy_state(legacy)
        save_rr_state(state)
        crawl_mod.log.info("🧩 Da migrate state cu sang %s", RR_STATE_FILE.name)
        return state

    state = {"layout": RR_STATE_LAYOUT, "workers": {}}
    for worker_id in WORKER_IDS:
        state["workers"][worker_id] = _default_worker_state(worker_id)
    save_rr_state(state)
    return state


def get_worker_task(worker_id: str, worker_state: dict) -> Task | None:
    tasks = WORKER_CONFIGS[worker_id]["tasks"]
    idx = worker_state["task_index"]
    if idx >= len(tasks):
        return None
    return tasks[idx]


def advance_task(worker_id: str, worker_state: dict):
    worker_state["task_index"] += 1
    worker_state["empty_streak"] = 0
    worker_state["empty_task_key"] = ""

    next_task = get_worker_task(worker_id, worker_state)
    if next_task is None:
        worker_state["completed"] = True
        return
    worker_state["next_page"] = next_task.start_page
    worker_state["completed"] = False


def process_one_page(driver, worker_id: str, task: Task, page: int, max_docs_per_page: int) -> int:
    suffix = crawl_mod.get_file_suffix(task.start_page, task.end_page)
    state_file = crawl_mod.get_state_file(task.org_id, suffix)
    output_file = crawl_mod.get_output_file(task.org_id, suffix)

    urls = crawl_mod.collect_urls(driver, task.org_id, start_page=page, end_page=page)
    if not urls:
        crawl_mod.log.info("[%s] org=%d page=%d: khong co URL moi", worker_id, task.org_id, page)
        return -1

    state = crawl_mod.load_state(state_file)
    visited_set = set(state.get("visited_urls", []))
    crawled_urls = crawl_mod.get_crawled_urls(output_file)
    pending = [u for u in urls if u not in visited_set and u not in crawled_urls]

    if not pending:
        crawl_mod.log.info("[%s] org=%d page=%d: URL da crawl het", worker_id, task.org_id, page)
        return 0

    written = 0
    with open(output_file, "a", encoding="utf-8") as out_f:
        for url in pending[:max_docs_per_page]:
            doc = crawl_mod.scrape_document(driver, url)
            state.setdefault("visited_urls", []).append(url)

            if doc and doc.get("content"):
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                written += 1

        out_f.flush()

    crawl_mod.save_state(state_file, state)
    return written


def run_worker_turn(worker_id: str, driver, worker_state: dict, pages_per_turn: int, max_docs_per_page: int):
    if worker_state.get("completed"):
        crawl_mod.log.info("[%s] da hoan thanh", worker_id)
        return

    if worker_state["task_index"] >= len(WORKER_CONFIGS[worker_id]["tasks"]):
        worker_state["completed"] = True
        return

    pages_done = 0
    docs_written = 0

    while pages_done < pages_per_turn and not worker_state.get("completed"):
        task = get_worker_task(worker_id, worker_state)
        if task is None:
            worker_state["completed"] = True
            break

        page = worker_state["next_page"]
        if page > task.end_page:
            crawl_mod.log.info("[%s] org=%d da xong range %d-%d",
                               worker_id, task.org_id, task.start_page, task.end_page)
            advance_task(worker_id, worker_state)
            continue

        crawl_mod.log.info("[%s] Luot page org=%d page=%d", worker_id, task.org_id, page)
        result = process_one_page(driver, worker_id, task, page, max_docs_per_page)

        if result == -1:
            task_key = f"{task.org_id}:{task.start_page}:{task.end_page}"
            if worker_state.get("empty_task_key") != task_key:
                worker_state["empty_task_key"] = task_key
                worker_state["empty_streak"] = 0

            worker_state["empty_streak"] = worker_state.get("empty_streak", 0) + 1
            worker_state["next_page"] = page + 1
            pages_done += 1

            # 2 trang rong lien tiep moi ket luan task da het ket qua
            if worker_state["empty_streak"] >= 2:
                crawl_mod.log.info("[%s] org=%d ket thuc som (2 trang rong lien tiep, den page=%d)",
                                   worker_id, task.org_id, page)
                advance_task(worker_id, worker_state)
            continue

        worker_state["empty_streak"] = 0
        worker_state["empty_task_key"] = ""
        docs_written += max(result, 0)
        worker_state["next_page"] = page + 1
        pages_done += 1

        time.sleep(1.0)

    worker_state["total_docs"] = worker_state.get("total_docs", 0) + docs_written
    worker_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    crawl_mod.log.info("[%s] Ket thuc luot: pages=%d, docs=%d", worker_id, pages_done, docs_written)


def all_completed(state: dict) -> bool:
    for worker_id in WORKER_IDS:
        if not state["workers"][worker_id].get("completed"):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Round-robin crawl 3 workers")
    parser.add_argument("--pages-per-turn", type=int, default=10, help="So trang moi worker moi luot")
    parser.add_argument("--delay-between-workers", type=float, default=2.0, help="Nghi giua 2 worker")
    parser.add_argument("--max-docs-per-page", type=int, default=999, help="Gioi han VB moi trang")
    parser.add_argument("--delay", type=float, default=8.0, help="Delay crawl goc (gan cho 05_crawl_by_org)")
    parser.add_argument("--once", action="store_true", help="Chi chay 1 vong w1->w3 roi dung")
    parser.add_argument(
        "--start-from-worker",
        type=int,
        default=1,
        help="Worker bat dau vong (1..3). Vi du 2 => thu tu 2-3-1",
    )
    args = parser.parse_args()

    try:
        worker_order = build_worker_order(args.start_from_worker)
    except ValueError as e:
        parser.error(str(e))
        return

    crawl_mod.DELAY = args.delay
    state = ensure_rr_state()

    drivers = {}
    for worker_id in WORKER_IDS:
        port = WORKER_CONFIGS[worker_id]["port"]
        drivers[worker_id] = crawl_mod.make_driver(port)

    round_no = 0
    try:
        while True:
            round_no += 1
            crawl_mod.log.info("=" * 70)
            crawl_mod.log.info("ROUND %d - Round Robin %s", round_no, " -> ".join(worker_order).upper())
            crawl_mod.log.info("=" * 70)

            for worker_id in worker_order:
                worker_state = state["workers"][worker_id]
                run_worker_turn(
                    worker_id,
                    drivers[worker_id],
                    worker_state,
                    pages_per_turn=args.pages_per_turn,
                    max_docs_per_page=args.max_docs_per_page,
                )
                save_rr_state(state)
                time.sleep(args.delay_between_workers)

            if args.once:
                crawl_mod.log.info("--once bat: dung sau 1 vong")
                break

            if all_completed(state):
                crawl_mod.log.info("Tat ca workers da hoan thanh")
                break

    except KeyboardInterrupt:
        crawl_mod.log.info("Dung boi Ctrl+C, da luu state")
    finally:
        save_rr_state(state)
        crawl_mod.log.info("State da luu: %s", RR_STATE_FILE)


if __name__ == "__main__":
    main()
