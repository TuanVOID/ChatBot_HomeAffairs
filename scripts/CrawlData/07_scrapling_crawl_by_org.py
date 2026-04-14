"""
07_scrapling_crawl_by_org.py
============================
Scrape TVPL with Scrapling Spider and export JSONL records compatible with this project.

Highlights:
- Uses Scrapling Spider + AsyncDynamicSession
- Connects to existing Chrome via CDP (for persistent profile/cookies)
- Handles TVPL check.aspx captcha with OCR inside page_action
- Stores output as JSONL with fields used by the chatbot pipeline
- Supports pause/resume via Scrapling checkpoint (crawldir)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import pickle
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import anyio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRAPLING_ROOT = PROJECT_ROOT / "03.Scrapling"
if SCRAPLING_ROOT.exists():
    sys.path.insert(0, str(SCRAPLING_ROOT))

try:
    from scrapling.fetchers import AsyncDynamicSession, AsyncStealthySession
    from scrapling.spiders import Request, Response, Spider
except ImportError as exc:
    raise SystemExit(
        "Cannot import Scrapling. Install dependencies first:\n"
        "  pip install \"f:/SpeechToText-indti/ChatBot2_Opus/03.Scrapling[fetchers]\"\n"
        "  scrapling install"
    ) from exc


try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageOps

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    HAS_OCR = True
except Exception:
    HAS_OCR = False


BASE_URL = "https://thuvienphapluat.vn"
SEARCH_TPL = (
    BASE_URL
    + "/page/tim-van-ban.aspx?keyword=&area=0&type=0&status=0&lan=1&org={org}"
    + "&signer=0&match=True&sort=1&bdate=12/04/1946&edate=13/04/2026&page={page}"
)

ORG_MAP = {
    1: {"slug": "co-quan-tw", "name": "Co quan TW"},
    3: {"slug": "bhxh-viet-nam", "name": "BHXH Viet Nam"},
    6: {"slug": "bo-giao-duc-dao-tao", "name": "Bo Giao duc va Dao tao"},
    12: {"slug": "bo-noi-vu", "name": "Bo Noi vu"},
    19: {"slug": "bo-van-hoa-tt-dl", "name": "Bo Van hoa, TT va DL"},
    22: {"slug": "chinh-phu", "name": "Chinh phu"},
    23: {"slug": "chu-tich-nuoc", "name": "Chu tich nuoc"},
    26: {"slug": "quoc-hoi", "name": "Quoc hoi"},
    33: {"slug": "thu-tuong-chinh-phu", "name": "Thu tuong Chinh phu"},
    95: {"slug": "tong-ldld-viet-nam", "name": "Tong LDLD Viet Nam"},
    97: {"slug": "uy-ban-tvqh", "name": "Uy ban TVQH"},
    98: {"slug": "van-phong-chinh-phu", "name": "Van phong Chinh phu"},
    104: {"slug": "bo-dan-toc-ton-giao", "name": "Bo Dan toc va Ton giao"},
}

LISTING_LINK_SELECTORS = [
    "p.vblist > a::attr(href)",
    "p.vblist a::attr(href)",
    ".nqTitle a::attr(href)",
    "a[href*='/van-ban/']::attr(href)",
]

CONTENT_SELECTORS = [
    "div.content1 div.content1",
    "div.content1",
    ".box-content-vb",
    ".toan-van-container",
    "#toanvancontent",
    "article",
]

METADATA_ROW_SELECTORS = [".boxTTVB .item", ".right-col p", ".ttvb .item"]

TRIM_MARKERS = [
    "Luu tru\nGhi chu\nY kien",
    "Bai lien quan:",
    "Hoi dap phap luat",
    "Ban an lien quan",
    "Facebook\nEmail\nIn",
]

CLOUDFLARE_AUTO_ATTEMPTS = 3
CLOUDFLARE_MANUAL_WAIT_SEC = 90
MAX_CHECK_RETRIES = 8
MAX_CF_PAGE_RETRIES = 6


def apply_scrapling_runtime_patches() -> None:
    """
    Patch Scrapling behavior at runtime so this project is stable on Windows
    without requiring direct edits inside 03.Scrapling.

    Patches:
    1) CheckpointManager.save -> use replace() instead of rename()
       to avoid WinError 183 on checkpoint overwrite.
    2) Async CDP sessions -> prefer existing browser context when available,
       so worker requests reuse profile cookies/session.
    """
    log = logging.getLogger(__name__)

    try:
        from scrapling.spiders.checkpoint import CheckpointManager
        from scrapling.engines._browsers import _controllers as ctrl_mod
        from scrapling.engines._browsers import _stealth as stealth_mod
    except Exception as exc:
        log.warning("Khong the nap module noi bo Scrapling de patch runtime: %s", exc)
        return

    if not getattr(CheckpointManager.save, "__tvpl_patch__", False):

        async def patched_checkpoint_save(self, data) -> None:
            await self.crawldir.mkdir(parents=True, exist_ok=True)
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            try:
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
                async with await anyio.open_file(temp_path, "wb") as f:
                    await f.write(serialized)
                # Windows-safe overwrite.
                await temp_path.replace(self._checkpoint_path)
                log.info("Checkpoint saved: %d requests, %d seen URLs", len(data.requests), len(data.seen))
            except Exception:
                if await temp_path.exists():
                    await temp_path.unlink()
                raise

        patched_checkpoint_save.__tvpl_patch__ = True
        CheckpointManager.save = patched_checkpoint_save
        log.info("Applied runtime patch: CheckpointManager.save (replace on Windows)")

    async def _patched_async_start_common(self, async_playwright_factory):
        if self.playwright:
            raise RuntimeError("Session has been already started")

        self.playwright = await async_playwright_factory().start()
        try:
            if self._config.cdp_url:
                self.browser = await self.playwright.chromium.connect_over_cdp(endpoint_url=self._config.cdp_url)
                if not self._config.proxy_rotator and self.browser:
                    if self.browser.contexts and not self._context_options.get("proxy"):
                        self.context = self.browser.contexts[0]
                    else:
                        self.context = await self.browser.new_context(**self._context_options)
            elif self._config.proxy_rotator:
                self.browser = await self.playwright.chromium.launch(**self._browser_options)
            else:
                persistent_options = self._browser_options | self._context_options | {"user_data_dir": self._user_data_dir}
                self.context = await self.playwright.chromium.launch_persistent_context(**persistent_options)

            if self.context:
                self.context = await self._initialize_context(self._config, self.context)
            self._is_alive = True
        except Exception:
            await self.playwright.stop()
            self.playwright = None
            raise

    if not getattr(AsyncDynamicSession.start, "__tvpl_patch__", False):

        async def patched_dynamic_start(self):
            return await _patched_async_start_common(self, ctrl_mod.async_playwright)

        patched_dynamic_start.__tvpl_patch__ = True
        AsyncDynamicSession.start = patched_dynamic_start
        log.info("Applied runtime patch: AsyncDynamicSession.start (reuse CDP context)")

    if not getattr(AsyncStealthySession.start, "__tvpl_patch__", False):

        async def patched_stealth_start(self):
            return await _patched_async_start_common(self, stealth_mod.async_playwright)

        patched_stealth_start.__tvpl_patch__ = True
        AsyncStealthySession.start = patched_stealth_start
        log.info("Applied runtime patch: AsyncStealthySession.start (reuse CDP context)")


def parse_orgs(orgs_text: str) -> list[int]:
    orgs = []
    for part in orgs_text.split(","):
        part = part.strip()
        if not part:
            continue
        org = int(part)
        if org not in ORG_MAP:
            raise ValueError(f"Unknown org id: {org}")
        orgs.append(org)
    if not orgs:
        raise ValueError("Empty --orgs")
    return orgs


def parse_ranges(ranges_text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in ranges_text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid range: {part}")
        start_s, end_s = part.split("-", 1)
        start = int(start_s.strip())
        end = int(end_s.strip())
        if start < 1 or end < start:
            raise ValueError(f"Invalid range: {part}")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("Empty --ranges")
    return ranges


def get_file_suffix(start_page: int, end_page: int) -> str:
    if start_page > 1:
        return f"_p{start_page:03d}"
    if end_page < 999:
        return "_p001"
    return ""


def get_output_file(output_dir: Path, org_id: int, start_page: int, end_page: int) -> Path:
    suffix = get_file_suffix(start_page, end_page)
    slug = ORG_MAP[org_id]["slug"]
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"org_{org_id:03d}_{slug}{suffix}.jsonl"


def normalize_url(url: str) -> str:
    return url.split("#")[0].split("?")[0].rstrip("/")


def load_crawled_urls(output_file: Path) -> set[str]:
    urls: set[str] = set()
    if not output_file.exists():
        return urls
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = row.get("url")
            if isinstance(url, str) and url:
                urls.add(normalize_url(url))
    return urls


def build_ocr_variants(raw_img: Image.Image) -> list[Image.Image]:
    variants: list[Image.Image] = []
    gray = ImageOps.autocontrast(raw_img.convert("L"))
    for scale in (2, 3):
        up = gray.resize((gray.width * scale, gray.height * scale), Image.Resampling.LANCZOS)
        denoise = up.filter(ImageFilter.MedianFilter(size=3))
        for threshold in (90, 110, 130, 150, 170):
            bw = denoise.point(lambda x, t=threshold: 255 if x > t else 0)
            variants.append(bw)
            variants.append(ImageOps.invert(bw))
    return variants


def ocr_captcha_code(raw_img: Image.Image) -> str | None:
    if not HAS_OCR:
        return None

    configs = (
        "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789",
        "--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789",
    )
    score_map: dict[str, int] = {}

    for img in build_ocr_variants(raw_img):
        for cfg in configs:
            text = pytesseract.image_to_string(img, config=cfg).strip()
            digits = re.sub(r"\D", "", text)
            if not digits:
                continue
            if not (5 <= len(digits) <= 8):
                continue
            score = 2 if len(digits) == 6 else 1
            score_map[digits] = score_map.get(digits, 0) + score

    if not score_map:
        return None

    return max(score_map.items(), key=lambda kv: kv[1])[0]


async def tvpl_page_action(page: Any) -> None:
    """Run after each navigation; try Cloudflare first, then solve TVPL captcha."""
    await handle_cloudflare(page)

    if "/check.aspx" not in (page.url or ""):
        return

    for attempt in range(1, 6):
        img = page.locator("img[src*='RegistImage']").first
        if await img.count() == 0:
            return

        png = await img.screenshot()
        code = ocr_captcha_code(Image.open(io.BytesIO(png)))
        if not code:
            await page.wait_for_timeout(700)
            continue

        inp = page.locator("#ctl00_Content_txtSecCode").first
        btn = page.locator("#ctl00_Content_CheckButton").first
        if await inp.count() == 0 or await btn.count() == 0:
            return

        await inp.fill(code)
        await btn.click()
        await page.wait_for_timeout(2800)
        if "/check.aspx" not in (page.url or ""):
            return

        # Try refreshing captcha image before next attempt
        try:
            await img.click(timeout=500)
        except Exception:
            pass
        await page.wait_for_timeout(1000)


async def is_cloudflare_challenge(page: Any) -> bool:
    try:
        title = (await page.title() or "").strip().lower()
    except Exception:
        title = ""

    if "just a moment" in title:
        return True

    try:
        html = (await page.content() or "").lower()
    except Exception:
        html = ""

    if "verifying you are human" in html:
        return True
    if "challenges.cloudflare.com" in html:
        return True
    if "cf-turnstile" in html or "cf_turnstile" in html:
        return True
    return False


async def click_cloudflare_widget(page: Any) -> bool:
    selectors = [
        "iframe[src*='challenges.cloudflare.com']",
        "#cf-turnstile",
        "#cf_turnstile",
        ".cf-turnstile",
        ".main-content p+div>div>div",
    ]
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.count() == 0:
                continue
            box = await loc.bounding_box()
            if box:
                x = box["x"] + max(5, box["width"] * 0.5)
                y = box["y"] + max(5, box["height"] * 0.5)
                await page.mouse.click(x, y, delay=random.randint(90, 180))
                return True
            await loc.click(timeout=1200)
            return True
        except Exception:
            continue
    return False


async def handle_cloudflare(page: Any) -> None:
    if not await is_cloudflare_challenge(page):
        return

    for _ in range(CLOUDFLARE_AUTO_ATTEMPTS):
        clicked = await click_cloudflare_widget(page)
        await page.wait_for_timeout(3500 if clicked else 1800)
        if not await is_cloudflare_challenge(page):
            return

    # Fallback: cho phép người dùng tự solve trong tab Chrome hiện tại
    if CLOUDFLARE_MANUAL_WAIT_SEC <= 0:
        return

    logging.getLogger(__name__).warning(
        "Cloudflare challenge van ton tai. Cho %ds de manual solve tren browser...",
        CLOUDFLARE_MANUAL_WAIT_SEC,
    )
    for sec in range(CLOUDFLARE_MANUAL_WAIT_SEC):
        if not await is_cloudflare_challenge(page):
            logging.getLogger(__name__).info("Cloudflare da duoc manual solve.")
            return
        if sec > 0 and sec % 10 == 0:
            logging.getLogger(__name__).info("Dang cho manual solve... %ds", sec)
        await page.wait_for_timeout(1000)


def is_cloudflare_response(response: Response) -> bool:
    title = response.css("title::text").get("").strip().lower()
    if "just a moment" in title:
        return True
    body = response.body.decode(response.encoding or "utf-8", errors="ignore").lower()
    return (
        "verifying you are human" in body
        or "challenges.cloudflare.com" in body
        or "cf-turnstile" in body
        or "cf-browser-verification" in body
    )


def extract_title(response: Response) -> str:
    for sel in ("h1::text", ".doc-title::text", ".title::text", "title::text"):
        title = " ".join(response.css(sel).get("").strip().split())
        title = title.replace(" - THƯ VIỆN PHÁP LUẬT", "").replace(" - THU VIEN PHAP LUAT", "")
        title = repair_mojibake(title)
        if title:
            return title
    return ""


def repair_mojibake(text: str) -> str:
    if not text:
        return text

    bad_markers = ("Ã", "Ä", "áº", "á»", "â€", "Æ", "Ð")
    if not any(m in text for m in bad_markers):
        return text

    def score(s: str) -> int:
        good = sum(1 for ch in s if ch in "ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
        bad = sum(s.count(m) for m in bad_markers)
        return good * 3 - bad

    candidates = [text]
    for src in ("latin-1", "cp1252"):
        try:
            candidates.append(text.encode(src, errors="ignore").decode("utf-8", errors="ignore"))
        except Exception:
            continue

    return max(candidates, key=score)


def selector_text(node: Any) -> str:
    parts = [x.strip() for x in node.css("::text").getall() if x and x.strip()]
    return repair_mojibake(" ".join(parts))


def extract_content(response: Response) -> str:
    for sel in CONTENT_SELECTORS:
        nodes = response.css(sel)
        if not nodes:
            continue
        for node in nodes:
            text = selector_text(node)
            if len(text) > 200:
                for marker in TRIM_MARKERS:
                    idx = text.find(marker)
                    if idx > 0:
                        text = text[:idx].rstrip()
                return text
    return ""


def extract_meta(response: Response) -> dict[str, str]:
    meta: dict[str, str] = {}
    labels = [
        ("Loai van ban", "loai_van_ban"),
        ("Loại văn bản", "loai_van_ban"),
        ("So hieu", "so_hieu"),
        ("Số hiệu", "so_hieu"),
        ("Co quan ban hanh", "co_quan"),
        ("Cơ quan ban hành", "co_quan"),
        ("Nguoi ky", "nguoi_ky"),
        ("Người ký", "nguoi_ky"),
        ("Ngay ban hanh", "ngay_ban_hanh"),
        ("Ngày ban hành", "ngay_ban_hanh"),
        ("Ngay hieu luc", "ngay_hieu_luc"),
        ("Ngày hiệu lực", "ngay_hieu_luc"),
        ("Tinh trang", "tinh_trang"),
        ("Tình trạng", "tinh_trang"),
        ("Linh vuc", "linh_vuc"),
        ("Lĩnh vực", "linh_vuc"),
    ]
    for sel in METADATA_ROW_SELECTORS:
        rows = response.css(sel)
        if not rows:
            continue
        for row in rows:
            text = selector_text(row)
            if not text:
                continue
            for label, key in labels:
                if label in text:
                    val = text.replace(label, "").strip().lstrip(":").strip()
                    if val:
                        meta[key] = val
        if meta:
            return meta
    return meta


def infer_document_number(title: str, content: str, url: str) -> str:
    patterns = [
        r"\bSố[:\s]+([0-9][0-9A-Za-zÀ-ỹĐđ/\-\.]{2,60})",
        r"\bSo[:\s]+([0-9][0-9A-Za-z/\-\.]{2,60})",
    ]
    for pat in patterns:
        m = re.search(pat, content, flags=re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" .;:,")
            if re.search(r"\d", cand):
                return cand

    m = re.search(r"\b(\d{1,5}/[^\s]{1,40})", title)
    if m:
        cand = m.group(1).strip(" .;:,")
        if re.search(r"\d", cand) and "/" in cand:
            return cand

    m = re.search(r"/([A-Za-z0-9\-]+)-\d+\.aspx$", url)
    if m:
        token = m.group(1)
        m2 = re.search(r"(\d{1,5}-[A-Za-z0-9\-]{2,60})", token)
        if m2:
            return m2.group(1)
    return ""


class TVPLScraplingSpider(Spider):
    name = "tvpl_scrapling"
    allowed_domains = {"thuvienphapluat.vn"}
    concurrent_requests = 1
    concurrent_requests_per_domain = 1
    max_blocked_retries = 3
    robots_txt_obey = False
    logging_level = logging.INFO
    logging_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    logging_date_format = "%H:%M:%S"

    def __init__(
        self,
        worker: str,
        cdp_url: str,
        org_ids: list[int],
        ranges: list[tuple[int, int]],
        delay: float,
        proxy: str | None,
        use_stealth: bool,
        output_dir: Path,
        crawldir: Path,
        max_blocked_retries: int,
    ):
        self.name = f"tvpl_scrapling_{worker}"
        self.worker = worker
        self.cdp_url = cdp_url
        self.org_ids = org_ids
        self.ranges = ranges
        self.base_delay = delay
        self.proxy = proxy
        self.use_stealth = use_stealth
        self.max_blocked_retries = max(1, int(max_blocked_retries))
        self.download_delay = max(0.5, delay)
        self.output_dir = output_dir
        self.crawldir_path = crawldir
        self._written = 0
        self._outputs: dict[tuple[int, int, int], Path] = {}
        self._seen_urls: dict[str, set[str]] = {}

        for org_id in self.org_ids:
            for start_page, end_page in self.ranges:
                out = get_output_file(self.output_dir, org_id, start_page, end_page)
                key = str(out)
                self._outputs[(org_id, start_page, end_page)] = out
                self._seen_urls[key] = load_crawled_urls(out)

        super().__init__(crawldir=self.crawldir_path, interval=120.0)

    def _request_wait_ms(self) -> int:
        low = max(1000, int((self.base_delay - 2) * 1000))
        high = int((self.base_delay + 2) * 1000)
        if high < low:
            high = low
        return random.randint(low, high)

    def configure_sessions(self, manager) -> None:
        session_kwargs = dict(
            cdp_url=self.cdp_url,
            proxy=self.proxy,
            timeout=65000,
            retries=2,
            retry_delay=1,
            network_idle=False,
            load_dom=True,
            google_search=False,
            wait=700,
            wait_selector="body",
            page_action=tvpl_page_action,
            max_pages=1,
        )

        if self.use_stealth:
            session = AsyncStealthySession(
                **session_kwargs,
                solve_cloudflare=False,
                block_webrtc=True,
                hide_canvas=True,
            )
        else:
            session = AsyncDynamicSession(**session_kwargs)

        mode = "stealth" if self.use_stealth else "dynamic"
        self.logger.info(
            "[W%s] session=%s proxy=%s max_blocked_retries=%s",
            self.worker,
            mode,
            "on" if self.proxy else "off",
            self.max_blocked_retries,
        )

        manager.add("browser", session, default=True)

    async def parse(self, response: Response):
        # Fallback callback required by Spider ABC; route to listing parser.
        async for item in self.parse_listing(response):
            yield item

    async def start_requests(self) -> AsyncGenerator[Request, None]:
        for org_id in self.org_ids:
            for start_page, end_page in self.ranges:
                page = start_page
                url = SEARCH_TPL.format(org=org_id, page=page)
                output_file = self._outputs[(org_id, start_page, end_page)]
                yield Request(
                    url=url,
                    sid="browser",
                    callback=self.parse_listing,
                    meta={
                        "org_id": org_id,
                        "page": page,
                        "start_page": start_page,
                        "end_page": end_page,
                        "output_file": str(output_file),
                    },
                    wait=self._request_wait_ms(),
                )

    async def is_blocked(self, response: Response) -> bool:
        if "/check.aspx" in response.url:
            return True
        return response.status in {401, 403, 407, 429, 500, 502, 503, 504}

    async def retry_blocked_request(self, request: Request, response: Response) -> Request:
        retry = request.copy()
        backoff_sec = self.base_delay + min(14, 2 * retry._retry_count) + random.uniform(0.5, 1.8)
        retry._session_kwargs["wait"] = int(backoff_sec * 1000)
        retry._session_kwargs["page_action"] = tvpl_page_action
        self.logger.warning(
            "[W%s] blocked status=%s retry=%s/%s wait=%.1fs url=%s",
            self.worker,
            response.status,
            retry._retry_count,
            self.max_blocked_retries,
            backoff_sec,
            request.url,
        )
        return retry

    async def parse_listing(self, response: Response):
        org_id = int(response.meta["org_id"])
        page = int(response.meta["page"])
        start_page = int(response.meta["start_page"])
        end_page = int(response.meta["end_page"])
        output_file = str(response.meta["output_file"])
        seen = self._seen_urls.setdefault(output_file, set())
        check_retry = int(response.meta.get("check_retry", 0))
        cf_retry = int(response.meta.get("cf_retry", 0))

        if "/check.aspx" in response.url:
            check_retry += 1
            if check_retry > MAX_CHECK_RETRIES:
                self.logger.error(
                    "[W%s] listing org=%d page=%d bi loop captcha qua %d lan, bo qua.",
                    self.worker,
                    org_id,
                    page,
                    MAX_CHECK_RETRIES,
                )
                return
            self.logger.warning(
                "[W%s] Captcha page while listing org=%d page=%d (retry %d/%d)",
                self.worker,
                org_id,
                page,
                check_retry,
                MAX_CHECK_RETRIES,
            )
            next_meta = response.meta.copy()
            next_meta["check_retry"] = check_retry
            yield Request(
                SEARCH_TPL.format(org=org_id, page=page),
                sid="browser",
                callback=self.parse_listing,
                meta=next_meta,
                dont_filter=True,
                wait=self._request_wait_ms() + min(20000, check_retry * 1500),
            )
            return

        if is_cloudflare_response(response):
            cf_retry += 1
            if cf_retry > MAX_CF_PAGE_RETRIES:
                self.logger.error(
                    "[W%s] listing org=%d page=%d bi Cloudflare qua %d lan, bo qua page.",
                    self.worker,
                    org_id,
                    page,
                    MAX_CF_PAGE_RETRIES,
                )
                return
            self.logger.warning(
                "[W%s] Cloudflare interstitial org=%d page=%d (retry %d/%d)",
                self.worker,
                org_id,
                page,
                cf_retry,
                MAX_CF_PAGE_RETRIES,
            )
            next_meta = response.meta.copy()
            next_meta["cf_retry"] = cf_retry
            yield Request(
                SEARCH_TPL.format(org=org_id, page=page),
                sid="browser",
                callback=self.parse_listing,
                meta=next_meta,
                dont_filter=True,
                wait=self._request_wait_ms() + min(25000, cf_retry * 2500),
            )
            return

        found_urls: list[str] = []
        for sel in LISTING_LINK_SELECTORS:
            hrefs = response.css(sel).getall()
            if not hrefs:
                continue
            for href in hrefs:
                if not href:
                    continue
                abs_url = normalize_url(response.urljoin(href))
                if "/van-ban/" not in abs_url or ".aspx" not in abs_url:
                    continue
                if abs_url not in found_urls:
                    found_urls.append(abs_url)
            if found_urls:
                break

        if found_urls:
            self.logger.info(
                "[W%s] org=%d page=%d -> +%d links", self.worker, org_id, page, len(found_urls)
            )
        else:
            self.logger.info("[W%s] org=%d page=%d -> no new links (stop this range)", self.worker, org_id, page)

        for doc_url in found_urls:
            if doc_url in seen:
                continue
            yield Request(
                doc_url,
                sid="browser",
                callback=self.parse_document,
                meta={
                    "org_id": org_id,
                    "page": page,
                    "start_page": start_page,
                    "end_page": end_page,
                    "output_file": output_file,
                    "doc_url": doc_url,
                },
                wait=self._request_wait_ms(),
            )

        if found_urls and page < end_page:
            next_page = page + 1
            next_url = SEARCH_TPL.format(org=org_id, page=next_page)
            next_meta = response.meta.copy()
            next_meta["page"] = next_page
            yield Request(
                next_url,
                sid="browser",
                callback=self.parse_listing,
                meta=next_meta,
                wait=self._request_wait_ms(),
            )

    async def parse_document(self, response: Response):
        output_file = str(response.meta["output_file"])
        org_id = int(response.meta["org_id"])
        seen = self._seen_urls.setdefault(output_file, set())
        target_url = normalize_url(str(response.meta.get("doc_url") or response.url))
        check_retry = int(response.meta.get("check_retry", 0))
        cf_retry = int(response.meta.get("cf_retry", 0))

        if "/check.aspx" in response.url:
            check_retry += 1
            if check_retry > MAX_CHECK_RETRIES:
                self.logger.error(
                    "[W%s] doc %s loop captcha qua %d lan, bo qua.",
                    self.worker,
                    target_url,
                    MAX_CHECK_RETRIES,
                )
                return
            self.logger.warning(
                "[W%s] Captcha page while opening doc: %s (retry %d/%d)",
                self.worker,
                target_url,
                check_retry,
                MAX_CHECK_RETRIES,
            )
            next_meta = response.meta.copy()
            next_meta["check_retry"] = check_retry
            yield Request(
                target_url,
                sid="browser",
                callback=self.parse_document,
                meta=next_meta,
                dont_filter=True,
                wait=self._request_wait_ms() + min(20000, check_retry * 1500),
            )
            return

        if is_cloudflare_response(response):
            cf_retry += 1
            if cf_retry > MAX_CF_PAGE_RETRIES:
                self.logger.error(
                    "[W%s] doc %s bi Cloudflare qua %d lan, bo qua.",
                    self.worker,
                    target_url,
                    MAX_CF_PAGE_RETRIES,
                )
                return
            self.logger.warning(
                "[W%s] Cloudflare while opening doc: %s (retry %d/%d)",
                self.worker,
                target_url,
                cf_retry,
                MAX_CF_PAGE_RETRIES,
            )
            next_meta = response.meta.copy()
            next_meta["cf_retry"] = cf_retry
            yield Request(
                target_url,
                sid="browser",
                callback=self.parse_document,
                meta=next_meta,
                dont_filter=True,
                wait=self._request_wait_ms() + min(25000, cf_retry * 2500),
            )
            return

        final_url = normalize_url(response.url)
        if final_url in seen:
            return

        title = extract_title(response)
        content = extract_content(response)
        if not content:
            return

        meta = extract_meta(response)
        doc_number = (meta.get("so_hieu") or "").strip()
        if not doc_number:
            doc_number = infer_document_number(title, content, final_url)
            if doc_number:
                meta["so_hieu"] = doc_number

        row: dict[str, Any] = {
            "url": final_url,
            "title": title,
            "content": content,
            "meta": meta,
            "document_number": doc_number,
            "source": "thuvienphapluat",
            "crawl_time": datetime.now().isoformat(timespec="seconds"),
            "_output_file": output_file,
            "_org_id": org_id,
        }
        yield row

    async def on_scraped_item(self, item: dict[str, Any]) -> dict[str, Any] | None:
        output_file = Path(str(item.get("_output_file", "")))
        org_id = item.get("_org_id")
        if not output_file:
            return item

        clean_url = normalize_url(str(item.get("url", "")))
        key = str(output_file)
        seen = self._seen_urls.setdefault(key, set())
        if clean_url in seen:
            return None

        item_to_write = dict(item)
        item_to_write.pop("_output_file", None)
        item_to_write.pop("_org_id", None)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item_to_write, ensure_ascii=False) + "\n")

        seen.add(clean_url)
        self._written += 1
        self.logger.info(
            "[W%s] wrote #%d org=%s -> %s",
            self.worker,
            self._written,
            org_id,
            output_file.name,
        )
        return item_to_write


def build_crawldir(base_state_dir: Path, worker: str, org_ids: list[int], ranges: list[tuple[int, int]]) -> Path:
    org_tag = "-".join(str(x) for x in org_ids)
    range_tag = "_".join(f"{a}-{b}" for a, b in ranges)
    return base_state_dir / "scrapling" / worker / f"orgs_{org_tag}__ranges_{range_tag}"


def resolve_cdp_ws_url(cdp_url: str, port: int) -> str:
    candidate = cdp_url.strip()
    if candidate.startswith("ws://") or candidate.startswith("wss://"):
        return candidate

    if candidate:
        base_http = candidate.rstrip("/")
    else:
        base_http = f"http://127.0.0.1:{port}"

    version_url = f"{base_http}/json/version"
    try:
        with urlopen(version_url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except URLError as exc:
        raise RuntimeError(f"Cannot reach CDP endpoint: {version_url}") from exc
    except Exception as exc:
        raise RuntimeError(f"Cannot read CDP metadata from: {version_url}") from exc

    ws_url = data.get("webSocketDebuggerUrl")
    if not isinstance(ws_url, str) or not ws_url.startswith(("ws://", "wss://")):
        raise RuntimeError(f"Invalid webSocketDebuggerUrl from: {version_url}")
    return ws_url


def parse_proxy_value(proxy_text: str) -> str | None:
    """Accept proxy formats:
    - ip:port:user:password
    - ip:port
    - http://user:pass@ip:port (or socks5://...)
    """
    raw = proxy_text.strip()
    if not raw:
        return None

    if raw.startswith(("http://", "https://", "socks5://", "socks5h://")):
        return raw

    # user:pass@ip:port (without scheme)
    if "@" in raw and "://" not in raw:
        creds, hostport = raw.rsplit("@", 1)
        if ":" not in creds or ":" not in hostport:
            raise ValueError("Invalid --proxy user:pass@ip:port format.")
        user, password = creds.split(":", 1)
        host, port = hostport.rsplit(":", 1)
        user_q = quote(user, safe="")
        pwd_q = quote(password, safe="")
        return f"http://{user_q}:{pwd_q}@{host}:{port}"

    parts = raw.split(":")
    if len(parts) == 2:
        host, port = parts
        return f"http://{host}:{port}"

    if len(parts) == 4:
        host, port, user, password = parts
        user_q = quote(user, safe="")
        pwd_q = quote(password, safe="")
        return f"http://{user_q}:{pwd_q}@{host}:{port}"

    raise ValueError(
        "Invalid --proxy format. Use ip:port:user:password, ip:port, or full proxy URL."
    )


def main() -> int:
    apply_scrapling_runtime_patches()

    parser = argparse.ArgumentParser(description="TVPL crawl using Scrapling spider")
    parser.add_argument("--worker", type=str, default="w1", help="Worker id label, e.g. w1")
    parser.add_argument("--port", type=int, default=9222, help="Chrome remote-debugging port")
    parser.add_argument("--cdp-url", type=str, default="", help="CDP endpoint (if set, overrides --port)")
    parser.add_argument("--orgs", type=str, default="", help="Org ids, comma-separated (e.g. 1,22)")
    parser.add_argument("--ranges", type=str, default="1-999", help="Page ranges, e.g. 1-50,101-150")
    parser.add_argument("--delay", type=float, default=10.0, help="Base delay in seconds")
    parser.add_argument("--proxy", type=str, default="", help="Proxy: ip:port:user:pass or full URL")
    parser.add_argument(
        "--force-proxy-context",
        action="store_true",
        help="When using CDP, force creating a new browser context with proxy (less stable for challenge-heavy sites).",
    )
    parser.add_argument("--dynamic", action="store_true", help="Backward-compatible alias to run in dynamic mode")
    parser.add_argument("--stealth", action="store_true", help="Use AsyncStealthySession (default is dynamic)")
    parser.add_argument(
        "--max-blocked-retries",
        type=int,
        default=6,
        help="Retries for blocked responses (401/403/429/5xx/check.aspx).",
    )
    parser.add_argument(
        "--cf-manual-wait",
        type=int,
        default=90,
        help="Seconds to wait for manual Cloudflare solve before giving up (0=disable).",
    )
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--state-dir", type=str, default=str(Path(__file__).resolve().parent / "state"))
    parser.add_argument("--list", action="store_true", help="List org ids")
    args = parser.parse_args()

    if args.list:
        print("\nOrg list:")
        for oid in sorted(ORG_MAP.keys()):
            print(f"  {oid:>3} | {ORG_MAP[oid]['name']}")
        return 0

    if not args.orgs.strip():
        print("Argument error: --orgs is required unless --list is used.")
        return 2

    try:
        org_ids = parse_orgs(args.orgs)
        ranges = parse_ranges(args.ranges)
    except Exception as exc:
        print(f"Argument error: {exc}")
        return 2

    try:
        cdp_url = resolve_cdp_ws_url(args.cdp_url, args.port)
    except Exception as exc:
        print(f"CDP error: {exc}")
        return 2

    global CLOUDFLARE_MANUAL_WAIT_SEC
    CLOUDFLARE_MANUAL_WAIT_SEC = max(0, int(args.cf_manual_wait))

    try:
        proxy_url = parse_proxy_value(args.proxy)
    except Exception as exc:
        print(f"Proxy error: {exc}")
        return 2

    if proxy_url and cdp_url and not args.force_proxy_context:
        print(
            "Info: CDP profile mode dang bat, bo qua --proxy de tai su dung cookie/session profile. "
            "Neu can ep dung proxy-level context, them --force-proxy-context.",
            flush=True,
        )
        proxy_url = None

    output_dir = Path(args.output_dir)
    state_dir = Path(args.state_dir)
    crawldir = build_crawldir(state_dir, args.worker, org_ids, ranges)
    crawldir.mkdir(parents=True, exist_ok=True)

    spider = TVPLScraplingSpider(
        worker=args.worker,
        cdp_url=cdp_url,
        org_ids=org_ids,
        ranges=ranges,
        delay=args.delay,
        proxy=proxy_url,
        use_stealth=bool(args.stealth and not args.dynamic),
        output_dir=output_dir,
        crawldir=crawldir,
        max_blocked_retries=args.max_blocked_retries,
    )
    print(
        f"[START] worker={args.worker} cdp={cdp_url} orgs={org_ids} ranges={ranges} "
        f"proxy={'on' if proxy_url else 'off'} crawldir={crawldir}",
        flush=True,
    )
    result = spider.start()
    print(
        f"\nDone. completed={result.completed}, paused={result.paused}, "
        f"items={result.stats.items_scraped}, requests={result.stats.requests_count}, "
        f"elapsed={result.stats.elapsed_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
