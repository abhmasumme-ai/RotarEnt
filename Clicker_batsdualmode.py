"""
clicker.py - v23 (wicket-only, strict-full-name + multi-first-3-words)
=======================================================================
Usage:
  python clicker.py <match_id>          -> STRICT mode: FULL name must match
  python clicker.py <match_id> multi    -> MULTI mode : any of first 3 words

STRICT: set(ws_words) == set(ocr_words)   (order-independent, no extras)
MULTI : any word (>=3 chars) of first 3 words in WS name appears in OCR name

Wicket-only flow:
  1. Only process events where ball_info() == "wicket"
  2. Read dismissed batsman from WS (batsman_out / striker)
  3. Screenshot panel below anchor (batch: 1 shot, N crops)
  4. OCR each UNDER box -> extract name + run
  5. Match name (strict/multi)
  6. Click UNDER box (highest run if multiple matches)

v23 fixes over v22:
  1. dedup key includes ball/xball  (two wickets in same over)
  2. str(m.get("over", ""))         (over=0 -> "0" not "")
  3. _last_dedup_key module-level + cleared on RESET
  4. ball_info: wicket checked BEFORE is_intermediate ("Batsman out" works)
  5. MULTI: first-3-words matching (per user spec), 3+ chars both sides
  6. Calibration loop: try/except on page.evaluate
  7. Name-only match allowed when run missing (single match) - logged as WARN
  8. Removed dead code (panel_cache, refresh_panel_header, CACHE_*)

Env:
  CLICKER_STRICT_MATCH   1 = strict (default), 0 = fuzzy
  CLICKER_WS_URL         default ws://127.0.0.1:5000/ws
  CLICKER_WS_TOKEN       optional WS auth token
  CLICKER_HEADLESS       1 = headless browser
  CLICKER_DEBUG          1 = verbose logs
  CLICKER_SITE_URL       optional (else prompt)
  CLICKER_MATCH_URL      optional (else prompt)
  CLICKER_PROXY_URL      optional proxy URL
  TESSERACT_CMD          custom tesseract path
"""
import asyncio
import json
import os
import queue
import random
import re
import sys
import threading
import time
import urllib.parse

import websockets
from playwright.sync_api import sync_playwright
import numpy as np
import cv2
import pytesseract

# ============================================================
#  Config
# ============================================================
_WS_URL_BASE = os.environ.get("CLICKER_WS_URL", "ws://127.0.0.1:5000/ws")
_WS_TOKEN = os.environ.get("CLICKER_WS_TOKEN", "").strip()
if _WS_TOKEN:
    _sep = "&" if "?" in _WS_URL_BASE else "?"
    WS_URL = f"{_WS_URL_BASE}{_sep}token={urllib.parse.quote(_WS_TOKEN)}"
else:
    WS_URL = _WS_URL_BASE

RECONNECT_SEC = 3
PING_SEC = 30
HEADLESS = os.environ.get("CLICKER_HEADLESS") == "1"
DEBUG = os.environ.get("CLICKER_DEBUG") == "1"
STRICT_MATCH = os.environ.get("CLICKER_STRICT_MATCH", "1") == "1"

TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

# Timing
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.05
EVENT_Q_MAXSIZE = 500
DEDUP_WINDOW_S = 600     # 10 min per batsman

# OCR
COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
OCR_CFG_HEADER = '--oem 1 --psm 6'

# Viewport
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 120
ANCHOR_H = 34

NAME_STOPWORDS = {
    'over', 'under', 'ovr', 'undr', 'delivery', 'deliveries', 'del',
    'runs', 'run', 'wicket', 'out', 'live', 'match', 'next', 'ball',
    'the', 'and', 'vs'
}

# State
ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = []
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0

_last_clicked = {"name": None, "at": 0.0}
_last_dedup_key = None

_pending_reset = threading.Event()
_stop_event = threading.Event()

WS_STATE = {"match_filter": "", "up": False, "msg_count": 0, "last_ts": 0.0}
event_q = queue.Queue(maxsize=EVENT_Q_MAXSIZE)

INTERMEDIATE_PATTERNS = [
    r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)',
    r'^ball in ?air', r'^boundary ?check',
    r'^(third ?umpire|3rd ?umpire)',
    r'^(confirm(ing)?|checking)', r'^(review|drs)',
    r'^(batsman|bowler|new ?batter|misfield|dropped|overthrow)',
]


# ============================================================
#  Utilities
# ============================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dbg(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] [DBG] {msg}", flush=True)


def is_intermediate(ev):
    return any(re.search(p, ev.lower().strip()) for p in INTERMEDIATE_PATTERNS)


def is_ball_chalu(ev):
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)',
                          ev.lower().strip()))


def handle_reset(from_id, to_id):
    _pending_reset.set()
    log(f"[RESET] {from_id} -> {to_id}")


def check_pending_reset():
    global _last_clicked, _last_dedup_key
    if _pending_reset.is_set():
        _pending_reset.clear()
        _last_clicked = {"name": None, "at": 0.0}
        _last_dedup_key = None
        log("[RESET] click dedup + event dedup key cleared")


# ============================================================
#  OCR name / run extraction
# ============================================================
def extract_name_from_text(text):
    if not text:
        return None
    t = text
    for kw in ('over', 'under', 'ovr', 'undr', 'deliveries', 'delivery', 'del'):
        t = re.sub(rf'\b{kw}\b', ' ', t, flags=re.I)
    t = re.sub(r'\b(runs?|wicket|out|live|match|next|ball)\b', ' ', t, flags=re.I)
    t = re.sub(r'\d+(\.\d+)?', ' ', t)
    t = re.sub(r'[^A-Za-z\s]', ' ', t)
    # Keep 2-char tokens (MS, AB, De) but drop 1-char noise
    words = [w for w in t.split() if len(w) >= 2 and w.lower() not in NAME_STOPWORDS]
    return ' '.join(words[:4]).strip() if words else None


def extract_run_from_text(text):
    if not text:
        return 0.0
    m = re.search(r'\(?(\d+\.\d+)\)?', text)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 0.0


def is_name_match(ws_name, ocr_name):
    """Unified name matcher.

    STRICT: set(ws_words) == set(ocr_words).  Extra/missing words -> NO.
    MULTI : any word (>=3 chars) of WS name's first 3 words appears in OCR name.
    """
    if not ws_name or not ocr_name:
        return False

    def clean(s):
        return re.sub(r'[^a-z\s]', ' ', s.lower()).strip()

    a = clean(ws_name)
    b = clean(ocr_name)
    if not a or not b:
        return False

    # -------------------- STRICT: full symmetric name match ------------------
    if STRICT_MATCH:
        return set(a.split()) == set(b.split())

    # -------------------- MULTI: first 3 words of WS name --------------------
    # Per user spec: 3+ chars only (MS, AB, De are too short -> filtered)
    ws_words = a.split()[:3]
    ocr_words = b.split()
    for wa in ws_words:
        if len(wa) < 3:
            continue
        for wb in ocr_words:
            if len(wb) < 3:
                continue
            if wa == wb:
                return True
            # Allow prefix match for longer words (OCR truncation)
            if len(wa) >= 5 and len(wb) >= 5:
                if wa.startswith(wb) or wb.startswith(wa):
                    return True
    return False


# ============================================================
#  Calibration UI (JS)
# ============================================================
SHOW_TARGETS_JS = """
() => {
    let box = document.getElementById('__pw_target_guide');
    if (!box) {
        box = document.createElement('div');
        box.id = '__pw_target_guide';
        box.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:999999;background:rgba(0,0,0,0.2);';
        document.body.appendChild(box);
    }
    box.innerHTML = '';
    let rowCount = window.__row_count || 3;
    let baseY = 400, anchorW = 120, anchorH = 34, leftX = 70, rightX = 425, boxW = 240, boxH = 30, rowGap = 3.34;

    const blueBox = document.createElement('div');
    blueBox.style.cssText = `position:absolute;left:${leftX}px;top:${baseY}px;width:${anchorW}px;height:${anchorH}px;border:2px solid #00aaff;background:rgba(0,170,255,0.4);box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:10px;`;
    blueBox.textContent = "Anchor";
    box.appendChild(blueBox);

    let firstRowY = baseY + anchorH;
    let finalTargets = [];

    for (let i = 0; i < rowCount; i++) {
        let y = firstRowY + i * (boxH + rowGap);
        const leftBox = document.createElement('div');
        leftBox.style.cssText = `position:absolute;left:${leftX}px;top:${y}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;`;
        box.appendChild(leftBox);
        const rightBox = document.createElement('div');
        rightBox.style.cssText = `position:absolute;left:${rightX}px;top:${y}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;`;
        box.appendChild(rightBox);
        finalTargets.push({side: 'over', x: leftX, y: y, w: boxW, h: boxH});
        finalTargets.push({side: 'under', x: rightX, y: y, w: boxW, h: boxH});
    }

    let btnTop = firstRowY + rowCount * (boxH + rowGap) + 15;
    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = `position:absolute;left:${leftX}px;top:${btnTop}px;pointer-events:auto;display:flex;gap:10px;`;

    const doneBtn = document.createElement('button');
    doneBtn.textContent = 'DONE';
    doneBtn.style.cssText = 'padding:10px 30px;font-size:16px;font-weight:bold;background:#00c853;color:white;border:none;border-radius:6px;cursor:pointer;';
    doneBtn.onclick = () => { window.__pw_final_targets = finalTargets; window.__pw_action = 'done'; };

    const plusBtn = document.createElement('button');
    plusBtn.textContent = '+ 2 Rows';
    plusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#2196f3;color:white;border:none;border-radius:6px;cursor:pointer;';
    plusBtn.onclick = () => { window.__row_count = Math.min(10, rowCount + 2); window.__pw_action = 'update'; };

    const minusBtn = document.createElement('button');
    minusBtn.textContent = '- 2 Rows';
    minusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff9800;color:white;border:none;border-radius:6px;cursor:pointer;';
    minusBtn.onclick = () => { window.__row_count = Math.max(2, rowCount - 2); window.__pw_action = 'update'; };

    btnContainer.appendChild(doneBtn);
    btnContainer.appendChild(plusBtn);
    btnContainer.appendChild(minusBtn);
    box.appendChild(btnContainer);
}
"""


# ============================================================
#  Calibration
# ============================================================
def setup_from_done(page, final_targets):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    log("Saving anchor + target box positions...")

    page.evaluate("() => { let b = document.getElementById('__pw_target_guide'); if(b) b.remove(); }")
    page.wait_for_timeout(1000)

    raw = page.screenshot(type="png", scale="css")
    src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    anchor_img = src_img[ANCHOR_TOP:ANCHOR_TOP + ANCHOR_H, ANCHOR_LEFT:ANCHOR_LEFT + ANCHOR_W]
    ANCHOR_TEMPLATE.update({"img": anchor_img, "w": ANCHOR_W, "h": ANCHOR_H})
    res = cv2.matchTemplate(src_img, anchor_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= COMPARE_THRESHOLD:
        CALIBRATED_ANCHOR_POS = max_loc
        last_anchor_pos = max_loc
        _anchor_trust_until = time.time() + ANCHOR_TRUST_SEC
    log(f"   Anchor saved (conf={max_val:.2f})")

    MAPPED_TARGETS.clear()
    for i, t in enumerate(final_targets):
        side = t.get('side')
        x, y, w, h = int(t['x']), int(t['y']), int(t['w']), int(t['h'])
        MAPPED_TARGETS.append({"side": side, "x": x, "y": y, "w": w, "h": h})

    log(f"   Mapped {len(MAPPED_TARGETS)} boxes total")


# ============================================================
#  Anchor detection
# ============================================================
def find_anchor_fast(page):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    now = time.time()
    cached = CALIBRATED_ANCHOR_POS or last_anchor_pos
    if cached and now < _anchor_trust_until:
        return cached
    if cached:
        ax, ay = cached
        try:
            raw = page.screenshot(
                clip={"x": ax, "y": ay, "width": ANCHOR_W, "height": ANCHOR_H},
                type="png", scale="css")
            crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if crop is not None and crop.shape == ANCHOR_TEMPLATE["img"].shape:
                res = cv2.matchTemplate(crop, ANCHOR_TEMPLATE["img"],
                                        cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= CACHE_VERIFY_THRESHOLD:
                    _anchor_trust_until = now + ANCHOR_TRUST_SEC
                    return ax, ay
        except Exception:
            pass
    try:
        raw = page.screenshot(type="png", scale="css")
        src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(src_img, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= COMPARE_THRESHOLD:
            CALIBRATED_ANCHOR_POS = max_loc
            last_anchor_pos = max_loc
            _anchor_trust_until = now + ANCHOR_TRUST_SEC
            return max_loc[0], max_loc[1]
    except Exception:
        pass
    return None, None


# ============================================================
#  Batch OCR: one screenshot for all under boxes
# ============================================================
def ocr_all_under_boxes(page, under_boxes, ax, ay):
    if not under_boxes:
        return []

    coords = []
    for b in under_boxes:
        x = ax + (b["x"] - ANCHOR_LEFT)
        y = ay + (b["y"] - ANCHOR_TOP)
        coords.append((int(x), int(y), int(b["w"]), int(b["h"])))

    x_min = max(0, min(c[0] for c in coords))
    y_min = max(0, min(c[1] for c in coords))
    x_max = min(VIEW_W, max(c[0] + c[2] for c in coords))
    y_max = min(VIEW_H, max(c[1] + c[3] for c in coords))

    clip_w = x_max - x_min
    clip_h = y_max - y_min
    if clip_w <= 0 or clip_h <= 0:
        return [(b, "") for b in under_boxes]

    try:
        raw = page.screenshot(
            clip={"x": x_min, "y": y_min, "width": clip_w, "height": clip_h},
            type="png", scale="css")
        big = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        dbg(f"Batch screenshot error: {e}")
        return [(b, "") for b in under_boxes]

    results = []
    for box, (x, y, w, h) in zip(under_boxes, coords):
        cx0 = max(0, x - x_min)
        cy0 = max(0, y - y_min)
        crop = big[cy0:cy0 + h, cx0:cx0 + w]
        if crop.size == 0:
            results.append((box, ""))
            continue
        try:
            crop_r = cv2.resize(crop, None, fx=3, fy=3,
                                interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                crop_r, config=OCR_CFG_HEADER).lower()
        except Exception as e:
            dbg(f"OCR error: {e}")
            text = ""
        results.append((box, text))
    return results


def do_click(page, target_box, ax, ay):
    x = ax + (target_box["x"] - ANCHOR_LEFT)
    y = ay + (target_box["y"] - ANCHOR_TOP)
    cx = x + target_box["w"] / 2
    cy = y + target_box["h"] / 2
    try:
        page.mouse.click(cx, cy, delay=random.randint(2, 5))
    except Exception as e:
        log(f"   Click error: {e}")
        return None, None
    return cx, cy


# ============================================================
#  Parse helpers
# ============================================================
def parse_over_ball(d):
    ov = d.get("over")
    if ov is None:
        return None, None, None, None

    over_num = None
    over_dec = 0
    ball = 1

    if isinstance(ov, (int, float)) and not isinstance(ov, bool):
        over_num = int(ov)
        frac = ov - over_num
        if frac > 0.01:
            try:
                over_dec = int(round(frac * 10))
                ball = over_dec
            except (ValueError, TypeError):
                over_dec = 0
                ball = 1
        else:
            over_dec = 0
            ball = 1
    else:
        ov_str = str(ov).strip()
        try:
            if "." in ov_str:
                parts = ov_str.split(".")
                over_num = int(parts[0])
                over_dec = int(parts[1]) if parts[1] else 1
                ball = over_dec
            else:
                over_num = int(ov_str)
                over_dec = 0
                ball = 1
        except (ValueError, IndexError):
            return None, None, None, None

    b = d.get("ball")
    if b is not None:
        try:
            ball = int(b)
        except (ValueError, TypeError):
            pass

    xball = d.get("xball")
    if xball is not None:
        try:
            xball = int(xball)
        except (ValueError, TypeError):
            xball = None

    return over_num, over_dec, ball, xball


def get_striker(m):
    """Returns (name, source_key). Prefer explicit 'batsman_out'."""
    for key in ("batsman_out", "dismissed_batsman", "out_batsman",
                "batsman_out_name", "dismissed_player"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip(), key
    for key in ("striker", "striker_name", "batsman", "batsman_name",
                "player", "player_name", "batsmanName", "strikerName"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip(), key
    return None, None


# ============================================================
#  Wicket handling
# ============================================================
def handle_wicket(page, striker):
    global _last_clicked

    now = time.time()
    if not striker:
        return False

    if (_last_clicked["name"] == striker.lower()
            and now - _last_clicked["at"] < DEDUP_WINDOW_S):
        log(f"[SKIP-DUP] '{striker}' clicked {(now - _last_clicked['at']):.0f}s ago")
        return False

    t0 = time.perf_counter()
    ax, ay = find_anchor_fast(page)
    if ax is None:
        log("[NO-ANCHOR]")
        return False

    under_boxes = [t for t in MAPPED_TARGETS if t["side"] == "under"]
    log(f"[SCAN] {len(under_boxes)} under boxes for '{striker}' "
        f"(mode={'STRICT' if STRICT_MATCH else 'MULTI'})")

    ocr_results = ocr_all_under_boxes(page, under_boxes, ax, ay)

    matched_options = []
    for i, (box, live_text) in enumerate(ocr_results):
        if not live_text:
            dbg(f"   [BOX {i}] OCR empty")
            continue

        live_name = extract_name_from_text(live_text)
        live_run = extract_run_from_text(live_text)

        if live_name and is_name_match(striker, live_name):
            matched_options.append((live_run, box, live_name, live_text))
            log(f"   [MATCH] '{striker}' -> '{live_name}' (run={live_run})")
        elif live_name:
            dbg(f"   [SKIP] '{striker}' vs '{live_name}'")

    if not matched_options:
        log(f"[NO-MATCH] '{striker}' not found in panel")
        return False

    matched_options.sort(key=lambda x: x[0], reverse=True)
    best_run, best_box, best_name, best_text = matched_options[0]

    if len(matched_options) > 1:
        log(f"[MULTI-MATCH] {len(matched_options)} candidates for '{striker}' "
            f"- picking run={best_run}")
    if best_run == 0:
        log(f"[WARN] '{striker}' matched but no run extracted (name-only click)")

    log(f"[BEST] '{striker}' -> run={best_run} (OCR='{best_name}')")

    cx, cy = do_click(page, best_box, ax, ay)
    if cx is None:
        return False

    ms = round((time.perf_counter() - t0) * 1000)
    log(f"[CLICK] UNDER {best_run} for '{striker}' @({cx:.0f},{cy:.0f}) [+{ms}ms]")

    _last_clicked = {"name": striker.lower(), "at": time.time()}
    return True


# ============================================================
#  Ball classification (wicket patterns FIRST)
# ============================================================
def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev:
        return (ev, "intermediate", 0)

    # Wicket patterns FIRST (some overlap with intermediate "^batsman")
    if re.match(r'^(wicket|out|w|batsman (out|dismissed))[.!]*$', ev_low):
        return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|'
                r'retired|handled|obstruct)', ev_low):
        return (ev, "wicket", 0)

    if is_intermediate(ev):
        return (ev, "intermediate", 0)
    return (ev, "intermediate", 0)


# ============================================================
#  WebSocket client
# ============================================================
async def ws_loop():
    while not _stop_event.is_set():
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(WS_URL, ping_interval=None,
                                          compression=None, max_size=None) as ws:
                WS_STATE["up"] = True
                log(f"[WS] connected: {WS_URL}")
                if mf:
                    try:
                        await ws.send(f"watch:{mf}")
                        log(f"Watching match: {mf}")
                    except Exception:
                        pass

                async def pinger():
                    while True:
                        await asyncio.sleep(PING_SEC)
                        try:
                            await ws.send("ping")
                        except Exception:
                            break

                ptask = asyncio.create_task(pinger())
                try:
                    async for raw in ws:
                        if raw == "pong" or raw == b"pong":
                            continue
                        if isinstance(raw, (bytes, bytearray)):
                            try:
                                raw = raw.decode("utf-8", errors="replace")
                            except Exception:
                                continue
                        try:
                            m = json.loads(raw)
                        except (ValueError, TypeError):
                            continue

                        mid_msg = m.get("id")
                        if mf and mid_msg and str(mid_msg) != str(mf):
                            continue

                        if m.get("t") == "reset":
                            handle_reset(m.get("from"), m.get("to"))
                            continue

                        WS_STATE["msg_count"] += 1
                        WS_STATE["last_ts"] = time.time()
                        try:
                            event_q.put_nowait(m)
                        except queue.Full:
                            dbg("event_q full - dropping message")
                finally:
                    ptask.cancel()
        except asyncio.CancelledError:
            return
        except Exception as e:
            log(f"[WS] disconnected: {type(e).__name__}: {e}")
        finally:
            WS_STATE["up"] = False
        if _stop_event.is_set():
            return
        await asyncio.sleep(RECONNECT_SEC)


def start_ws_thread():
    def _run():
        try:
            asyncio.run(ws_loop())
        except Exception as e:
            log(f"[WS] thread crashed: {e}")

    threading.Thread(target=_run, daemon=True).start()


# ============================================================
#  Main
# ============================================================
def monitor_and_click():
    global STRICT_MATCH, _last_dedup_key

    if len(sys.argv) < 2:
        print("Usage: python clicker.py <match_key> [multi]")
        print("  (no 2nd arg)  -> STRICT : FULL name must match")
        print("  multi         -> MULTI  : any of first 3 words of name matches")
        return

    match_filter = sys.argv[1].strip()

    if len(sys.argv) >= 3:
        arg2 = sys.argv[2].strip().lower()
        if arg2 == "multi":
            STRICT_MATCH = False
        elif arg2 in ("strict", "full"):
            STRICT_MATCH = True
        elif arg2:
            log(f"Unknown mode '{arg2}' - using default")

    WS_STATE["match_filter"] = match_filter

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    site_url = os.environ.get("CLICKER_SITE_URL", "").strip()
    if not site_url:
        site_url = input("\nSite URL: ").strip() or "https://example.com"

    proxy_config = None
    proxy_url = os.environ.get("CLICKER_PROXY_URL", "").strip()
    if not proxy_url:
        print("\n=== Proxy Settings ===")
        proxy_url = input("Proxy URL [Enter to skip]: ").strip()
    if proxy_url:
        try:
            if "@" in proxy_url:
                proto, rest = (proxy_url.split("://") if "://" in proxy_url
                               else ("http://", proxy_url))
                auth, host_port = rest.split("@")
                user, pwd = auth.split(":")
                proxy_config = {"server": f"{proto}://{host_port}",
                                "username": user, "password": pwd}
            else:
                proxy_config = {"server": proxy_url}
            log(f"Proxy: {proxy_config['server']}")
        except Exception as e:
            log(f"Proxy error: {e}")
    else:
        log("Proxy skipped.")

    try:
        pytesseract.image_to_string(np.zeros((30, 100), dtype=np.uint8),
                                    config=OCR_CFG_HEADER)
        log("Tesseract warmed up")
    except Exception:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=[f"--window-size={WIN_W},{WIN_H}", "--window-position=0,0"],
        )
        context = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/122.0.0.0 Safari/537.36",
            proxy=proxy_config,
        )
        page = context.new_page()
        page.goto(site_url)

        match_url = os.environ.get("CLICKER_MATCH_URL", "").strip()
        if not match_url:
            print("\n=== log in manually ===")
            match_url = input("Match URL: ").strip()
        if not match_url:
            browser.close()
            return
        page.goto(match_url)
        log(f"Match page ready: {page.url}")

        print("\n" + "=" * 65)
        print("Go to browser and select 'Players, Total Runs' market")
        print("Adjust red boxes. Use '+ 2 Rows' if more players.")
        print("Click 'DONE' when ready")
        print("=" * 65 + "\n")

        # Crash-proof calibration loop
        while not _stop_event.is_set():
            try:
                page.evaluate("() => { window.__pw_action = null; }")
                page.evaluate(SHOW_TARGETS_JS)
            except Exception as e:
                log(f"Calibration setup error: {e}")
                break
            print("Waiting for calibration (DONE / + / -)...")
            action = None
            while not _stop_event.is_set():
                try:
                    action = page.evaluate("() => window.__pw_action")
                except Exception as e:
                    log(f"Calibration poll error: {e}")
                    break
                if action in ('done', 'update'):
                    break
                page.wait_for_timeout(200)

            if action == 'update':
                continue
            if action == 'done':
                try:
                    final_targets = page.evaluate("() => window.__pw_final_targets")
                    setup_from_done(page, final_targets)
                except Exception as e:
                    log(f"Calibration save error: {e}")
                break
            # action None or browser closed -> abort
            break

        if not MAPPED_TARGETS:
            log("No targets mapped - aborting.")
            browser.close()
            return

        start_ws_thread()
        log(f"[WS] waiting... (match_key={match_filter})")
        log(f"   v23 - wicket-only mode")
        if STRICT_MATCH:
            log("   Mode: STRICT - FULL name must match (all words)")
        else:
            log("   Mode: MULTI  - first 3 words of WS name")
        log(f"   Batch OCR: 1 screenshot for all under boxes")
        log(f"   Event timeout: {EVENT_TIMEOUT_SEC*1000:.0f}ms")
        log(f"   Dedup window: {DEDUP_WINDOW_S}s per batsman\n")

        _last_dedup_key = None

        try:
            while not _stop_event.is_set():
                check_pending_reset()

                try:
                    m = event_q.get(timeout=EVENT_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if m.get("t") != "ev":
                    continue

                ev = str(m.get("ev") or "").strip()
                if not ev:
                    continue
                if is_ball_chalu(ev):
                    _last_dedup_key = None
                    continue

                text, kind, _ = ball_info(m)
                if kind != "wicket":
                    continue

                # Dedup key (FIX #1+#2): includes over + ball/xball + ev
                over_str = str(m.get("over", ""))
                bkey = m.get("xball") or m.get("ball") or ""
                dedup_key = f"{over_str}|{bkey}|{text}"
                if dedup_key == _last_dedup_key:
                    dbg(f"[DUP] {dedup_key}")
                    continue
                _last_dedup_key = dedup_key

                striker, src = get_striker(m)
                if not striker:
                    log("[WICKET] no batsman name in payload - skip "
                        "(main.py must send 'batsman_out' or 'striker')")
                    continue

                log(f"[WICKET] '{striker}' (from '{src}') | "
                    f"over={m.get('over')} ball={m.get('ball')} ev='{text}'")
                handle_wicket(page, striker)

        except KeyboardInterrupt:
            print("\nClosing...")
        finally:
            _stop_event.set()
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    monitor_and_click()
