"""
clicker.py - v19 (v18 core logic + v15 compatibility, English only)
====================================================================
Core logic (preserved from v18):
  1. Multi-row panel support (2, 4, 6, or more rows).
  2. On wicket event, read dismissed batsman name from WS.
  3. OCR-scan all Under boxes in the panel.
  4. Match name strictly (exact First + Last name, ignoring middle names).
  5. Click the Under box with the HIGHEST run value among matches.

Compatibility fixes (from v15):
  - Binary WS frames (bytes + str)
  - WS auth token via ?token= (env CLICKER_WS_TOKEN)
  - Thread-safe reset handling (threading.Event)
  - Bounded event queue (max 500)
  - Graceful shutdown via threading.Event
  - Env/CLI overrides for input() calls (headless friendly)
  - Fast single mouse.click() action
  - English-only logs

Bug fixes (from v18):
  - extract_name_from_text: len(w) >= 3 -> len(w) >= 2
    (fixes "MS Dhoni", "AB De Villiers", "F du Plessis", "BJ Watling")
  - is_name_match: add subset-with-last-name-match check
    (fixes single-char first names)
  - _last_clicked_batsman replaced with time-windowed dict
    (allows the same batsman to be clicked in 2nd innings)
  - handle_wicket: require live_run > 0 (skip if OCR failed to read run)
  - WS thread crash now logged (was silent)

Server-side recommendation (main.py v8/v9/v10):
  In the ball_event output, add:
    "batsman_out": d.get("dismissed_batsman")
                   or d.get("batsman_out")
                   or d.get("out_batsman")
                   or d.get("striker_name")
  Without this, clicker falls back to "striker" field, which is
  correct for bowled/lbw/caught/stumped but may be wrong for run-outs.

Env:
  CLICKER_WS_URL         default ws://127.0.0.1:5000/ws
  CLICKER_WS_TOKEN       optional WS auth token
  CLICKER_HEADLESS       1 = headless browser
  CLICKER_DEBUG          1 = verbose
  CLICKER_STRICT_MATCH   1 = strict exact name match (default 1)
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
CACHE_FRESH_SEC = 0.2
IDLE_REFRESH_SEC = 0.15
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.05
EVENT_Q_MAXSIZE = 500
DEDUP_WINDOW_S = 600  # same batsman cannot be out twice within 10 minutes

# OCR
COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
HEADER_CROP_W = 240
OCR_RESIZE = 3
OCR_CFG_HEADER = '--oem 1 --psm 6'
OCR_CFG_HEADER_FALLBACK = '--oem 1 --psm 7'

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

panel_cache = {"over": None, "del": None, "ts": 0.0, "ax": None, "ay": None}

# Time-windowed dedup: allows same batsman in 2nd innings (>10 min later)
_last_clicked = {"name": None, "at": 0.0}

# Thread-safe flags
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
    """Called from WS thread. Sets flag only (thread-safe)."""
    _pending_reset.set()
    log(f"[RESET] {from_id} -> {to_id}")


def check_pending_reset():
    """Called from main thread. Clears dedup state safely."""
    global _last_clicked
    if _pending_reset.is_set():
        _pending_reset.clear()
        _last_clicked = {"name": None, "at": 0.0}
        log("[RESET] click dedup cleared")


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
    # FIX: len(w) >= 2 (was >= 3) to preserve 2-char names like MS, AB, BJ, De
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
    """Strict name matcher (exact First + Last name, ignoring middle)."""
    if not ws_name or not ocr_name:
        return False

    def clean(s):
        return re.sub(r'[^a-z\s]', ' ', s.lower()).strip()

    a_clean = clean(ws_name)
    b_clean = clean(ocr_name)
    if not a_clean or not b_clean:
        return False

    # 1. Full exact match
    if a_clean == b_clean:
        return True

    a_parts = a_clean.split()
    b_parts = b_clean.split()

    # 2. Same words, different order
    if sorted(a_parts) == sorted(b_parts):
        return True

    # 3. First + Last name exact match (ignores middle names)
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        if a_parts[0] == b_parts[0] and a_parts[-1] == b_parts[-1]:
            return True

    # 4. FIX: OCR captured only part of the name (short first names / missing words)
    #    but last name matches exactly.
    #    Example: ws="F du Plessis", ocr="DU PLESSIS"
    #             ws="MS Dhoni", ocr="DHONI"
    if b_parts and a_parts:
        if b_parts[-1] == a_parts[-1] and all(w in a_parts for w in b_parts):
            return True

    # 5. Non-strict fuzzy fallback (opt-in via CLICKER_STRICT_MATCH=0)
    if not STRICT_MATCH:
        if len(a_parts[0]) >= 4 and a_parts[0] in b_parts:
            return True
        if len(a_parts[-1]) >= 4 and a_parts[-1] in b_parts:
            return True

    return False


# ============================================================
#  Calibration UI (JS) - multi-row with +/- buttons
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

    log(f"   Mapped {len(MAPPED_TARGETS)} boxes total (Strict Match ready)")


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
                res = cv2.matchTemplate(crop, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
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
#  OCR and click
# ============================================================
def get_live_ocr_data(page, box, ax, ay):
    """Screenshot one box and OCR it to lowercase text."""
    tx = ax + (box["x"] - ANCHOR_LEFT)
    ty = ay + (box["y"] - ANCHOR_TOP)
    try:
        raw = page.screenshot(
            clip={"x": tx, "y": ty, "width": box["w"], "height": box["h"]},
            type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(crop, config=OCR_CFG_HEADER).lower()
    except Exception as e:
        dbg(f"OCR error: {e}")
        return ""


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
    """Return (over_num, over_dec, ball, xball). Handles int, float, str."""
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
    """Return dismissed batsman name. Prefer explicit out fields, fallback to striker."""
    # Priority 1: explicit dismissed batsman fields (from server patch)
    for key in ("dismissed_batsman", "batsman_out", "out_batsman",
                "batsman_out_name", "dismissed_player"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip()
    # Priority 2: current striker (main.py v8 fallback)
    for key in ("striker", "striker_name", "batsman", "batsman_name",
                "player", "player_name", "batsmanName", "strikerName"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip()
    return None


# ============================================================
#  Wicket handling
# ============================================================
def handle_wicket(page, striker, over_num, over_dec, ball, xball):
    global _last_clicked

    now = time.time()
    if not striker:
        return False

    # Dedup: same batsman cannot be out twice within 10 minutes
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
    log(f"[SCAN] Checking {len(under_boxes)} under boxes for '{striker}'")

    matched_options = []
    for i, box in enumerate(under_boxes):
        live_text = get_live_ocr_data(page, box, ax, ay)
        if not live_text:
            dbg(f"   [BOX {i}] OCR empty")
            continue

        live_name = extract_name_from_text(live_text)
        live_run = extract_run_from_text(live_text)

        if live_name and is_name_match(striker, live_name):
            if live_run > 0:
                matched_options.append((live_run, box, live_name, live_text))
                log(f"   [MATCH] '{striker}' == '{live_name}' (run={live_run})")
            else:
                log(f"   [WARN] '{striker}' name matched but run value missing")
        elif live_name:
            dbg(f"   [SKIP] '{striker}' != '{live_name}'")

    if not matched_options:
        log(f"[NO-MATCH] '{striker}' not found in panel (or name mismatch)")
        return False

    # Sort descending by run value, pick highest
    matched_options.sort(key=lambda x: x[0], reverse=True)
    best_run, best_box, best_name, best_text = matched_options[0]
    log(f"[BEST] '{striker}' -> run={best_run}")

    cx, cy = do_click(page, best_box, ax, ay)
    if cx is None:
        return False

    ms = round((time.perf_counter() - t0) * 1000)
    log(f"[CLICK] UNDER {best_run} for '{striker}' @({cx:.0f},{cy:.0f}) [+{ms}ms]")

    _last_clicked = {"name": striker.lower(), "at": time.time()}
    return True


def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev or is_intermediate(ev):
        return (ev, "intermediate", 0)
    if re.match(r'^(wicket|out|w)[.!]*$', ev_low):
        return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|'
                r'retired|handled|obstruct)', ev_low):
        return (ev, "wicket", 0)
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
                        # Handle str + bytes (WS_BINARY=1 sends bytes)
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
    if len(sys.argv) < 2:
        print("Usage: python clicker.py <match_key>")
        return
    match_filter = sys.argv[1].strip()
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
                proto, rest = proxy_url.split("://") if "://" in proxy_url else ("http://", proxy_url)
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

        while not _stop_event.is_set():
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("Waiting for calibration (DONE / + / -)...")
            while not _stop_event.is_set():
                action = page.evaluate("() => window.__pw_action")
                if action in ('done', 'update'):
                    break
                page.wait_for_timeout(200)

            if action == 'update':
                continue

            final_targets = page.evaluate("() => window.__pw_final_targets")
            setup_from_done(page, final_targets)
            break

        if not MAPPED_TARGETS:
            log("No targets mapped - aborting.")
            browser.close()
            return

        start_ws_thread()
        log(f"[WS] waiting... (match_key={match_filter})")
        log(f"   v19 - compatible with main.py v8/v9/v10")
        log(f"   Strict name match: {'ON' if STRICT_MATCH else 'OFF'}")
        log(f"   On wicket: scans all Under boxes, clicks highest run match\n")

        last_dedup_key = None

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
                if not ev or is_ball_chalu(ev) or is_intermediate(ev):
                    continue

                over_str = str(m.get("over") or "")
                dedup_key = f"{over_str}|{ev}"
                if dedup_key == last_dedup_key:
                    dbg(f"[DUP] {dedup_key}")
                    continue
                last_dedup_key = dedup_key

                text, kind, run = ball_info(m)
                if kind != "wicket":
                    continue

                over_num, over_dec, main_ball, xball = parse_over_ball(m)
                if over_num is None:
                    continue

                striker = get_striker(m)
                if striker:
                    log(f"[WICKET] '{striker}' out | "
                        f"over={m.get('over')} ball={m.get('ball')}")
                    handle_wicket(page, striker, over_num, over_dec, main_ball, xball)

        except KeyboardInterrupt:
            print("\nClosing...")
        finally:
            _stop_event.set()
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    monitor_and_click()
