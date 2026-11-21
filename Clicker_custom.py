"""
clicker_custom.py - v2 (dynamic over/threshold - set boxes ONCE, runs forever)
================================================================================

Key facts:
  * Blue anchor box  -> position saved ONCE at DONE. Read each click for log.
  * Red boxes (1 row x 2 cols default) -> positions saved ONCE at DONE.
  * Text INSIDE red boxes is DYNAMIC:
        "25 over total run under 136"  -> today over 25 / threshold 136
        "26 over total run under 141"  -> later over 26 / threshold 141 (auto-detected)
    Script re-OCRs boxes every click, parses numbers dynamically.
    You do NOT touch DONE again during the match.

Click rules:
  Wicket                       -> click matching-over panel with HIGHEST threshold
  4 / 6                        -> click matching-over panel with HIGHEST threshold
  wide+4 / wide+6              -> click matching-over panel with HIGHEST threshold
  noball / noball+4 / noball+6 -> click matching-over panel with HIGHEST threshold
  dot / 1 / 2 / 3 / plain wide -> skip (no click)

Multiple panels for same over -> highest threshold wins (your examples: 136 both times).

Modes:
  python clicker_custom.py <match_id>          # SINGLE: fresh OCR (safe)
  python clicker_custom.py <match_id> multi    # MULTI:  1.5s cache (fast)

Env:
  CLICKER_WS_URL / CLICKER_WS_TOKEN / CLICKER_HEADLESS / CLICKER_DEBUG
  CLICKER_NO_IDLE / CLICKER_MULTI / CLICKER_SITE_URL / CLICKER_MATCH_URL
  CLICKER_PROXY_URL / CLICKER_DEDUP_WINDOW / TESSERACT_CMD
"""
import asyncio
import collections
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
NO_IDLE = os.environ.get("CLICKER_NO_IDLE") == "1"
MULTI_MODE = os.environ.get("CLICKER_MULTI") == "1"
DEDUP_WINDOW = int(os.environ.get("CLICKER_DEDUP_WINDOW", "30"))
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

# Timing
PANEL_CACHE_FRESH_SEC = 0.3           # single mode: fresh OCR essentially always
PANEL_CACHE_FRESH_SEC_MULTI = 1.5     # multi mode: aggressive cache
IDLE_REFRESH_SEC = 0.5
EVENT_TIMEOUT_SEC = 0.01
POST_CLICK_COOLDOWN = 0.1
IDLE_SKIP_WS_RECENT_SEC = 1.0
EVENT_Q_MAXSIZE = 500

# OCR
OCR_RESIZE = 3
OCR_CFG_HEADER = '--oem 1 --psm 7'          # single-line faster + more accurate
OCR_CFG_HEADER_FALLBACK = '--oem 1 --psm 6' # multi-line fallback

# Sanity ranges for parsed values
MAX_OVER = 60
MAX_THRESHOLD = 3000

# Viewport
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

# Anchor (fixed reference)
ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 100
ANCHOR_H = 34

# State
ANCHOR_RECT = None
MAPPED_TARGETS = []
PANEL_CACHE = {"panels": [], "ts": 0.0}
_last_click_at = 0.0
_recent_clicked = collections.deque(maxlen=DEDUP_WINDOW)
_pending_reset = threading.Event()
_last_dedup_key = None
_prev_panel_snapshot = None

WS_STATE = {"match_filter": "", "up": False, "msg_count": 0, "last_ts": 0.0}
event_q = queue.Queue(maxsize=EVENT_Q_MAXSIZE)
_stop_event = threading.Event()


# ============================================================
#  Logging
# ============================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dbg(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] [DBG] {msg}", flush=True)


def handle_reset(from_id, to_id):
    _pending_reset.set()
    log(f"[RESET] {from_id} -> {to_id}")


def check_pending_reset():
    global _last_dedup_key
    if _pending_reset.is_set():
        _pending_reset.clear()
        _recent_clicked.clear()
        _last_dedup_key = None
        log("[RESET] dedup cleared")


# ============================================================
#  Event classification
# ============================================================
def classify_event(ev_text):
    """Return 'wicket', 'over', or None (skip)."""
    ev = (ev_text or "").lower().strip()
    if not ev:
        return None

    # Wicket
    if re.match(r'^(wicket|out|w)[.!]*$', ev):
        return 'wicket'
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|'
                r'retired|handled|obstruct)', ev):
        return 'wicket'

    # No-ball (any variant incl. no-ball wide)
    if re.search(r'\b(no[\s_-]?ball|nb)\b', ev):
        return 'over'

    # Wide (only if +4 or +6, else skip)
    if re.match(r'^wide\b', ev):
        if re.search(r'\b(four|six|4|6)\b', ev):
            return 'over'
        return None

    # Four / Six
    if re.match(r'^(four|six|4|6)\b', ev):
        return 'over'

    return None


# ============================================================
#  Over/ball parse
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


# ============================================================
#  OCR
# ============================================================
def _ocr_region(page, x, y, w, h, cfg):
    try:
        raw = page.screenshot(
            clip={"x": x, "y": y, "width": w, "height": h},
            type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.size == 0:
            return ""
        crop_big = cv2.resize(crop, None, fx=OCR_RESIZE, fy=OCR_RESIZE,
                              interpolation=cv2.INTER_CUBIC)
        for c in (cfg, OCR_CFG_HEADER_FALLBACK):
            try:
                t = pytesseract.image_to_string(crop_big, config=c).lower().strip()
            except Exception:
                t = ""
            if t:
                return t
    except Exception as e:
        dbg(f"OCR err: {e}")
    return ""


def parse_panel_text(text):
    """
    DYNAMIC parse - handles any over / any threshold.

    Handles:
      "25 over total run under 136"
      "26 over total run over 141"
      "25.0 over total run under 136"
      "over 25 total run under 136"
      "25 ovr total runs undr 136"
      "total run under 136 at 25 over"
    """
    if not text:
        return None
    t = re.sub(r'\s+', ' ', text).strip().lower()

    SIDE_RX = r'\b(u[nm]d[ea]r|ov[ea]r|undr|ovr)\b'
    OVER_RX = r'[o0]v[ea]?r\b'

    # Try pattern 1: "<N> over ... <side> <M>"
    for pat in (
        rf'(\d+)(?:\.\d+)?\s*{OVER_RX}\D{{0,50}}?{SIDE_RX}\D{{0,10}}?(\d+)',
        rf'{OVER_RX}\s*(\d+)(?:\.\d+)?\D{{0,50}}?{SIDE_RX}\D{{0,10}}?(\d+)',
    ):
        m = re.search(pat, t)
        if m:
            try:
                over_num = int(m.group(1))
                side_word = m.group(2)
                threshold = int(m.group(3))
                side = 'under' if side_word[0] == 'u' else 'over'
                if 0 <= over_num <= MAX_OVER and 1 <= threshold <= MAX_THRESHOLD:
                    return {'over': over_num, 'side': side, 'threshold': threshold}
            except (ValueError, TypeError):
                pass

    # Try pattern 2: "<side> <M> ... <N> over"
    m = re.search(
        rf'{SIDE_RX}\D{{0,10}}?(\d+)\D{{0,50}}?(\d+)(?:\.\d+)?\s*{OVER_RX}', t)
    if m:
        try:
            side_word = m.group(1)
            threshold = int(m.group(2))
            over_num = int(m.group(3))
            side = 'under' if side_word[0] == 'u' else 'over'
            if 0 <= over_num <= MAX_OVER and 1 <= threshold <= MAX_THRESHOLD:
                return {'over': over_num, 'side': side, 'threshold': threshold}
        except (ValueError, TypeError):
            pass

    return None


def ocr_anchor_text(page):
    """Read text at the fixed anchor box (for log/verify only)."""
    if not ANCHOR_RECT:
        return ""
    return _ocr_region(page, ANCHOR_RECT['x'], ANCHOR_RECT['y'],
                       ANCHOR_RECT['w'], ANCHOR_RECT['h'], OCR_CFG_HEADER)


def refresh_panels(page):
    """OCR all red boxes. parse_panel_text handles dynamic over/threshold."""
    global _prev_panel_snapshot
    if not ANCHOR_RECT or not MAPPED_TARGETS:
        return []
    t0 = time.perf_counter()
    panels = []
    ax = ANCHOR_RECT['x']
    ay = ANCHOR_RECT['y']
    for idx, t in enumerate(MAPPED_TARGETS):
        px = ax + (t['x'] - ANCHOR_LEFT)
        py = ay + (t['y'] - ANCHOR_TOP)
        text = _ocr_region(page, px, py, t['w'], t['h'], OCR_CFG_HEADER)
        info = parse_panel_text(text)
        if info:
            info['x'] = t['x']
            info['y'] = t['y']
            info['w'] = t['w']
            info['h'] = t['h']
            info['idx'] = idx
            info['raw'] = text[:80]
            panels.append(info)
        else:
            dbg(f"Panel[{idx}] unparsed: {text[:80]!r}")

    PANEL_CACHE['panels'] = panels
    PANEL_CACHE['ts'] = time.time()

    # Log only when panel state changes (avoid spam)
    snap = tuple((p['over'], p['side'], p['threshold']) for p in panels)
    if snap != _prev_panel_snapshot:
        log(f"[PANELS] {list(snap)}")
        _prev_panel_snapshot = snap

    ms = round((time.perf_counter() - t0) * 1000)
    dbg(f"refresh_panels: {len(panels)}/{len(MAPPED_TARGETS)} in {ms}ms")
    return panels


def get_panels(page):
    age = time.time() - PANEL_CACHE['ts']
    ttl = PANEL_CACHE_FRESH_SEC_MULTI if MULTI_MODE else PANEL_CACHE_FRESH_SEC
    if age < ttl and PANEL_CACHE['panels']:
        dbg(f"panels from cache (age={age*1000:.0f}ms, ttl={ttl*1000:.0f}ms)")
        return PANEL_CACHE['panels']
    return refresh_panels(page)


# ============================================================
#  Panel selection: HIGHEST threshold for matching over
# ============================================================
def pick_best_panel(panels, target_over):
    matching = [p for p in panels if p['over'] == target_over]
    if not matching:
        return None
    return max(matching, key=lambda p: p['threshold'])


# ============================================================
#  Click
# ============================================================
def click_panel(page, panel):
    cx = ANCHOR_RECT['x'] + (panel['x'] - ANCHOR_LEFT) + panel['w'] / 2
    cy = ANCHOR_RECT['y'] + (panel['y'] - ANCHOR_TOP) + panel['h'] / 2
    try:
        page.mouse.click(cx, cy, delay=random.randint(2, 4))
        return cx, cy
    except Exception as e:
        log(f"   Click error: {e}")
        return None, None


# ============================================================
#  Event processing (dynamic over/threshold)
# ============================================================
def process_event(page, ev_text, event_type, over_num, over_dec, ball, xball):
    global _last_click_at

    t0 = time.perf_counter()

    # Event's over determines target over. Always try current first.
    # Fallback to next over in case site advanced immediately.
    target_candidates = [over_num, over_num + 1]

    # First attempt: read cache / OCR
    panels = get_panels(page)
    best = None
    for tc in target_candidates:
        b = pick_best_panel(panels, tc)
        if b:
            best = b
            break

    # Safety net: if nothing matched, force a fresh OCR (cache might be stale)
    if not best:
        dbg("no match in current cache - forcing fresh OCR")
        panels = refresh_panels(page)
        for tc in target_candidates:
            b = pick_best_panel(panels, tc)
            if b:
                best = b
                break

    if not best:
        summary = [(p['over'], p['side'], p['threshold']) for p in panels]
        log(f"[NO-MATCH] type={event_type} want_overs={target_candidates} "
            f"visible_panels={summary}")
        return False

    # Dedup
    dedup_key = (best['over'], best['side'], best['threshold'])
    if dedup_key in _recent_clicked:
        log(f"[ALREADY-CLICKED] panel={dedup_key}")
        return False

    # Click
    cx, cy = click_panel(page, best)
    if cx is None:
        return False

    ms = round((time.perf_counter() - t0) * 1000)
    log(f"[CLICK] {event_type} -> over={best['over']} "
        f"{best['side']} {best['threshold']} @({cx:.0f},{cy:.0f}) [+{ms}ms]")

    _recent_clicked.append(dedup_key)
    _last_click_at = time.time()

    # Invalidate + pre-warm (short delay so browser processes click state first)
    PANEL_CACHE['ts'] = 0
    try:
        page.wait_for_timeout(80)     # let panel update
        refresh_panels(page)
    except Exception:
        pass

    return True


# ============================================================
#  Calibration save
# ============================================================
def setup_from_done(page, final_targets):
    global ANCHOR_RECT, MAPPED_TARGETS

    page.evaluate(
        "() => { let b = document.getElementById('__pw_target_guide'); if(b) b.remove(); }")
    page.wait_for_timeout(800)

    ANCHOR_RECT = {'x': ANCHOR_LEFT, 'y': ANCHOR_TOP,
                   'w': ANCHOR_W, 'h': ANCHOR_H}
    log(f"   Anchor rect: {ANCHOR_RECT}")
    at = ocr_anchor_text(page)
    if at:
        log(f"   Anchor text: {at!r}")

    MAPPED_TARGETS = []
    for t in final_targets:
        MAPPED_TARGETS.append({
            'x': int(t['x']), 'y': int(t['y']),
            'w': int(t['w']), 'h': int(t['h']),
        })
    log(f"   Saved {len(MAPPED_TARGETS)} panel boxes")

    log("   Initial panel OCR:")
    panels = refresh_panels(page)
    for p in panels:
        log(f"      [{p['idx']}] over={p['over']} {p['side']} "
            f"{p['threshold']}  raw={p.get('raw','')!r}")


# ============================================================
#  Calibration UI JS (default 1 row, +1/-1, blue anchor fixed)
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
    let rowCount = window.__row_count || 1;
    let baseY = 400, anchorW = 100, anchorH = 34;
    let leftX = 70, rightX = 425, boxW = 220, boxH = 30, rowGap = 3.34;

    // Blue anchor
    const blueBox = document.createElement('div');
    blueBox.style.cssText = `position:absolute;left:${leftX}px;top:${baseY}px;width:${anchorW}px;height:${anchorH}px;border:2px solid #00aaff;background:rgba(0,170,255,0.4);box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:10px;`;
    blueBox.textContent = "Anchor";
    box.appendChild(blueBox);

    let firstRowY = baseY + anchorH;
    let finalTargets = [];
    for (let i = 0; i < rowCount; i++) {
        let y = firstRowY + i * (boxH + rowGap);
        const leftBox = document.createElement('div');
        leftBox.style.cssText = `position:absolute;left:${leftX}px;top:${y}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:red;font-weight:bold;font-size:9px;`;
        leftBox.textContent = `R${i+1}L`;
        box.appendChild(leftBox);
        const rightBox = document.createElement('div');
        rightBox.style.cssText = `position:absolute;left:${rightX}px;top:${y}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:red;font-weight:bold;font-size:9px;`;
        rightBox.textContent = `R${i+1}R`;
        box.appendChild(rightBox);
        finalTargets.push({x: leftX,  y: y, w: boxW, h: boxH});
        finalTargets.push({x: rightX, y: y, w: boxW, h: boxH});
    }

    let btnTop = firstRowY + rowCount * (boxH + rowGap) + 15;
    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = `position:absolute;left:${leftX}px;top:${btnTop}px;pointer-events:auto;display:flex;gap:10px;align-items:center;`;

    const doneBtn = document.createElement('button');
    doneBtn.textContent = 'DONE';
    doneBtn.style.cssText = 'padding:10px 30px;font-size:16px;font-weight:bold;background:#00c853;color:white;border:none;border-radius:6px;cursor:pointer;';
    doneBtn.onclick = () => { window.__pw_final_targets = finalTargets; window.__pw_action = 'done'; };

    const plusBtn = document.createElement('button');
    plusBtn.textContent = '+ 1 Row';
    plusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#2196f3;color:white;border:none;border-radius:6px;cursor:pointer;';
    plusBtn.onclick = () => { window.__row_count = Math.min(9, rowCount + 1); window.__pw_action = 'update'; };

    const minusBtn = document.createElement('button');
    minusBtn.textContent = '- 1 Row';
    minusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff9800;color:white;border:none;border-radius:6px;cursor:pointer;';
    minusBtn.onclick = () => { window.__row_count = Math.max(1, rowCount - 1); window.__pw_action = 'update'; };

    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'RESET';
    resetBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff3d00;color:white;border:none;border-radius:6px;cursor:pointer;';
    resetBtn.onclick = () => { window.__pw_action = 'reset'; };

    const info = document.createElement('span');
    info.style.cssText = 'color:white;font-size:13px;padding:0 10px;';
    info.textContent = `Rows: ${rowCount}  (default: 1 row × 2 columns)`;

    btnContainer.appendChild(doneBtn);
    btnContainer.appendChild(plusBtn);
    btnContainer.appendChild(minusBtn);
    btnContainer.appendChild(resetBtn);
    btnContainer.appendChild(info);
    box.appendChild(btnContainer);

    const infoWrap = document.createElement('div');
    infoWrap.style.cssText = 'position:absolute;left:10px;top:10px;background:rgba(0,0,0,0.75);padding:10px 14px;border-radius:6px;color:white;font-size:13px;font-family:sans-serif;';
    infoWrap.innerHTML =
        '<b>Quick Events Calibration</b><br>' +
        '• Blue box = anchor (fixed reference)<br>' +
        '• Red boxes = clickable panels<br>' +
        '• Default: 1 row × 2 columns<br>' +
        '• +1 / -1 to add/remove rows<br>' +
        '• Click DONE once — panel text change handled automatically';
    box.appendChild(infoWrap);
}
"""


# ============================================================
#  WebSocket
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
                            handle_reset(m.get('from'), m.get('to'))
                            continue
                        WS_STATE["msg_count"] += 1
                        WS_STATE["last_ts"] = time.time()
                        try:
                            event_q.put_nowait(m)
                        except queue.Full:
                            dbg("event_q full")
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
    global MULTI_MODE, _last_dedup_key

    if len(sys.argv) < 2:
        print("Usage: python clicker_custom.py <match_key> [multi|single]")
        return
    match_filter = sys.argv[1].strip()

    if len(sys.argv) >= 3:
        arg2 = sys.argv[2].strip().lower()
        if arg2 == "multi":
            MULTI_MODE = True
        elif arg2 in ("single", "v15"):
            MULTI_MODE = False

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
        log(f"Match page: {page.url}")

        print("\n" + "=" * 65)
        print("Go to browser and select 'Quick Events'")
        print("Blue anchor fixed, red boxes = 1 row × 2 columns by default")
        print("Use '+1 Row' / '-1 Row' to adjust. Click DONE once.")
        print("=" * 65 + "\n")

        while not _stop_event.is_set():
            try:
                page.evaluate("() => { window.__pw_action = null; }")
                page.evaluate(SHOW_TARGETS_JS)
            except Exception as e:
                log(f"Calibration setup error: {e}")
                break
            print("Waiting for calibration (DONE / +1 / -1 / RESET)...")
            action = None
            while not _stop_event.is_set():
                try:
                    action = page.evaluate("() => window.__pw_action")
                except Exception as e:
                    log(f"Calibration poll error: {e}")
                    break
                if action in ('done', 'update', 'reset'):
                    break
                page.wait_for_timeout(200)
            if action == 'done':
                try:
                    final_targets = page.evaluate("() => window.__pw_final_targets")
                    setup_from_done(page, final_targets)
                except Exception as e:
                    log(f"Calibration save error: {e}")
                break
            elif action == 'reset':
                try:
                    page.evaluate("() => { window.__row_count = 1; }")
                except Exception:
                    break
                continue
            elif action == 'update':
                continue
            else:
                break

        if not MAPPED_TARGETS:
            log("No targets mapped - aborting.")
            browser.close()
            return

        start_ws_thread()
        log(f"[WS] waiting... (match_key={match_filter})")
        log(f"   Mode: {'MULTI (fast, 1.5s cache)' if MULTI_MODE else 'SINGLE (fresh OCR)'}")
        log("   Panel text is DYNAMIC - over/threshold read fresh each time")
        log("   Wicket/4/6/wide+nb/noball -> click panel with HIGHEST threshold")
        log("   Other events (dot/1/2/3/plain wide) -> skip")
        log(f"   Dedup window: {DEDUP_WINDOW}\n")

        _last_dedup_key = None

        try:
            while not _stop_event.is_set():
                check_pending_reset()

                try:
                    m = event_q.get(timeout=EVENT_TIMEOUT_SEC)
                except queue.Empty:
                    if NO_IDLE:
                        continue
                    if event_q.qsize() > 0:
                        continue
                    if time.time() - _last_click_at < POST_CLICK_COOLDOWN:
                        continue
                    if time.time() - WS_STATE["last_ts"] < IDLE_SKIP_WS_RECENT_SEC:
                        continue
                    if time.time() - PANEL_CACHE['ts'] < IDLE_REFRESH_SEC:
                        continue
                    try:
                        refresh_panels(page)
                    except Exception:
                        pass
                    continue

                if m.get("t") != "ev":
                    continue

                ev = str(m.get("ev") or "").strip()
                if not ev:
                    continue

                over_str = str(m.get("over", ""))
                bkey = m.get("xball") or m.get("ball") or ""
                dedup_key = f"{over_str}|{bkey}|{ev}"
                if dedup_key == _last_dedup_key:
                    log(f"[DUP] {dedup_key}")
                    continue
                _last_dedup_key = dedup_key

                over_num, over_dec, ball, xball = parse_over_ball(m)
                if over_num is None:
                    continue

                lag_str = ""
                srv_ts = m.get("_server_ts")
                if srv_ts:
                    lag_str = f" | lag={round(time.time()*1000 - srv_ts, 1)}ms"
                xb_str = f" xball={xball}" if xball is not None else ""

                event_type = classify_event(ev)
                if event_type is None:
                    log(f"[EV] ev={ev} | over={m.get('over')} "
                        f"ball={m.get('ball')}{xb_str}{lag_str} -> skip")
                    continue

                log(f"[EV] ev={ev} | over={m.get('over')} over_dec={over_dec} "
                    f"ball={m.get('ball')}{xb_str}{lag_str} -> {event_type}")

                try:
                    process_event(page, ev, event_type,
                                  over_num, over_dec, ball, xball)
                except Exception as e:
                    log(f"[ERR] process_event: {type(e).__name__}: {e}")

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
