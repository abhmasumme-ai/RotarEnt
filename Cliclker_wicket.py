"""
clicker.py - v25 (v24 dynamic Under-finder + v15 compatibility, English only)
==============================================================================
Core logic (preserved from v24):
  1. Two vertical boxes on screen (Over top, Under bottom).
  2. On wicket event, scan the panel for the "Under" text dynamically.
  3. Click wherever "Under" is found (survives box swaps).
  4. First pass uses OCR; caches "Under" text template for fast template-match
     on subsequent wickets.

Compatibility fixes (from v15):
  - Binary WS frames (bytes + str) handled
  - WS auth token via ?token= (env CLICKER_WS_TOKEN)
  - Thread-safe reset (threading.Event)
  - Bounded event queue (max 500)
  - Graceful shutdown via threading.Event
  - Env/CLI overrides for input() calls (headless friendly)
  - Fast single mouse.click() action
  - English-only logs
  - parse_over_ball handles int / float / str
  - WS thread crash logged
  - Template cache invalidation on reset

Bug fixes (from v24):
  - Wicket throttle reduced from 3s to 1.5s (still prevents double-click)
  - Template match threshold configurable via env
  - Template image invalidated if too small (OCR garbage)
  - Combined region uses min/max coordinates correctly
  - Fallback always clicks the configured under box (never aborts)

Env:
  CLICKER_WS_URL                default ws://127.0.0.1:5000/ws
  CLICKER_WS_TOKEN              optional WS auth token
  CLICKER_HEADLESS              1 = headless browser
  CLICKER_DEBUG                 1 = verbose
  CLICKER_SITE_URL              optional (else prompt)
  CLICKER_MATCH_URL             optional (else prompt)
  CLICKER_PROXY_URL             optional proxy URL
  CLICKER_TEMPLATE_THRESHOLD    default 0.82 (template match score)
  TESSERACT_CMD                 custom tesseract path
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
TEMPLATE_MATCH_THRESHOLD = float(os.environ.get("CLICKER_TEMPLATE_THRESHOLD", "0.82"))
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

# Timing
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.05
EVENT_Q_MAXSIZE = 500
WICKET_THROTTLE_S = 1.5
COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82

# OCR
OCR_CFG_HEADER = '--oem 1 --psm 7'
OCR_CFG_MULTI = '--oem 1 --psm 6'
OCR_UPSCALE = 2

# Viewport
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 240
ANCHOR_H = 34

# State
ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = []
UNDER_TEMPLATE = {}
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0
_last_wicket_time = 0.0

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
    """Called from main thread. Clears template cache safely."""
    if _pending_reset.is_set():
        _pending_reset.clear()
        UNDER_TEMPLATE.clear()
        log("[RESET] template cache cleared")


# ============================================================
#  Calibration UI (JS) - vertical 2-box layout
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
    let baseY = 400, anchorW = 240, anchorH = 34, leftX = 70, boxW = 420, boxH = 30, rowGap = 4;

    const blueBox = document.createElement('div');
    blueBox.style.cssText = `position:absolute;left:${leftX}px;top:${baseY}px;width:${anchorW}px;height:${anchorH}px;border:2px solid #00aaff;background:rgba(0,170,255,0.4);box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:10px;`;
    blueBox.textContent = "Anchor";
    box.appendChild(blueBox);

    let y1 = baseY + anchorH;
    let y2 = y1 + boxH + rowGap;

    const box1 = document.createElement('div');
    box1.style.cssText = `position:absolute;left:${leftX}px;top:${y1}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;`;
    box1.textContent = "OVER (Top)";
    box1.style.color = "red"; box1.style.fontSize = "12px"; box1.style.padding = "4px";
    box.appendChild(box1);

    const box2 = document.createElement('div');
    box2.style.cssText = `position:absolute;left:${leftX}px;top:${y2}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;`;
    box2.textContent = "UNDER (Bottom)";
    box2.style.color = "red"; box2.style.fontSize = "12px"; box2.style.padding = "4px";
    box.appendChild(box2);

    let finalTargets = [
        {side: 'over', x: leftX, y: y1, w: boxW, h: boxH},
        {side: 'under', x: leftX, y: y2, w: boxW, h: boxH}
    ];

    let btnTop = y2 + boxH + 15;
    const btnContainer = document.createElement('div');
    btnContainer.style.cssText = `position:absolute;left:${leftX}px;top:${btnTop}px;pointer-events:auto;display:flex;gap:10px;`;

    const doneBtn = document.createElement('button');
    doneBtn.textContent = 'DONE';
    doneBtn.style.cssText = 'padding:10px 30px;font-size:16px;font-weight:bold;background:#00c853;color:white;border:none;border-radius:6px;cursor:pointer;';
    doneBtn.onclick = () => { window.__pw_final_targets = finalTargets; window.__pw_action = 'done'; };

    btnContainer.appendChild(doneBtn);
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
    for t in final_targets:
        MAPPED_TARGETS.append({
            "side": t.get('side'),
            "x": int(t['x']), "y": int(t['y']),
            "w": int(t['w']), "h": int(t['h'])
        })

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
#  Dynamic Under-finder (fastest path)
# ============================================================
def _combined_region(ax, ay):
    """Return bounding rect covering all mapped boxes in screen coords."""
    if not MAPPED_TARGETS:
        return None
    xs = [ax + (t["x"] - ANCHOR_LEFT) for t in MAPPED_TARGETS]
    ys = [ay + (t["y"] - ANCHOR_TOP) for t in MAPPED_TARGETS]
    min_x = int(min(xs))
    min_y = int(min(ys))
    max_x = int(max(x + t["w"] for x, t in zip(xs, MAPPED_TARGETS)))
    max_y = int(max(y + t["h"] for y, t in zip(ys, MAPPED_TARGETS)))
    if max_x <= min_x or max_y <= min_y:
        return None
    return min_x, min_y, max_x - min_x, max_y - min_y


def find_under_position(page, ax, ay):
    """
    Fast path: single screenshot of combined region, then either
      (a) template match cached "under" text (~10ms), or
      (b) OCR one pass across the region (~40ms), caching template for next time.
    Returns (cx, cy) screen coords of the "under" text center, or None.
    """
    region = _combined_region(ax, ay)
    if region is None:
        return None
    min_x, min_y, region_w, region_h = region

    try:
        raw = page.screenshot(
            clip={"x": min_x, "y": min_y, "width": region_w, "height": region_h},
            type="png", scale="css")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Fast path: template match
        tpl = UNDER_TEMPLATE.get("img")
        if (tpl is not None
                and tpl.shape[0] <= img.shape[0]
                and tpl.shape[1] <= img.shape[1]):
            res = cv2.matchTemplate(img, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= TEMPLATE_MATCH_THRESHOLD:
                cx = min_x + max_loc[0] + tpl.shape[1] / 2
                cy = min_y + max_loc[1] + tpl.shape[0] / 2
                dbg(f"template match conf={max_val:.2f}")
                return (cx, cy)

        # Slow path: OCR once over whole region
        scale = OCR_UPSCALE
        big = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        data = pytesseract.image_to_data(
            big, config=OCR_CFG_MULTI, output_type=pytesseract.Output.DICT
        )

        for i, word in enumerate(data['text']):
            if 'under' in word.lower().strip():
                wx = data['left'][i] / scale
                wy = data['top'][i] / scale
                ww = data['width'][i] / scale
                wh = data['height'][i] / scale

                # Cache "under" text template (with small padding)
                pad = 2
                tx1 = max(0, int(wx) - pad)
                ty1 = max(0, int(wy) - pad)
                tx2 = min(img.shape[1], int(wx + ww) + pad)
                ty2 = min(img.shape[0], int(wy + wh) + pad)
                if tx2 > tx1 and ty2 > ty1 and (tx2 - tx1) >= 10 and (ty2 - ty1) >= 6:
                    UNDER_TEMPLATE["img"] = img[ty1:ty2, tx1:tx2].copy()

                cx = min_x + wx + ww / 2
                cy = min_y + wy + wh / 2
                dbg(f"OCR found 'under' -> ({cx:.0f},{cy:.0f})")
                return (cx, cy)
    except Exception as e:
        log(f"[WARN] find_under_position error: {e}")

    return None


def do_click_xy(page, cx, cy):
    try:
        page.mouse.click(cx, cy, delay=random.randint(2, 5))
    except Exception as e:
        log(f"   Click error: {e}")
        return None, None
    return cx, cy


def do_click_box(page, box, ax, ay):
    x = ax + (box["x"] - ANCHOR_LEFT)
    y = ay + (box["y"] - ANCHOR_TOP)
    cx = x + box["w"] / 2
    cy = y + box["h"] / 2
    return do_click_xy(page, cx, cy)


# ============================================================
#  Wicket handling
# ============================================================
def handle_wicket(page):
    global _last_wicket_time
    t0 = time.perf_counter()

    if time.time() - _last_wicket_time < WICKET_THROTTLE_S:
        dbg("wicket throttle - skipping")
        return False
    _last_wicket_time = time.time()

    ax, ay = find_anchor_fast(page)
    if ax is None:
        log("[NO-ANCHOR] anchor not found")
        return False

    pos = find_under_position(page, ax, ay)
    if pos is not None:
        cx, cy = do_click_xy(page, pos[0], pos[1])
        if cx is None:
            return False
        ms = round((time.perf_counter() - t0) * 1000)
        log(f"[CLICK] UNDER @({cx:.0f},{cy:.0f}) [+{ms}ms]")
        return True

    # Fallback: configured under box
    under_box = next((t for t in MAPPED_TARGETS if t["side"] == "under"), None)
    if under_box:
        cx, cy = do_click_box(page, under_box, ax, ay)
        if cx is None:
            return False
        ms = round((time.perf_counter() - t0) * 1000)
        log(f"[FALLBACK] 'Under' text not found -> configured Under box "
            f"@({cx:.0f},{cy:.0f}) [+{ms}ms]")
        return True

    log("[FAIL] no Under box available")
    return False


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
        print("Go to browser and select 'Next Wicket Fall' market")
        print("Adjust red boxes: TOP box over Over, BOTTOM box over Under")
        print("Click 'DONE' when ready")
        print("=" * 65 + "\n")

        while not _stop_event.is_set():
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("Waiting for calibration (DONE)...")
            while not _stop_event.is_set():
                action = page.evaluate("() => window.__pw_action")
                if action == 'done':
                    break
                page.wait_for_timeout(200)

            final_targets = page.evaluate("() => window.__pw_final_targets")
            setup_from_done(page, final_targets)
            break

        if not MAPPED_TARGETS:
            log("No targets mapped - aborting.")
            browser.close()
            return

        start_ws_thread()
        log(f"[WS] waiting... (match_key={match_filter})")
        log("   v25 - compatible with main.py v8/v9/v10")
        log("   On wicket: dynamic search for 'Under' text, click wherever found\n")

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

                text, kind, _ = ball_info(m)
                if kind != "wicket":
                    continue

                log(f"[WICKET] Out detected | over={m.get('over')} "
                    f"ball={m.get('ball')}")
                handle_wicket(page)

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
