"""
clicker.py — Playwright auto-clicker (v6: strict + cached-header = fast + accurate)

Strategy:
  - STRICT: click only when panel header == WS (over AND delivery exact)
  - FAST: cache panel header during idle; instant click if cache fresh
  - Background OCR polls panel between events (~300ms per poll)
  - Anchor trust 30s; auto re-anchor on garbage

Expected:
  - Latency: ~50-150ms (cache hit) / ~400-600ms (needs fresh OCR)
  - Wrong-delivery clicks: 0%
  - Miss rate: ~10-20% (WS lag/drops)

Usage:
    python clicker.py <match_key>
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

import websockets
from playwright.sync_api import sync_playwright
import numpy as np
import cv2
import pytesseract

# ================= CONFIG =================
WS_URL = "ws://127.0.0.1:5000/ws"
RECONNECT_SEC = 3
PING_SEC = 30
HEADLESS = os.environ.get("CLICKER_HEADLESS") == "1"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
ANCHOR_TRUST_SEC = 30
CACHE_FRESH_SEC = 0.8             # cache younger than this = trusted
HEADER_CROP_W = 150               # "N over, M delivery" area width
MAX_ATTEMPTS = 3
RETRY_SLEEP = 0.06

VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 100
ANCHOR_H = 34

ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = []
market_del_counter = 0
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0
TARGET_ROWS = 3

# ⚡ Fast panel cache
panel_cache = {
    "over": None,
    "del": None,
    "ts": 0.0,
    "ax": None,
    "ay": None,
}

WS_STATE = {"match_filter": "", "up": False, "msg_count": 0, "last_ts": 0.0}
event_q = queue.SimpleQueue()

INTERMEDIATE_PATTERNS = [
    r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)',
    r'^ball in ?air', r'^boundary ?check',
    r'^(third ?umpire|3rd ?umpire)',
    r'^(confirm(ing)?|checking)', r'^(review|drs)',
    r'^(batsman|bowler|new ?batter|misfield|dropped|overthrow)',
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_intermediate(ev):
    ev_low = ev.lower().strip()
    return any(re.search(p, ev_low) for p in INTERMEDIATE_PATTERNS)


def is_ball_chalu(ev):
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)', ev.lower().strip()))


# ============ JS ============
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
    let baseY = 400, anchorW = 100, anchorH = 34, leftX = 70, rightX = 425, boxW = 220, boxH = 30, rowGap = 3.34;
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
    plusBtn.textContent = '+ 3 Rows';
    plusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#2196f3;color:white;border:none;border-radius:6px;cursor:pointer;';
    plusBtn.onclick = () => { window.__row_count = Math.min(9, rowCount + 3); window.__pw_action = 'update'; };
    const minusBtn = document.createElement('button');
    minusBtn.textContent = '- 3 Rows';
    minusBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff9800;color:white;border:none;border-radius:6px;cursor:pointer;';
    minusBtn.onclick = () => { window.__row_count = Math.max(3, rowCount - 3); window.__pw_action = 'update'; };
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'RESET';
    resetBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff3d00;color:white;border:none;border-radius:6px;cursor:pointer;';
    resetBtn.onclick = () => { window.__pw_action = 'reset'; };
    btnContainer.appendChild(doneBtn);
    btnContainer.appendChild(plusBtn);
    btnContainer.appendChild(minusBtn);
    btnContainer.appendChild(resetBtn);
    box.appendChild(btnContainer);
}
"""

SCROLL_JS = """
(dy) => {
    let x = 400, y = 500;
    let el = document.elementFromPoint(x, y);
    while (el) {
        let style = window.getComputedStyle(el);
        if ((style.overflowY === 'auto' || style.overflowY === 'scroll') && el.scrollHeight > el.clientHeight) {
            el.scrollBy(0, dy);
            return true;
        }
        el = el.parentElement;
    }
    return false;
}
"""


# ============ Calibration ============
def setup_from_done(page, final_targets):
    global market_del_counter, CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    log("⚙️  স্ক্রিন থেকে নীল বক্স (Anchor) এবং লাল বক্সের লেখা সেভ করা হচ্ছে...")

    page.evaluate("() => { let b = document.getElementById('__pw_target_guide'); if(b) b.remove(); }")
    page.wait_for_timeout(1000)

    raw = page.screenshot(type="png", scale="css")
    src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    anchor_img = src_img[ANCHOR_TOP:ANCHOR_TOP+ANCHOR_H, ANCHOR_LEFT:ANCHOR_LEFT+ANCHOR_W]
    ANCHOR_TEMPLATE.update({"img": anchor_img, "w": ANCHOR_W, "h": ANCHOR_H})
    res = cv2.matchTemplate(src_img, anchor_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= COMPARE_THRESHOLD:
        CALIBRATED_ANCHOR_POS = max_loc
        last_anchor_pos = max_loc
        _anchor_trust_until = time.time() + ANCHOR_TRUST_SEC
    log("   ✅ Anchor (Blue Box) সফলভাবে সেভ হয়েছে।")

    MAPPED_TARGETS.clear()
    max_del = 0
    KNOWN_LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    max_boxes = TARGET_ROWS * 2

    for i, t in enumerate(final_targets[:max_boxes]):
        x, y, w, h = int(t['x']), int(t['y']), int(t['w']), int(t['h'])
        side_hint = (t.get('side') or '').lower()
        crop = src_img[y:y+h, x:x+w]
        if crop.size == 0:
            continue
        crop_r = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(crop_r, config='--psm 6').lower()
        log(f"      🔍 [BOX {i}] side={side_hint or '?'}  OCR={text!r}")

        m_del = re.search(r'(\d+)\s*-?\s*deliver', text)
        if m_del:
            max_del = max(max_del, int(m_del.group(1)))

        side = side_hint if side_hint in ('over', 'under') else ('over' if i % 2 == 0 else 'under')
        line = None
        m_ou = re.search(r'\b(over|under|ovr|undr)\b\s*(\d+\.?\d*)', text)
        if not m_ou:
            m_ou = re.search(r'(\d+\.?\d*)\s*\b(over|under|ovr|undr)\b', text)
        if m_ou:
            for g in m_ou.groups():
                if g and re.fullmatch(r'\d+\.?\d*', g):
                    line = float(g)
                elif g and g[0] == 'o':
                    side = 'over'
                elif g and g[0] == 'u':
                    side = 'under'
        if line is None:
            floats = [float(n) for n in re.findall(r'(\d+\.\d+)', text)]
            for f in floats:
                if f in KNOWN_LINES:
                    line = f
                    break
            if line is None and floats:
                line = floats[0]
        if line is None:
            line = KNOWN_LINES[i // 2] if (i // 2) < len(KNOWN_LINES) else 0.5

        MAPPED_TARGETS.append({"side": side, "line": line, "x": x, "y": y, "w": w, "h": h})
        log(f"      ✅ ম্যাপিং: {side.upper()} {line}")

    market_del_counter = max_del
    log(f"   📊 বর্তমান ডেলিভারি: {market_del_counter}")
    log(f"   🎯 ম্যাপ হয়েছে {len(MAPPED_TARGETS)}/{max_boxes}টি টার্গেট")
    if not MAPPED_TARGETS:
        log("   ❌ কোনো বক্স ম্যাপ হয়নি")


# ============ Decide click target ============
def decide_side_line(run, kind):
    if kind == 'wide':
        if run == 1: return ("over", 0.5)
        elif 2 <= run <= 4: return ("over", 1.5)
        elif run >= 5: return ("over", 3.5)
    elif kind == 'noball':
        if run == 1: return ("over", 0.5)
        elif 2 <= run <= 4: return ("over", 1.5)
        elif run >= 5: return ("over", 3.5)
    elif kind == 'wicket':
        if run == 0: return ("under", 0.5)
        elif 1 <= run <= 3: return ("over", 1.5)
        elif run >= 4: return ("over", 3.5)
    else:
        if run == 0: return ("under", 0.5)
        known = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        return ("over", max([ln for ln in known if run > ln], default=0.5))
    return ("over", 0.5)


# ============ Anchor ============
def invalidate_anchor():
    global _anchor_trust_until
    _anchor_trust_until = 0


def find_anchor_fast(page):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    now = time.time()

    cached = CALIBRATED_ANCHOR_POS or last_anchor_pos
    if cached and now < _anchor_trust_until:
        return cached

    if cached:
        ax, ay = cached
        if 0 <= ax <= VIEW_W - ANCHOR_W and 0 <= ay <= VIEW_H - ANCHOR_H:
            try:
                raw = page.screenshot(clip={"x": ax, "y": ay, "width": ANCHOR_W, "height": ANCHOR_H},
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

    raw = page.screenshot(type="png", scale="css")
    src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    res = cv2.matchTemplate(src_img, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= COMPARE_THRESHOLD:
        CALIBRATED_ANCHOR_POS = max_loc
        last_anchor_pos = max_loc
        _anchor_trust_until = now + ANCHOR_TRUST_SEC
        return max_loc[0], max_loc[1]
    return None, None


# ============ ⚡ Header-only OCR → fast (~200-400ms) ============
def refresh_panel_header(page, probe_x, probe_y, probe_h):
    """OCR only 'N over, M delivery' area. Updates panel_cache."""
    global panel_cache

    ax, ay = find_anchor_fast(page)
    if ax is None:
        return False

    try:
        raw = page.screenshot(
            clip={"x": probe_x, "y": probe_y, "width": HEADER_CROP_W, "height": probe_h},
            type="png", scale="css",
        )
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.size == 0:
            return False
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(crop, config='--psm 6').lower()

        # Try "N over, M delivery"
        m = re.search(r'(\d+)\s*-?\s*over\D+?(\d+)\s*-?\s*deliver', text, re.IGNORECASE)
        if not m:
            # Fallback: just two numbers
            m = re.search(r'(\d+)\D+?(\d+)', text)

        if m:
            panel_cache["over"] = int(m.group(1))
            panel_cache["del"] = int(m.group(2))
            panel_cache["ts"] = time.time()
            panel_cache["ax"] = ax
            panel_cache["ay"] = ay
            return True
    except Exception:
        pass
    return False


def do_click(page, target_box, ax, ay):
    x = ax + (target_box["x"] - ANCHOR_LEFT)
    y = ay + (target_box["y"] - ANCHOR_TOP)
    cx = x + target_box["w"] / 2
    cy = y + target_box["h"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.wait_for_timeout(random.randint(3, 6))
    page.mouse.up()
    return cx, cy


def process_second_push(page, ball_key, run, kind, ws_over, market_del):
    global market_del_counter
    t_start = time.time()
    side_want, line_want = decide_side_line(run, kind)
    log(f"🎯 [TARGET] ball={ball_key} run={run} ({kind}) → {side_want} {line_want} | WS: {ws_over}-{market_del}")

    target_box = None
    for t in MAPPED_TARGETS:
        if t["side"] == side_want and abs(t["line"] - line_want) < 0.01:
            target_box = t
            break

    if not target_box:
        log(f"❌ [NO-CLICK] ball={ball_key} — {side_want} {line_want} not in mapping")
        return False

    for attempt in range(MAX_ATTEMPTS):
        cache_age = time.time() - panel_cache["ts"]

        # ⚡ FAST PATH: cache fresh + match → click instantly
        if (cache_age < CACHE_FRESH_SEC and
            panel_cache["over"] == ws_over and
            panel_cache["del"] == market_del and
            panel_cache["ax"] is not None):
            cx, cy = do_click(page, target_box, panel_cache["ax"], panel_cache["ay"])
            elapsed = round((time.time() - t_start) * 1000)
            log(f"✅ [CLICK] FAST cache age={cache_age*1000:.0f}ms panel={ws_over}-{market_del} ({cx},{cy}) [+{elapsed}ms]")
            market_del_counter = panel_cache["del"]
            panel_cache["ts"] = 0   # invalidate for next event
            return True

        # Cache stale or mismatch → refresh via OCR
        ax, ay = find_anchor_fast(page)
        if ax is not None:
            tx = ax + (target_box["x"] - ANCHOR_LEFT)
            ty = ay + (target_box["y"] - ANCHOR_TOP)
            refresh_panel_header(page, tx, ty, target_box["h"])

            # Re-check after refresh
            if (panel_cache["over"] == ws_over and
                panel_cache["del"] == market_del and
                panel_cache["ax"] is not None):
                cx, cy = do_click(page, target_box, panel_cache["ax"], panel_cache["ay"])
                elapsed = round((time.time() - t_start) * 1000)
                log(f"✅ [CLICK] FRESH OCR panel={ws_over}-{market_del} ({cx},{cy}) [+{elapsed}ms]")
                market_del_counter = panel_cache["del"]
                panel_cache["ts"] = 0
                return True

        log(f"      🔍 [VERIFY] try{attempt+1} | panel={panel_cache['over']}-{panel_cache['del']} vs WS={ws_over}-{market_del}")

        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(RETRY_SLEEP)

    log(f"   ⚠️ [NO-CLICK] {ball_key} — panel {panel_cache['over']}-{panel_cache['del']} ≠ WS {ws_over}-{market_del}")
    return False


# ============ WS ============
async def ws_loop():
    while True:
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(WS_URL, ping_interval=None, compression=None, max_size=None) as ws:
                WS_STATE["up"] = True
                log(f"🔌 [WS] connected: {WS_URL}")
                if mf:
                    try:
                        await ws.send(f"watch:{mf}")
                        log(f"👁️  Watching match: {mf}")
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
                        if raw == "pong":
                            continue
                        try:
                            m = json.loads(raw)
                        except (ValueError, TypeError):
                            continue
                        mid_msg = m.get("id")
                        if mf and mid_msg and str(mid_msg) != str(mf):
                            continue
                        WS_STATE["msg_count"] += 1
                        WS_STATE["last_ts"] = time.time()
                        event_q.put(m)
                finally:
                    ptask.cancel()
        except asyncio.CancelledError:
            return
        except Exception:
            pass
        finally:
            WS_STATE["up"] = False
        await asyncio.sleep(RECONNECT_SEC)


def start_ws_thread():
    threading.Thread(target=lambda: asyncio.run(ws_loop()), daemon=True).start()


# ============ Parsing ============
def get_over_delivery(d):
    """WS '11.2' → (12, 2). Panel shows '12 - over, 2 - delivery'."""
    ov = d.get("over")
    if ov is None:
        return None, None
    try:
        s = str(ov)
        if "." in s:
            p = s.split(".")
            return int(p[0]) + 1, int(p[1])
        return int(s) + 1, 0
    except (ValueError, IndexError):
        return None, None


def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev or is_intermediate(ev):
        return (ev, "intermediate", 0)
    if re.match(r'^(wicket|out|w)[.!]*$', ev_low):
        return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|retired|handled|obstruct)', ev_low):
        return (ev, "wicket", 0)
    if re.match(r'^(no[\s_-]?ball|nb)\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        return (ev, "noball", int(m.group(1)) + 1 if m else 1)
    if re.match(r'^wide\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        return (ev, "wide", int(m.group(1)) + 1 if m else 1)
    if re.match(r'^(six|6)\b', ev_low):
        return ("6", "six", 6)
    if re.match(r'^(four|4)\b', ev_low):
        return ("4", "four", 4)
    if re.match(r'^(dot|0)\b', ev_low):
        return ("0", "dot", 0)
    m = re.match(r'^(\d+)', ev_low)
    if m:
        n = int(m.group(1))
        return (str(n), "run", n)
    return (ev, "intermediate", 0)


# ============ MAIN ============
def monitor_and_click():
    global market_del_counter

    if len(sys.argv) < 2:
        print("Usage: python clicker.py <match_key>")
        return
    match_filter = sys.argv[1].strip()
    WS_STATE["match_filter"] = match_filter

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    site_url = input("\nSite link din: ").strip() or "https://example.com"

    proxy_config = None
    print("\n=== Proxy Settings (IPRoyal) ===")
    proxy_url = input("Proxy URL (e.g., http://user:pass@geo.iproyal.com:12321) [Leave blank to skip]: ").strip()
    if proxy_url:
        try:
            if "@" in proxy_url:
                proto, rest = proxy_url.split("://") if "://" in proxy_url else ("http://", proxy_url)
                auth, host_port = rest.split("@")
                user, pwd = auth.split(":")
                proxy_config = {"server": f"{proto}://{host_port}", "username": user, "password": pwd}
            else:
                proxy_config = {"server": proxy_url}
            log(f"✅ Proxy: {proxy_config['server']}")
        except Exception as e:
            log(f"❌ Proxy error: {e}")
    else:
        log("⏭️ Proxy skipped.")

    try:
        pytesseract.image_to_string(np.zeros((30, 100), dtype=np.uint8), config='--psm 6')
        log("   🔥 Tesseract warmed up")
    except Exception:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=HEADLESS,
            args=[f"--window-size={WIN_W},{WIN_H}", "--window-position=0,0"],
        )
        context = browser.new_context(
            viewport={"width": VIEW_W, "height": VIEW_H},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            proxy=proxy_config,
        )
        page = context.new_page()
        page.goto(site_url)

        print("\n=== log in manually ===")
        match_url = input("Match URL: ").strip()
        if not match_url:
            browser.close()
            return
        page.goto(match_url)
        log(f"✅ Match page: {page.url}")

        print("\n" + "=" * 65)
        print("👉 go to browser and select 'Quick Events'")
        print("👉 adjust Red Box below Blue Box")
        print("👉 if all is well click to 'DONE' button")
        print("=" * 65 + "\n")

        while True:
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("⏳ set target on your browser (DONE / + / - / RESET)...")
            while True:
                action = page.evaluate("() => window.__pw_action")
                if action in ('done', 'update', 'reset'):
                    break
                page.wait_for_timeout(200)
            if action == 'done':
                final_targets = page.evaluate("() => window.__pw_final_targets")
                setup_from_done(page, final_targets)
                break
            elif action == 'reset':
                page.evaluate("() => { window.__row_count = 3; }")
                continue
            elif action == 'update':
                continue

        if not MAPPED_TARGETS:
            log("⚠️ কোনো টার্গেট ম্যাপ হয়নি — বন্ধ।")
            browser.close()
            return

        start_ws_thread()
        log(f"👂 [WS] waiting... (match_key={match_filter})")
        log(f"   💡 Panel header cached every ~400ms during idle; click latency ~50-150ms\n")

        last_ws_over = None
        clicked_over = None

        try:
            while True:
                try:
                    m = event_q.get(timeout=0.1)
                except queue.Empty:
                    # ⚡ IDLE: refresh cache for the next event
                    box = MAPPED_TARGETS[0]
                    ax, ay = find_anchor_fast(page)
                    if ax is not None:
                        px = ax + (box["x"] - ANCHOR_LEFT)
                        py = ay + (box["y"] - ANCHOR_TOP)
                        refresh_panel_header(page, px, py, box["h"])
                    continue

                lag_ms = None
                srv_ts = m.get("_server_ts")
                if srv_ts:
                    lag_ms = round(time.time() * 1000 - srv_ts, 1)

                if m.get("t") != "ev":
                    continue

                ev = str(m.get("ev") or "").strip()
                if not ev:
                    continue
                if is_ball_chalu(ev):
                    clicked_over = None
                    continue
                if is_intermediate(ev):
                    continue

                over_str = str(m.get("over") or "")
                if over_str and over_str == clicked_over:
                    continue
                clicked_over = over_str

                ws_over, _ = get_over_delivery(m)
                if ws_over is None:
                    continue

                text, kind, run = ball_info(m)
                if kind == "intermediate":
                    continue

                # Counter: wide/nb count as delivery (panel does same)
                if last_ws_over is None:
                    last_ws_over = ws_over
                    log(f"   ⚙️ [INIT] counter={market_del_counter} (from setup)")
                elif ws_over != last_ws_over:
                    last_ws_over = ws_over
                    market_del_counter = 1
                    log(f"   🔄 [NEW OVER {ws_over}] counter reset → 1")
                else:
                    market_del_counter += 1

                lag_str = f" | lag={lag_ms}ms" if lag_ms is not None else ""
                log(f"🏏 [EV] over={ws_over}.{market_del_counter} ev={text} kind={kind} run={run}{lag_str}")

                process_second_push(page, f"{over_str}_{ev}", run, kind, ws_over, market_del_counter)

        except KeyboardInterrupt:
            print("\n closing...")
        finally:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    monitor_and_click()
