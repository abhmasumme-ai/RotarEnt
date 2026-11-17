"""
clicker.py — Playwright auto-clicker v11 (fastest + strict equality)

Fixes in v11:
  - want_del = ball + 1 (Entity ball already counts wides; no wide_count add)
  - find_anchor_fast: cache blindly trusted (no verify → no slow path)
  - Idle loop: trust renew only, no force_verify
  - Target latency: ~80-130ms per event (was ~1000ms)

Formula (strict):
  want_over = Entity.over + 1                    (0-index → 1-index)
  want_del  = Entity.ball + 1                    (Entity ball includes wides)
  if want_del > 6: want_del = 1; want_over += 1  (over rollover)
  → panel must show exactly (want_over, want_del) to click

Env:
  CLICKER_HEADLESS   1 = headless (default 0)
  CLICKER_STRICT     1 = strict equality (default 1)
                     0 = also accept want+1 (prediction bet)
  TESSERACT_CMD      custom tesseract path
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
STRICT_EQUALITY = os.environ.get("CLICKER_STRICT", "1") != "0"
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

COMPARE_THRESHOLD = 0.70
ANCHOR_TRUST_SEC = 60
HEADER_CROP_W = 150
OCR_RESIZE = 2
IDLE_POLL_MS = 20

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
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0
TARGET_ROWS = 3

comm_state = {
    "over": None,
    "wide_count": 0,
    "has_data": False,
}

live_over_num = None
live_wide_count = 0

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
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
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


# ============ Anchor (blind trust, no verify) ============
def find_anchor_fast(page, force_verify=False):
    """Return cached anchor instantly. Full search only if never calibrated."""
    global CALIBRATED_ANCHOR_POS, last_anchor_pos
    cached = CALIBRATED_ANCHOR_POS or last_anchor_pos
    if cached:
        return cached
    # First-time search only (only if calibration somehow failed)
    try:
        raw = page.screenshot(type="png", scale="css")
        src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(src_img, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= COMPARE_THRESHOLD:
            CALIBRATED_ANCHOR_POS = max_loc
            last_anchor_pos = max_loc
            return max_loc[0], max_loc[1]
    except Exception:
        pass
    return None, None


# ============ OCR (fast, single line) ============
def ocr_header(page, x, y, h):
    """Small clip + 2x resize + --psm 7 → ~50-80ms."""
    try:
        raw = page.screenshot(
            clip={"x": x, "y": y, "width": HEADER_CROP_W, "height": h},
            type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.size == 0:
            return ""
        crop = cv2.resize(crop, None, fx=OCR_RESIZE, fy=OCR_RESIZE,
                          interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(crop, config='--psm 7').lower()
    except Exception:
        return ""


# ============ Click (direct) ============
def do_click(page, target_box, ax, ay):
    x = ax + (target_box["x"] - ANCHOR_LEFT)
    y = ay + (target_box["y"] - ANCHOR_TOP)
    cx = x + target_box["w"] / 2
    cy = y + target_box["h"] / 2
    page.mouse.move(cx, cy, steps=1)
    page.mouse.down()
    page.mouse.up()
    return cx, cy


# ============ Strict-equality click ============
def process_second_push(page, ball_key, run, kind, want_over, want_del):
    """
    Single-shot strict check:
      1. Get cached anchor (0ms — blind trust)
      2. Fresh OCR header (~60ms)
      3. Panel == (want_over, want_del)? → click
      4. Else skip immediately
    """
    t0 = time.perf_counter()
    side_want, line_want = decide_side_line(run, kind)

    log(f"🎯 [TARGET] {ball_key} run={run}({kind}) → {side_want} {line_want} | "
        f"want={want_over}-{want_del} | strict={STRICT_EQUALITY}")

    target_box = None
    for t in MAPPED_TARGETS:
        if t["side"] == side_want and abs(t["line"] - line_want) < 0.01:
            target_box = t
            break
    if not target_box:
        log(f"❌ [NO-MAP] {side_want} {line_want}")
        return False

    ax, ay = find_anchor_fast(page)
    if ax is None:
        log(f"❌ [NO-ANCHOR] {ball_key}")
        return False

    tx = ax + (target_box["x"] - ANCHOR_LEFT)
    ty = ay + (target_box["y"] - ANCHOR_TOP)
    text = ocr_header(page, tx, ty, target_box["h"])

    m = re.search(r'(\d+)\D+(\d+)', text)
    if not m:
        log(f"⏭️ [OCR-FAIL] text={text.strip()!r} "
            f"[+{round((time.perf_counter()-t0)*1000)}ms]")
        return False

    po, pd = int(m.group(1)), int(m.group(2))

    matched = (po == want_over and pd == want_del)

    if not matched and not STRICT_EQUALITY:
        nxt_del, nxt_over = want_del + 1, want_over
        if nxt_del > 6:
            nxt_del, nxt_over = 1, want_over + 1
        matched = (po == nxt_over and pd == nxt_del)

    if not matched:
        log(f"⏭️ [MISS] panel={po}-{pd} want={want_over}-{want_del} "
            f"[+{round((time.perf_counter()-t0)*1000)}ms]")
        return False

    cx, cy = do_click(page, target_box, ax, ay)
    latency = round((time.perf_counter() - t0) * 1000)
    log(f"✅ [CLICK] panel={po}-{pd} target={side_want} {line_want} "
        f"@({cx:.0f},{cy:.0f}) [+{latency}ms]")
    return True


# ============ Comm processing ============
def update_comm_state(m):
    """From a `match` message, extract current over + wide/noball count."""
    global comm_state
    comm = m.get("comm") or []
    if not comm:
        return
    overs = [e.get("ov") for e in comm if e.get("ov") is not None]
    if not overs:
        return
    max_ov = max(overs)
    curr = [e for e in comm if e.get("ov") == max_ov]
    wc = sum(1 for e in curr if (e.get("wd") or 0) > 0 or (e.get("nb") or 0) > 0)
    if comm_state["over"] != max_ov or comm_state["wide_count"] != wc:
        log(f"   📊 [COMM] over={max_ov} deliveries={len(curr)} wide_nb={wc}")
    comm_state["over"] = max_ov
    comm_state["wide_count"] = wc
    comm_state["has_data"] = True


# ============ WS ============
async def ws_loop():
    while True:
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(WS_URL, ping_interval=None,
                                          compression=None, max_size=None) as ws:
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
                        m["_recv_ts"] = time.perf_counter()
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
def parse_over_ball(d):
    ov = d.get("over")
    if ov is None:
        return None, None, None
    ov_str = str(ov).strip()
    try:
        if "." in ov_str:
            parts = ov_str.split(".")
            over_num = int(parts[0])
            ball = int(parts[1]) if parts[1] else 1
        else:
            over_num = int(ov_str)
            ball = 1
    except (ValueError, IndexError):
        return None, None, None

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

    return over_num, ball, xball


def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev or is_intermediate(ev):
        return (ev, "intermediate", 0)
    if re.match(r'^(wicket|out|w)[.!]*$', ev_low):
        runs = int(d.get("runs", 0) or 0)
        return (ev, "wicket", runs)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|retired|handled|obstruct)', ev_low):
        runs = int(d.get("runs", 0) or 0)
        return (ev, "wicket", runs)
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
    global live_over_num, live_wide_count, _anchor_trust_until

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
    proxy_url = input("Proxy URL (e.g., http://user:pass@geo.iproyal.com:12321) "
                      "[Leave blank to skip]: ").strip()
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
            log(f"✅ Proxy: {proxy_config['server']}")
        except Exception as e:
            log(f"❌ Proxy error: {e}")
    else:
        log("⏭️ Proxy skipped.")

    # Tesseract warm-up
    try:
        pytesseract.image_to_string(np.zeros((30, 100), dtype=np.uint8), config='--psm 7')
        log("   🔥 Tesseract warmed up (psm 7)")
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
        log(f"   💡 STRICT_EQUALITY={STRICT_EQUALITY}")
        log(f"   💡 want_over = Entity.over + 1 | want_del = Entity.ball + 1 (rollover > 6)")
        log(f"   💡 Entity ball already counts wides — no extra wide_count\n")

        last_dedup_key = None

        try:
            while True:
                try:
                    m = event_q.get(timeout=IDLE_POLL_MS / 1000)
                except queue.Empty:
                    # Idle: renew trust only (no slow-path verify)
                    if last_anchor_pos:
                        _anchor_trust_until = time.time() + ANCHOR_TRUST_SEC
                    continue

                t = m.get("t")

                if t == "match":
                    update_comm_state(m)
                    continue
                if t != "ev":
                    continue

                ev = str(m.get("ev") or "").strip()
                if not ev:
                    continue
                if is_ball_chalu(ev):
                    last_dedup_key = None
                    continue
                if is_intermediate(ev):
                    continue

                over_str = str(m.get("over") or "")
                dedup_key = f"{over_str}|{ev}"
                if dedup_key == last_dedup_key:
                    log(f"⏭️ [DUP] skip {dedup_key}")
                    continue
                last_dedup_key = dedup_key

                text, kind, run = ball_info(m)
                if kind == "intermediate":
                    continue

                over_num, main_ball, xball = parse_over_ball(m)
                if over_num is None:
                    continue

                if live_over_num != over_num:
                    live_wide_count = 0
                    live_over_num = over_num

                # ── Strict target: NEXT upcoming delivery ──
                # Entity data.ball already includes wides (total deliveries)
                want_over = over_num + 1                 # 0-idx → 1-idx
                want_del = main_ball + 1                 # next delivery
                if want_del > 6:
                    want_del = 1
                    want_over += 1

                srv_ts = m.get("_server_ts")
                lag_ms = round(time.time() * 1000 - srv_ts, 1) if srv_ts else None
                lag_str = f" | lag={lag_ms}ms" if lag_ms is not None else ""

                log(f"🏏 [EV] ev={text} kind={kind} run={run} | "
                    f"raw over={m.get('over')} ball={m.get('ball')} | "
                    f"want={want_over}-{want_del} (ball+1){lag_str}")

                process_second_push(
                    page, f"{over_str}_{ev}", run, kind, want_over, want_del
                )

                if kind in ('wide', 'noball'):
                    live_wide_count += 1
                    log(f"   🔵 [WIDE/NB] live_wide_count → {live_wide_count}")

        except KeyboardInterrupt:
            print("\n closing...")
        finally:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    monitor_and_click()
