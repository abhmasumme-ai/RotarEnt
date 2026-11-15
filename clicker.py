"""
clicker.py — Playwright auto-clicker for cricket control panel

Receives data from main.py exactly like script.js does:
    - WebSocket ws://127.0.0.1:5000/ws
    - On connect: send "watch:<match_key>"
    - Every 30s: send "ping"
    - Filter messages by m.id === match_key
    - Process m.t === "ev" (ball events) → OCR → click

Usage:
    python clicker.py <match_key>

Example:
    python clicker.py 12345

Requirements:
    pip install playwright websockets opencv-python numpy pytesseract pillow
    python -m playwright install chromium
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
from collections import OrderedDict

import websockets
from playwright.sync_api import sync_playwright
import numpy as np
import cv2
import pytesseract

# ================= CONFIG (same as script.js) =================
WS_URL = "ws://127.0.0.1:5000/ws"
RECONNECT_SEC = 3          # script.js: RECONNECT_MS = 3000
PING_SEC = 30              # script.js: setInterval ping 30000
HEADLESS = os.environ.get("CLICKER_HEADLESS") == "1"
CHROME_PATH = os.environ.get("CLICKER_CHROME_PATH") or None
TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

COMPARE_THRESHOLD = 0.70
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = []
market_del_counter = 0

CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None

TARGET_ROWS = 3

# ⭐ Shared state between WS thread and main Playwright thread
# (like S.matchKey in script.js)
WS_STATE = {
    "match_filter": "",
    "up": False,
    "msg_count": 0,
    "last_ts": 0.0,
}

# ⭐ FastQueue (SimpleQueue = lock-free, faster than Queue)
event_q = queue.SimpleQueue()

# ⭐ Intermediate events (skip, no click, no counter)
INTERMEDIATE_PATTERNS = [
    r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)',
    r'^ball in ?air',
    r'^boundary ?check',
    r'^(third ?umpire|3rd ?umpire)',
    r'^(confirm(ing)?|checking)',
    r'^(review|drs)',
    r'^(batsman|bowler|new ?batter|misfield|dropped|overthrow)',
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_intermediate(ev):
    ev_low = ev.lower().strip()
    return any(re.search(p, ev_low) for p in INTERMEDIATE_PATTERNS)


def is_ball_chalu(ev):
    ev_low = ev.lower().strip()
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)', ev_low))


# ============ JS snippets (unchanged) ============
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
    let baseY = 400;
    let anchorW = 100;
    let anchorH = 34;
    let leftX = 70;
    let rightX = 425;
    let boxW = 220;
    let boxH = 30;
    let rowGap = 3.34;

    const blueBox = document.createElement('div');
    blueBox.style.cssText = `position:absolute;left:${leftX}px;top:${baseY}px;width:${anchorW}px;height:${anchorH}px;` +
      `border:2px solid #00aaff;background:rgba(0,170,255,0.4);box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:10px;`;
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


# ============ Calibration setup (unchanged) ============
def setup_from_done(page, final_targets):
    global market_del_counter, CALIBRATED_ANCHOR_POS, last_anchor_pos
    log("⚙️  স্ক্রিন থেকে নীল বক্স (Anchor) এবং লাল বক্সের লেখা সেভ করা হচ্ছে...")

    page.evaluate("() => { let b = document.getElementById('__pw_target_guide'); if(b) b.remove(); }")
    page.wait_for_timeout(300)

    raw = page.screenshot(type="png", scale="css")
    np_arr = np.frombuffer(raw, np.uint8)
    src_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    anchor_img = src_img[400:434, 70:170]
    ANCHOR_TEMPLATE.update({"img": anchor_img, "w": 100, "h": 34})

    res = cv2.matchTemplate(src_img, anchor_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= COMPARE_THRESHOLD:
        CALIBRATED_ANCHOR_POS = max_loc
        last_anchor_pos = max_loc

    log("   ✅ Anchor (Blue Box) সফলভাবে সেভ হয়েছে।")

    MAPPED_TARGETS.clear()
    max_del = 0

    max_boxes = TARGET_ROWS * 2
    for t in final_targets[:max_boxes]:
        x, y, w, h = int(t['x']), int(t['y']), int(t['w']), int(t['h'])

        crop = src_img[y:y+h, x:x+w]
        crop_resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        config = '--psm 7 -c tessedit_char_whitelist=0123456789.ovrdeli'
        text = pytesseract.image_to_string(crop_resized, config=config).lower()

        m_del = re.search(r'(\d+)\s*-?\s*delivery', text, re.IGNORECASE)
        if m_del:
            d_num = int(m_del.group(1))
            if d_num > max_del:
                max_del = d_num

        matches = re.findall(r'(over|under)\s*(\d+\.?\d*)', text)
        if matches:
            side, line_str = matches[-1]
            line = float(line_str)
            MAPPED_TARGETS.append({"side": side, "line": line, "x": x, "y": y, "w": w, "h": h})
            log(f"      ✅ ম্যাপিং: {side.upper()} {line} সেভ হয়েছে।")

    market_del_counter = max_del
    log(f"   📊 স্ক্রিন থেকে বর্তমান ডেলিভারি সেট করা হলো: {market_del_counter}")
    log(f"   🎯 শুধুমাত্র প্রথম {TARGET_ROWS}টি রো টার্গেট হিসেবে ম্যাপ করা হয়েছে।")


# ============ Decide click target (unchanged) ============
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
        known_lines = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        best = max([ln for ln in known_lines if run > ln], default=0.5)
        return ("over", best)
    return ("over", 0.5)


def ocr_crop(src_img, x, y, w, h):
    crop = src_img[y:y+h, x:x+w]
    if crop.size == 0:
        return ""
    crop_resized = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    config = '--psm 7 -c tessedit_char_whitelist=0123456789.ovrdeli'
    return pytesseract.image_to_string(crop_resized, config=config).lower()


def find_anchor(page):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos

    raw = page.screenshot(type="png", scale="css")
    np_arr = np.frombuffer(raw, np.uint8)
    src_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

    if CALIBRATED_ANCHOR_POS:
        cx, cy = CALIBRATED_ANCHOR_POS
        if cx + ANCHOR_TEMPLATE["w"] < VIEW_W and cy + ANCHOR_TEMPLATE["h"] < VIEW_H:
            crop = src_img[cy:cy+ANCHOR_TEMPLATE["h"], cx:cx+ANCHOR_TEMPLATE["w"]]
            if crop.size > 0:
                res = cv2.matchTemplate(crop, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= 0.90:
                    return src_img, cx, cy

    if last_anchor_pos:
        ax, ay = last_anchor_pos
        if ax + ANCHOR_TEMPLATE["w"] < VIEW_W and ay + ANCHOR_TEMPLATE["h"] < VIEW_H:
            crop = src_img[ay:ay+ANCHOR_TEMPLATE["h"], ax:ax+ANCHOR_TEMPLATE["w"]]
            if crop.size > 0:
                res = cv2.matchTemplate(crop, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= 0.90:
                    CALIBRATED_ANCHOR_POS = (ax, ay)
                    return src_img, ax, ay

    res = cv2.matchTemplate(src_img, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= COMPARE_THRESHOLD:
        last_anchor_pos = max_loc
        CALIBRATED_ANCHOR_POS = max_loc
        return src_img, max_loc[0], max_loc[1]

    page.evaluate(SCROLL_JS, -2000)
    page.wait_for_timeout(200)
    for _ in range(10):
        raw = page.screenshot(type="png", scale="css")
        np_arr = np.frombuffer(raw, np.uint8)
        src_img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        res = cv2.matchTemplate(src_img, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= COMPARE_THRESHOLD:
            last_anchor_pos = max_loc
            CALIBRATED_ANCHOR_POS = max_loc
            return src_img, max_loc[0], max_loc[1]
        page.evaluate(SCROLL_JS, 300)
        page.wait_for_timeout(200)

    return None, None, None


def process_second_push(page, ball_key, run, kind, ws_over, market_del):
    side_want, line_want = decide_side_line(run, kind)
    log(f"🎯 [TARGET] ball={ball_key} run={run} ({kind}) → {side_want} {line_want} | WS: {ws_over}-{market_del}")

    for attempt in range(3):
        src_img, ax, ay = find_anchor(page)

        if src_img is None:
            log(f"❌ [NO-CLICK] ball={ball_key} — নীল বক্স (Anchor) পাওয়া যায়নি!")
            return False

        target_box = None
        for t in MAPPED_TARGETS:
            if t["side"] == side_want and abs(t["line"] - line_want) < 0.01:
                target_box = t
                break

        if not target_box:
            log(f"❌ [NO-CLICK] ball={ball_key} — ম্যাপিং-এ {side_want} {line_want} নেই!")
            return False

        rel_y = target_box["y"] - 434
        actual_y = ay + rel_y
        actual_x = ax + (target_box["x"] - 70)

        text = ocr_crop(src_img, actual_x, actual_y, target_box["w"], target_box["h"])
        clean_text = re.sub(r'[^a-z0-9]', '', text)

        if str(ws_over) in clean_text and str(market_del) in clean_text:
            center_x = actual_x + (target_box["w"] / 2)
            center_y = actual_y + (target_box["h"] / 2)

            log(f"   🎯 Verified! Clicking at ({center_x},{center_y})")
            try:
                page.mouse.move(center_x, center_y)
                page.mouse.down()
                page.wait_for_timeout(random.randint(5, 10))
                page.mouse.up()
                log(f"✅ [CLICK] ball={ball_key}")
                return True
            except Exception as e:
                log(f"⚠️ [CLICK] error: {e}")
                return False
        else:
            if attempt < 2:
                log(f"   ⏳ [RETRY] WS: {ws_over}-{market_del}, Screen: {clean_text[:10]}... অপেক্ষা করছি")
                page.wait_for_timeout(100)
            else:
                log(f"   ⚠️ [OCR Mismatch] ৩ বার চেষ্টার পরেও মেলেনি। ক্লিক বাতিল।")
                return False


# ============ WS LOOP (script.js pattern) ============
async def ws_loop():
    """
    Exactly like script.js's startLiveListener():
        onopen  → send watch:<matchKey>, start 30s ping
        message → if "pong" skip, else filter by m.id, else put to queue
        onclose → reconnect after RECONNECT_MS
    """
    while True:
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=None,      # script.js: no auto-ping, uses own interval
                compression=None,        # main.py: ws_per_message_deflate=False
                max_size=None,
            ) as ws:
                WS_STATE["up"] = True
                log(f"🔌 [WS] connected: {WS_URL}")

                # ⭐ onopen → sendWatch()  (script.js)
                if mf:
                    try:
                        await ws.send(f"watch:{mf}")
                        log(f"👁️  Watching match: {mf}")
                    except Exception as e:
                        log(f"⚠️ watch send failed: {e}")

                # ⭐ 30s ping  (script.js: setInterval ping 30000)
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
                        # ⭐ pong skip (script.js: `if (event.data === "pong") return;`)
                        if raw == "pong":
                            continue

                        # ⭐ parse + filter by m.id (script.js onmessage logic)
                        try:
                            m = json.loads(raw)
                        except (ValueError, TypeError):
                            continue

                        # script.js: `if (S.matchKey && m.id && String(m.id) !== S.matchKey) return;`
                        mid_msg = m.get("id")
                        if mf and mid_msg and str(mid_msg) != str(mf):
                            continue

                        WS_STATE["msg_count"] += 1
                        WS_STATE["last_ts"] = time.time()

                        # SimpleQueue.put — lock-free, ~1μs
                        event_q.put(m)
                finally:
                    ptask.cancel()

        except asyncio.CancelledError:
            return
        except Exception as e:
            # script.js: onclose → silent, retry after 3s
            pass
        finally:
            WS_STATE["up"] = False

        await asyncio.sleep(RECONNECT_SEC)


def start_ws_thread():
    threading.Thread(target=lambda: asyncio.run(ws_loop()), daemon=True).start()


# ============ Parsing ============
def get_over_delivery(d):
    """main.py 'over' string like '5.3' → (over_1indexed=6, ball_1indexed=4)."""
    ov = d.get("over")
    if ov is None:
        return None, None
    try:
        ov_str = str(ov)
        if "." in ov_str:
            parts = ov_str.split(".")
            return int(parts[0]) + 1, int(parts[1]) + 1
        return int(ov_str) + 1, 1
    except (ValueError, IndexError):
        return None, None


def ball_info(d):
    """main.py 'ev' string → (text, kind, runs)."""
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()

    if not ev:
        return ("", "intermediate", 0)

    if is_intermediate(ev):
        return (ev, "intermediate", 0)

    if re.match(r'^(wicket|out|w)[.!]*$', ev_low):
        return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|retired|handled|obstruct)', ev_low):
        return (ev, "wicket", 0)

    if re.match(r'^(no[\s_-]?ball|nb)\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        runs = int(m.group(1)) + 1 if m else 1
        return (ev, "noball", runs)

    if re.match(r'^wide\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        runs = int(m.group(1)) + 1 if m else 1
        return (ev, "wide", runs)

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


def make_seen_lru(maxlen):
    od = OrderedDict()

    def remember(k):
        if k in od:
            od.move_to_end(k)
        else:
            od[k] = True
        while len(od) > maxlen:
            od.popitem(last=False)

    return remember, lambda k: k in od, lambda k: od.pop(k, None)


# ============ MAIN ============
def monitor_and_click():
    global market_del_counter, last_anchor_pos, CALIBRATED_ANCHOR_POS

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
            log(f"✅ Proxy configured: {proxy_config['server']}")
        except Exception as e:
            log(f"❌ Proxy format error: {e}. Skipping proxy.")
            proxy_config = None
    else:
        log("⏭️ Proxy skipped. Running on local IP.")

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

        print("\n=== ম্যানুয়ালি লগইন করুন ===")
        match_url = input("Match URL: ").strip()
        if not match_url:
            browser.close()
            return
        page.goto(match_url)
        log(f"✅ Match page: {page.url}")

        print("\n" + "=" * 65)
        print("👉 ব্রাউজারে গিয়ে ম্যানুয়ালি 'Quick Events' সিলেক্ট করুন।")
        print("👉 নীল বক্সের নিচে লাল বক্সগুলো এডজাস্ট করুন।")
        print("👉 সব ঠিক থাকলে 'DONE' বাটনে ক্লিক করুন।")
        print("=" * 65 + "\n")

        while True:
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)

            print("⏳ ব্রাউজারে আপনার টার্গেট সেট করুন (DONE / + / - / RESET)...")
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
            log("⚠️ কোনো টার্গেট ম্যাপিং করা যায়নি — স্ক্রিপ্ট বন্ধ।")
            browser.close()
            return

        # ⭐ Start WS thread (like script.js startLiveListener())
        start_ws_thread()
        log(f"👂 [WS] wait korchi... (match_key={match_filter})\n")

        remember_seen, was_seen, forget_seen = make_seen_lru(300)

        last_ws_over = None
        clicked_over = None

        try:
            while True:
                try:
                    m = event_q.get(timeout=0.5)
                except queue.Empty:
                    continue

                # ⭐ Measure pipeline lag (backend serialize → clicker receive)
                now_ms = time.time() * 1000
                srv_ts = m.get("_server_ts")
                lag_ms = round(now_ms - srv_ts, 1) if srv_ts else None

                t_type = m.get("t")
                mid_msg = m.get("id")

                # script.js: only process `t === "ev"` (ball events)
                if t_type == "ev":
                    ev = str(m.get("ev") or "").strip()
                    if not ev:
                        continue

                    # Ball Chalu → reset per-delivery dedup
                    if is_ball_chalu(ev):
                        clicked_over = None
                        continue

                    # Skip intermediate
                    if is_intermediate(ev):
                        continue

                    # Dedup by over (skip corrections: 4→6 same over)
                    over_str = str(m.get("over") or "")
                    if over_str and over_str == clicked_over:
                        lag_str = f" | lag={lag_ms}ms" if lag_ms is not None else ""
                        log(f"⏭️ [SKIP] correction over={over_str} ev={ev}{lag_str}")
                        continue
                    clicked_over = over_str

                    ws_over, ws_del_legal = get_over_delivery(m)
                    if ws_over is None:
                        continue

                    text, kind, run = ball_info(m)
                    if kind == "intermediate":
                        continue

                    # ⭐ Delivery counter sync
                    if last_ws_over is None:
                        last_ws_over = ws_over
                        if kind in ('wide', 'noball'):
                            market_del_counter += 1
                        elif ws_del_legal > market_del_counter:
                            market_del_counter = ws_del_legal
                    elif ws_over != last_ws_over:
                        last_ws_over = ws_over
                        market_del_counter = 1
                        if ws_del_legal > market_del_counter:
                            market_del_counter = ws_del_legal
                        last_anchor_pos = None
                        CALIBRATED_ANCHOR_POS = None
                    else:
                        market_del_counter += 1
                        if ws_del_legal > market_del_counter:
                            market_del_counter = ws_del_legal

                    bkey = f"{over_str}_{ev}"
                    lag_str = f" | lag={lag_ms}ms" if lag_ms is not None else ""
                    log(f"🏏 [EV] over={ws_over}.{market_del_counter} ev={text} kind={kind} run={run}{lag_str}")

                    process_second_push(page, bkey, run, kind, ws_over, market_del_counter)

                # script.js handles `t === "match"` for score/team updates
                # clicker doesn't need score, but we can log for diagnostics
                elif t_type == "match":
                    # Optional: log only (clicker doesn't use score)
                    # score = m.get("score")
                    # if score:
                    #     log(f"📊 [MATCH] score={score.get('runs')}/{score.get('wk')} ov={score.get('ov')}")
                    pass

        except KeyboardInterrupt:
            print("\nবন্ধ করা হচ্ছে...")
        finally:
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    monitor_and_click()
