"""
clicker.py — v17 (Multi-Row + Highest Run Under)
===============================================
লজিক:
  ১. প্যানেলে ২, ৪, ৬ বা তার বেশি রো (বক্স) থাকতে পারে।
  ২. উইকেট পড়লে WS থেকে আউট হওয়া ব্যাটসম্যানের নাম নিবে।
  ৩. প্যানেলের সবগুলো Under বক্স লাইভ স্ক্যান করবে।
  ৪. যে বক্সগুলোতে ওই ব্যাটসম্যানের নাম আছে, সেগুলোর রান চেক করবে।
  ৫. সবচেয়ে বেশি রানের (Highest Target) Under বক্সটিতে ক্লিক করবে।
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
DEBUG = os.environ.get("CLICKER_DEBUG") == "1"

TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

CACHE_FRESH_SEC = 0.2
IDLE_REFRESH_SEC = 0.15
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.1

COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
HEADER_CROP_W = 240
OCR_RESIZE = 3
OCR_CFG_HEADER = '--oem 1 --psm 6'
OCR_CFG_HEADER_FALLBACK = '--oem 1 --psm 7'

# ── Viewport / anchor ──
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 120
ANCHOR_H = 34
TARGET_ROWS = 3 # Default row (JS theke dynamic ashbe)

NAME_STOPWORDS = {
    'over', 'under', 'ovr', 'undr', 'delivery', 'deliveries', 'del',
    'runs', 'run', 'wicket', 'out', 'live', 'match', 'next', 'ball',
    'the', 'and', 'vs', 'next'
}

ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = [] 
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0

panel_cache = {"over": None, "del": None, "ts": 0.0, "ax": None, "ay": None}
_last_clicked_batsman = None

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

def dbg(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] [DBG] {msg}", flush=True)

def is_intermediate(ev):
    return any(re.search(p, ev.lower().strip()) for p in INTERMEDIATE_PATTERNS)

def is_ball_chalu(ev):
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)', ev.lower().strip()))

def extract_name_from_text(text):
    if not text: return None
    t = text
    for kw in ('over', 'under', 'ovr', 'undr', 'deliveries', 'delivery', 'del'):
        t = re.sub(rf'\b{kw}\b', ' ', t, flags=re.I)
    t = re.sub(r'\b(runs?|wicket|out|live|match|next|ball)\b', ' ', t, flags=re.I)
    t = re.sub(r'\d+(\.\d+)?', ' ', t) 
    t = re.sub(r'[^A-Za-z\s]', ' ', t) 
    words = [w for w in t.split() if len(w) >= 3 and w.lower() not in NAME_STOPWORDS]
    return ' '.join(words[:4]).strip() if words else None

def extract_run_from_text(text):
    """লেখা থেকে রানের মানটা বের করে (যেমন: 35.5)"""
    if not text: return 0.0
    m = re.search(r'\(?(\d+\.\d+)\)?', text)
    if m:
        try: return float(m.group(1))
        except: pass
    return 0.0

def fuzzy_match(a, b):
    if not a or not b: return False
    a_low, b_low = a.lower().strip(), b.lower().strip()
    if a_low == b_low: return True
    if len(a_low) >= 4 and a_low in b_low: return True
    if len(b_low) >= 4 and b_low in a_low: return True
    a_parts = [p for p in a_low.split() if len(p) >= 3]
    b_parts = [p for p in b_low.split() if len(p) >= 3]
    for pa in a_parts:
        if pa in b_parts: return True
    return False

# ══════════════════════════════════════════════════════════════
#  JS (Multi-Row Support with + / - buttons)
# ══════════════════════════════════════════════════════════════
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
    let rowCount = window.__row_count || 3; // Default 3 rows
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

# ══════════════════════════════════════════════════════════════
#  CALIBRATION
# ══════════════════════════════════════════════════════════════
def setup_from_done(page, final_targets):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    log("⚙️  Anchor + Targets (Over/Under) পজিশন সেভ করছি...")

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
    log(f"   ✅ Anchor সেভ (conf={max_val:.2f})")

    MAPPED_TARGETS.clear()
    
    for i, t in enumerate(final_targets):
        side = t.get('side')
        x, y, w, h = int(t['x']), int(t['y']), int(t['w']), int(t['h'])
        
        MAPPED_TARGETS.append({
            "side": side, "x": x, "y": y, "w": w, "h": h
        })
        
    log(f"   🎯 মোট {len(MAPPED_TARGETS)}টি বক্স সেভ হয়েছে। (ডাইনামিক নাম এবং হাইয়েস্ট রান রেডি)")

# ══════════════════════════════════════════════════════════════
#  ANCHOR & OCR HELPERS
# ══════════════════════════════════════════════════════════════
def find_anchor_fast(page):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    now = time.time()
    cached = CALIBRATED_ANCHOR_POS or last_anchor_pos
    if cached and now < _anchor_trust_until: return cached
    if cached:
        ax, ay = cached
        try:
            raw = page.screenshot(clip={"x": ax, "y": ay, "width": ANCHOR_W, "height": ANCHOR_H}, type="png", scale="css")
            crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            if crop is not None and crop.shape == ANCHOR_TEMPLATE["img"].shape:
                res = cv2.matchTemplate(crop, ANCHOR_TEMPLATE["img"], cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= CACHE_VERIFY_THRESHOLD:
                    _anchor_trust_until = now + ANCHOR_TRUST_SEC
                    return ax, ay
        except: pass
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
    except: pass
    return None, None

def get_live_ocr_data(page, box, ax, ay):
    """নির্দিষ্ট বক্সের ছবি তুলে OCR টেক্সট রিটার্ন করে"""
    tx = ax + (box["x"] - ANCHOR_LEFT)
    ty = ay + (box["y"] - ANCHOR_TOP)
    try:
        raw = page.screenshot(clip={"x": tx, "y": ty, "width": box["w"], "height": box["h"]}, type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return pytesseract.image_to_string(crop, config=OCR_CFG_HEADER).lower()
    except:
        return ""

def do_click(page, target_box, ax, ay):
    x = ax + (target_box["x"] - ANCHOR_LEFT)
    y = ay + (target_box["y"] - ANCHOR_TOP)
    cx = x + target_box["w"] / 2
    cy = y + target_box["h"] / 2
    page.mouse.move(cx, cy, steps=1)
    page.mouse.down()
    page.wait_for_timeout(random.randint(2, 5))
    page.mouse.up()
    return cx, cy

# ══════════════════════════════════════════════════════════════
#  PARSE & WICKET LOGIC
# ══════════════════════════════════════════════════════════════
def parse_over_ball(d):
    ov = d.get("over")
    if ov is None: return None, None, None, None
    ov_str = str(ov).strip()
    try:
        if "." in ov_str:
            parts = ov_str.split(".")
            over_num, over_dec = int(parts[0]), int(parts[1]) if parts[1] else 1
            ball = over_dec
        else: over_num, over_dec, ball = int(ov_str), 0, 1
    except: return None, None, None, None
    b, xball = d.get("ball"), d.get("xball")
    if b is not None:
        try: ball = int(b)
        except: pass
    if xball is not None:
        try: xball = int(xball)
        except: xball = None
    return over_num, over_dec, ball, xball

def get_striker(m):
    for key in ("batsman", "striker", "batsman_name", "striker_name",
                "player", "player_name", "batsmanName", "strikerName",
                "batsman_out", "out_batsman", "dismissed_batsman"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip()
    return None

def handle_wicket(page, striker, over_num, over_dec, ball, xball):
    global _last_clicked_batsman

    t0 = time.perf_counter()
    if not striker or _last_clicked_batsman == striker.lower(): return False

    ax, ay = find_anchor_fast(page)
    if ax is None: return False

    # প্যানেলের সবগুলো Under বক্স নিব
    under_boxes = [t for t in MAPPED_TARGETS if t["side"] == "under"]
    
    matched_options = []
    for box in under_boxes:
        live_text = get_live_ocr_data(page, box, ax, ay)
        if not live_text: continue

        live_name = extract_name_from_text(live_text)
        live_run = extract_run_from_text(live_text)
        
        if live_name and fuzzy_match(striker, live_name):
            matched_options.append((live_run, box, live_name, live_text))
            log(f"   🔍 [FOUND] '{striker}' er jonno option: {live_run} (Text: '{live_text.strip()}')")

    if not matched_options:
        log(f"❌ [NO-MATCH] '{striker}' প্যানেলে নেই!")
        return False

    # সবচেয়ে বেশি রান অনুযায়ী সর্ট করব (Descending order)
    matched_options.sort(key=lambda x: x[0], reverse=True)
    
    best_run, best_box, best_name, best_text = matched_options[0]
    log(f"🏆 [HIGHEST RUN SELECTED] '{striker}' এর জন্য সবচেয়ে বড় রান: {best_run}")

    cx, cy = do_click(page, best_box, ax, ay)
    ms = round((time.perf_counter() - t0) * 1000)
    log(f"✅ [CLICK-SUCCESS] UNDER ({best_run}) for '{striker}' @({cx:.0f},{cy:.0f}) [+{ms}ms]")
    
    _last_clicked_batsman = striker.lower()
    return True

def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev or is_intermediate(ev): return (ev, "intermediate", 0)
    if re.match(r'^(wicket|out|w)[.!]*$', ev_low): return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|retired|handled|obstruct)', ev_low): return (ev, "wicket", 0)
    return (ev, "intermediate", 0)

# ══════════════════════════════════════════════════════════════
#  WS
# ══════════════════════════════════════════════════════════════
async def ws_loop():
    while True:
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(WS_URL, ping_interval=None, compression=None, max_size=None) as ws:
                WS_STATE["up"] = True
                log(f"🔌 [WS] connected")
                if mf: await ws.send(f"watch:{mf}")
                async def pinger():
                    while True:
                        await asyncio.sleep(PING_SEC)
                        try: await ws.send("ping")
                        except: break
                ptask = asyncio.create_task(pinger())
                try:
                    async for raw in ws:
                        if raw == "pong": continue
                        try: m = json.loads(raw)
                        except: continue
                        mid_msg = m.get("id")
                        if mf and mid_msg and str(mid_msg) != str(mf): continue
                        WS_STATE["msg_count"] += 1
                        event_q.put(m)
                finally:
                    ptask.cancel()
        except asyncio.CancelledError: return
        except: pass
        finally: WS_STATE["up"] = False
        await asyncio.sleep(RECONNECT_SEC)

def start_ws_thread():
    threading.Thread(target=lambda: asyncio.run(ws_loop()), daemon=True).start()

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def monitor_and_click():
    if len(sys.argv) < 2:
        print("Usage: python clicker.py <match_key>")
        return
    match_filter = sys.argv[1].strip()
    WS_STATE["match_filter"] = match_filter

    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

    site_url = input("\nSite link din: ").strip() or "https://example.com"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=[f"--window-size={WIN_W},{WIN_H}", "--window-position=0,0"])
        context = browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H})
        page = context.new_page()
        page.goto(site_url)

        print("\n=== log in manually ===")
        match_url = input("Match URL: ").strip()
        if not match_url: browser.close(); return
        page.goto(match_url)
        log(f"✅ Match page ready")

        print("\n" + "=" * 65)
        print("👉 go to browser and select 'Players, Total Runs' Market")
        print("👉 Adjust Red Boxes. (Use + 2 Rows button if more players)")
        print("👉 if all is well click 'DONE'")
        print("=" * 65 + "\n")

        while True:
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("⏳ set target on your browser (DONE / + / -)...")
            while True:
                action = page.evaluate("() => window.__pw_action")
                if action in ('done', 'update'): break
                page.wait_for_timeout(200)
            
            if action == 'update':
                continue
                
            final_targets = page.evaluate("() => window.__pw_final_targets")
            setup_from_done(page, final_targets)
            break

        start_ws_thread()
        log(f"👂 [WS] waiting... (match_key={match_filter})")
        log(f"   💡 উইকেট পড়লে লাইভ স্ক্যান হবে। সবচেয়ে বেশি রানের Under-এ ক্লিক হবে।\n")

        last_dedup_key = None

        try:
            while True:
                try:
                    m = event_q.get(timeout=EVENT_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if m.get("t") != "ev": continue

                ev = str(m.get("ev") or "").strip()
                if not ev or is_ball_chalu(ev) or is_intermediate(ev): continue

                over_str = str(m.get("over") or "")
                dedup_key = f"{over_str}|{ev}"
                if dedup_key == last_dedup_key: continue
                last_dedup_key = dedup_key

                text, kind, run = ball_info(m)
                if kind != "wicket": continue

                over_num, over_dec, main_ball, xball = parse_over_ball(m)
                if over_num is None: continue

                striker = get_striker(m)
                if striker:
                    log(f"🏏 [WICKET] {striker} out hoyeche!")
                    handle_wicket(page, striker, over_num, over_dec, main_ball, xball)

        except KeyboardInterrupt:
            print("\n closing...")
        finally:
            try: browser.close()
            except: pass

if __name__ == "__main__":
    monitor_and_click()
