"""
clicker.py — v23 (Top/Bottom Layout + Fast Under Check)
=====================================================
লজিক:
  ১. স্ক্রিনে ২টি লম্বা বক্স থাকবে (উপরে Over, নিচে Under)।
  ২. WS থেকে Wicket আসলে স্ক্রিপ্ট নিচের বক্সে ক্লিক করবে।
  ৩. ভুল হওয়ার ভয় থাকবে না, কারণ ক্লিক করার ঠিক আগে স্ক্রিপ্ট দেখে নিবে 
     বক্সের ভেতরে "Under" লেখা আছে কিনা (সুপার ফাস্ট চেক)।
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

# ── Timing ──
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.1
COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
OCR_CFG_HEADER = '--oem 1 --psm 7' # Fast single line OCR

# ── Viewport / anchor ──
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 240
ANCHOR_H = 34

ANCHOR_TEMPLATE = {}
MAPPED_TARGETS = [] 
CALIBRATED_ANCHOR_POS = None
last_anchor_pos = None
_anchor_trust_until = 0.0
_last_wicket_time = 0 

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
    return any(re.search(p, ev.lower().strip()) for p in INTERMEDIATE_PATTERNS)

def is_ball_chalu(ev):
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)', ev.lower().strip()))

# ══════════════════════════════════════════════════════════════
#  JS (Top/Bottom Layout - Vertical Boxes)
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
    
    // ২টি লম্বা বক্স (উপরে Over, নিচে Under)
    let baseY = 400, anchorW = 240, anchorH = 34, leftX = 70, boxW = 420, boxH = 30, rowGap = 4;
    
    const blueBox = document.createElement('div');
    blueBox.style.cssText = `position:absolute;left:${leftX}px;top:${baseY}px;width:${anchorW}px;height:${anchorH}px;border:2px solid #00aaff;background:rgba(0,170,255,0.4);box-sizing:border-box;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:10px;`;
    blueBox.textContent = "Anchor";
    box.appendChild(blueBox);
    
    let y1 = baseY + anchorH; // Over Box
    let y2 = y1 + boxH + rowGap; // Under Box
    
    // Box 0: Over (Top)
    const box1 = document.createElement('div');
    box1.style.cssText = `position:absolute;left:${leftX}px;top:${y1}px;width:${boxW}px;height:${boxH}px;border:2px solid red;background:transparent;box-sizing:border-box;`;
    box1.textContent = "OVER (Top)";
    box1.style.color = "red"; box1.style.fontSize = "12px"; box1.style.padding = "4px";
    box.appendChild(box1);
    
    // Box 1: Under (Bottom)
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
        
    log(f"   🎯 মোট {len(MAPPED_TARGETS)}টি বক্স সেভ হয়েছে (১টি Over, ১টি Under)।")

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

def check_is_under(page, box, ax, ay):
    """দ্রুত চেক করে বক্সের ভেতরে 'Under' লেখা আছে কিনা"""
    tx = ax + (box["x"] - ANCHOR_LEFT)
    ty = ay + (box["y"] - ANCHOR_TOP)
    try:
        raw = page.screenshot(clip={"x": tx, "y": ty, "width": box["w"], "height": box["h"]}, type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(crop, config=OCR_CFG_HEADER).lower()
        return "under" in text
    except:
        return False

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
#  WICKET LOGIC (Smart Check)
# ══════════════════════════════════════════════════════════════
def handle_wicket(page):
    global _last_wicket_time
    t0 = time.perf_counter()
    
    if time.time() - _last_wicket_time < 3.0: return False
    _last_wicket_time = time.time()

    ax, ay = find_anchor_fast(page)
    if ax is None: return False

    # আমাদের কাছে ২টি বক্স আছে: index 0 = Over, index 1 = Under
    under_boxes = [t for t in MAPPED_TARGETS if t["side"] == "under"]
    
    if not under_boxes:
        log(f"❌ [NO-MATCH] Under বক্স পাওয়া যায়নি!")
        return False

    # ডিফল্টভাবে নিচের বক্সটাকে টার্গেট করব
    best_box = under_boxes[0]
    
    # ⭐ সেফটি চেক: বক্সে সত্যিই Under লেখা আছে কিনা দেখা (যদি না থাকে তবে Over হবে)
    is_under = check_is_under(page, best_box, ax, ay)
    
    if is_under:
        cx, cy = do_click(page, best_box, ax, ay)
        ms = round((time.perf_counter() - t0) * 1000)
        log(f"✅ [CLICK-SUCCESS] UNDER clicked @({cx:.0f},{cy:.0f}) [+{ms}ms]")
    else:
        # যদি Under না পায়, তার মানে সাইটের ডিজাইন উল্টে গেছে, তবে Over বক্সে ক্লিক করবে
        over_boxes = [t for t in MAPPED_TARGETS if t["side"] == "over"]
        if over_boxes:
            cx, cy = do_click(page, over_boxes[0], ax, ay)
            log(f"⚠️ [SWAPPED] Under পাওয়া যায়নি! Over-এ ক্লিক করা হলো @({cx:.0f},{cy:.0f})")
            
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
        print("👉 go to browser and select 'Next Wicket Fall' Market")
        print("👉 Adjust Red Boxes: TOP box over Over, BOTTOM box over Under")
        print("👉 if all is well click 'DONE'")
        print("=" * 65 + "\n")

        while True:
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("⏳ set target on your browser (DONE)...")
            while True:
                action = page.evaluate("() => window.__pw_action")
                if action == 'done': break
                page.wait_for_timeout(200)
            
            final_targets = page.evaluate("() => window.__pw_final_targets")
            setup_from_done(page, final_targets)
            break

        start_ws_thread()
        log(f"👂 [WS] waiting... (match_key={match_filter})")
        log(f"   💡 উইকেট পড়লে Under বক্সে ক্লিক করবে (স্মার্ট চেক সহ)।\n")

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

                log(f"🏏 [WICKET] Out hoyeche! Clicking Under...")
                handle_wicket(page)

        except KeyboardInterrupt:
            print("\n closing...")
        finally:
            try: browser.close()
            except: pass

if __name__ == "__main__":
    monitor_and_click()
