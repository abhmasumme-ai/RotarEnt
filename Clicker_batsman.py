"""
clicker.py — v15 (Batsman Per-Market Wicket Mode)
=================================================
লজিক: 
  ১. Wicket হলে WS থেকে Striker-এর নাম নিবে।
  ২. প্যানেলের Under বক্সগুলো থেকে ব্যাটসম্যানের নাম OCR করবে।
  ৩. নাম মিললে সেই ব্যাটসম্যানের Under-এ ক্লিক করবে।

Env Flags:
  CLICKER_WICKET_ONLY    1 = শুধু wicket-এ কাজ করবে (default 1)
  CLICKER_DEBUG          1 = ডিটেইলস লগ দেখাবে (default 0)
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
NO_IDLE = os.environ.get("CLICKER_NO_IDLE") == "1"
WICKET_ONLY = os.environ.get("CLICKER_WICKET_ONLY", "1") == "1"

TESSERACT_CMD = os.environ.get("TESSERACT_CMD") or (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe") else None
)

# ── Timing ──
CACHE_FRESH_SEC = 0.2
IDLE_REFRESH_SEC = 0.15
ANCHOR_TRUST_SEC = 300
EVENT_TIMEOUT_SEC = 0.1

# ── OCR & Crop Settings ──
COMPARE_THRESHOLD = 0.70
CACHE_VERIFY_THRESHOLD = 0.82
HEADER_CROP_W = 240
OCR_RESIZE = 3
OCR_CFG_HEADER = '--oem 1 --psm 6'
OCR_CFG_HEADER_FALLBACK = '--oem 1 --psm 7'
OCR_CFG_SETUP = '--oem 1 --psm 6'

# ── Viewport / anchor ──
VIEW_W = 1080
VIEW_H = 797
WIN_W = VIEW_W + 100
WIN_H = VIEW_H + 120

ANCHOR_LEFT = 70
ANCHOR_TOP = 400
ANCHOR_W = 120   # নতুন অনুযায়ী
ANCHOR_H = 34
TARGET_ROWS = 2  # নতুন অনুযায়ী

# স্টপওয়ার্ড যা নামের সাথে থাকবে না
NAME_STOPWORDS = {
    'over', 'under', 'ovr', 'undr', 'delivery', 'deliveries', 'del',
    'runs', 'run', 'wicket', 'out', 'live', 'match', 'next', 'ball',
    'the', 'and', 'vs', 'next'
}

# ── State ──
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

# ══════════════════════════════════════════════════════════════
#  UTIL
# ══════════════════════════════════════════════════════════════
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def dbg(msg):
    if DEBUG:
        print(f"[{time.strftime('%H:%M:%S')}] [DBG] {msg}", flush=True)

def is_intermediate(ev):
    ev_low = ev.lower().strip()
    return any(re.search(p, ev_low) for p in INTERMEDIATE_PATTERNS)

def is_ball_chalu(ev):
    return bool(re.search(r'^ball\b.*\b(chalu|start|chalo|shuru|shuro)', ev.lower().strip()))

def extract_name_from_text(text):
    """'kamindu mendis under (33.5)' -> 'kamindu mendis'"""
    if not text:
        return None
    t = text
    for kw in ('over', 'under', 'ovr', 'undr', 'deliveries', 'delivery', 'del'):
        t = re.sub(rf'\b{kw}\b', ' ', t, flags=re.I)
    t = re.sub(r'\b(runs?|wicket|out|live|match|next|ball)\b', ' ', t, flags=re.I)
    t = re.sub(r'\d+(\.\d+)?', ' ', t)  # নাম্বার বাদ
    t = re.sub(r'[^A-Za-z\s]', ' ', t)  # স্পেশাল ক্যারেক্টার বাদ
    words = [w for w in t.split() if len(w) >= 3 and w.lower() not in NAME_STOPWORDS]
    return ' '.join(words[:4]).strip() if words else None

def fuzzy_match(a, b):
    """OCR এর ছোটখাটো ভুল বা শর্ট নেম (K Mendis vs Kamindu Mendis) হ্যান্ডেল করে"""
    if not a or not b:
        return False
    a_low = a.lower().strip()
    b_low = b.lower().strip()
    if a_low == b_low:
        return True
    if len(a_low) >= 4 and a_low in b_low:
        return True
    if len(b_low) >= 4 and b_low in a_low:
        return True
    a_parts = [p for p in a_low.split() if len(p) >= 3]
    b_parts = [p for p in b_low.split() if len(p) >= 3]
    for pa in a_parts:
        if pa in b_parts:
            return True
    return False

# ══════════════════════════════════════════════════════════════
#  JS (আপনার দেওয়া ডাইমেনশন অনুযায়ী)
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
    let rowCount = window.__row_count || 2;
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
    
    const resetBtn = document.createElement('button');
    resetBtn.textContent = 'RESET';
    resetBtn.style.cssText = 'padding:10px 20px;font-size:16px;font-weight:bold;background:#ff3d00;color:white;border:none;border-radius:6px;cursor:pointer;';
    resetBtn.onclick = () => { window.__pw_action = 'reset'; };
    
    btnContainer.appendChild(doneBtn);
    btnContainer.appendChild(resetBtn);
    box.appendChild(btnContainer);
}
"""

# ══════════════════════════════════════════════════════════════
#  CALIBRATION (Under Box theke direct Name pora)
# ══════════════════════════════════════════════════════════════
def setup_from_done(page, final_targets):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    log("⚙️  Anchor + Targets সেভ করছি...")

    page.evaluate("() => { let b = document.getElementById('__pw_target_guide'); if(b) b.remove(); }")
    page.wait_for_timeout(1000)

    raw = page.screenshot(type="png", scale="css")
    src_img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)

    # Save Anchor
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
    max_boxes = TARGET_ROWS * 2

    for row_idx in range(TARGET_ROWS):
        under_idx = row_idx * 2 + 1
        if under_idx >= len(final_targets):
            break
        u_t = final_targets[under_idx]

        ux, uy = int(u_t['x']), int(u_t['y'])
        uw, uh = int(u_t['w']), int(u_t['h'])

        # Under box theke direct text pora
        crop = src_img[uy:uy+uh, ux:ux+uw]
        if crop.size == 0:
            continue
            
        crop_r = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(crop_r, config=OCR_CFG_SETUP).lower()
        log(f"      🔍 [ROW {row_idx} UNDER] OCR={text!r}")

        # নাম বের করা
        batsman = extract_name_from_text(text)
        
        # লাইন বের করা (যেমন 33.5)
        m_line = re.search(r'\(?(\d+\.\d+)\)?', text)
        line = float(m_line.group(1)) if m_line else 0.0

        # শুধু UNDER টার্গেট সেভ করছি (কারন Wicket হলে Under-এ ক্লিক করতে হবে)
        MAPPED_TARGETS.append({
            "side": "under", "batsman": batsman, "line": line,
            "x": ux, "y": uy, "w": uw, "h": uh
        })
        log(f"      ✅ ম্যাপিং: UNDER | Batsman='{batsman}' | Line={line}")

    log(f"   🎯 ম্যাপ হয়েছে {len(MAPPED_TARGETS)}টি Under টার্গেট")

# ══════════════════════════════════════════════════════════════
#  ANCHOR & OCR
# ══════════════════════════════════════════════════════════════
def find_anchor_fast(page):
    global CALIBRATED_ANCHOR_POS, last_anchor_pos, _anchor_trust_until
    now = time.time()
    cached = CALIBRATED_ANCHOR_POS or last_anchor_pos
    if cached and now < _anchor_trust_until:
        return cached
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

def refresh_panel_header(page, ax, ay, probe_x, probe_y, probe_h):
    if ax is None:
        return False
    try:
        raw = page.screenshot(clip={"x": probe_x, "y": probe_y, "width": HEADER_CROP_W, "height": probe_h}, type="png", scale="css")
        crop = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        if crop is None or crop.size == 0: return False
        crop_big = cv2.resize(crop, None, fx=OCR_RESIZE, fy=OCR_RESIZE, interpolation=cv2.INTER_CUBIC)

        text = ""
        for cfg in (OCR_CFG_HEADER, OCR_CFG_HEADER_FALLBACK):
            try: text = pytesseract.image_to_string(crop_big, config=cfg).lower().strip()
            except Exception: text = ""
            if text: break

        if not text: return False

        # Over/Delivery ম্যাচ
        m = (re.search(r'(\d+)\D{0,4}over\D{0,8}(\d+)\D{0,4}del', text) or
             re.search(r'(\d+)\D{0,4}over\D{0,8}(\d+)\D{0,4}de[l1]', text))
        if not m: return False

        po, pd = int(m.group(1)), int(m.group(2))
        if not (0 <= po <= 50 and 1 <= pd <= 50): return False

        panel_cache["over"] = po
        panel_cache["del"] = pd
        panel_cache["ts"] = time.time()
        panel_cache["ax"] = ax
        panel_cache["ay"] = ay
        return True
    except Exception as e:
        dbg(f"OCR err: {e}")
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
#  PARSE / WANT CANDIDATES
# ══════════════════════════════════════════════════════════════
def parse_over_ball(d):
    ov = d.get("over")
    if ov is None: return None, None, None, None
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
        try: ball = int(b)
        except: pass

    xball = d.get("xball")
    if xball is not None:
        try: xball = int(xball)
        except: xball = None

    return over_num, over_dec, ball, xball

def compute_want_candidates(over_num, over_dec, ball, xball):
    base_over = over_num + 1
    delivery = xball if xball is not None else ball
    if over_dec >= 6:
        cands = [(base_over, delivery), (base_over + 1, 1)]
    else:
        cands = [(base_over, delivery), (base_over, delivery + 1)]
    out, seen = [], set()
    for c in cands:
        if c not in seen and (0 <= c[0] <= 50 and 1 <= c[1] <= 50):
            seen.add(c)
            out.append(c)
    return out

def get_striker(m):
    # Entity Sports data theke batsman ber korar chesta
    for key in ("batsman", "striker", "batsman_name", "striker_name",
                "player", "player_name", "batsmanName", "strikerName",
                "batsman_out", "out_batsman", "dismissed_batsman"):
        v = m.get(key)
        if v and isinstance(v, str) and len(v.strip()) >= 2:
            return v.strip()
    return None

# ══════════════════════════════════════════════════════════════
#  WICKET HANDLER
# ══════════════════════════════════════════════════════════════
def handle_wicket(page, ball_key, striker, over_num, over_dec, ball, xball):
    global _last_clicked_batsman

    t0 = time.perf_counter()
    cands = compute_want_candidates(over_num, over_dec, ball, xball)
    cand_str = " | ".join(f"{o}-{d}" for o, d in cands)

    log(f"🎯 [WICKET] {ball_key} striker='{striker}' | want panel: {cand_str}")

    if not striker:
        log(f"❌ [NO-STRIKER] Raw data te batsman field paen ni! RAW MSG check korun.")
        return False

    # MAPPED_TARGETS ekhon shudhu UNDER box rakha ache
    under_boxes = [t for t in MAPPED_TARGETS if t["side"] == "under"]
    
    matched_box = None
    for box in under_boxes:
        stored = box.get("batsman")
        if stored and fuzzy_match(striker, stored):
            log(f"🎯 [NAME MATCH] '{striker}' == '{stored}' (Line: {box['line']})")
            matched_box = box
            break

    if not matched_box:
        log(f"❌ [NO-BATSMAN-MATCH] '{striker}' panel e nei!")
        return False

    # Dedup - ekoi batsman er jonno ekoi wicket e 2 bar click na korte
    if _last_clicked_batsman == striker.lower():
        log(f"⏭️ [ALREADY-CLICKED] '{striker}' er jonno click kora hoise!")
        return False

    # Panel verify
    ax, ay = find_anchor_fast(page)
    if ax is None:
        log(f"❌ [NO-ANCHOR]")
        return False

    tx = ax + (matched_box["x"] - ANCHOR_LEFT)
    ty = ay + (matched_box["y"] - ANCHOR_TOP)
    refresh_panel_header(page, ax, ay, tx, ty, matched_box["h"])

    po, pd = panel_cache["over"], panel_cache["del"]
    if (po, pd) in cands:
        cx, cy = do_click(page, matched_box, ax, ay)
        ms = round((time.perf_counter() - t0) * 1000)
        log(f"✅ [CLICK-SUCCESS] UNDER ({matched_box['line']}) for '{striker}' "
            f"panel={po}-{pd} @({cx:.0f},{cy:.0f}) [+{ms}ms]")
        _last_clicked_batsman = striker.lower()
        panel_cache["ts"] = 0
        return True
    else:
        log(f"⏭️ [MISS] panel={po}-{pd} ∉ {cand_str} (Panel ekhono update hoy ni)")
        return False

# ══════════════════════════════════════════════════════════════
#  BALL INFO & WS
# ══════════════════════════════════════════════════════════════
def ball_info(d):
    ev = str(d.get("ev") or "").strip()
    ev_low = ev.lower()
    if not ev or is_intermediate(ev): return (ev, "intermediate", 0)
    if re.match(r'^(wicket|out|w)[.!]*$', ev_low): return (ev, "wicket", 0)
    if re.match(r'^(bowled|lbw|caught|c ?& ?b|run ?out|stumped?|hit ?wicket|retired|handled|obstruct)', ev_low): return (ev, "wicket", 0)
    if re.match(r'^(no[\s_-]?ball|nb)\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        return (ev, "noball", int(m.group(1)) + 1 if m else 1)
    if re.match(r'^wide\b', ev_low):
        m = re.search(r'(\d+)', ev_low)
        return (ev, "wide", int(m.group(1)) + 1 if m else 1)
    if re.match(r'^(six|6)\b', ev_low): return ("6", "six", 6)
    if re.match(r'^(four|4)\b', ev_low): return ("4", "four", 4)
    if re.match(r'^(dot|0)\b', ev_low): return ("0", "dot", 0)
    m = re.match(r'^(\d+)', ev_low)
    if m:
        n = int(m.group(1))
        return (str(n), "run", n)
    return (ev, "intermediate", 0)

async def ws_loop():
    while True:
        mf = WS_STATE.get("match_filter") or ""
        try:
            async with websockets.connect(WS_URL, ping_interval=None, compression=None, max_size=None) as ws:
                WS_STATE["up"] = True
                log(f"🔌 [WS] connected: {WS_URL}")
                if mf:
                    try: await ws.send(f"watch:{mf}")
                    except: pass
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
    
    # Proxy
    proxy_config = None
    proxy_url = input("Proxy URL [Enter to skip]: ").strip()
    if proxy_url:
        try:
            if "@" in proxy_url:
                proto, rest = proxy_url.split("://") if "://" in proxy_url else ("http://", proxy_url)
                auth, host_port = rest.split("@")
                user, pwd = auth.split(":")
                proxy_config = {"server": f"{proto}://{host_port}", "username": user, "password": pwd}
            else: proxy_config = {"server": proxy_url}
        except: pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=[f"--window-size={WIN_W},{WIN_H}", "--window-position=0,0"])
        context = browser.new_context(viewport={"width": VIEW_W, "height": VIEW_H}, proxy=proxy_config)
        page = context.new_page()
        page.goto(site_url)

        print("\n=== log in manually ===")
        match_url = input("Match URL: ").strip()
        if not match_url: browser.close(); return
        page.goto(match_url)
        log(f"✅ Match page ready")

        print("\n" + "=" * 65)
        print("👉 go to browser and select 'Players, Total Runs' Market")
        print("👉 adjust Red Box over the batsman name box")
        print("👉 if all is well click 'DONE'")
        print("=" * 65 + "\n")

        while True:
            page.evaluate("() => { window.__pw_action = null; }")
            page.evaluate(SHOW_TARGETS_JS)
            print("⏳ set target on your browser (DONE / RESET)...")
            while True:
                action = page.evaluate("() => window.__pw_action")
                if action in ('done', 'reset'): break
                page.wait_for_timeout(200)
            if action == 'done':
                final_targets = page.evaluate("() => window.__pw_final_targets")
                setup_from_done(page, final_targets)
                break
            elif action == 'reset':
                page.evaluate("() => { window.__row_count = 2; }")
                continue

        if not MAPPED_TARGETS:
            log("⚠️ কোনো টার্গেট ম্যাপ হয়নি — বন্ধ।")
            browser.close(); return

        start_ws_thread()
        log(f"👂 [WS] waiting... (match_key={match_filter})")
        log(f"   💡 Wicket হলে, striker-এর নাম মিলিয়ে Under-এ ক্লিক করবে\n")

        last_dedup_key = None

        try:
            while True:
                try:
                    m = event_q.get(timeout=EVENT_TIMEOUT_SEC)
                except queue.Empty:
                    continue

                if m.get("t") != "ev": continue

                # RAW DEBUG
                if DEBUG and WS_STATE["msg_count"] <= 10:
                    print(f"[DBG RAW] {json.dumps(m, default=str)[:400]}")

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
                handle_wicket(page, f"{over_str}_{ev}", striker, over_num, over_dec, main_ball, xball)

        except KeyboardInterrupt:
            print("\n closing...")
        finally:
            try: browser.close()
            except: pass

if __name__ == "__main__":
    monitor_and_click()
