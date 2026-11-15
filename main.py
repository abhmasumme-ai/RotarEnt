"""
Cricket control panel backend  (Entity Sports edition)

Architecture:
    Entity push → main loop → per-match worker → per-conn writer → client

Fixture sources:
    1. REST poll every 25s → live + upcoming + finished
    2. Entity push (live only) → overlays FIX for real-time status

Baseline (single match):
    Cache → REST (with detailed logging) → FIX fallback → 404
"""
import asyncio
import gzip
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import uvloop
    uvloop.install()
except ImportError:
    pass

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def _dumps(obj) -> str:
    if _orjson is not None:
        try:
            return _orjson.dumps(obj).decode("utf-8")
        except Exception:
            pass
    return json.dumps(obj, separators=(",", ":"), default=str)


log = logging.getLogger("uvicorn.error")
_tasks = set()

PUSH_TOKEN = os.environ.get("ES_PUSH_TOKEN")
REST_TOKEN = os.environ.get("ES_REST_TOKEN")
PUSH_URL = os.environ.get("ES_PUSH_URL", "ws://webhook.entitysport.com:8087/connect")

_REST_RAW = os.environ.get("ES_REST_BASE", "https://restapi.entitysport.com/v2").strip()
REST_BASE = _REST_RAW.split("?")[0].rstrip("/")
REST_BASE = re.sub(r"/matches/?$", "", REST_BASE).rstrip("/")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

client = httpx.AsyncClient(timeout=15)

UP = {"connected": False, "msgs": 0, "last": 0.0}
LATEST_OBJ = {}
FIX = {}
_fix_pending: set = set()

# REST state
_rest = {
    "ts": 0.0,
    "rows": [],
    "last_error": "",
    "last_ok_ts": 0.0,
    "refreshing": False,
}
_rest_lock = asyncio.Lock()
REST_REFRESH_S = 25

MAX_CACHE = 300
CACHE_FRESH_MS = 60_000
MATCH_QUEUE_SIZE = 300
CONN_QUEUE_SIZE = 500
WORKER_IDLE_TIMEOUT_S = 60

STATUS_MAP = {1: "not_started", 2: "completed", 3: "started", 4: "cancelled"}


def _spawn(coro):
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


def _i(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return d


def _decode(raw):
    if isinstance(raw, bytes):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        raw = raw.decode("utf-8", errors="replace")
    return json.loads(raw)


_W_RE = re.compile(r"^\d*w$", re.I)


def fixture_row(mi):
    ta, tb = mi.get("teama") or {}, mi.get("teamb") or {}
    return {
        "key": str(mi.get("match_id")),
        "title": mi.get("title") or mi.get("short_title") or "",
        "format": mi.get("format_str") or "",
        "status": STATUS_MAP.get(_i(mi.get("status")), "not_started"),
        "start_at": mi.get("timestamp_start"),
        "teams": {"a": {"name": ta.get("name")}, "b": {"name": tb.get("name")}},
    }


def slim_comm(live):
    out = []
    for e in (live.get("commentaries") or []):
        if not isinstance(e, dict):
            continue
        if e.get("event") not in ("ball", "wicket"):
            continue
        eid = str(e.get("event_id") or "")
        if not eid:
            continue
        out.append({
            "id": eid,
            "ov": _i(e.get("over")),
            "b": _i(e.get("ball")),
            "x": _i(e.get("xball")),
            "run": _i(e.get("run")),
            "bat": _i(e.get("bat_run")),
            "wd": _i(e.get("wide_run")),
            "nb": _i(e.get("noball_run")),
            "by": _i(e.get("bye_run")),
            "lb": _i(e.get("legbye_run")),
            "four": bool(e.get("four")),
            "six": bool(e.get("six")),
            "w": e.get("event") == "wicket" or bool(_W_RE.match(str(e.get("score", "")).strip())),
            "dis": str(e.get("dismissal") or ""),
            "fh": bool(e.get("freehit")),
        })
    return out[-30:]


def slim_match(msg, now):
    r = msg.get("response") or {}
    mi = r.get("match_info") or {}
    live = r.get("live") if isinstance(r.get("live"), dict) else {}
    ta, tb = mi.get("teama") or {}, mi.get("teamb") or {}
    li = live.get("live_inning") if isinstance(live.get("live_inning"), dict) else {}
    ls = live.get("live_score") if isinstance(live.get("live_score"), dict) else {}
    inn_no = _i(live.get("live_inning_number"))

    bat_id = str(li.get("batting_team_id", ""))
    bat = "a" if bat_id and bat_id == str(ta.get("team_id")) else (
        "b" if bat_id and bat_id == str(tb.get("team_id")) else None)

    mx = _i(li.get("max_over"))
    if not mx:
        for inn in ((r.get("scorecard") or {}).get("innings") or []):
            if _i(inn.get("number")) == inn_no or not mx:
                mx = _i(inn.get("max_over")) or mx

    score = None
    if ls and ls.get("runs") is not None:
        score = {
            "runs": _i(ls.get("runs")),
            "wk": _i(ls.get("wickets")),
            "ov": str(ls.get("overs", 0)),
            "target": _i(ls.get("target")),
            "rrr": ls.get("required_runrate"),
        }

    lw = li.get("last_wicket") if isinstance(li.get("last_wicket"), dict) else None
    status = STATUS_MAP.get(_i(mi.get("status")), "started")
    result = mi.get("result") or (mi.get("status_note") if status == "completed" else "") or ""

    def team(t):
        return {"id": t.get("team_id"), "name": t.get("name"),
                "short": t.get("short_name"), "sc": t.get("scores_full") or ""}

    ts = msg.get("timestamp")
    out = {
        "t": "match",
        "id": str(r.get("match_id") or mi.get("match_id") or live.get("mid") or ""),
        "ts": ts if isinstance(ts, (int, float)) else None,
        "_server_ts": now * 1000,
        "status": status,
        "game": str(mi.get("game_state_str") or live.get("game_state_str") or ""),
        "fmt": mi.get("format_str") or "",
        "title": mi.get("title") or "",
        "result": result,
        "toss": (mi.get("toss") or {}).get("text") or "",
        "day": mi.get("day"),
        "session": mi.get("session"),
        "start": mi.get("timestamp_start"),
        "a": team(ta),
        "b": team(tb),
        "inn": inn_no,
        "bat": bat,
        "max": mx,
        "score": score,
        "lw": ({"ov": str(lw.get("overs_at_dismissal", "")),
                "dis": str(lw.get("dismissal") or ""),
                "sc": _i(lw.get("score_at_dismissal"))} if lw else None),
        "comm": slim_comm(live),
    }
    if isinstance(ts, (int, float)):
        lag = round(now * 1000 - ts)
        if 0 <= lag < 60000:
            out["_lag_ms"] = lag
    return out


def _fresh(obj):
    d = dict(obj)
    d["_server_ts"] = time.time() * 1000
    d.pop("_lag_ms", None)
    return d


class ConnectionManager:
    def __init__(self):
        self.conns = {}
        self.queues = {}
        self.writers = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.conns[ws] = None
        self.queues[ws] = asyncio.Queue(maxsize=CONN_QUEUE_SIZE)
        self.writers[ws] = asyncio.create_task(self._writer(ws))

    async def _writer(self, ws: WebSocket):
        q = self.queues.get(ws)
        if q is None:
            return
        try:
            while True:
                msg = await q.get()
                try:
                    await asyncio.wait_for(ws.send_text(msg), timeout=2)
                except Exception:
                    self.disconnect(ws)
                    return
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            return

    def disconnect(self, ws: WebSocket):
        self.conns.pop(ws, None)
        self.queues.pop(ws, None)
        w = self.writers.pop(ws, None)
        if w and not w.done():
            w.cancel()

    def watch(self, ws: WebSocket, mid):
        if ws in self.conns:
            self.conns[ws] = mid or None

    async def send(self, ws: WebSocket, msg: str):
        q = self.queues.get(ws)
        if q is None:
            return
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    def broadcast_sync(self, message: str, match_id: str):
        for c, w in list(self.conns.items()):
            if w == match_id:
                q = self.queues.get(c)
                if q is not None:
                    try:
                        q.put_nowait(message)
                    except asyncio.QueueFull:
                        pass

    async def broadcast(self, message: str, match_id=None):
        if match_id is None:
            targets = list(self.conns.keys())
        else:
            targets = [c for c, w in self.conns.items() if w == match_id]
        for c in targets:
            await self.send(c, message)


manager = ConnectionManager()


def _watched_ids() -> set:
    return {w for w in manager.conns.values() if w}


# =====================================================================
# REST — fixture for live + upcoming + finished
# =====================================================================
async def _fetch_status(status: int, per_page: int):
    try:
        r = await client.get(
            f"{REST_BASE}/matches/",
            params={"status": status, "per_page": per_page, "paged": 1, "token": REST_TOKEN},
        )
        if not r.is_success:
            return {"status": status, "error": f"HTTP {r.status_code}: {r.text[:160]}", "rows": []}
        resp = r.json().get("response")
        if not isinstance(resp, dict):
            return {"status": status, "error": "unexpected response type", "rows": []}
        items = resp.get("items") or []
        rows = [fixture_row(i) for i in items if isinstance(i, dict) and i.get("match_id")]
        return {"status": status, "error": "", "rows": rows}
    except Exception as e:
        return {"status": status, "error": f"exception: {e!r}", "rows": []}


async def _do_rest_fetch():
    if not REST_TOKEN:
        return
    async with _rest_lock:
        if _rest["refreshing"]:
            return
        _rest["refreshing"] = True
        try:
            parts = await asyncio.gather(
                _fetch_status(3, 50),   # live
                _fetch_status(1, 50),   # upcoming
                _fetch_status(2, 50),   # finished
            )
            errors = [p["error"] for p in parts if p["error"]]

            merged = {}
            for p in parts:
                for row in p["rows"]:
                    if row["key"] not in merged:
                        merged[row["key"]] = row
            rows = list(merged.values())

            now = time.time()
            _rest["rows"] = rows
            _rest["ts"] = now
            if not errors:
                _rest["last_error"] = ""
                _rest["last_ok_ts"] = now
            else:
                _rest["last_error"] = "; ".join(errors)
                log.warning("REST partial failure: %s", _rest["last_error"])

            log.warning(
                "REST refresh: %d fixtures (live=%d upcoming=%d finished=%d)",
                len(rows),
                sum(1 for r in rows if r["status"] == "started"),
                sum(1 for r in rows if r["status"] == "not_started"),
                sum(1 for r in rows if r["status"] == "completed"),
            )
        finally:
            _rest["refreshing"] = False


async def _fixture_refresh_loop():
    if not REST_TOKEN:
        log.warning("ES_REST_TOKEN missing: upcoming/finished matches dekhabe na")
        return
    try:
        await _do_rest_fetch()
    except Exception as e:
        log.warning("initial REST fetch failed: %r", e)
    while True:
        await asyncio.sleep(REST_REFRESH_S)
        try:
            await _do_rest_fetch()
        except Exception as e:
            log.warning("periodic REST fetch failed: %r", e)


# =====================================================================
# FIX — per-match metadata (from push)
# =====================================================================
def _create_stub_fix(mid: str):
    if mid in FIX:
        return
    FIX[mid] = {
        "key": mid,
        "title": f"Match {mid}",
        "format": "",
        "status": "started",
        "start_at": None,
        "teams": {"a": {}, "b": {}},
    }
    if len(FIX) > MAX_CACHE:
        FIX.pop(next(iter(FIX)), None)


async def _fetch_fix_metadata(mid: str):
    try:
        if not REST_TOKEN:
            _create_stub_fix(mid)
            return
        r = await client.get(
            f"{REST_BASE}/matches/{mid}/info",
            params={"token": REST_TOKEN},
        )
        if not r.is_success:
            log.warning("FIX fetch %s HTTP %s", mid, r.status_code)
            _create_stub_fix(mid)
            return
        resp = r.json().get("response") or {}
        mi = resp.get("match_info") if isinstance(resp.get("match_info"), dict) else resp
        if isinstance(mi, dict) and mi.get("match_id"):
            FIX[mid] = fixture_row(mi)
            log.warning("FIX populated via REST: %s → %s", mid, FIX[mid].get("title"))
        else:
            _create_stub_fix(mid)
    except Exception as e:
        log.warning("FIX fetch %s failed: %r", mid, e)
        _create_stub_fix(mid)
    finally:
        _fix_pending.discard(mid)


def _ensure_fix(mid: str):
    if mid in FIX:
        return
    if REST_TOKEN:
        if mid not in _fix_pending:
            _fix_pending.add(mid)
            _spawn(_fetch_fix_metadata(mid))
        return
    _create_stub_fix(mid)


# =====================================================================
# PER-MATCH WORKER
# =====================================================================
_match_queues: dict = {}
_match_workers: dict = {}


def _ensure_worker(mid: str):
    w = _match_workers.get(mid)
    if w and not w.done():
        return
    q = asyncio.Queue(maxsize=MATCH_QUEUE_SIZE)
    _match_queues[mid] = q
    _match_workers[mid] = asyncio.create_task(_match_worker(mid, q))


async def _match_worker(mid: str, q: asyncio.Queue):
    while True:
        try:
            msg = await asyncio.wait_for(q.get(), timeout=WORKER_IDLE_TIMEOUT_S)
        except asyncio.TimeoutError:
            if mid not in _watched_ids():
                _match_workers.pop(mid, None)
                _match_queues.pop(mid, None)
                return
            continue
        except asyncio.CancelledError:
            return
        try:
            _process_match_msg(msg, mid)
        except Exception as e:
            log.warning("match %s process error: %r", mid, e)
        finally:
            q.task_done()


def _process_match_msg(msg: dict, mid: str):
    now = time.time()
    r = msg.get("response")
    if not isinstance(r, dict):
        return

    if "ball_event" in r:
        d = r.get("data") if isinstance(r.get("data"), dict) else {}
        out = {
            "t": "ev", "id": mid,
            "ev": str(r.get("ball_event", "")),
            "code": str(r.get("code", "")),
            "over": d.get("over"), "ball": d.get("ball"),
            "inn": _i(d.get("inn_num")), "iid": d.get("iid"),
            "striker": d.get("striker_name"), "bowler": d.get("bowler_name"),
            "_server_ts": now * 1000,
        }
        manager.broadcast_sync(_dumps(out), mid)
        return

    if "match_info" in r or "live" in r:
        out = slim_match(msg, now)
        if not out["id"] or out["id"] != mid:
            return
        LATEST_OBJ[mid] = out
        if len(LATEST_OBJ) > MAX_CACHE:
            LATEST_OBJ.pop(next(iter(LATEST_OBJ)), None)
        manager.broadcast_sync(_dumps(out), mid)


# =====================================================================
# UPSTREAM — Entity push
# =====================================================================
def handle_upstream(raw):
    try:
        msg = _decode(raw)
    except Exception:
        return
    if not isinstance(msg, dict):
        return
    r = msg.get("response")
    if not isinstance(r, dict):
        return

    mi0 = r.get("match_info") if isinstance(r.get("match_info"), dict) else {}
    live0 = r.get("live") if isinstance(r.get("live"), dict) else {}
    mid = str(
        r.get("match_id")
        or mi0.get("match_id")
        or live0.get("mid")
        or ""
    )
    if not mid:
        return

    if isinstance(mi0, dict) and mi0.get("match_id"):
        FIX[mid] = fixture_row(mi0)
        _fix_pending.discard(mid)
        if len(FIX) > MAX_CACHE:
            FIX.pop(next(iter(FIX)), None)
    else:
        _ensure_fix(mid)

    if mid not in _watched_ids():
        return

    _ensure_worker(mid)
    q = _match_queues.get(mid)
    if q is None:
        return
    try:
        q.put_nowait(msg)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
            q.task_done()
        except Exception:
            pass
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def upstream_loop():
    url = f"{PUSH_URL}?token={PUSH_TOKEN}"
    delay = 3
    while True:
        t0 = time.time()
        msgs0 = UP["msgs"]
        try:
            async with websockets.connect(url, ping_interval=None, max_size=None) as ws:
                UP["connected"] = True
                log.warning("Entity push connected")
                async for raw in ws:
                    UP["msgs"] += 1
                    UP["last"] = time.time()
                    try:
                        handle_upstream(raw)
                    except Exception as e:
                        log.warning("push message skip: %r", e)
                    await asyncio.sleep(0)
                log.warning("Entity push: server close (code=%s reason=%r)",
                            ws.close_code, ws.close_reason)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("Entity push disconnected: %r", e)
        UP["connected"] = False
        if time.time() - t0 > 30 or UP["msgs"] > msgs0:
            delay = 3
        else:
            log.warning("Push token reject/limit — invalid token? 2+ connections?")
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    push_task = None
    refresh_task = None
    if PUSH_TOKEN:
        push_task = asyncio.create_task(upstream_loop())
    else:
        log.warning("ES_PUSH_TOKEN missing: live ball-by-ball cholbe na")
    if REST_TOKEN:
        refresh_task = asyncio.create_task(_fixture_refresh_loop())
    else:
        log.warning("ES_REST_TOKEN missing: upcoming/finished fixture dekhabe na")
    log.warning("REST base: %s | orjson: %s", REST_BASE, _orjson is not None)
    yield
    for t in (push_task, refresh_task):
        if t:
            t.cancel()
            try:
                await t
            except BaseException:
                pass
    for w in list(_match_workers.values()):
        w.cancel()
    for w in list(manager.writers.values()):
        w.cancel()
    await client.aclose()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _send_baseline(ws: WebSocket, mid: str):
    """Watch এ cache miss হলে REST থেকে baseline আনে।"""
    if not REST_TOKEN:
        return
    try:
        r = await client.get(f"{REST_BASE}/matches/{mid}/info",
                             params={"token": REST_TOKEN})
        resp = r.json().get("response") or {}
        mi = resp.get("match_info") if isinstance(resp.get("match_info"), dict) else resp
        if not isinstance(mi, dict) or not mi:
            return
        if manager.conns.get(ws) != mid:
            return
        obj = slim_match({"response": {"match_id": mid, "match_info": mi}}, time.time())
        obj["id"] = str(mid)
        LATEST_OBJ[mid] = obj
        await manager.send(ws, _dumps(_fresh(obj)))
    except Exception as e:
        log.warning("baseline fetch failed for %s: %r", mid, e)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await manager.send(websocket, "pong")
            elif data.startswith("watch:"):
                mid = data[6:].strip()
                manager.watch(websocket, mid)
                if not mid:
                    continue
                obj = LATEST_OBJ.get(mid)
                stale = obj and (time.time() * 1000 - obj.get("_server_ts", 0) > CACHE_FRESH_MS)
                if obj and not stale:
                    await manager.send(websocket, _dumps(_fresh(obj)))
                elif REST_TOKEN:
                    _spawn(_send_baseline(websocket, mid))
                elif obj:
                    await manager.send(websocket, _dumps(_fresh(obj)))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


async def send_telegram(text: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        r = await client.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text},
        )
        return r.is_success
    except Exception:
        return False


@app.post("/api/click")
async def handle_click(data: dict):
    action = data.get("action")
    match = data.get("match", "")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        _spawn(send_telegram(f"Action: {action}\nMatch: {match}"))
    return {"status": "ok", "action": action}


@app.get("/api/matches")
async def get_matches():
    merged = {}

    for row in _rest["rows"]:
        merged[row["key"]] = dict(row)

    for k, row in FIX.items():
        is_stub = str(row.get("title", "")).startswith("Match ") and not row.get("format")
        if is_stub and k in merged:
            continue
        if k in merged:
            base = merged[k]
            for kk, vv in row.items():
                if vv not in (None, "", {}):
                    base[kk] = vv
        else:
            merged[k] = dict(row)

    order = {"started": 0, "not_started": 1, "completed": 2, "cancelled": 3}
    matches = sorted(merged.values(), key=lambda r: order.get(r.get("status"), 9))

    return {"status": "ok", "data": {"matches": matches}}


# =====================================================================
# ★ FIXED: /api/match/{key} — REST fail হলে 502 নয়, debug log + FIX fallback
# =====================================================================
@app.get("/api/match/{match_key}")
async def get_match(match_key: str):
    # 1. In-memory cache (push থেকে আসা freshest data)
    obj = LATEST_OBJ.get(match_key)
    if obj:
        return {"status": "ok", "data": _fresh(obj)}

    # 2. REST fetch (detailed logging)
    if REST_TOKEN:
        try:
            r = await client.get(
                f"{REST_BASE}/matches/{match_key}/info",
                params={"token": REST_TOKEN},
            )
            log.warning(
                "baseline %s: REST HTTP %s body=%s",
                match_key, r.status_code, r.text[:300],
            )

            if r.is_success:
                try:
                    resp = r.json().get("response") or {}
                except Exception as je:
                    log.warning("baseline %s: json parse failed: %r", match_key, je)
                    resp = {}

                # Entity error wrapper: {"error": true, "message": "..."}
                if isinstance(resp, dict) and resp.get("error"):
                    log.warning("baseline %s: REST error: %s", match_key, str(resp)[:200])
                else:
                    mi = resp.get("match_info") if isinstance(resp.get("match_info"), dict) else resp
                    if isinstance(mi, dict) and mi.get("match_id"):
                        obj = slim_match(
                            {"response": {"match_id": match_key, "match_info": mi}},
                            time.time(),
                        )
                        obj["id"] = str(match_key)
                        LATEST_OBJ[match_key] = obj   # cache করার জন্য পরের বার কাজে লাগবে
                        return {"status": "ok", "data": _fresh(obj)}
                    else:
                        log.warning(
                            "baseline %s: no match_info (resp keys=%s)",
                            match_key,
                            list(resp.keys())[:10] if isinstance(resp, dict)
                            else type(resp).__name__,
                        )
        except Exception as e:
            log.warning("baseline %s: REST exception: %r", match_key, e)

    # 3. FIX fallback — viewport এ অন্তত teams দেখাবে
    fix = FIX.get(match_key)
    if fix:
        return {
            "status": "ok",
            "data": {
                "t": "match",
                "id": str(match_key),
                "_server_ts": time.time() * 1000,
                "status": fix.get("status", "started"),
                "fmt": fix.get("format", ""),
                "title": fix.get("title", ""),
                "a": {"name": fix.get("teams", {}).get("a", {}).get("name", "")},
                "b": {"name": fix.get("teams", {}).get("b", {}).get("name", "")},
                "score": None,
                "comm": [],
            },
        }

    # 4. Really nothing — 404 (frontend retry logic better handle করবে)
    return JSONResponse(
        {"status": "error", "message": "match info not available (no cache + REST failed)"},
        status_code=404,
    )


BASE_DIR = Path(__file__).resolve().parent


def _file(name: str):
    f = BASE_DIR / name
    if not f.exists():
        return JSONResponse({"detail": f"{name} not found"}, status_code=404)
    return FileResponse(f)


@app.get("/")
async def index():
    return _file("index.html")


@app.get("/index.css")
async def css():
    return _file("index.css")


@app.get("/script.js")
async def js():
    return _file("script.js")


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ws_clients": len(manager.conns),
        "watched_matches": sorted(_watched_ids()),
        "match_workers": sorted(_match_workers.keys()),
        "push_connected": UP["connected"],
        "push_msgs": UP["msgs"],
        "push_last_age_s": round(time.time() - UP["last"], 1) if UP["last"] else None,
        "matches_cached": len(LATEST_OBJ),
        "fixtures_cached": len(FIX),
        "rest_token_set": bool(REST_TOKEN),
        "rest_base": REST_BASE,
        "rest_cache_age_s": round(time.time() - _rest["ts"], 1) if _rest["ts"] else None,
        "rest_cache_rows": len(_rest["rows"]),
        "rest_last_error": _rest["last_error"] or None,
        "rest_last_ok_age_s": round(time.time() - _rest["last_ok_ts"], 1) if _rest["last_ok_ts"] else None,
        "fix_pending": list(_fix_pending),
        "orjson": _orjson is not None,
    }


@app.get("/api/debug")
async def debug():
    counts = {"started": 0, "not_started": 0, "completed": 0, "cancelled": 0}
    for r in _rest["rows"]:
        counts[r.get("status", "not_started")] = counts.get(r.get("status", "not_started"), 0) + 1
    return {
        "push": UP,
        "rest": {
            "base": REST_BASE,
            "token_set": bool(REST_TOKEN),
            "cache_age_s": round(time.time() - _rest["ts"], 1) if _rest["ts"] else None,
            "rows": len(_rest["rows"]),
            "counts": counts,
            "last_error": _rest["last_error"] or None,
            "sample": _rest["rows"][:3],
        },
        "fix": {
            "count": len(FIX),
            "keys": list(FIX.keys())[:20],
            "sample": next(iter(FIX.values()), None),
            "pending": list(_fix_pending),
        },
        "watched_matches": sorted(_watched_ids()),
        "match_workers": sorted(_match_workers.keys()),
        "latest_obj_keys": list(LATEST_OBJ.keys())[:20],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=1,
                access_log=False, log_level="warning",
                ws_per_message_deflate=False)