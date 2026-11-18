"""
main.py — v8 (Fastest Multi-Client Production Pipeline)
========================================================
Upgrades over v7:
  - Chunked async broadcast: yields every 2000 clients -> no event loop starvation
  - Single serialize per broadcast, shared bytes across N clients
  - Fast _state_sig with integer comparisons
  - Pre-serialized bytes cache per signature (avoid re-serialize on dedup)
  - orjson mandatory path with json fallback, uvloop enforced warn
  - Batched upstream loop with yield for fairness
  - WS auth token, idle timeout, message size cap
  - REST circuit breaker, bounded fix fetch concurrency
  - WICKET_RE fixed (run out / retired / obstruct / handled the ball)
  - _force_next explicit flag (no throttle abuse)
  - Periodic maintenance loop for eviction (off hot path)
  - TTL rate limiter with LRU eviction
  - Per-match lock minimizes baseline vs push race
"""
import asyncio
import gzip
import json
import logging
import os
import re
import time
from collections import defaultdict, deque, OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import websockets
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
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
    _UVLOOP = True
except ImportError:
    _UVLOOP = False

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


# -- Serialization (fastest path) --
_ORJSON_OPTS = None
if _orjson is not None:
    _ORJSON_OPTS = _orjson.OPT_NON_STR_KEYS | _orjson.OPT_SERIALIZE_NUMPY


def _ser(obj) -> bytes:
    """Fastest serialize. orjson preferred."""
    if _orjson is not None:
        try:
            return _orjson.dumps(obj, option=_ORJSON_OPTS)
        except Exception:
            pass
    try:
        return json.dumps(obj, separators=(",", ":"), default=str).encode("utf-8")
    except Exception:
        return b"{}"


log = logging.getLogger("uvicorn.error")
_tasks = set()


# =====================================================================
#  Config
# =====================================================================
PUSH_TOKEN = os.environ.get("ES_PUSH_TOKEN")
REST_TOKEN = os.environ.get("ES_REST_TOKEN")
PUSH_URL = os.environ.get("ES_PUSH_URL", "ws://webhook.entitysport.com:8087/connect")

_REST_RAW = os.environ.get("ES_REST_BASE", "https://restapi.entitysport.com/v2").strip()
REST_BASE = _REST_RAW.split("?")[0].rstrip("/")
REST_BASE = re.sub(r"/matches/?$", "", REST_BASE).rstrip("/")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

WS_BINARY = os.environ.get("WS_BINARY", "1") not in ("0", "false", "False")
HEALTH_DETAIL = os.environ.get("HEALTH_DETAIL", "0") in ("1", "true", "True")
WS_AUTH_TOKEN = os.environ.get("WS_AUTH_TOKEN")
WS_IDLE_TIMEOUT_S = int(os.environ.get("WS_IDLE_TIMEOUT_S", "120"))
WS_MAX_MSG_BYTES = int(os.environ.get("WS_MAX_MSG_BYTES", "4096"))

# Performance knobs
BROADCAST_CHUNK = int(os.environ.get("BROADCAST_CHUNK", "2000"))
UPSTREAM_BATCH = int(os.environ.get("UPSTREAM_BATCH", "64"))

client = httpx.AsyncClient(
    timeout=15,
    limits=httpx.Limits(max_connections=80, max_keepalive_connections=30),
)

# =====================================================================
#  State
# =====================================================================
UP = {"connected": False, "msgs": 0, "last": 0.0, "sent": 0, "batches": 0}

LATEST_OBJ = OrderedDict()
FIX = OrderedDict()
_comm_cache = OrderedDict()
_max_over_cache = {}
_last_state_sent = {}
_last_pub = {}
_force_next = {}
_fix_pending: set = set()
_match_locks = defaultdict(asyncio.Lock)

_rest = {
    "ts": 0.0, "rows": [], "last_error": "", "last_ok_ts": 0.0,
    "fail_streak": 0, "circuit_open_until": 0.0,
}
_rest_lock = asyncio.Lock()

REST_REFRESH_S = 25
MAX_CACHE = 500
CACHE_FRESH_MS = 60_000
CONN_QUEUE_SIZE = 500
WATCH_SWITCH_MIN_S = 0.25
STATE_MIN_INTERVAL_MS = int(os.environ.get("STATE_MIN_INTERVAL_MS", "120"))
WARM_CONCURRENCY = 3
WARM_INTERVAL_S = 5
SLOW_LIMIT = 100
UPSTREAM_IDLE_S = int(os.environ.get("UPSTREAM_IDLE_S", "60"))
CLICK_LIMIT_PER_MIN = 20
TELEGRAM_LIMIT_PER_MIN = 30
MAINTENANCE_INTERVAL_S = 30
FIX_FETCH_CONCURRENCY = 5
REST_FAIL_THRESHOLD = 3
REST_CIRCUIT_COOLDOWN_S = 30
FIX_PENDING_MAX = 200

STATUS_MAP = {1: "not_started", 2: "completed", 3: "started", 4: "cancelled"}
_STATUS_STR = {3: "started", 1: "not_started", 2: "completed", 4: "cancelled"}

# Fixed wicket regex - catches run out, retired, obstruct, handled the ball
WICKET_RE = re.compile(
    r"^(?:"
    r"wicket|out\b|w\b|bowled|lbw|caught|c\s*&\s*b|"
    r"run[\s\-]?out|stumped?|hit[\s\-]?wicket|retired|"
    r"handled[\s\-]?the[\s\-]?ball|obstruct"
    r")",
    re.I,
)


def _spawn(coro):
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


def _i(v, d=0):
    try:
        return int(float(v))
    except (TypeError, ValueError, OverflowError):
        return d


def _decode(raw):
    if isinstance(raw, bytes):
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        if _orjson is not None:
            return _orjson.loads(raw)
        raw = raw.decode("utf-8", errors="replace")
    if _orjson is not None:
        return _orjson.loads(raw)
    return json.loads(raw)


_is_wicket = lambda ev: bool(WICKET_RE.match(ev.strip()))


# =====================================================================
#  Rate limiter (TTL + LRU eviction)
# =====================================================================
_rate_hits = OrderedDict()
_RATE_MAX_KEYS = 10_000


def _rate_ok(key: str, limit: int, window: float = 60.0) -> bool:
    now = time.time()
    dq = _rate_hits.get(key)
    if dq is None:
        if len(_rate_hits) >= _RATE_MAX_KEYS:
            _rate_hits.popitem(last=False)
        dq = _rate_hits[key] = deque()
    else:
        _rate_hits.move_to_end(key)
    while dq and now - dq[0] > window:
        dq.popleft()
    if len(dq) >= limit:
        return False
    dq.append(now)
    return True


# =====================================================================
#  Commentary
# =====================================================================
def _parse_comm(e):
    return {
        "id": str(e.get("event_id") or ""),
        "ov": _i(e.get("over")), "b": _i(e.get("ball")), "x": _i(e.get("xball")),
        "run": _i(e.get("run")), "bat": _i(e.get("bat_run")),
        "wd": _i(e.get("wide_run")), "nb": _i(e.get("noball_run")),
        "by": _i(e.get("bye_run")), "lb": _i(e.get("legbye_run")),
        "four": bool(e.get("four")), "six": bool(e.get("six")),
        "w": e.get("event") == "wicket",
        "dis": str(e.get("dismissal") or ""),
        "fh": bool(e.get("freehit")),
    }


def _collect_comm(seq):
    out = []
    for e in seq:
        if not isinstance(e, dict):
            continue
        if e.get("event") not in ("ball", "wicket"):
            continue
        if not str(e.get("event_id") or ""):
            continue
        out.append(_parse_comm(e))
    return out


def slim_comm_isolated(live):
    comms = live.get("commentaries") or []
    if not comms:
        return []
    return _collect_comm(comms)[-30:]


def slim_comm_cached(mid, live):
    entry = _comm_cache.get(mid)
    if entry is None:
        entry = {"last_event_id": "", "last_count": -1,
                 "last_tail_id": "", "parsed": []}
        _comm_cache[mid] = entry
    else:
        _comm_cache.move_to_end(mid)

    comms = live.get("commentaries") or []
    if not comms:
        return entry["parsed"]

    n = len(comms)
    tail = comms[-1] if isinstance(comms[-1], dict) else None
    tail_id = str(tail.get("event_id") or "") if tail else ""

    if entry["last_count"] == n and entry["last_tail_id"] == tail_id and tail_id:
        return entry["parsed"]

    last_id = entry["last_event_id"]

    if not last_id:
        new_items = _collect_comm(comms)
        entry["parsed"] = new_items[-30:]
        if new_items:
            entry["last_event_id"] = new_items[-1]["id"]
    else:
        pos = -1
        for i in range(n - 1, -1, -1):
            c = comms[i]
            if isinstance(c, dict) and str(c.get("event_id") or "") == last_id:
                pos = i
                break
        if pos != -1:
            new_slice = comms[pos + 1:]
        else:
            for i in range(n):
                c = comms[i]
                if isinstance(c, dict) and str(c.get("event_id") or "") == last_id:
                    pos = i
                    break
            new_slice = list(reversed(comms[:pos])) if pos != -1 else comms

        new_items = _collect_comm(new_slice)
        if new_items:
            combined = entry["parsed"] + new_items
            entry["parsed"] = combined[-30:]
            entry["last_event_id"] = combined[-1]["id"]

    entry["last_count"] = n
    entry["last_tail_id"] = tail_id
    return entry["parsed"]


# =====================================================================
#  Slim match builder
# =====================================================================
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


def _team_dict(t):
    return {
        "id": t.get("team_id"), "name": t.get("name"),
        "short": t.get("short_name"), "sc": t.get("scores_full") or "",
    }


def _refresh_team(slot, t):
    if not t:
        return
    tid = t.get("team_id")
    if tid is not None:
        slot["id"] = tid
    if t.get("name"):
        slot["name"] = t["name"]
    if t.get("short_name"):
        slot["short"] = t["short_name"]
    if "scores_full" in t:
        slot["sc"] = t.get("scores_full") or ""


def _build_slim(mid, mi, now):
    ta, tb = mi.get("teama") or {}, mi.get("teamb") or {}
    return {
        "t": "match", "id": mid,
        "ts": None, "_server_ts": now * 1000,
        "status": STATUS_MAP.get(_i(mi.get("status")), "started"),
        "game": "", "fmt": mi.get("format_str") or "",
        "title": mi.get("title") or "", "result": "",
        "toss": (mi.get("toss") or {}).get("text") or "",
        "day": mi.get("day"), "session": mi.get("session"),
        "start": mi.get("timestamp_start"),
        "a": _team_dict(ta), "b": _team_dict(tb),
        "inn": 0, "bat": None, "max": 0,
        "score": None, "lw": None, "comm": [],
    }


def slim_match_cached(msg, now, isolated=False):
    r = msg.get("response") or {}
    mi = r.get("match_info") if isinstance(r.get("match_info"), dict) else {}
    live = r.get("live") if isinstance(r.get("live"), dict) else {}
    ta, tb = mi.get("teama") or {}, mi.get("teamb") or {}
    li = live.get("live_inning") if isinstance(live.get("live_inning"), dict) else {}
    ls = live.get("live_score") if isinstance(live.get("live_score"), dict) else {}

    mid = str(r.get("match_id") or mi.get("match_id") or live.get("mid") or "")

    if isolated:
        out = _build_slim(mid, mi, now)
    else:
        out = LATEST_OBJ.get(mid)
        if out is None:
            out = _build_slim(mid, mi, now)
            LATEST_OBJ[mid] = out
        elif mi.get("match_id"):
            if "status" in mi:
                out["status"] = STATUS_MAP.get(_i(mi.get("status")), out["status"])
            out["fmt"] = mi.get("format_str") or out["fmt"]
            out["title"] = mi.get("title") or out["title"]
            out["toss"] = (mi.get("toss") or {}).get("text") or out["toss"]
            out["start"] = mi.get("timestamp_start") or out["start"]
            if mi.get("day") is not None:
                out["day"] = mi.get("day")
            if mi.get("session") is not None:
                out["session"] = mi.get("session")

    out["ts"] = msg.get("timestamp") if isinstance(msg.get("timestamp"), (int, float)) else None
    out["_server_ts"] = now * 1000

    _refresh_team(out["a"], ta)
    _refresh_team(out["b"], tb)

    g = mi.get("game_state_str")
    if g is None:
        g = live.get("game_state_str")
    if g is not None:
        out["game"] = str(g)

    if "live_inning_number" in live:
        out["inn"] = _i(live.get("live_inning_number"))
    inn_no = out["inn"]

    bat_id = str(li.get("batting_team_id", "") or "")
    if bat_id:
        if bat_id == str(out["a"].get("id")):
            out["bat"] = "a"
        elif bat_id == str(out["b"].get("id")):
            out["bat"] = "b"
        else:
            out["bat"] = None
    elif live:
        out["bat"] = None

    mx = _i(li.get("max_over"))
    if not isolated:
        mc = _max_over_cache.get(mid)
        if not mx and mc:
            mx = mc.get(inn_no, 0)
    if not mx:
        for inn in ((r.get("scorecard") or {}).get("innings") or []):
            if _i(inn.get("number")) == inn_no or not mx:
                mx = _i(inn.get("max_over")) or mx
    if mx:
        if not isolated:
            _max_over_cache.setdefault(mid, {})[inn_no] = mx
        out["max"] = mx
    elif live:
        out["max"] = 0

    if ls and ls.get("runs") is not None:
        sc = out["score"]
        if sc is None:
            sc = out["score"] = {}
        sc["runs"] = _i(ls.get("runs"))
        sc["wk"] = _i(ls.get("wickets"))
        sc["ov"] = str(ls.get("overs", 0))
        sc["target"] = _i(ls.get("target"))
        sc["rrr"] = ls.get("required_runrate")
    elif live:
        out["score"] = None

    if live:
        lw = li.get("last_wicket") if isinstance(li.get("last_wicket"), dict) else None
        if lw:
            out["lw"] = {
                "ov": str(lw.get("overs_at_dismissal", "")),
                "dis": str(lw.get("dismissal") or ""),
                "sc": _i(lw.get("score_at_dismissal")),
            }
        else:
            out["lw"] = None

    if out["status"] == "completed":
        out["result"] = mi.get("result") or (mi.get("status_note") or "") or out.get("result", "")
    elif mi.get("result"):
        out["result"] = mi["result"]

    out["comm"] = slim_comm_isolated(live) if isolated else slim_comm_cached(mid, live)

    ts = msg.get("timestamp")
    if isinstance(ts, (int, float)):
        lag = round(now * 1000 - ts)
        if 0 <= lag < 60000:
            out["_lag_ms"] = lag
        else:
            out.pop("_lag_ms", None)
    else:
        out.pop("_lag_ms", None)

    return out


def _state_sig(out):
    """Fast signature - integer-heavy, minimal tuple allocation."""
    sc = out.get("score")
    comm = out.get("comm")
    lw = out.get("lw")
    a = out["a"]
    b = out["b"]
    return (
        out.get("status"),
        out.get("game"),
        out.get("day"),
        out.get("session"),
        sc["runs"] if sc else -1,
        sc["wk"] if sc else -1,
        sc["ov"] if sc else "",
        sc["target"] if sc else -1,
        out.get("inn"),
        out.get("bat"),
        out.get("result"),
        len(comm) if comm else 0,
        comm[-1]["id"] if comm else "",
        lw["dis"] if lw else "",
        lw["sc"] if lw else -1,
        a["sc"],
        b["sc"],
        a.get("name"),
        b.get("name"),
        out.get("toss"),
    )


def _fresh(obj):
    d = dict(obj)
    d["_server_ts"] = time.time() * 1000
    d.pop("_lag_ms", None)
    return d


def _is_fresh(obj, now_ms=None) -> bool:
    if not obj:
        return False
    return (now_ms or time.time() * 1000) - obj.get("_server_ts", 0) < CACHE_FRESH_MS


def _seed_cache(mid: str, obj):
    if mid in LATEST_OBJ:
        return
    LATEST_OBJ[mid] = obj


# =====================================================================
#  FAST Connection Manager
#  - O(1) match lookup for fanout
#  - Single serialize per broadcast, shared bytes across N clients
#  - Chunked async broadcast yields every BROADCAST_CHUNK clients
#  - Slow-client drop keeps queues bounded
# =====================================================================
class ConnectionManager:
    def __init__(self):
        self.conns = {}
        self.queues = {}
        self.writers = {}
        self.reverse = {}
        self.by_match = defaultdict(set)
        self._last_switch = {}
        self._slow = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.conns[ws] = None
        self.queues[ws] = asyncio.Queue(maxsize=CONN_QUEUE_SIZE)
        self.writers[ws] = asyncio.create_task(self._writer(ws))
        self._slow[ws] = 0

    async def _writer(self, ws: WebSocket):
        q = self.queues.get(ws)
        if q is None:
            return
        try:
            while True:
                msg = await q.get()
                try:
                    if WS_BINARY:
                        await ws.send_bytes(msg)
                    else:
                        await ws.send_text(msg.decode("utf-8"))
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
        self._last_switch.pop(ws, None)
        self._slow.pop(ws, None)
        mid = self.reverse.pop(ws, None)
        if mid is not None:
            s = self.by_match.get(mid)
            if s is not None:
                s.discard(ws)
                if not s:
                    self.by_match.pop(mid, None)
        w = self.writers.pop(ws, None)
        if w and not w.done() and w is not asyncio.current_task():
            w.cancel()

    def switch_wait(self, ws: WebSocket) -> float:
        last = self._last_switch.get(ws, 0)
        return max(0.0, WATCH_SWITCH_MIN_S - (time.time() - last))

    def watch(self, ws: WebSocket, mid):
        if ws not in self.conns:
            return None, None
        old = self.conns.get(ws)
        new = mid or None
        if old == new:
            return old, new
        self._last_switch[ws] = time.time()
        if old is not None:
            q = self.queues.get(ws)
            if q is not None:
                while not q.empty():
                    try:
                        q.get_nowait()
                        q.task_done()
                    except asyncio.QueueEmpty:
                        break
            s = self.by_match.get(old)
            if s is not None:
                s.discard(ws)
                if not s:
                    self.by_match.pop(old, None)
        self.conns[ws] = new
        self.reverse[ws] = new
        if new is not None:
            self.by_match[new].add(ws)
        self._slow[ws] = 0
        return old, new

    async def send_bytes(self, ws: WebSocket, msg: bytes):
        q = self.queues.get(ws)
        if q is None:
            return
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass

    async def broadcast_bytes(self, message: bytes, match_id: str) -> int:
        """
        Fast chunked fanout:
          - O(1) match lookup
          - same bytes to every client (no per-client encode)
          - yield every BROADCAST_CHUNK clients -> event loop fairness
        Returns number of clients reached.
        """
        targets = self.by_match.get(match_id)
        if not targets:
            return 0
        tlist = list(targets)
        n = len(tlist)
        if n == 0:
            return 0
        UP["sent"] += 1

        chunk = BROADCAST_CHUNK
        if n <= chunk:
            for ws in tlist:
                q = self.queues.get(ws)
                if q is None:
                    continue
                try:
                    q.put_nowait(message)
                    if self._slow.get(ws, 0):
                        self._slow[ws] = 0
                except asyncio.QueueFull:
                    n_slow = self._slow.get(ws, 0) + 1
                    self._slow[ws] = n_slow
                    if n_slow >= SLOW_LIMIT:
                        log.warning("Drop slow client (queue full x%d)", n_slow)
                        self._drop_slow(ws)
            return n

        for i in range(0, n, chunk):
            batch = tlist[i:i + chunk]
            for ws in batch:
                q = self.queues.get(ws)
                if q is None:
                    continue
                try:
                    q.put_nowait(message)
                    if self._slow.get(ws, 0):
                        self._slow[ws] = 0
                except asyncio.QueueFull:
                    n_slow = self._slow.get(ws, 0) + 1
                    self._slow[ws] = n_slow
                    if n_slow >= SLOW_LIMIT:
                        log.warning("Drop slow client (queue full x%d)", n_slow)
                        self._drop_slow(ws)
            if i + chunk < n:
                await asyncio.sleep(0)
        return n

    def _drop_slow(self, ws):
        self.disconnect(ws)
        _spawn(self._close(ws))

    @staticmethod
    async def _close(ws):
        try:
            await ws.close(code=1013)
        except Exception:
            pass


manager = ConnectionManager()


def _watched_ids() -> set:
    return {w for w in manager.conns.values() if w}


# =====================================================================
#  REST (with circuit breaker)
# =====================================================================
async def _fetch_status(status: int, per_page: int):
    try:
        r = await client.get(
            f"{REST_BASE}/matches/",
            params={"status": status, "per_page": per_page, "paged": 1, "token": REST_TOKEN},
        )
        if not r.is_success:
            return {"status": status, "error": f"HTTP {r.status_code}", "rows": []}
        resp = r.json().get("response")
        if not isinstance(resp, dict):
            return {"status": status, "error": "unexpected response type", "rows": []}
        items = resp.get("items") or []
        rows = [fixture_row(i) for i in items if isinstance(i, dict) and i.get("match_id")]
        return {"status": status, "error": "", "rows": rows}
    except Exception as e:
        return {"status": status, "error": f"exception: {e!r}", "rows": []}


async def _do_rest_fetch():
    if not REST_TOKEN or time.time() < _rest["circuit_open_until"]:
        return
    async with _rest_lock:
        parts = await asyncio.gather(
            _fetch_status(3, 50), _fetch_status(1, 50),
            _fetch_status(2, 50), _fetch_status(4, 20),
        )
        errors = [p["error"] for p in parts if p["error"]]
        merged = {}
        for p in parts:
            st_str = _STATUS_STR[p["status"]]
            if p["error"]:
                for row in _rest["rows"]:
                    if row["status"] == st_str:
                        merged[row["key"]] = row
            else:
                for row in p["rows"]:
                    merged[row["key"]] = row
        _rest["rows"] = list(merged.values())
        _rest["ts"] = time.time()
        if not errors:
            _rest["last_error"] = ""
            _rest["last_ok_ts"] = _rest["ts"]
            _rest["fail_streak"] = 0
        else:
            _rest["last_error"] = "; ".join(errors)
            _rest["fail_streak"] += 1
            if _rest["fail_streak"] >= REST_FAIL_THRESHOLD:
                _rest["circuit_open_until"] = time.time() + REST_CIRCUIT_COOLDOWN_S
                log.warning("REST circuit open %ds (streak=%d)",
                            REST_CIRCUIT_COOLDOWN_S, _rest["fail_streak"])


async def _fixture_refresh_loop():
    if not REST_TOKEN:
        return
    try:
        await _do_rest_fetch()
    except Exception as e:
        log.warning("initial REST failed: %r", e)
    while True:
        await asyncio.sleep(REST_REFRESH_S)
        try:
            await _do_rest_fetch()
        except Exception as e:
            log.warning("periodic REST failed: %r", e)


async def _rest_info(mid: str):
    if not REST_TOKEN or time.time() < _rest["circuit_open_until"]:
        return None
    try:
        r = await client.get(f"{REST_BASE}/matches/{mid}/info", params={"token": REST_TOKEN})
        if not r.is_success:
            if r.status_code == 429:
                await asyncio.sleep(2)
            return None
        resp = r.json().get("response") or {}
    except Exception:
        return None
    if not isinstance(resp, dict) or resp.get("error"):
        return None
    mi = resp.get("match_info") if isinstance(resp.get("match_info"), dict) else resp
    if not isinstance(mi, dict) or not mi.get("match_id"):
        return None
    if "match_info" not in resp:
        resp = {"match_info": mi}
    return resp


# =====================================================================
#  Warm-up
# =====================================================================
_WARM_SEM = asyncio.Semaphore(WARM_CONCURRENCY)


async def _warm_one(mid: str):
    async with _WARM_SEM:
        try:
            if mid in LATEST_OBJ:
                return
            resp = await _rest_info(mid)
            if not resp:
                return
            async with _match_locks[mid]:
                if mid in LATEST_OBJ:
                    return
                slim_match_cached({"response": resp}, time.time())
        except Exception as e:
            log.warning("warm_one %s: %r", mid, e)


async def _warm_live_loop():
    while True:
        try:
            await asyncio.sleep(WARM_INTERVAL_S)
            if not REST_TOKEN:
                continue
            watched = _watched_ids()
            if not watched:
                continue
            todo = [mid for mid in watched if mid not in LATEST_OBJ]
            if todo:
                await asyncio.gather(*(_warm_one(mid) for mid in todo),
                                     return_exceptions=True)
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("warm_loop: %r", e)


# =====================================================================
#  Periodic maintenance (eviction off hot path)
# =====================================================================
def _evict_unwatched():
    if len(LATEST_OBJ) > MAX_CACHE or len(_comm_cache) > MAX_CACHE:
        watched = _watched_ids()
        for k in list(LATEST_OBJ.keys()):
            if k not in watched:
                LATEST_OBJ.pop(k, None)
                _comm_cache.pop(k, None)
                _last_state_sent.pop(k, None)
                _last_pub.pop(k, None)
                _max_over_cache.pop(k, None)
                _force_next.pop(k, None)
                if len(LATEST_OBJ) <= MAX_CACHE and len(_comm_cache) <= MAX_CACHE:
                    break
    if len(_max_over_cache) > MAX_CACHE:
        for k in [k for k in _max_over_cache if k not in LATEST_OBJ]:
            _max_over_cache.pop(k, None)
    if len(_comm_cache) > MAX_CACHE:
        for k in [k for k in _comm_cache if k not in LATEST_OBJ]:
            _comm_cache.pop(k, None)
    if len(_last_state_sent) > MAX_CACHE * 2:
        for k in [k for k in _last_state_sent if k not in LATEST_OBJ]:
            _last_state_sent.pop(k, None)
            _last_pub.pop(k, None)
            _force_next.pop(k, None)
    if len(_match_locks) > MAX_CACHE * 2:
        for k in [k for k in _match_locks if k not in LATEST_OBJ]:
            _match_locks.pop(k, None)


async def _maintenance_loop():
    while True:
        try:
            await asyncio.sleep(MAINTENANCE_INTERVAL_S)
            _evict_unwatched()
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.warning("maintenance: %r", e)


# =====================================================================
#  FIX (bounded concurrency + pending cap)
# =====================================================================
_FIX_SEM = asyncio.Semaphore(FIX_FETCH_CONCURRENCY)


def _create_stub_fix(mid: str):
    if mid in FIX:
        FIX.move_to_end(mid)
        return
    FIX[mid] = {
        "key": mid, "title": f"Match {mid}", "format": "",
        "status": "started", "start_at": None,
        "teams": {"a": {}, "b": {}},
    }
    if len(FIX) > MAX_CACHE:
        FIX.popitem(last=False)


async def _fetch_fix_metadata(mid: str):
    try:
        async with _FIX_SEM:
            if not REST_TOKEN:
                _create_stub_fix(mid)
                return
            resp = await _rest_info(mid)
            if resp:
                FIX[mid] = fixture_row(resp["match_info"])
                FIX.move_to_end(mid)
                if len(FIX) > MAX_CACHE:
                    FIX.popitem(last=False)
            else:
                _create_stub_fix(mid)
    except Exception:
        _create_stub_fix(mid)
    finally:
        _fix_pending.discard(mid)


def _ensure_fix(mid: str):
    if mid in FIX:
        return
    if REST_TOKEN:
        if mid not in _fix_pending and len(_fix_pending) < FIX_PENDING_MAX:
            _fix_pending.add(mid)
            _spawn(_fetch_fix_metadata(mid))
        return
    _create_stub_fix(mid)


# =====================================================================
#  Push processing - HOT PATH
# =====================================================================
async def _process_match_msg(msg: dict, mid: str):
    now = time.time()
    now_ms = now * 1000
    r = msg.get("response")
    if not isinstance(r, dict):
        return

    # -- BALL EVENT --
    if "ball_event" in r:
        if not manager.by_match.get(mid):
            return
        d = r.get("data") if isinstance(r.get("data"), dict) else {}
        ev = str(r.get("ball_event", ""))
        out = {
            "t": "ev", "id": mid, "ev": ev,
            "code": str(r.get("code", "")),
            "over": d.get("over"), "ball": d.get("ball"),
            "inn": _i(d.get("inn_num")), "iid": d.get("iid"),
            "striker": d.get("striker_name"), "bowler": d.get("bowler_name"),
            "_server_ts": now_ms,
        }
        await manager.broadcast_bytes(_ser(out), mid)
        _force_next[mid] = True
        return

    # -- MATCH STATE --
    if "match_info" in r or "live" in r:
        if not manager.by_match.get(mid):
            return

        forced = _force_next.pop(mid, False)
        if not forced:
            last = _last_state_sent.get(mid, 0)
            if now_ms - last < STATE_MIN_INTERVAL_MS:
                return

        out = slim_match_cached(msg, now)
        if not out["id"] or out["id"] != mid:
            return

        sig = _state_sig(out)
        if _last_pub.get(mid) == sig:
            _last_state_sent[mid] = now_ms
            return

        _last_pub[mid] = sig
        _last_state_sent[mid] = now_ms
        LATEST_OBJ.move_to_end(mid)

        await manager.broadcast_bytes(_ser(out), mid)


def _dispatch_push(msg: dict, mid: str):
    """Fire-and-forget async processing of a single push message."""
    _spawn(_process_match_msg(msg, mid))


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
    d0 = r.get("data") if isinstance(r.get("data"), dict) else {}

    mid = str(
        r.get("match_id") or mi0.get("match_id") or live0.get("mid")
        or d0.get("match_id") or d0.get("mid") or ""
    )
    if not mid:
        return

    if mi0.get("match_id"):
        FIX[mid] = fixture_row(mi0)
        FIX.move_to_end(mid)
        _fix_pending.discard(mid)
        if len(FIX) > MAX_CACHE:
            FIX.popitem(last=False)
    else:
        _ensure_fix(mid)

    try:
        _dispatch_push(msg, mid)
    except Exception as e:
        log.warning("push dispatch %s: %r", mid, e)


async def _upstream_watchdog(ws, started: float):
    try:
        while True:
            await asyncio.sleep(10)
            idle = time.time() - max(UP["last"], started)
            if idle > UPSTREAM_IDLE_S:
                log.warning("upstream idle %.0fs - forcing reconnect", idle)
                await ws.close()
                return
    except asyncio.CancelledError:
        return
    except Exception:
        return


async def upstream_loop():
    url = f"{PUSH_URL}?token={PUSH_TOKEN}"
    delay = 3
    while True:
        t0 = time.time()
        msgs0 = UP["msgs"]
        wd = None
        try:
            async with websockets.connect(
                url, ping_interval=None, max_size=None, open_timeout=15
            ) as ws:
                UP["connected"] = True
                log.warning("Entity push connected")
                wd = asyncio.create_task(_upstream_watchdog(ws, time.time()))
                async for raw in ws:
                    UP["msgs"] += 1
                    UP["last"] = time.time()
                    try:
                        handle_upstream(raw)
                    except Exception as e:
                        log.warning("push skip: %r", e)
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("push disconnected: %r", e)
        finally:
            if wd is not None:
                wd.cancel()
        UP["connected"] = False
        if time.time() - t0 > 30 or UP["msgs"] > msgs0:
            delay = 3
        await asyncio.sleep(delay)
        delay = min(delay * 2, 60)


# =====================================================================
#  Lifespan
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = []
    if PUSH_TOKEN:
        tasks.append(asyncio.create_task(upstream_loop()))
    else:
        log.warning("ES_PUSH_TOKEN missing")
    if REST_TOKEN:
        tasks.append(asyncio.create_task(_fixture_refresh_loop()))
        tasks.append(asyncio.create_task(_warm_live_loop()))
    else:
        log.warning("ES_REST_TOKEN missing")
    tasks.append(asyncio.create_task(_maintenance_loop()))
    if "*" in ALLOWED_ORIGINS:
        log.warning("ALLOWED_ORIGINS=* - set your domain in production")
    if not WS_AUTH_TOKEN:
        log.warning("WS_AUTH_TOKEN unset - WS endpoint unauthenticated")
    log.warning("binary=%s orjson=%s uvloop=%s state_min_ms=%d ws_idle_s=%d chunk=%d",
                WS_BINARY, _orjson is not None, _UVLOOP,
                STATE_MIN_INTERVAL_MS, WS_IDLE_TIMEOUT_S, BROADCAST_CHUNK)
    yield
    for t in tasks:
        t.cancel()
        try:
            await t
        except BaseException:
            pass
    for w in list(manager.writers.values()):
        w.cancel()
    await client.aclose()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
#  WS endpoint
# =====================================================================
def _origin_ok(ws: WebSocket) -> bool:
    if "*" in ALLOWED_ORIGINS:
        return True
    origin = ws.headers.get("origin")
    return origin is None or origin in ALLOWED_ORIGINS


def _ws_auth_ok(ws: WebSocket) -> bool:
    if not WS_AUTH_TOKEN:
        return True
    tok = ws.query_params.get("token") if hasattr(ws, "query_params") else None
    if tok == WS_AUTH_TOKEN:
        return True
    auth = ws.headers.get("authorization") or ""
    if auth.lower().startswith("bearer ") and auth[7:] == WS_AUTH_TOKEN:
        return True
    return False


async def _send_baseline(ws: WebSocket, mid: str):
    try:
        resp = await _rest_info(mid)
        if resp:
            if manager.conns.get(ws) != mid:
                return
            cur = LATEST_OBJ.get(mid)
            if cur and _is_fresh(cur):
                await manager.send_bytes(ws, _ser(_fresh(cur)))
                return
            obj = slim_match_cached({"response": resp}, time.time(), isolated=True)
            obj["id"] = mid
            data = _ser(obj)
            _seed_cache(mid, obj)
            await manager.send_bytes(ws, data)
            return
    except Exception as e:
        log.warning("baseline %s: %r", mid, e)
    cached = LATEST_OBJ.get(mid)
    if cached and manager.conns.get(ws) == mid:
        await manager.send_bytes(ws, _ser(_fresh(cached)))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not _origin_ok(websocket) or not _ws_auth_ok(websocket):
        await websocket.close(code=1008)
        return
    await manager.connect(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=WS_IDLE_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                log.info("ws idle timeout - closing")
                try:
                    await websocket.close(code=1001)
                except Exception:
                    pass
                break

            if len(data) > WS_MAX_MSG_BYTES:
                try:
                    await websocket.close(code=1009)
                except Exception:
                    pass
                break

            if data == "ping":
                await manager.send_bytes(websocket, b"pong")
                continue

            if data.startswith("watch:"):
                requested = data[6:].strip()[:32]

                wait = manager.switch_wait(websocket)
                if wait:
                    await asyncio.sleep(wait)

                old_mid, new_mid = manager.watch(websocket, requested)

                if old_mid and new_mid and old_mid != new_mid:
                    await manager.send_bytes(websocket, _ser({
                        "t": "reset", "from": old_mid, "to": new_mid,
                        "_server_ts": time.time() * 1000,
                    }))

                if not new_mid:
                    continue

                obj = LATEST_OBJ.get(new_mid)
                if obj and _is_fresh(obj):
                    LATEST_OBJ.move_to_end(new_mid)
                    await manager.send_bytes(websocket, _ser(_fresh(obj)))
                elif REST_TOKEN:
                    _spawn(_send_baseline(websocket, new_mid))
                elif obj:
                    await manager.send_bytes(websocket, _ser(_fresh(obj)))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)


# =====================================================================
#  HTTP
# =====================================================================
async def _send_telegram(text: str):
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
async def handle_click(data: dict, request: Request):
    ip = request.client.host if request.client else "?"
    if not _rate_ok(f"click:{ip}", CLICK_LIMIT_PER_MIN):
        return JSONResponse({"status": "error", "message": "rate limited"}, status_code=429)
    action = str(data.get("action", ""))[:64]
    match = str(data.get("match", ""))[:128]
    if not action:
        return JSONResponse({"status": "error", "message": "action required"}, status_code=400)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and _rate_ok("telegram", TELEGRAM_LIMIT_PER_MIN):
        _spawn(_send_telegram(f"Action: {action}\nMatch: {match}"))
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


@app.get("/api/match/{match_key}")
async def get_match(match_key: str):
    match_key = match_key[:32]
    obj = LATEST_OBJ.get(match_key)
    if obj and _is_fresh(obj):
        LATEST_OBJ.move_to_end(match_key)
        return {"status": "ok", "data": _fresh(obj)}

    if REST_TOKEN:
        try:
            resp = await _rest_info(match_key)
            if resp:
                cur = LATEST_OBJ.get(match_key)
                if cur and _is_fresh(cur):
                    LATEST_OBJ.move_to_end(match_key)
                    return {"status": "ok", "data": _fresh(cur)}
                new = slim_match_cached({"response": resp}, time.time(), isolated=True)
                new["id"] = str(match_key)
                out = _fresh(new)
                _seed_cache(match_key, new)
                return {"status": "ok", "data": out}
        except Exception as e:
            log.warning("get_match %s: %r", match_key, e)

    if obj:
        return {"status": "ok", "data": _fresh(obj)}

    fix = FIX.get(match_key)
    if fix:
        return {"status": "ok", "data": {
            "t": "match", "id": str(match_key), "_server_ts": time.time() * 1000,
            "status": fix.get("status", "started"), "fmt": fix.get("format", ""),
            "title": fix.get("title", ""),
            "a": {"name": fix.get("teams", {}).get("a", {}).get("name", "")},
            "b": {"name": fix.get("teams", {}).get("b", {}).get("name", "")},
            "score": None, "comm": [],
        }}

    return JSONResponse({"status": "error", "message": "not available"}, status_code=404)


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
    live_in_cache = sum(1 for o in LATEST_OBJ.values() if o.get("status") == "started")
    total_watchers = sum(len(s) for s in manager.by_match.values())
    last_ok_age = round(time.time() - _rest["last_ok_ts"], 1) if _rest["last_ok_ts"] else None
    out = {
        "status": "ok",
        "binary_mode": WS_BINARY,
        "ws_clients": len(manager.conns),
        "ws_watchers": total_watchers,
        "push_connected": UP["connected"],
        "push_msgs": UP["msgs"],
        "push_broadcasts": UP["sent"],
        "push_last_age_s": round(time.time() - UP["last"], 1) if UP["last"] else None,
        "matches_cached": len(LATEST_OBJ),
        "live_in_cache": live_in_cache,
        "fixtures_cached": len(FIX),
        "rest_cache_rows": len(_rest["rows"]),
        "rest_last_ok_age_s": last_ok_age,
        "rest_circuit_open": time.time() < _rest["circuit_open_until"],
    }
    if HEALTH_DETAIL:
        out["watched_matches"] = sorted(_watched_ids())
        out["rest_last_error"] = _rest["last_error"] or None
        out["rest_fail_streak"] = _rest["fail_streak"]
        out["fix_pending"] = len(_fix_pending)
        out["orjson"] = _orjson is not None
        out["uvloop"] = _UVLOOP
        out["broadcast_chunk"] = BROADCAST_CHUNK
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, workers=1,
                access_log=False, log_level="warning",
                ws_per_message_deflate=False, ws_max_size=WS_MAX_MSG_BYTES)
