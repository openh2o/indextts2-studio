"""Regression tests for the /tts job lifecycle endpoints (server.py).

These cover the three production bugs found in the job machinery:
  1. worker deadlock: run_infer used to call record_event() while holding
     _JOBS_LOCK, and record_event() re-acquires that same non-reentrant lock.
  2. /tts/{id}/cancel and /stop returned the stale state "queued" after
     flipping a queued job to "cancelled".
  3. a model-load failure inside the worker left the SSE stream hanging
     forever (no terminal event, job stuck in "running").

No GPU / httpx needed: server.py is imported with argv patched, the CWD
moved to a tmp dir, and the endpoint functions called directly. Inference
itself is monkeypatched out; only the job bookkeeping and SSE framing are
under test.

Run: uv run --extra test pytest tests/test_server_jobs.py -v
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# server.py parses argv at import time and side-effects the CWD
# (prompts/outputs dirs, legacy-prompt cleanup). Import it once, isolated
# inside a tmp working directory.
@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("server_root")
    for d in ("prompts", "outputs"):
        (tmp / d).mkdir(exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(tmp)
    real_argv = sys.argv
    sys.argv = ["server.py", "--port", "7860"]
    try:
        import server as server_mod
        yield server_mod
    finally:
        sys.argv = real_argv
        os.chdir(prev_cwd)
        if "server" in sys.modules:
            sys.modules["server"]._JOBS.clear()


# ---------- _register_job / _job_public / _prune_finished_jobs ----------

def test_register_job_and_public_view(server):
    job_id, ahead = server._register_job("client-a")
    assert ahead == 0
    with server._JOBS_LOCK:
        job = server._JOBS[job_id]
        data = server._job_public(job)
    assert data["state"] == "queued"
    assert data["client_id"] == "client-a"
    assert "stop_event" not in data and "events" not in data
    server._JOBS.clear()


def test_prune_finished_jobs_drops_expired(server):
    server._JOBS.clear()
    old_id, _ = server._register_job("old")
    with server._JOBS_LOCK:
        server._JOBS[old_id]["state"] = "done"
        server._JOBS[old_id]["finished_at"] = time.time() - (server.JOB_RETENTION_SECONDS + 5)
    new_id, _ = server._register_job("new")  # _register_job prunes
    assert old_id not in server._JOBS
    assert new_id in server._JOBS
    server._JOBS.clear()


# ---------- /tts/{id}/cancel and /tts/{id}/stop state transitions ----------

def test_cancel_queued_job_returns_cancelled(server):
    server._JOBS.clear()
    job_id, _ = server._register_job("client-a")
    resp = server.tts_cancel(job_id)
    assert resp == {"ok": True, "state": "cancelled"}, \
        "queued job must report the NEW state, not the pre-transition 'queued'"
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "cancelled"
    server._JOBS.clear()


def test_stop_queued_job_returns_cancelled(server):
    server._JOBS.clear()
    job_id, _ = server._register_job("client-a")
    resp = server.tts_stop(job_id)
    assert resp == {"ok": True, "state": "cancelled"}
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "cancelled"
        # the stop flag must NOT be set for a queued job - the worker
        # short-circuits on the state alone
        assert not server._JOBS[job_id]["stop_event"].is_set()
    server._JOBS.clear()


def test_stop_running_job_requests_stop(server):
    server._JOBS.clear()
    job_id, _ = server._register_job("client-a")
    with server._JOBS_LOCK:
        server._JOBS[job_id]["state"] = "running"
    resp = server.tts_stop(job_id)
    assert resp == {"ok": True, "state": "stop_requested"}
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "stop_requested"
        assert server._JOBS[job_id]["stop_event"].is_set()
    server._JOBS.clear()


def test_stop_unknown_job_404(server):
    from fastapi import HTTPException
    server._JOBS.clear()
    with pytest.raises(HTTPException) as ei:
        server.tts_stop("nosuchjob")
    assert ei.value.status_code == 404


# ---------- worker: no deadlock on queued-cancel, SSE terminates ----------

class _SSEConsumer(threading.Thread):
    """Drain the raw SSE generator in a thread with a timeout.

    Uses the sync generator captured via the StreamingResponse shim below,
    avoiding starlette's async body_iterator wrapper. If the worker never
    emits a terminal event (the regression under test), the generator blocks
    on progress_q.get() forever and the thread stays alive -> join times out
    -> wait_for_terminal returns None -> the test fails with a clear message.
    """

    def __init__(self, response):
        super().__init__(daemon=True)
        self.response = response
        self.events = []

    def run(self):
        try:
            for chunk in self.response.body_iterator:
                for line in chunk.splitlines():
                    if line.startswith("data:"):
                        self.events.append(json.loads(line[5:].strip()))
                if self.events and self.events[-1]["type"] in ("done", "error"):
                    break
        except Exception as e:  # pragma: no cover - surfaced via events list
            self.events.append({"type": "error", "detail": f"consumer crashed: {e}"})

    def wait_for_terminal(self, timeout=5.0):
        self.join(timeout)
        if self.is_alive():
            return None
        return self.events


def _sse_events_from(response, timeout=5.0):
    c = _SSEConsumer(response)
    c.start()
    events = c.wait_for_terminal(timeout)
    assert events is not None, "SSE stream did not terminate within timeout - hung forever"
    return events


def _start_tts(server, monkeypatch, tmp_path, client_id="client-a", infer_impl=None):
    """Call do_tts() with model inference stubbed; returns (job_id, response)."""
    out_file = tmp_path / "out.wav"
    out_file.write_bytes(b"RIFF....WAVEfmt ")
    out_name = str(out_file)

    if infer_impl is not None:
        class _StubTTS:
            gr_progress = None
            def ensure_loaded(self):
                return self
            def infer(self, **kwargs):
                return infer_impl(kwargs, out_name)
        monkeypatch.setattr(server.tts, "ensure_loaded", lambda: _StubTTS())
    monkeypatch.setattr(server, "_output_name", lambda *a, **k: out_name)
    # _history_add is called on success; keep it from touching real outputs/
    monkeypatch.setattr(server, "_history_add", lambda *a, **k: {"id": "test"})
    # Capture the raw sync SSE generator instead of letting StreamingResponse
    # wrap it into an async body_iterator (needs a running event loop).
    class _RawResponse:
        def __init__(self, gen, media_type=None, headers=None):
            self.body_iterator = gen
    monkeypatch.setattr(server, "StreamingResponse", _RawResponse)

    class _File:
        def __init__(self, payload):
            self.file = self
            self._payload = payload
            self.read = lambda: payload
    spk = _File(b"spk-bytes")
    # Direct call (no TestClient/httpx): FastAPI does not resolve Form() or
    # File(None) defaults here, so every defaulted parameter must be passed.
    response = server.do_tts(
        text="你好", client_id=client_id, spk_audio=spk,
        emo_mode="0", emo_audio=None, emo_weight=0.65,
        vec1=0.0, vec2=0.0, vec3=0.0, vec4=0.0,
        vec5=0.0, vec6=0.0, vec7=0.0, vec8=0.0,
        emo_text="", use_random=False,
        max_text_tokens_per_segment=120, do_sample=True,
        top_p=0.8, top_k=30, temperature=0.8,
        length_penalty=0.0, num_beams=3, repetition_penalty=10.0,
        max_mel_tokens=1500, file_naming="title_time",
    )
    # job_id is generated inside do_tts; recover it from the registry
    with server._JOBS_LOCK:
        job_id = max(
            (jid for jid, j in server._JOBS.items() if j.get("client_id") == client_id),
            key=lambda jid: server._JOBS[jid]["created_at"],
        )
    return job_id, response


def test_queued_cancel_short_circuits_without_deadlock(server, monkeypatch, tmp_path):
    """The original bug: worker called record_event() while holding _JOBS_LOCK
    -> deadlock + _INFER_LOCK held forever. The queue-wait loop must notice the
    cancel and emit a terminal error event instead."""
    server._JOBS.clear()

    # Hold _INFER_LOCK so do_tts's worker parks in the queue-wait loop
    assert server._INFER_LOCK.acquire(timeout=1)
    try:
        job_id, response = _start_tts(server, monkeypatch, tmp_path)
        # while queued, cancel it
        resp = server.tts_cancel(job_id)
        assert resp["state"] == "cancelled"
    finally:
        server._INFER_LOCK.release()

    events = _sse_events_from(response, timeout=5.0)
    assert events[-1]["type"] == "error", f"expected terminal error, got {events}"
    assert "已取消" in events[-1]["detail"]
    # the lock must be free again (worker exited without holding it)
    assert server._INFER_LOCK.acquire(timeout=1), "worker left _INFER_LOCK held - deadlock"
    server._INFER_LOCK.release()
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "error"
    server._JOBS.clear()


def test_model_load_failure_emits_terminal_error(server, monkeypatch, tmp_path):
    """The original bug: ensure_loaded() ran outside the try block, so a model
    load failure killed the worker silently - SSE hung, job stuck 'running'."""
    server._JOBS.clear()

    def boom():
        raise RuntimeError("simulated model load failure")
    monkeypatch.setattr(server.tts, "ensure_loaded", boom)

    job_id, response = _start_tts(server, monkeypatch, tmp_path)
    events = _sse_events_from(response, timeout=5.0)
    assert events[-1]["type"] == "error"
    assert "simulated model load failure" in events[-1]["detail"]
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "error"
        assert server._JOBS[job_id]["finished_at"] is not None
    server._JOBS.clear()


def test_stop_while_queued_after_lock_wait(server, monkeypatch, tmp_path):
    """Stop (not cancel) on a queued job while the GPU is busy: the queue-wait
    loop must honor stop_requested too and short-circuit."""
    server._JOBS.clear()

    assert server._INFER_LOCK.acquire(timeout=1)
    try:
        job_id, response = _start_tts(server, monkeypatch, tmp_path)
        resp = server.tts_stop(job_id)
        assert resp["state"] == "cancelled"
    finally:
        server._INFER_LOCK.release()

    events = _sse_events_from(response, timeout=5.0)
    assert events[-1]["type"] == "error"
    with server._JOBS_LOCK:
        assert server._JOBS[job_id]["state"] == "error"
    server._JOBS.clear()
