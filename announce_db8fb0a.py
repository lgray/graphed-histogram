"""The job side of a cluster-hosted service: ``python3 announce.py service.json`` in the job's scratch dir.

Stdlib only and python >= 3.9, since a service's image need not carry our venv: the job runs this file by
path, and ``python -m graphed_executors.htcondor_backend.announce`` runs the same code. ``service.json``
holds ``argv``, ``env``, ``check``, ``ports``, ``key``, ``url``, ``watch``, ``python``, ``timeout_s``,
``lease_s`` and ``beat_s``; the announce secret is no field.

It reads the secret, starts the recipe's child in ``service/`` (the transferred directory holding exactly
the recipe's inputs) on the first free port of ``ports``, checks it where it runs, and posts ``key
host:port identity``, signed, to ``<url>/announce``. ``timeout_s`` bounds the whole start. A child that
exits moves on to the next port only when another process took its port; otherwise the start fails (exit
3). Attached (``url`` set): the secret file is read and unlinked before the child starts, the announce
repeats every ``beat_s``, and a 403 or no 200 for ``lease_s`` (counted from readiness) stops the service
and exits 0, as an orphaned pilot does. Watch mode (``watch`` a directory): ``<watch>/driver.url`` and
``<watch>/graphed-secret`` are read each second and each new pair is announced until it answers 200.
Otherwise the exit code is the child's; SIGTERM reaps the child and exits 143.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, NoReturn

SIG_HEADER = "X-Graphed-Sig"
RUN_DIR = "service"
SECRET_FILE = "graphed-secret"
URL_FILE = "driver.url"
REAP_S = 5.0
CHECK_S = 5.0

# the self-check dials this job's own port: an environment proxy must not sit in between
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class _Stop(BaseException):
    """Raised by the SIGTERM handler into the main thread, whose top level does the one reap."""


CHILD: list[subprocess.Popen[bytes] | None] = [None]  # the child being started or served


def log(msg: str) -> None:
    print(f"[announce {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def host_identity() -> str:
    """``Machine`` from ``$_CONDOR_MACHINE_AD``, else ``socket.getfqdn()`` (the engine's rule, copied)."""
    path = os.environ.get("_CONDOR_MACHINE_AD")
    if path and os.path.isfile(path):
        with open(path) as ad:
            for line in ad:
                name, _, value = line.partition("=")
                if name.strip() == "Machine":
                    return value.strip().strip('"')
    return socket.getfqdn()


def self_check(check: str, host: str, port: int, timeout: float) -> str | None:
    """``None`` when ready: ``http:<path>`` wants a 2xx whose content-type is not ``application/grpc*``;
    ``tcp`` and ``grpc:`` a connect."""
    try:
        if check.startswith("http:"):
            with _DIRECT.open(f"http://{host}:{port}{check[5:] or '/'}", timeout=timeout) as resp:
                ctype = str(resp.headers.get("Content-Type", ""))
                if not ctype.startswith("application/grpc"):
                    return None
                return f"answered {resp.status} {ctype}"
        socket.create_connection((host, port), timeout=timeout).close()
        return None
    except Exception as exc:  # a refusal, a non-2xx (HTTPError), a server not yet speaking HTTP
        return repr(exc)


def free(port: int) -> bool:
    """A bind with ``SO_REUSEADDR``: a listener refuses it, a ``TIME_WAIT`` the self-check left does not."""
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def post(url: str, secret: bytes, body: bytes) -> int | None:
    """The status of ``body`` posted, signed, to ``<url>/announce``; ``None`` when nothing answered."""
    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()
    request = urllib.request.Request(
        url.rstrip("/") + "/announce", data=body, method="POST", headers={SIG_HEADER: sig}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except OSError:
        return None


def reap(child: subprocess.Popen[bytes]) -> None:
    """The one bounded reap: terminate, wait at most ``REAP_S``, kill (neither signals a reaped child)."""
    child.terminate()
    try:
        child.wait(REAP_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def hard_reap(pid: int | None) -> None:
    """The reap after a SIGTERM, on the pid: ``Popen``'s waitpid lock may be held by the wait it
    interrupted. SIGTERM, ``waitpid(WNOHANG)`` for at most ``REAP_S``, then SIGKILL and a blocking wait."""
    if pid is None or sys.platform == "win32":  # after the None return: the reaped path runs everywhere
        return
    try:
        os.kill(pid, signal.SIGTERM)
        end = time.monotonic() + REAP_S
        while time.monotonic() < end:
            if os.waitpid(pid, os.WNOHANG) != (0, 0):
                return
            time.sleep(0.05)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
    except (ProcessLookupError, ChildProcessError):
        return


def on_sigterm(signum: int, frame: object) -> NoReturn:
    # never waits: the interrupted main thread may hold Popen's waitpid lock
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise _Stop


def unblock_sigterm() -> None:
    if sys.platform != "win32":
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def interpreter(name: str) -> str:
    """``{python}`` as the job dir names it: a path made absolute here (the child starts in
    ``service/``), a bare name looked up on ``PATH`` or ``os.defpath``, else left bare."""
    if os.sep in name:
        return os.path.abspath(name)
    return shutil.which(name, path=os.environ.get("PATH", os.defpath)) or name


def start(cfg: dict[str, Any], ident: str) -> tuple[subprocess.Popen[bytes], int] | str:
    """The ready child and its port, or why none is."""
    low, high = cfg["ports"]
    deadline = time.monotonic() + float(cfg["timeout_s"])
    subs = {"{python}": interpreter(cfg["python"]), "{host}": ident}
    for port in range(low, high + 1):
        if not free(port):
            log(f"port {port} taken, next")
            continue
        argv = []
        for arg in cfg["argv"]:
            for token, value in {**subs, "{port}": str(port)}.items():
                arg = arg.replace(token, value)
            argv.append(arg)
        # SIGTERM stays blocked until the child is recorded; the child unblocks it before exec
        if sys.platform != "win32":
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            child = subprocess.Popen(
                argv, env={**os.environ, **cfg["env"]}, cwd=RUN_DIR, preexec_fn=unblock_sigterm
            )
            CHILD[0] = child
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return f"cannot start {argv[0]}: {exc!r}"
        finally:
            unblock_sigterm()
        while True:
            if child.poll() is not None:
                if free(port):
                    return f"the child exited with returncode {child.returncode} on port {port}"
                log(f"port {port} taken after the scan (the child exited {child.returncode}), next")
                break
            why = self_check(cfg["check"], ident, port, max(0.1, min(CHECK_S, deadline - time.monotonic())))
            if why is None and child.poll() is None:
                return child, port
            if time.monotonic() >= deadline:
                reap(child)
                return f"not ready within timeout_s={cfg['timeout_s']}: {why}"
            time.sleep(0.5)
    return f"no free port in {low}-{high}"


def _watched(watch: str) -> tuple[str, bytes] | None:
    """``(url, secret)`` from the watch dir, or ``None`` while either is missing or unreadable."""
    try:
        with open(os.path.join(watch, URL_FILE)) as f:
            url = f.read().strip()
        with open(os.path.join(watch, SECRET_FILE)) as f:
            secret = bytes.fromhex(f.read().strip())
    except (OSError, ValueError):
        return None
    return (url, secret) if url else None


def serve() -> int:
    with open(sys.argv[1]) as f:
        cfg: dict[str, Any] = json.load(f)
    ident = host_identity()
    secret = b""
    if not cfg["watch"]:
        with open(SECRET_FILE) as f:
            secret = bytes.fromhex(f.read().strip())
        os.unlink(SECRET_FILE)
    if os.path.ismount(RUN_DIR):
        log(f"not ready: {RUN_DIR} is a mount point")
        return 3
    os.makedirs(RUN_DIR, exist_ok=True)  # a recipe without inputs transfers none
    started = start(cfg, ident)
    if isinstance(started, str):
        log(f"not ready: {started}")
        return 3
    child, port = started
    body = f"{cfg['key']} {ident}:{port} {ident}".encode()
    log(f"ready pid={child.pid} body={body.decode()!r}")
    last: tuple[str, bytes] | None = None
    last_ok, announced = time.monotonic(), False
    while child.poll() is None:
        if cfg["watch"]:
            pair = _watched(cfg["watch"])
            if pair is not None and pair != last:
                status = post(pair[0], pair[1], body)
                log(f"announce to {pair[0]} -> {status}")
                if status == 200:
                    last = pair
            time.sleep(1.0)
            continue
        status = post(cfg["url"], secret, body)
        if status == 200:
            if not announced:
                log(f"announced to {cfg['url']}")
            announced, last_ok = True, time.monotonic()
        elif status == 403 or time.monotonic() - last_ok > cfg["lease_s"]:
            log(f"orphaned ({status}, last 200 {time.monotonic() - last_ok:.1f}s ago): stopping the service")
            reap(child)
            return 0
        time.sleep(cfg["beat_s"] if announced else 1.0)
    log(f"the child exited {child.returncode}")
    return int(child.returncode)


def main() -> int:
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        return serve()
    except _Stop:
        # a pid Popen already reaped (returncode set) may be reused: it is not signalled
        child = CHILD[0]
        hard_reap(child.pid if child is not None and child.returncode is None else None)
        log("SIGTERM: reaped")
        sys.exit(143)  # through SystemExit, so coverage saves its data


if __name__ == "__main__":
    sys.exit(main())
