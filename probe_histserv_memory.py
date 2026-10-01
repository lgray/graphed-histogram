"""histserv 0.2.1 server memory, measured on the server process (Linux RSS), and the
sizing model's constants fitted to it (the MODEL line). Peak = the kernel's high-water mark (VmHWM), reset
between phases by writing 5 to /proc/<pid>/clear_refs.

Run: probes/m69b/run_memory_probe.sh [amd64]  (python:3.12-slim container, histserv==0.2.1 from PyPI)
Each scenario starts a fresh `python -m histserv`. "above warm" = peak RSS minus the warm idle server's RSS.
The model a server is packed against:
    peak <= B + sum_h (O + I*tasks + (chunks_h + 1)*dense_h) + (a + b*workers)*M
B the warm server plus the condor job's announce.py, O per histogram, I per recorded unique_id, dense_h one
chunk's dense flow view, M the largest histogram's stored bytes (= its FillMany message); a covers one fill's or
one snapshot's transient plus the heap it leaves behind, b each further fill in flight.
`python probe_histserv_memory.py connections` measures K alone, the server memory per open client connection (one
per worker process in a run, each channel on its own subchannel pool as a separate process would be).
"""

from __future__ import annotations

import math
import re
import socket
import subprocess
import sys
import threading
import time

import boost_histogram as bh
import grpc
import hist
import numpy as np
import psutil
from histserv.chunked_hist import ChunkedHist
from histserv.client import Client
from histserv.protos import hist_pb2
from histserv.serialize import serialize_chunk_payload, serialize_unique_id

MiB = 1 << 20
TIMEOUT = 120
NEED: list[tuple[str, int, float]] = []  # (scenario, fills in flight, (peak above warm - static) / M)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Server:
    def __init__(self, *extra: str) -> None:
        self.port = free_port()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "histserv", "--port", str(self.port), "--log-level", "ERROR",
             "--stats-interval-seconds", "3600", *extra]
        )
        deadline = time.monotonic() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        self.ps = psutil.Process(self.proc.pid)
        self.client = Client(f"127.0.0.1:{self.port}")
        warm(self)
        self.warm = self.reset_peak()

    def status(self, field: str) -> int:
        with open(f"/proc/{self.proc.pid}/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) * 1024
        raise KeyError(field)

    def settle(self) -> int:
        time.sleep(0.6)
        return self.status("VmRSS")

    def reset_peak(self) -> int:
        now = self.settle()
        with open(f"/proc/{self.proc.pid}/clear_refs", "w") as f:
            f.write("5")
        return now

    def above(self) -> int:
        return self.status("VmHWM") - self.warm

    def close(self) -> None:
        self.proc.terminate()
        self.proc.wait()


def dense_hist(nbytes: int, storage: object, labels: tuple[str, ...] = ()) -> ChunkedHist:
    """An EMPTY ChunkedHist (no chunks) whose dense flow view is `nbytes`; `labels` = a variation chunk axis."""
    item = 16 if isinstance(storage, bh.storage.Weight) else 8
    axes = [hist.axis.Regular(nbytes // item - 2, 0.0, 1.0, name="a0")]
    if labels:
        axes.append(hist.axis.StrCategory(list(labels), name="variation"))
    return ChunkedHist(*axes, storage=storage)


def ones_view(remote) -> np.ndarray:
    t = remote._template
    v = np.zeros(t.dense_view_shape, dtype=t.dense_view_dtype)
    if v.dtype.fields:
        for f in v.dtype.names:
            v[f] = 1
    else:
        v[...] = 1
    return v


def fill_many(remote, uid: str) -> hist_pb2.FillManyRequest:
    """The request m69b sends: every chunk of one partition's partial in one FillMany, one unique_id."""
    t = remote._template
    view = ones_view(remote)
    keys = [(k,) for k in t.chunk_axes[0].known_keys] if t.chunk_axes else [()]
    chunks = [serialize_chunk_payload(k, view, shape=t.dense_view_shape, dtype=t.dense_view_dtype) for k in keys]
    req = hist_pb2.FillManyRequest(hist_id=remote.hist_id, chunks=chunks)
    req.unique_id = serialize_unique_id(uid)
    return req


def warm(srv: Server) -> None:
    r = srv.client.init(dense_hist(64, bh.storage.Double()))
    srv.client.stub.FillMany(fill_many(r, "w"), timeout=TIMEOUT)
    r.snapshot(delete_from_server=True)


def need(name: str, in_flight: int, above: int, static: int, m: int) -> None:
    NEED.append((name, in_flight, (above - static) / m))
    print(f"  {name}: peak above warm {above / MiB:.1f} MiB, static {static / MiB:.1f} MiB, M {m / MiB:.0f} MiB"
          f" -> (peak - static)/M = {(above - static) / m:.2f} with {in_flight} fill(s) in flight")


def scenario_baseline() -> int:
    srv = Server()
    print(f"A warm idle server: rss={srv.warm} ({srv.warm / MiB:.1f} MiB)")
    srv.close()
    return srv.warm


def scenario_overhead() -> float:
    worst = 0.0
    for storage in (bh.storage.Double(), bh.storage.Weight()):
        srv = Server()
        k = 2000
        remotes = [srv.client.init(dense_hist(64, storage)) for _ in range(k)]
        for i, r in enumerate(remotes):
            srv.client.stub.FillMany(fill_many(r, f"p{i}"), timeout=TIMEOUT)
        per = (srv.settle() - srv.warm) / k - 2 * 64  # less the histogram's own (chunk + scratch) dense bytes
        worst = max(worst, per)
        print(f"B per histogram beyond its dense bytes ({type(storage).__name__}, K={k}, one fill each): {per:.0f}")
        srv.close()
    return worst


def scenario_unique_ids() -> float:
    srv = Server()
    r = srv.client.init(dense_hist(64, bh.storage.Double()))
    srv.client.stub.FillMany(fill_many(r, "first"), timeout=TIMEOUT)
    base = srv.settle()
    n = 40000
    for i in range(n):
        srv.client.stub.FillMany(fill_many(r, f"Partition(uri='f{i}.root', tree='Events')"), timeout=TIMEOUT)
    per = (srv.settle() - base) / n
    print(f"D per recorded unique_id ({n} fills of one histogram): {per:.1f}")
    srv.close()
    return per


def scenario_fills_and_snapshot() -> None:
    print("C/F one worker: two FillMany of one histogram, then a snapshot (kept, then deleted)")
    for labels, dense in (((), 64 * MiB), (tuple(f"l{i}" for i in range(4)), 16 * MiB), (tuple(f"l{i}" for i in range(8)), 8 * MiB)):
        srv = Server()
        r = srv.client.init(dense_hist(dense, bh.storage.Double(), labels))
        chunks = max(1, len(labels))
        static, m = (chunks + 1) * dense, chunks * dense
        srv.client.stub.FillMany(fill_many(r, "a"), timeout=TIMEOUT)
        srv.client.stub.FillMany(fill_many(r, "b"), timeout=TIMEOUT)
        need(f"fills chunks={chunks}", 1, srv.above(), static, m)
        r.snapshot(delete_from_server=False)
        need(f"snapshot chunks={chunks}", 1, srv.above(), static, m)
        r.snapshot(delete_from_server=True)
        need(f"snapshot+delete chunks={chunks}", 1, srv.above(), static, m)
        srv.close()


def scenario_concurrent() -> None:
    print("E n workers: n FillMany of one 32 MiB histogram in flight at once (worst of 3)")
    dense = 32 * MiB
    for n in (1, 2, 4, 8):
        worst = 0
        for _rep in range(3):
            srv = Server()
            r = srv.client.init(dense_hist(dense, bh.storage.Double()))
            srv.client.stub.FillMany(fill_many(r, "seed"), timeout=TIMEOUT)
            reqs = [fill_many(r, f"c{i}") for i in range(n)]
            clients = [Client(f"127.0.0.1:{srv.port}") for _ in range(n)]
            for c in clients:
                grpc.channel_ready_future(c.channel).result(timeout=10)
            barrier = threading.Barrier(n)

            def send(i: int) -> None:
                barrier.wait()
                clients[i].stub.FillMany(reqs[i], timeout=TIMEOUT)

            ts = [threading.Thread(target=send, args=(i,)) for i in range(n)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            worst = max(worst, srv.above())
            srv.close()
        need(f"{n} in flight", n, worst, 2 * dense, dense)


def scenario_ceiling() -> None:
    srv = Server()
    r = srv.client.init(dense_hist(64, bh.storage.Double()))
    for size in ((1 << 29) - 1024, 1 << 29):
        req = hist_pb2.FillRequest(hist_id=r.hist_id, dense_view=b"\0" * size)
        req.unique_id = serialize_unique_id(f"big{size}")
        try:
            srv.client.stub.Fill(req, timeout=TIMEOUT)
            print(f"H ceiling: dense_view {size} B (message {req.ByteSize()} B): accepted")
        except grpc.RpcError as e:
            print(f"H ceiling: dense_view {size} B (message {req.ByteSize()} B): {e.code().name} {e.details()[:80]}")
    srv.close()


def scenario_prune() -> None:
    srv = Server("--prune-after-seconds", "2", "--prune-interval-seconds", "0.5")
    idle = srv.client.init(dense_hist(64, bh.storage.Double()))
    busy = srv.client.init(dense_hist(64, bh.storage.Double()))
    for i in range(8):  # busy is touched every 0.5 s for 4 s; idle is not
        srv.client.stub.FillMany(fill_many(busy, f"t{i}"), timeout=TIMEOUT)
        time.sleep(0.5)
    print(f"I prune after 2 s: untouched histogram exists={idle.exists()}, one filled every 0.5 s exists={busy.exists()}")
    srv.close()
    out = subprocess.run([sys.executable, "-m", "histserv", "--help"], capture_output=True, text=True).stdout
    m = re.search(r"--prune-after-seconds PRUNE_AFTER_SECONDS (.*?\(default: [^)]*\))", " ".join(out.split()))
    print(f"I `python -m histserv --help`: --prune-after-seconds {m.group(1) if m else '?'}")


def scenario_wrapper() -> int:
    """The condor ServiceJob's announce.py beside the server (executors db8fb0a), imported and idle."""
    code = (
        "import importlib.util as u, time; s = u.spec_from_file_location('announce', '/p/announce_db8fb0a.py');"
        " m = u.module_from_spec(s); s.loader.exec_module(m); print('ready', flush=True); time.sleep(3)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    proc.stdout.readline()
    time.sleep(0.5)
    rss = psutil.Process(proc.pid).memory_info().rss
    proc.wait()
    print(f"J announce.py (db8fb0a) imported, idle: rss={rss} ({rss / MiB:.1f} MiB)")
    return rss


def scenario_connections() -> None:
    from histserv.protos import hist_pb2_grpc

    print("K n client connections, each its own channel and subchannel pool, one Stats RPC each, held open")
    worst = 0.0
    for n in (64, 256, 512):
        srv = Server()
        opts = [("grpc.use_local_subchannel_pool", 1)]
        channels = []
        try:
            for _ in range(n):
                ch = grpc.insecure_channel(f"127.0.0.1:{srv.port}", options=opts)
                hist_pb2_grpc.HistogrammerServiceStub(ch).Stats(hist_pb2.StatsRequest(), timeout=TIMEOUT)
                channels.append(ch)
            srv.settle()
            above = srv.above()
            worst = max(worst, above / n)
            print(f"  {n} connections: peak above warm {above / MiB:.1f} MiB -> {above / n / 1024:.1f} KiB per connection")
        finally:
            for ch in channels:
                ch.close()
            srv.close()
    print(f"MODEL K={up(worst / 1024, 1):.0f} KiB  (K covers every row above by construction)")


def up(x: float, step: float) -> float:
    return math.ceil(x / step) * step


if __name__ == "__main__":
    from importlib.metadata import version

    print(f"histserv {version('histserv')} grpcio {version('grpcio')} python {sys.version.split()[0]} {sys.platform}")
    if sys.argv[1:] == ["connections"]:
        scenario_connections()
        sys.exit(0)
    base = scenario_baseline()
    per_hist = scenario_overhead()
    per_id = scenario_unique_ids()
    scenario_fills_and_snapshot()
    scenario_concurrent()
    scenario_ceiling()
    scenario_prune()
    wrapper = scenario_wrapper()
    one = max(v for name, w, v in NEED if w == 1 and "in flight" in name)
    b = max((v - one) / (w - 1) for name, w, v in NEED if w > 1)
    a = max(v - b * w for _name, w, v in NEED)
    print(
        f"MODEL B={up((base + wrapper) / MiB, 1):.0f} MiB O={up(per_hist, 100):.0f} B I={up(per_id, 10):.0f} B"
        f" a={up(a, 0.5):.1f} b={up(b, 0.5):.1f}  (B = A + J; a, b cover every C/E/F row above by construction)"
    )
