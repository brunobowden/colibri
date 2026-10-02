#!/usr/bin/env python3
"""Registration and discovery control plane for local expert workers."""

import argparse
import json
import os
import re
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import openai_server


PROTOCOL_VERSION = 1

# A worker's host ends up in CLUSTER_WORKERS as "host:port", comma-separated,
# and the engine splits that list on ',' and the entry on its last ':'. So a
# host is a DNS name or an IPv4/IPv6 literal (IPv6 unbracketed, optionally
# with a %zone): no separators, no whitespace.
_HOST_RE = re.compile(r"[A-Za-z0-9._:%-]{1,253}")
_MAX_NODE_ID = 256


class ClusterRegistry:
    def __init__(self, stale_after=30.0):
        self.stale_after = float(stale_after)
        self._nodes = {}
        self._lock = threading.Lock()

    def register(self, node):
        required = {"node_id", "host", "port", "role"}
        missing = sorted(required - set(node))
        if missing:
            raise ValueError("missing node fields: " + ", ".join(missing))
        node_id, host = node["node_id"], node["host"]
        if not isinstance(node_id, str) or not 0 < len(node_id) <= _MAX_NODE_ID:
            raise ValueError(f"node_id must be a string of 1 to {_MAX_NODE_ID} characters")
        if not isinstance(host, str) or not _HOST_RE.fullmatch(host):
            raise ValueError("host must be a hostname or an IP address")
        port = int(node["port"])
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        role = str(node["role"])
        if role not in ("expert", "dense", "coordinator"):
            raise ValueError("role must be expert, dense, or coordinator")
        now = time.time()
        record = dict(node)
        record.update(protocol_version=PROTOCOL_VERSION, port=port, last_seen=now)
        with self._lock:
            self._prune(now)
            self._nodes[node_id] = record
        return record

    def _prune(self, now):
        """Forget nodes that missed their heartbeats. snapshot() already hid them;
        keeping the records only grew the map with every worker that ever came and
        went. A pruned worker's next heartbeat is a 404 and it registers again
        (heartbeat_or_register), which is also how workers recover from a
        coordinator restart. Caller holds the lock."""
        stale = [key for key, node in self._nodes.items()
                 if now - node["last_seen"] > self.stale_after]
        for key in stale:
            del self._nodes[key]

    def heartbeat(self, node_id):
        now = time.time()
        with self._lock:
            self._prune(now)
            node = self._nodes.get(str(node_id))
            if node is None:
                raise KeyError(node_id)
            node["last_seen"] = now
            return dict(node)

    def snapshot(self):
        now = time.time()
        with self._lock:
            self._prune(now)
            nodes = [dict(node) for node in self._nodes.values()]
        nodes.sort(key=lambda node: (node["role"], node["node_id"]))
        return {"protocol_version": PROTOCOL_VERSION, "nodes": nodes}

    def expert_endpoints(self):
        return [f"{node['host']}:{node['port']}"
                for node in self.snapshot()["nodes"] if node["role"] == "expert"]


class _Handler(openai_server.APIHandler):
    # The original registry served HTTP/1.0: one request per connection, no
    # keep-alive. Keep that wire shape; the shared APIHandler machinery (Host
    # guard, body cap, deadline reader, connection bookkeeping) still applies.
    protocol_version = "HTTP/1.0"
    server_version = "colibri-cluster/1"

    def log_message(self, *_args):
        return

    def _fail(self, error):
        self.send_json(error.status, {"error": error.message})

    def do_GET(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
        except openai_server.APIError as error:
            self._fail(error)
            return
        if self.path in ("/health", "/v1/cluster/topology"):
            self.send_json(200, self.server.registry.snapshot())
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802 - stdlib handler API
        try:
            self._check_host()
            body = self.read_json()
            if self.path == "/v1/cluster/register":
                self.send_json(200, self.server.registry.register(body))
            elif self.path == "/v1/cluster/heartbeat":
                if "node_id" not in body:
                    raise ValueError("missing node fields: node_id")
                try:
                    node = self.server.registry.heartbeat(body["node_id"])
                except KeyError:
                    # Not a malformed request: this coordinator does not know the
                    # node (it restarted, or the node went stale). 404 tells the
                    # worker to register again; a 400 left it heartbeating forever.
                    self.send_json(404, {"error": "unknown node; register again"})
                    return
                self.send_json(200, node)
            else:
                self.send_json(404, {"error": "not found"})
        except openai_server.APIError as error:
            self._fail(error)
        except (KeyError, ValueError, TypeError) as error:
            self.send_json(400, {"error": str(error)})


class ClusterServer(openai_server.APIServer):
    def __init__(self, address, registry, allowed_hosts=()):
        ThreadingHTTPServer.__init__(self, address, _Handler)
        self.registry = registry
        # Shared APIHandler plumbing reads these off the server. The registry has
        # no API key or CORS surface, and its Host guard stays loopback + bind
        # address (the same default openai_server enforces). A cross-host worker
        # registers from a LAN IP, so the operator opts those hosts in via
        # --allowed-host (the same #597 escape hatch coli serve exposes); the
        # default stays loopback + bind address.
        self.cors_origins = ()
        self.allowed_hosts = tuple(
            h.strip().lower() for h in allowed_hosts if h and h.strip())
        # Connection-cap bookkeeping for the inherited APIServer.process_request /
        # _release / close_request. There is no engine or model to carry, so
        # initialise the tracking state directly rather than APIServer.__init__.
        self._conn_lock = threading.Lock()
        self._conn_live = 0
        self._conn_by_ip = {}
        self._conn_owner = {}


def _endpoint(coordinator, path):
    return coordinator.rstrip("/") + "/v1/cluster/" + path


def serve(host="127.0.0.1", port=8765, stale_after=30.0, allowed_hosts=()):
    if allowed_hosts and "*" in allowed_hosts:
        print("WARNING: --allowed-host '*' accepts ANY Host header "
              "(DNS-rebinding guard disabled)", file=sys.stderr)
    server = ClusterServer((host, port), ClusterRegistry(stale_after), allowed_hosts)
    print(f"colibri cluster coordinator listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def register(coordinator, node):
    request = Request(_endpoint(coordinator, "register"),
                      data=json.dumps(node).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def heartbeat(coordinator, node_id):
    request = Request(_endpoint(coordinator, "heartbeat"),
                      data=json.dumps({"node_id": node_id}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        return json.load(response)


def heartbeat_or_register(coordinator, node):
    """One keep-alive tick for a worker: heartbeat, or register again when the
    coordinator answers 404 (it restarted, or dropped the node as stale)."""
    try:
        return heartbeat(coordinator, node["node_id"])
    except HTTPError as error:
        if error.code != 404:
            raise
        error.close()
    return register(coordinator, node)


def discover_workers(coordinator):
    with urlopen(_endpoint(coordinator, "topology"), timeout=5) as response:
        topology = json.load(response)
    return [f"{node['host']}:{int(node['port'])}"
            for node in topology.get("nodes", []) if node.get("role") == "expert"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stale-after", type=float, default=30.0)
    parser.add_argument("--allowed-host", action="append",
                        default=[h.strip() for h in os.environ.get("COLI_ALLOWED_HOSTS", "").split(",") if h.strip()],
                        help="additional Host header accepted by the DNS-rebinding guard; repeat as needed")
    args = parser.parse_args()
    serve(args.host, args.port, args.stale_after, args.allowed_host)


if __name__ == "__main__":
    main()
