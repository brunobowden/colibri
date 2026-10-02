import json
import sys
import threading
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cluster import (ClusterRegistry, ClusterServer, PROTOCOL_VERSION,
                     heartbeat_or_register, register)


class ClusterRegistryTests(unittest.TestCase):
    def test_registers_and_discovers_expert_nodes(self):
        registry = ClusterRegistry()
        registry.register({"node_id": "mac-a", "host": "10.0.0.2", "port": 9100,
                           "role": "expert", "layers": "all"})
        registry.register({"node_id": "mac-b", "host": "10.0.0.3", "port": 9101,
                           "role": "dense", "layers": "38-75"})
        self.assertEqual(registry.expert_endpoints(), ["10.0.0.2:9100"])
        self.assertEqual(registry.snapshot()["protocol_version"], PROTOCOL_VERSION)

    def test_stale_nodes_are_forgotten_not_just_hidden(self):
        registry = ClusterRegistry(stale_after=30)
        clock = [1000.0]
        with patch("cluster.time.time", lambda: clock[0]):
            for index in range(50):        # workers that came and went
                registry.register({"node_id": f"gone-{index}", "host": "10.0.0.9",
                                   "port": 9100 + index, "role": "expert"})
            clock[0] += 31
            registry.register({"node_id": "live", "host": "10.0.0.2", "port": 9100,
                               "role": "expert"})
            self.assertEqual(registry.expert_endpoints(), ["10.0.0.2:9100"])
            self.assertEqual(list(registry._nodes), ["live"])
            with self.assertRaises(KeyError):
                registry.heartbeat("gone-0")

    def test_register_refuses_hosts_that_would_corrupt_the_worker_list(self):
        """discover_workers() joins host:port with ',' into CLUSTER_WORKERS and the
        engine splits it back on ',' and the last ':'."""
        registry = ClusterRegistry()
        good = {"node_id": "n", "port": 9100, "role": "expert"}
        for host in ("10.0.0.2", "worker-3.lan", "fe80::1%eth0", "::1"):
            registry.register(dict(good, host=host))
        for host in ("", "10.0.0.2:9101,10.0.0.66", "a b", "evil\n", 7, None, ["x"]):
            with self.subTest(host=host), self.assertRaises(ValueError):
                registry.register(dict(good, host=host))
        for node_id in ("", 7, None, "x" * 257):
            with self.subTest(node_id=node_id), self.assertRaises(ValueError):
                registry.register(dict(good, host="10.0.0.2", node_id=node_id))

    def test_http_topology_registration_and_heartbeat(self):
        server = ClusterServer(("127.0.0.1", 0), ClusterRegistry())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = HTTPConnection(*server.server_address)
            body = json.dumps({"node_id": "mac-a", "host": "127.0.0.1",
                               "port": 9100, "role": "expert"})
            conn.request("POST", "/v1/cluster/register", body,
                         {"Content-Type": "application/json"})
            self.assertEqual(conn.getresponse().status, 200)
            conn.request("POST", "/v1/cluster/heartbeat", json.dumps({"node_id": "mac-a"}),
                         {"Content-Type": "application/json"})
            self.assertEqual(conn.getresponse().status, 200)
            conn.request("GET", "/v1/cluster/topology")
            response = conn.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(len(json.loads(response.read())["nodes"]), 1)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()

    def test_worker_registers_again_after_a_coordinator_restart(self):
        """A restarted coordinator does not know the worker. Its heartbeat used to be a
        400 that the worker loop swallowed (HTTPError is an OSError), so the worker was
        gone from the topology until someone restarted it too."""
        server = ClusterServer(("127.0.0.1", 0), ClusterRegistry())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            node = {"node_id": "mac-a", "host": "127.0.0.1", "port": 9100, "role": "expert"}
            register(base, node)
            server.registry = ClusterRegistry()            # the coordinator restarted
            conn = HTTPConnection(*server.server_address)
            conn.request("POST", "/v1/cluster/heartbeat", json.dumps({"node_id": "mac-a"}),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            response.read()
            conn.close()
            self.assertEqual(response.status, 404)
            self.assertEqual(server.registry.expert_endpoints(), [])
            heartbeat_or_register(base, node)               # the worker's next tick
            self.assertEqual(server.registry.expert_endpoints(), ["127.0.0.1:9100"])
            heartbeat_or_register(base, node)               # and plain heartbeats after
            self.assertEqual(server.registry.expert_endpoints(), ["127.0.0.1:9100"])
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
