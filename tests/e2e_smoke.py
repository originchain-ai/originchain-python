"""Self-contained live end-to-end smoke for the originchain SDK.

Unlike ``test_e2e_live.py`` (which assumes pre-existing schemas and skips
liberally), this script registers its own namespace, seeds data, and asserts
every feature — a real CI gate. Run it directly, not via pytest:

    OC_BASE=https://<tenant>.<region>.db.originchain.ai \\
    OC_BEARER=<bearer> OC_TENANT=<tenant-ulid> [OC_NS=shopdemo] \\
    python tests/e2e_smoke.py

Guarded: exits 0 without running when OC_BASE is unset (e.g. a fork PR without
secrets). Graph uses a self-relation FK (the real engine model): the edge runs
from a row's PK to the value in from_col.
"""
import os
import sys

from originchain import OriginChain

if not os.environ.get("OC_BASE"):
    print("OC_BASE unset — skipping live E2E (set OC_BASE/OC_BEARER/OC_TENANT to run).")
    sys.exit(0)

BASE = os.environ["OC_BASE"]
TOK = os.environ["OC_BEARER"]
T = os.environ["OC_TENANT"]
NS = os.environ.get("OC_NS", "shopdemo")
db = OriginChain(base_url=BASE, bearer=TOK, tenant=T)

results = []


def step(name, fn, expect_fail=False):
    try:
        v = fn()
        results.append((name, not expect_fail, str(v).replace("\n", " ")[:100]))
    except Exception as ex:  # noqa: BLE001 - a smoke step records any failure
        results.append((name, expect_fail, ("%s: %s" % (type(ex).__name__, ex))[:140]))


PRODUCTS_TOML = """version = 1
namespace = "%s"
table = "products"
primary_key = ["id"]
extractions = []
foreign_keys = []
check_constraints = []
triggers = []

[[columns]]
name = "id"
ty = "str"
required = true

[[columns]]
name = "name"
ty = "str"

[[columns]]
name = "price"
ty = "f64"

[[columns]]
name = "description"
ty = "str"

[[columns]]
name = "related_to"
ty = "str"

[[relations]]
name = "related"
from_col = "related_to"
bidirectional = true

[relations.target]
namespace = "%s"
table = "products"
pk = "id"
""" % (NS, NS)

VEC = [0.1, 0.2, 0.1, 0.0, 0.3, 0.1, 0.2, 0.0]
P = NS + ".products"


def check_neighbors():
    ns = db.graph.neighbors(P, rel="related", pk="p1")
    pks = [n.pk for n in ns]
    if pks != ["p2"]:
        raise AssertionError("expected ['p2'], got %r" % pks)
    return pks


step("health", lambda: db.health())
step("schemas.register(products)", lambda: db.schemas.register(PRODUCTS_TOML))
step("schemas.list", lambda: db.schemas.list())
step("rows.put p1", lambda: db.rows.put(P, {"id": "p1", "name": "Widget", "price": 9.99, "description": "a small blue widget for testing", "related_to": "p2"}))
step("rows.put p2", lambda: db.rows.put(P, {"id": "p2", "name": "Gadget", "price": 2.50, "description": "a tiny red gadget gizmo"}))
step("rows.get p1", lambda: db.rows.get(P, "p1"))
step("sql SELECT", lambda: db.sql("SELECT id, name, price FROM %s WHERE price > 5" % P))
step("sql_one COUNT", lambda: db.sql_one("SELECT COUNT(*) FROM %s" % P))
step("vector.put", lambda: db.vector.put(P, "p1", VEC))
step("vector.topk", lambda: db.vector.topk(P, VEC, k=5))
step("fts.index", lambda: db.fts.index(P, "description", "p1", "a small blue widget for testing"))
step("fts.search", lambda: db.fts.search(P, "description", "widget"))
step("graph.neighbors ->[p2]", check_neighbors)
step("ask", lambda: db.ask("how many products cost more than 5", schemas=[P]))
step("usage", lambda: db.usage())
step("sql SELECT 1 (want-fail)", lambda: db.sql("SELECT 1"), expect_fail=True)

print("\n=== PYTHON SDK %s E2E (ns=%s) ===" % (getattr(__import__("originchain"), "__version__", "?"), NS))
npass = 0
for name, ok, detail in results:
    print("%-4s %-28s %s" % ("PASS" if ok else "FAIL", name, detail))
    npass += ok
print("=== %d/%d passed ===" % (npass, len(results)))
sys.exit(0 if npass == len(results) else 1)
