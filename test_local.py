import json
import time
import urllib.request

BASE = "http://localhost:8080"
DATA = "expanded"

def post(path, obj):
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read())

def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, json.loads(r.read())

print("health", get("/v1/healthz"))
print("metadata", get("/v1/metadata"))

# Load the official expanded base dataset exactly as the judge does.
for path in ["expanded/categories/dentists.json", "expanded/categories/gyms.json", "expanded/categories/pharmacies.json", "expanded/categories/restaurants.json", "expanded/categories/salons.json"]:
    p = json.load(open(path))
    cid = p.get("slug")
    print(post("/v1/context", {"scope":"category","context_id":cid,"version":1,"payload":p,"delivered_at":"2026-04-26T09:45:00Z"}))

for fn in __import__('glob').glob("expanded/merchants/*.json"):
    p=json.load(open(fn)); print(post("/v1/context", {"scope":"merchant","context_id":p["merchant_id"],"version":1,"payload":p,"delivered_at":"2026-04-26T09:45:00Z"})[0], end=" ")
print()
for fn in __import__('glob').glob("expanded/customers/*.json"):
    p=json.load(open(fn)); post("/v1/context", {"scope":"customer","context_id":p["customer_id"],"version":1,"payload":p,"delivered_at":"2026-04-26T09:45:00Z"})

pairs=json.load(open("expanded/test_pairs.json"))["pairs"]
# Push and tick the first 30 official scenarios, one at a time.
for pair in pairs:
    fn=[x for x in __import__('glob').glob("expanded/triggers/*.json") if json.load(open(x)).get("id")==pair["trigger_id"]][0]
    trg=json.load(open(fn))
    status,_=post("/v1/context", {"scope":"trigger","context_id":trg["id"],"version":1,"payload":trg,"delivered_at":"2026-04-26T10:00:00Z"})
    status,res=post("/v1/tick", {"now":"2026-04-26T10:35:00Z","available_triggers":[trg["id"]]})
    print(pair["test_id"], status, len(res.get("actions",[])), res.get("actions", [{}])[0].get("body", "")[:160])

print("final health", get("/v1/healthz"))
