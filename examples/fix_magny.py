import sys, os

EX_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EX_DIR)
PYTHON_API_DIR = os.path.join(ROOT_DIR, "PythonAPI")

sys.path.insert(0, PYTHON_API_DIR)
sys.path.insert(0, EX_DIR)

import simstar
from simstar.road_network_generator import RoadNetworkGenerator

rng = RoadNetworkGenerator()

result = rng.api.query("""
    [out:xml][timeout:90];
    way(around:2500,46.8642,3.1631)["highway"="raceway"]
        ["name"!~"kart|Kart|pit|Pit",i]
        ["service"!="pit_lane"];
    out body;
    >;
    out skel qt;
""")

ways = sorted(result.ways, key=lambda w: len(w.nodes), reverse=True)

print("WAY COUNT:", len(ways))

for w in ways[:20]:
    print("=" * 60)
    print("Way ID:", w.id)
    print("Node count:", len(w.nodes))
    print("Tags:", w.tags)