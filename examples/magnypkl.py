import os
import sys
import pickle

EX_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EX_DIR)
PYTHON_API_DIR = os.path.join(ROOT_DIR, "PythonAPI")

sys.path.insert(0, PYTHON_API_DIR)
sys.path.insert(0, EX_DIR)

import simstar
from simstar.road_network_generator import RoadNetworkGenerator

gen = RoadNetworkGenerator()
waypoints = gen.get_way_points(simstar.TrackName.MagnyCours)

print("Waypoint count:", len(waypoints))

if len(waypoints) == 0:
    raise RuntimeError("MagnyCours waypoint boş geldi.")

out_path = os.path.join(EX_DIR, "MagnyCours.pkl")

with open(out_path, "wb") as f:
    pickle.dump(waypoints, f)

print("Saved:", out_path)
print("First:", vars(waypoints[0]))
print("Last:", vars(waypoints[-1]))