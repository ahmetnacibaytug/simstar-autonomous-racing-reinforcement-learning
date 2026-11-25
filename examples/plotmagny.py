# plot_magny.py
import os
import pickle
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(CURRENT_DIR, "MagnyCours.pkl")

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"PKL bulunamadı: {PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    waypoints = pickle.load(f)

print("Waypoint count:", len(waypoints))

if len(waypoints) == 0:
    raise RuntimeError("MagnyCours.pkl boş. Önce pkl dosyasını yeniden üretmen lazım.")

print("First waypoint type:", type(waypoints[0]))
print("First waypoint vars:", vars(waypoints[0]))

xs = []
ys = []

for wp in waypoints:
    if hasattr(wp, "X") and hasattr(wp, "Y"):
        xs.append(wp.X)
        ys.append(wp.Y)
    else:
        raise AttributeError("WayPoint içinde X/Y alanı bulunamadı.")

plt.figure(figsize=(8, 8))

plt.plot(xs, ys, "-", color="lightcoral", alpha=0.5, label="Path")
plt.scatter(xs, ys, s=12, color="red", label="Waypoints")

plt.scatter(xs[0], ys[0], s=130, color="green", label="Start", zorder=5)
plt.scatter(xs[-1], ys[-1], s=130, color="blue", label="End", zorder=5)

plt.title("Magny-Cours Map Visualization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.tight_layout()

OUT_PATH = os.path.join(CURRENT_DIR, "MagnyCours_Map_Visualization.png")
plt.savefig(OUT_PATH, dpi=300)

print("Saved:", OUT_PATH)