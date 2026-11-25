# plot_istanbulpark.py
import os
import pickle
import matplotlib.pyplot as plt

# Bu dosya examples klasöründeyse IstanbulPark.pkl de genelde burada oluşur
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(CURRENT_DIR, "IstanbulPark.pkl")

if not os.path.exists(PKL_PATH):
    raise FileNotFoundError(f"PKL bulunamadı: {PKL_PATH}")

with open(PKL_PATH, "rb") as f:
    waypoints = pickle.load(f)

print("Waypoint count:", len(waypoints))
print("First waypoint type:", type(waypoints[0]))
print("First waypoint vars:", vars(waypoints[0]))

# WayPoint objelerinde x-y alanlarını çek
xs = []
ys = []

for wp in waypoints:
    if hasattr(wp, "X") and hasattr(wp, "Y"):
        xs.append(wp.X)
        ys.append(wp.Y)
    else:
        raise AttributeError("WayPoint içinde x/y alanı bulunamadı. vars(wp) çıktısına bak.")

plt.figure(figsize=(8, 8))

# Path çizgisi
plt.plot(xs, ys, "-", color="lightcoral", alpha=0.5, label="Path")

# Waypoint noktaları
plt.scatter(xs, ys, s=14, color="red", label="Waypoints")

# Start ve End
plt.scatter(xs[0], ys[0], s=120, color="green", label="Start", zorder=5)
plt.scatter(xs[-1], ys[-1], s=120, color="red", label="End", zorder=5)

plt.title("Istanbul Park Map Visualization")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis("equal")

plt.tight_layout()
plt.savefig("IstanbulPark_Map_Visualization.png", dpi=300)
print("Saved: IstanbulPark_Map_Visualization.png")