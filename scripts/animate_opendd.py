"""Animate openDD road users as points moving over the dataset's own lane links.

Usage: python animate_opendd.py <example_data dir> <out.mp4|out.gif> [t_start t_end speed]
Links are the `trafficLane` centrelines (with successor/predecessor graph) from map_rdb1.sqlite.
"""
import sys, re, sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

base, out = sys.argv[1].rstrip("/"), sys.argv[2]
t_start, t_end, speed = (float(a) for a in (sys.argv[3:6] + ["0", "1e9", "4"][len(sys.argv) - 3:]))
FPS = 30 if out.endswith(".mp4") else 12
HZ = 1 / 0.0333667                                   # recording rate (29.97 Hz)

def coords(wkt):
    return np.array([[float(v) for v in p.split()] for p in re.search(r"\(+(.*?)\)+", wkt).group(1).split(",")])

tr = pd.read_sql("select TIMESTAMP t, OBJID id, UTM_X x, UTM_Y y, V v, CLASS cls from rdb1_4",
                 sqlite3.connect(f"{base}/rdb1_4.sqlite"))
tr = tr[~tr.cls.isin(["Pedestrian", "Trailer"])]     # trailers duplicate their towing vehicle
mp = pd.read_sql("select * from rdb1", sqlite3.connect(f"{base}/map_rdb1/map_rdb1.sqlite"))
lanes = [coords(g) for g in mp[mp.type == "trafficLane"].geometry]
bounds = [coords(g) for g in mp[mp.type == "boundary"].geometry.dropna()]
ox, oy = np.vstack(lanes).mean(0)
lanes = [l - (ox, oy) for l in lanes]; bounds = [b - (ox, oy) for b in bounds]
tr["x"] -= ox; tr["y"] -= oy
tr["f"] = (tr.t * HZ).round().astype(int)

step = max(1, round(speed * HZ / FPS))
frames = [f for f in range(int(max(t_start, tr.t.min()) * HZ), int(min(t_end, tr.t.max()) * HZ) + 1, step)]
by_f = {f: g for f, g in tr[tr.f.isin(frames)].groupby("f")}
seen = tr.groupby("id").f.min().sort_values().to_numpy()    # for the cumulative counter

style = {"Car": ("car", "#2a78d6", 30), "Medium Vehicle": ("van / medium", "#12a594", 40),
         "Heavy Vehicle": ("heavy vehicle", "#7a4fbf", 60), "Bus": ("bus", "#e07b00", 60),
         "Motorcycle": ("motorcycle", "#d6336c", 18), "Bicycle": ("bicycle", "#8a94a0", 12)}

fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=100)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.06)
ax.add_collection(LineCollection(bounds, colors="#dfe3e8", linewidths=0.8, zorder=1))
ax.add_collection(LineCollection(lanes, colors="#c5ccd5", linewidths=6, capstyle="round", zorder=2))
ax.add_collection(LineCollection(lanes, colors="#7f8995", linewidths=0.7, zorder=3))
for l in lanes:                                       # link ends = nodes; arrow = direction
    ax.plot(*l[0], "o", ms=2.6, color="#5b6570", zorder=3.5)
    m = len(l) // 2
    if m + 1 < len(l):
        ax.annotate("", l[m + 1], l[m], arrowprops=dict(arrowstyle="-|>", color="#5b6570", lw=0, mutation_scale=9), zorder=3.5)
scat = {k: ax.scatter([], [], s=s, color=c, edgecolor="white", linewidth=0.5, label=n, zorder=5)
        for k, (n, c, s) in style.items()}
allp = np.vstack(lanes)
ax.set_xlim(allp[:, 0].min() - 6, allp[:, 0].max() + 6); ax.set_ylim(allp[:, 1].min() - 6, allp[:, 1].max() + 6)
ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)
xs, ys = ax.get_xlim()[0] + 4, ax.get_ylim()[0] + 3
ax.plot([xs, xs + 20], [ys, ys], color="#333", lw=2); ax.text(xs + 10, ys + 1.2, "20 m", ha="center", fontsize=8)
ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
fig.suptitle("Observed road users over lane links: openDD roundabout rdb1 (unsignalised, drone, all tracked vehicles)",
             fontsize=10.5, x=0.02, ha="left")
clock = fig.text(0.02, 0.915, "", fontsize=10, va="top", family="monospace")
fig.text(0.02, 0.015, f"Data: openDD example recording rdb1_4 (Breuer et al. 2020, CC BY-ND 4.0). "
         f"{len(lanes)} lane links from the dataset's HD map. Playback {step * FPS / HZ:.1f}x.", fontsize=7, color="#5b6570")
empty = np.empty((0, 2))

def draw(i):
    f = frames[i]; g = by_f.get(f)
    for k, s in scat.items():
        s.set_offsets(empty if g is None else g.loc[g.cls == k, ["x", "y"]].to_numpy().reshape(-1, 2))
    clock.set_text(f"t = {f / HZ:6.1f} s   in view: {0 if g is None else len(g):2d}   seen so far: {np.searchsorted(seen, f, 'right'):3d}")
    return list(scat.values()) + [clock]

writer = FFMpegWriter(fps=FPS, bitrate=1800) if out.endswith(".mp4") else PillowWriter(fps=FPS)
FuncAnimation(fig, draw, frames=len(frames), blit=False).save(out, writer=writer)
draw(len(frames) // 2); fig.savefig(re.sub(r"\.\w+$", "_frame.png", out))
print(out, "frames", len(frames), "road users", tr.id.nunique(), "links", len(lanes))
