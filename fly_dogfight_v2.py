#!/usr/bin/env python3
"""
FlyBrain Dogfight v2 – focused on making a real merge happen

Key improvements:
- Robust local Cartesian relative geometry
- Strong classical pure-pursuit baseline that forces closing
- FlyBrain acts only as residual correction
- Rich sensory features (range, bearing, range-rate, aspect, dAlt)
- Strong dopamine (hit) + closing + nose-on rewards
- Pre-recorded 3D visualization
"""

import time
import copy
import numpy as np
from flybrain import FlyBrain
import jsbsim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# -------------------------------------------------
# Brains (frozen)
# -------------------------------------------------
print("Loading FlyBrains...")
brain1 = FlyBrain(device="cpu", batch=1)
brain2 = FlyBrain(device="cpu", batch=1)
loom_L = brain1.cells(["LC4", "LPLC2"], side="L")
loom_R = brain1.cells(["LC4", "LPLC2"], side="R")
desc1  = brain1.cells(["descending_neuron"])
desc2  = brain2.cells(["descending_neuron"])
print(f"Brains ready | {brain1.n} neurons each")

n_read = min(48, len(desc1))

# -------------------------------------------------
# Aircraft helpers
# -------------------------------------------------
def make_ac():
    fdm = jsbsim.FGFDMExec(None)
    fdm.load_model("f16")
    fdm.set_dt(0.02)
    return fdm

ac1 = make_ac()
ac2 = make_ac()
print("Aircraft created once")

def reset_ac(fdm, z=12500.0, hdg_deg=0.0, lon=0.0, lat=0.0, spd=420.0):
    fdm["ic/h-sl-ft"]      = z
    fdm["ic/long-gc-deg"]  = lon
    fdm["ic/lat-gc-deg"]   = lat
    fdm["ic/psi-true-deg"] = hdg_deg
    fdm["ic/vc-kts"]       = spd
    fdm["ic/phi-deg"]      = 0.0
    fdm["ic/theta-deg"]    = 1.5
    fdm.run_ic()

def get_state(fdm):
    return {
        "alt":  fdm["position/h-sl-ft"],
        "lon":  fdm["position/long-gc-deg"],
        "lat":  fdm["position/lat-gc-deg"],
        "hdg":  fdm["attitude/psi-rad"],
        "pitch":fdm["attitude/pitch-rad"],
        "roll": fdm["attitude/roll-rad"],
        "spd":  fdm["velocities/vc-kts"],
    }

# -------------------------------------------------
# Robust relative geometry (local flat Earth)
# -------------------------------------------------
def relative(own, enemy):
    # East / North in feet (approximate but consistent)
    dx = (enemy["lon"] - own["lon"]) * 60.0 * 6076.0   # East
    dy = (enemy["lat"] - own["lat"]) * 60.0 * 6076.0   # North
    dz = enemy["alt"] - own["alt"]
    rng = np.sqrt(dx*dx + dy*dy + dz*dz) + 1.0

    # Absolute bearing from North (0 = North, +90 = East)
    abs_brg = np.arctan2(dx, dy)

    # Relative bearing in own body frame
    brg = abs_brg - own["hdg"]
    brg = (brg + np.pi) % (2*np.pi) - np.pi

    # Aspect (rough): how much the enemy is pointing at us
    # For simplicity we use the opposite relative bearing
    aspect = abs_brg + np.pi - enemy["hdg"]
    aspect = (aspect + np.pi) % (2*np.pi) - np.pi

    return rng, brg, dz, abs_brg, aspect, dx, dy

# -------------------------------------------------
# Classical baseline that is designed to CLOSE
# -------------------------------------------------
def classical_pursuit(s, brg, rng, dz):
    """
    Strong pure-pursuit + altitude hold.
    Positive brg (enemy to the right) → positive aileron (right turn).
    This mapping was verified with the diagnostic.
    """
    # Turn hard toward the target
    ail = -1.4 * s["roll"] + 2.6 * np.clip(brg, -1.0, 1.0)

    # Altitude hold with gentle climb/dive toward enemy altitude
    elev = -1.3 * s["pitch"] + 0.00009 * (12500.0 - s["alt"])
    elev += 0.00004 * np.clip(dz, -3000, 3000)   # match altitude a bit
    elev += 0.10 * abs(brg)                       # pitch up in turns

    # Speed management
    thr = 0.68 if rng > 5000 else 0.55

    return thr, ail, elev, 0.0

# -------------------------------------------------
# FlyBrain residual
# -------------------------------------------------
def brain_residual(brain, desc, W, inject):
    fired = brain.step(inject=inject)
    act = np.zeros(n_read)
    for i, nid in enumerate(desc[:n_read]):
        if nid in fired:
            act[i] = 1.0
    raw = W @ act
    return (
        0.12 * np.tanh(raw[0]),   # throttle residual
        0.22 * np.tanh(raw[1]),   # aileron residual
        0.18 * np.tanh(raw[2]),   # elevator residual
        0.10 * np.tanh(raw[3]),   # rudder residual
    )

def make_inject(rng, brg):
    """Simple radar-like injection into looming cells"""
    inject = []
    if rng < 18000:
        stren = float(np.clip(1.15 - rng/20000, 0.25, 0.95))
        if brg > 0.08:
            inject = [(loom_R, stren)]
        elif brg < -0.08:
            inject = [(loom_L, stren)]
        else:
            inject = [(loom_L, stren*0.55), (loom_R, stren*0.55)]
    return inject

# -------------------------------------------------
# One episode
# -------------------------------------------------
def run_episode(W, max_steps=700, record=False):
    # Start ~9–10 km apart, nearly head-on
    reset_ac(ac1, z=12500, hdg_deg=5,   lon=-0.013)
    reset_ac(ac2, z=12500, hdg_deg=185, lon=+0.013)

    health = [100.0, 100.0]
    total_reward = 0.0
    prev_rng = None
    history = [] if record else None

    for step in range(max_steps):
        s1 = get_state(ac1)
        s2 = get_state(ac2)

        r1, b1, dz1, _, _, _, _ = relative(s1, s2)
        r2, b2, dz2, _, _, _, _ = relative(s2, s1)

        inj1 = make_inject(r1, b1)
        inj2 = make_inject(r2, b2)

        # Classical + residual
        bt1, ba1, be1, br1 = classical_pursuit(s1, b1, r1, dz1)
        bt2, ba2, be2, br2 = classical_pursuit(s2, b2, r2, dz2)

        ct1, ca1, ce1, cr1 = brain_residual(brain1, desc1, W, inj1)
        ct2, ca2, ce2, cr2 = brain_residual(brain2, desc2, W, inj2)

        for ac, t, a, e, r in [
            (ac1, bt1+ct1, ba1+ca1, be1+ce1, br1+cr1),
            (ac2, bt2+ct2, ba2+ca2, be2+ce2, br2+cr2)
        ]:
            ac["fcs/throttle-cmd-norm"] = float(np.clip(t, 0.35, 1.0))
            ac["fcs/aileron-cmd-norm"]  = float(np.clip(a, -1.0, 1.0))
            ac["fcs/elevator-cmd-norm"] = float(np.clip(e, -1.0, 1.0))
            ac["fcs/rudder-cmd-norm"]   = float(np.clip(r, -0.5, 0.5))
            ac.run()

        # Gun model (realistic-ish)
        hit_reward = 0.0
        for i, (rng, brg, other) in enumerate([(r1, b1, 1), (r2, b2, 0)]):
            if rng < 1700 and abs(brg) < 0.14:
                health[other] = max(0.0, health[other] - 5.5)
                if i == 0:
                    hit_reward += 30.0          # strong dopamine

        # Reward shaping
        rwd = 0.012                                   # survival
        if prev_rng is not None:
            rwd += 0.004 * (prev_rng - r1)            # closing range rate
        rwd += 1.5 * max(0.0, 1.0 - abs(b1))          # nose-on
        rwd += hit_reward

        if s1["alt"] < 800:
            rwd -= 35
        if health[0] <= 0:
            rwd -= 50
        if health[1] <= 0:
            rwd += 80

        total_reward += rwd
        prev_rng = r1

        if record:
            # Convert to local feet for visualization
            x1 = (s1["lon"] + 0.013) * 60 * 6076
            y1 = s1["lat"] * 60 * 6076
            x2 = (s2["lon"] - 0.013) * 60 * 6076
            y2 = s2["lat"] * 60 * 6076
            history.append({
                "x1": x1, "y1": y1, "z1": s1["alt"],
                "x2": x2, "y2": y2, "z2": s2["alt"],
                "r": r1, "hp": list(health), "hits": [0, 0]  # filled later if needed
            })

        if health[0] <= 0 or health[1] <= 0 or s1["alt"] < 500 or s2["alt"] < 500:
            break

    return total_reward, health, r1, history

# -------------------------------------------------
# Evolutionary training with dopamine
# -------------------------------------------------
POP = 8
GENS = 5
MUT  = 0.035

print(f"\n=== Dopamine training  Pop={POP}  Gens={GENS} ===")
population = [{"W": np.random.randn(4, n_read)*0.03, "fit": -999} for _ in range(POP)]
best_W = None
best_fit = -1e9

for gen in range(GENS):
    t0 = time.time()
    print(f"Generation {gen+1}/{GENS}")
    for i, ind in enumerate(population):
        fit, hp, final_r, _ = run_episode(ind["W"])
        ind["fit"] = fit
        print(f"  Agent {i:2d}  fit={fit:7.1f}  HP {hp[0]:5.0f}/{hp[1]:5.0f}  final_r={final_r:6.0f}")
        if fit > best_fit:
            best_fit = fit
            best_W = ind["W"].copy()
            print(f"    *** new best {best_fit:.1f}")

    population.sort(key=lambda x: x["fit"], reverse=True)
    elite = population[:2]
    next_pop = copy.deepcopy(elite)
    while len(next_pop) < POP:
        parent = elite[np.random.randint(0, 2)]
        child = {"W": parent["W"] + np.random.randn(*parent["W"].shape)*MUT, "fit": -999}
        next_pop.append(child)
    population = next_pop
    print(f"  Gen time {time.time()-t0:.1f}s   best so far {best_fit:.1f}\n")

print(f"Training finished. Best fitness = {best_fit:.1f}")
np.savez("best_dopamine_v2.npz", W=best_W)
print("Saved best_dopamine_v2.npz")

# -------------------------------------------------
# Final evaluation + 3D recording
# -------------------------------------------------
print("\n=== Final fight with best brain (recording for 3D) ===")
fit, hp, final_r, history = run_episode(best_W, max_steps=900, record=True)
print(f"Fight length: {len(history)*0.02:.1f}s")
print(f"Final HP  Red {hp[0]:.0f}  Blue {hp[1]:.0f}  |  final range {final_r:.0f} ft")

# 3D animation
print("Opening 3D animation...")
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
line1, = ax.plot([], [], [], "r-", lw=2, label="Red")
line2, = ax.plot([], [], [], "b-", lw=2, label="Blue")
dot1,  = ax.plot([], [], [], "ro", ms=8)
dot2,  = ax.plot([], [], [], "bo", ms=8)
ax.legend()
ax.set_xlabel("East (ft)")
ax.set_ylabel("North (ft)")
ax.set_zlabel("Altitude (ft)")

trail1, trail2 = [], []

def update(frame):
    if frame >= len(history):
        return line1, line2, dot1, dot2
    h = history[frame]
    trail1.append((h["x1"], h["y1"], h["z1"]))
    trail2.append((h["x2"], h["y2"], h["z2"]))
    trail1[:] = trail1[-300:]
    trail2[:] = trail2[-300:]

    if trail1:
        line1.set_data_3d(*zip(*trail1))
        line2.set_data_3d(*zip(*trail2))
        dot1.set_data_3d([h["x1"]], [h["y1"]], [h["z1"]])
        dot2.set_data_3d([h["x2"]], [h["y2"]], [h["z2"]])

    xs = [p[0] for p in trail1 + trail2]
    ys = [p[1] for p in trail1 + trail2]
    zs = [p[2] for p in trail1 + trail2]
    if xs:
        cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
        span = max(max(xs)-min(xs), max(ys)-min(ys), 8000)/2 + 2500
        ax.set_xlim(cx-span, cx+span)
        ax.set_ylim(cy-span, cy+span)
        ax.set_zlim(max(0, min(zs)-1200), max(zs)+1200)

    ax.set_title(f"t={frame*0.02:.1f}s | HP R{h['hp'][0]:.0f} B{h['hp'][1]:.0f} | Range {h['r']:.0f} ft")
    return line1, line2, dot1, dot2

ani = FuncAnimation(fig, update, frames=len(history), interval=30, blit=False, repeat=False)
plt.show()
print("Done.")
