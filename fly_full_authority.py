#!/usr/bin/env python3
"""
FlyBrain Dogfight – Full Authority version

Architecture
------------
- FlyBrain outputs high-level commands:
    desired bank angle, desired pitch angle, throttle
- Simple inner-loop controller converts those into
  aileron / elevator / rudder that actually produce a turn
  (the raw aileron test showed the aircraft can bank but needs
   coordination to change heading).
- Strong dopamine (hit) reward + closing + nose-on shaping
- Evolutionary training of the readout only
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
# Brains
# -------------------------------------------------
print("Loading FlyBrains...")
brain1 = FlyBrain(device="cpu", batch=1)
brain2 = FlyBrain(device="cpu", batch=1)
loom_L = brain1.cells(["LC4", "LPLC2"], side="L")
loom_R = brain1.cells(["LC4", "LPLC2"], side="R")
desc1  = brain1.cells(["descending_neuron"])
desc2  = brain2.cells(["descending_neuron"])
print(f"Brains ready | {brain1.n} neurons")

n_read = min(48, len(desc1))

# -------------------------------------------------
# Aircraft
# -------------------------------------------------
def make_ac():
    fdm = jsbsim.FGFDMExec(None)
    fdm.load_model("f16")
    fdm.set_dt(0.02)
    return fdm

ac1 = make_ac()
ac2 = make_ac()
print("Aircraft ready")

def reset_ac(fdm, z=12500.0, hdg_deg=0.0, lon=0.0):
    fdm["ic/h-sl-ft"]      = z
    fdm["ic/long-gc-deg"]  = lon
    fdm["ic/lat-gc-deg"]   = 0.0
    fdm["ic/psi-true-deg"] = hdg_deg
    fdm["ic/vc-kts"]       = 420.0
    fdm["ic/phi-deg"]      = 0.0
    fdm["ic/theta-deg"]    = 2.0
    fdm.run_ic()

def get_state(fdm):
    return {
        "alt":   fdm["position/h-sl-ft"],
        "lon":   fdm["position/long-gc-deg"],
        "lat":   fdm["position/lat-gc-deg"],
        "hdg":   fdm["attitude/psi-rad"],
        "pitch": fdm["attitude/pitch-rad"],
        "roll":  fdm["attitude/roll-rad"],
        "spd":   fdm["velocities/vc-kts"],
    }

def relative(own, enemy):
    dx = (enemy["lon"] - own["lon"]) * 60.0 * 6076.0
    dy = (enemy["lat"] - own["lat"]) * 60.0 * 6076.0
    dz = enemy["alt"] - own["alt"]
    rng = np.sqrt(dx*dx + dy*dy + dz*dz) + 1.0
    abs_brg = np.arctan2(dx, dy)
    brg = abs_brg - own["hdg"]
    brg = (brg + np.pi) % (2*np.pi) - np.pi
    return rng, brg, dz, abs_brg

# -------------------------------------------------
# Inner-loop controller
# Converts desired bank / pitch into surface commands
# that actually change heading
# -------------------------------------------------
def inner_loop(s, des_bank, des_pitch, thr):
    """
    des_bank, des_pitch in radians
    Returns throttle, aileron, elevator, rudder
    """
    # Bank hold (aileron)
    bank_err = des_bank - s["roll"]
    ail = 1.8 * bank_err - 0.4 * s["roll"]   # P + mild damping

    # Pitch hold (elevator)
    pitch_err = des_pitch - s["pitch"]
    elev = 1.6 * pitch_err - 0.5 * s["pitch"]

    # Simple coordinated rudder (helps turn rate)
    rud = 0.35 * s["roll"]   # approximate coordination

    return (
        float(np.clip(thr, 0.35, 1.0)),
        float(np.clip(ail, -1.0, 1.0)),
        float(np.clip(elev, -1.0, 1.0)),
        float(np.clip(rud, -0.6, 0.6)),
    )

# -------------------------------------------------
# FlyBrain high-level policy
# Outputs: desired bank, desired pitch, throttle
# -------------------------------------------------
def brain_policy(brain, desc, W, inject, brg, rng, dz):
    fired = brain.step(inject=inject)
    act = np.zeros(n_read)
    for i, nid in enumerate(desc[:n_read]):
        if nid in fired:
            act[i] = 1.0
    raw = W @ act

    # Map network output to high-level commands
    # Positive brg (enemy to the right) should produce positive desired bank (right bank)
    des_bank  = 0.9 * np.clip(brg, -1.0, 1.0) + 0.35 * np.tanh(raw[1])
    des_pitch = 0.08 * np.clip(dz / 2000.0, -1, 1) + 0.25 * np.tanh(raw[2])
    thr       = 0.62 + 0.25 * np.tanh(raw[0])

    # Extra urgency when far
    if rng > 6000:
        des_bank *= 1.25

    des_bank  = float(np.clip(des_bank, -1.1, 1.1))
    des_pitch = float(np.clip(des_pitch, -0.4, 0.35))
    thr       = float(np.clip(thr, 0.4, 0.95))

    return des_bank, des_pitch, thr

def make_inject(rng, brg):
    inject = []
    if rng < 18000:
        stren = float(np.clip(1.2 - rng / 20000, 0.25, 0.95))
        if brg > 0.07:
            inject = [(loom_R, stren)]
        elif brg < -0.07:
            inject = [(loom_L, stren)]
        else:
            inject = [(loom_L, stren * 0.5), (loom_R, stren * 0.5)]
    return inject

# -------------------------------------------------
# Episode
# -------------------------------------------------
def run_episode(W, max_steps=800, record=False):
    reset_ac(ac1, z=12500, hdg_deg=0,   lon=-0.012)
    reset_ac(ac2, z=12500, hdg_deg=180, lon=+0.012)

    health = [100.0, 100.0]
    total_reward = 0.0
    prev_rng = None
    history = [] if record else None

    for step in range(max_steps):
        s1 = get_state(ac1)
        s2 = get_state(ac2)
        r1, b1, dz1, _ = relative(s1, s2)
        r2, b2, dz2, _ = relative(s2, s1)

        inj1 = make_inject(r1, b1)
        inj2 = make_inject(r2, b2)

        # High-level commands from brains
        db1, dp1, thr1 = brain_policy(brain1, desc1, W, inj1, b1, r1, dz1)
        db2, dp2, thr2 = brain_policy(brain2, desc2, W, inj2, b2, r2, dz2)

        # Inner loops
        t1, a1, e1, r1_ = inner_loop(s1, db1, dp1, thr1)
        t2, a2, e2, r2_ = inner_loop(s2, db2, dp2, thr2)

        for ac, t, a, e, r in [(ac1, t1, a1, e1, r1_), (ac2, t2, a2, e2, r2_)]:
            ac["fcs/throttle-cmd-norm"] = t
            ac["fcs/aileron-cmd-norm"]  = a
            ac["fcs/elevator-cmd-norm"] = e
            ac["fcs/rudder-cmd-norm"]   = r
            ac.run()

        # Gun + dopamine
        hit_r = 0.0
        for i, (rng, brg, other) in enumerate([(r1, b1, 1), (r2, b2, 0)]):
            if rng < 1600 and abs(brg) < 0.13:
                health[other] = max(0.0, health[other] - 6.0)
                if i == 0:
                    hit_r += 35.0          # strong dopamine

        # Shaping
        rwd = 0.01
        if prev_rng is not None:
            rwd += 0.005 * (prev_rng - r1)     # closing
        rwd += 1.8 * max(0.0, 1.0 - abs(b1))  # nose-on
        rwd += hit_r

        if s1["alt"] < 700:
            rwd -= 40
        if health[0] <= 0:
            rwd -= 55
        if health[1] <= 0:
            rwd += 90

        total_reward += rwd
        prev_rng = r1

        if record:
            x1 = (s1["lon"] + 0.012) * 60 * 6076
            y1 = s1["lat"] * 60 * 6076
            x2 = (s2["lon"] - 0.012) * 60 * 6076
            y2 = s2["lat"] * 60 * 6076
            history.append({
                "x1": x1, "y1": y1, "z1": s1["alt"],
                "x2": x2, "y2": y2, "z2": s2["alt"],
                "r": r1, "hp": list(health)
            })

        if health[0] <= 0 or health[1] <= 0 or s1["alt"] < 500 or s2["alt"] < 500:
            break

    return total_reward, health, r1, history

# -------------------------------------------------
# Training
# -------------------------------------------------
POP = 8
GENS = 6
MUT  = 0.04

print(f"\n=== Full-authority training  Pop={POP}  Gens={GENS} ===")
population = [{"W": np.random.randn(4, n_read) * 0.03, "fit": -999} for _ in range(POP)]
best_W = None
best_fit = -1e9

for gen in range(GENS):
    t0 = time.time()
    print(f"Generation {gen+1}/{GENS}")
    for i, ind in enumerate(population):
        fit, hp, final_r, _ = run_episode(ind["W"])
        ind["fit"] = fit
        print(f"  Agent {i:2d}  fit={fit:7.1f}  HP {hp[0]:5.0f}/{hp[1]:5.0f}  rng={final_r:6.0f}")
        if fit > best_fit:
            best_fit = fit
            best_W = ind["W"].copy()
            print(f"    *** new best {best_fit:.1f}")

    population.sort(key=lambda x: x["fit"], reverse=True)
    elite = population[:2]
    next_pop = copy.deepcopy(elite)
    while len(next_pop) < POP:
        p = elite[np.random.randint(0, 2)]
        child = {"W": p["W"] + np.random.randn(*p["W"].shape) * MUT, "fit": -999}
        next_pop.append(child)
    population = next_pop
    print(f"  Time {time.time()-t0:.1f}s   best {best_fit:.1f}\n")

print(f"Training finished. Best fitness = {best_fit:.1f}")
np.savez("best_full_authority.npz", W=best_W)
print("Saved best_full_authority.npz")

# -------------------------------------------------
# Final fight + 3D
# -------------------------------------------------
print("\n=== Final fight (recording) ===")
fit, hp, final_r, history = run_episode(best_W, max_steps=1000, record=True)
print(f"Fight length: {len(history)*0.02:.1f}s")
print(f"Final HP Red {hp[0]:.0f}  Blue {hp[1]:.0f}  |  final range {final_r:.0f} ft")

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
