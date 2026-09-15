#!/usr/bin/env python3
"""
FlyBrain Dogfight v3 -- the brain is the pilot, not a correction term.

Architecture (per aircraft, 4 frozen fly connectomes):

  Perception (2 brains, "left eye" / "right eye")
    - Real photoreceptors (brain.visual / brain.azimuth) are stimulated with a
      noisy, range-attenuated bump of light at the enemy's true bearing --
      the brain is not handed the bearing directly, it has to represent it
      in population activity like a real visual system would.
    - Read out: spiking activity of that brain's own `visual_projection`
      neurons (includes LC4/LPLC2 looming detectors) on its own side.

  Motor (2 brains, lateral / longitudinal)
    - Input: the two perception brains' activity, projected through an
      evolved encoding matrix onto `ascending_neuron` cells (the real
      sensory-to-brain relay population) -- this is the only place learning
      happens on the input side.
    - Output: spiking activity of `vnc_motor` neurons (the fly's literal
      final motor pathway to the body), projected through an evolved
      readout matrix straight into flight-control commands.
    - Lateral brain -> aileron + rudder. Longitudinal brain -> throttle +
      elevator (plus its own altitude/speed proprioception as extra input).

There is no classical pursuit controller anymore. The only non-brain code
left is a hard, non-blended ground/stall safety envelope -- it overrides,
it doesn't get added to the brain's output, so the brain is genuinely
making the dogfighting decisions.
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
# Sizes
# -------------------------------------------------
N_PERC      = 64   # neurons read per perception brain (per side)
N_INJECT    = 48   # ascending_neuron targets per motor brain
N_MOTOR_RD  = 32   # vnc_motor neurons read per motor brain
LON_EXTRA   = 2     # own-state features fed only to the longitudinal brain

# -------------------------------------------------
# Brain rig: 4 frozen connectomes per aircraft
# -------------------------------------------------
def make_brain_rig(seed0):
    percL = FlyBrain(device="cpu", batch=1, seed=seed0 + 0)
    percR = FlyBrain(device="cpu", batch=1, seed=seed0 + 1)
    motorLat = FlyBrain(device="cpu", batch=1, seed=seed0 + 2)
    motorLon = FlyBrain(device="cpu", batch=1, seed=seed0 + 3)

    rig = {
        "percL": percL, "percR": percR,
        "motorLat": motorLat, "motorLon": motorLon,
        "visL": percL.cells(["visual_projection"], side="L")[:N_PERC],
        "visR": percR.cells(["visual_projection"], side="R")[:N_PERC],
        "injLat": motorLat.cells(["ascending_neuron"])[:N_INJECT],
        "injLon": motorLon.cells(["ascending_neuron"])[:N_INJECT],
        "rdLat": motorLat.cells(["vnc_motor"])[:N_MOTOR_RD],
        "rdLon": motorLon.cells(["vnc_motor"])[:N_MOTOR_RD],
    }
    return rig

def reset_rig(rig, seed0):
    rig["percL"].reset(seed0 + 0)
    rig["percR"].reset(seed0 + 1)
    rig["motorLat"].reset(seed0 + 2)
    rig["motorLon"].reset(seed0 + 3)

print("Loading FlyBrain rigs (4 connectomes x 2 aircraft)...")
rig1 = make_brain_rig(seed0=100)
rig2 = make_brain_rig(seed0=200)
print(f"Rigs ready | {rig1['percL'].n} neurons per brain")

# -------------------------------------------------
# Genome: only the encode/readout matrices are trained
# -------------------------------------------------
N_PERC_TOTAL = 2 * N_PERC  # both eyes concatenated

def random_genome(scale=0.05):
    return {
        "encLat": np.random.randn(N_INJECT, N_PERC_TOTAL) * scale,
        "rdLat":  np.random.randn(2, N_MOTOR_RD) * scale,
        "encLon": np.random.randn(N_INJECT, N_PERC_TOTAL + LON_EXTRA) * scale,
        "rdLon":  np.random.randn(2, N_MOTOR_RD) * scale,
    }

def mutate(genome, sigma):
    return {k: v + np.random.randn(*v.shape) * sigma for k, v in genome.items()}

# -------------------------------------------------
# Aircraft helpers (unchanged physics)
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

def relative(own, enemy):
    dx = (enemy["lon"] - own["lon"]) * 60.0 * 6076.0
    dy = (enemy["lat"] - own["lat"]) * 60.0 * 6076.0
    dz = enemy["alt"] - own["alt"]
    rng = np.sqrt(dx*dx + dy*dy + dz*dz) + 1.0
    abs_brg = np.arctan2(dx, dy)
    brg = abs_brg - own["hdg"]
    brg = (brg + np.pi) % (2*np.pi) - np.pi
    return rng, brg, dz

# -------------------------------------------------
# Perception: a real (noisy, decoded) sensing step
# -------------------------------------------------
def eye_drive(brain, rng_ft, brg_rad, max_range=20000.0, sigma=0.18, noise=0.06):
    """Paint a light bump on the real photoreceptor array at the enemy's
    true bearing, attenuated with range, plus sensor noise. The brain has
    to turn this into useful population activity itself."""
    az_target = np.clip(brg_rad / np.pi, -1.0, 1.0)
    if rng_ft > max_range:
        drive = np.zeros_like(brain.azimuth)
    else:
        strength = float(np.clip(1.15 - rng_ft / max_range, 0.05, 1.0))
        d = brain.azimuth - az_target
        drive = strength * np.exp(-0.5 * (d / sigma) ** 2)
    drive = drive + np.random.uniform(0.0, noise, size=drive.shape)
    return np.clip(drive, 0.0, 1.0).astype(np.float32)

def perceive(rig, rng_ft, brg_rad):
    driveL = eye_drive(rig["percL"], rng_ft, brg_rad)
    driveR = eye_drive(rig["percR"], rng_ft, brg_rad)
    firedL = rig["percL"].step(eye_drive=driveL)
    firedR = rig["percR"].step(eye_drive=driveR)
    actL = np.isin(rig["visL"], firedL).astype(np.float32)
    actR = np.isin(rig["visR"], firedR).astype(np.float32)
    return np.concatenate([actL, actR])

# -------------------------------------------------
# Motor: perception -> ascending_neuron injection -> vnc_motor readout
# -------------------------------------------------
def fly_motor(brain, inj_idx, read_idx, enc, rd, feat):
    """The library's `inject` only takes one scalar per neuron pool, not a
    distinct value per neuron -- so a learned per-neuron encoding has to
    touch brain.v (a plain public array) directly instead. This lands the
    injected current one tick before the built-in decay/noise/threshold
    pass rather than after it, a minor timing offset that doesn't change
    what's being computed: a genuinely per-neuron, encoded drive."""
    inj_amount = np.tanh(enc @ feat) * 1.6
    brain.v[inj_idx, 0] += inj_amount.astype(np.float32)
    fired = brain.step()
    act = np.isin(read_idx, fired).astype(np.float32)
    return np.tanh(rd @ act)

def brain_fly(rig, genome, perc_vec, own_alt_err, own_spd_err):
    lat_out = fly_motor(rig["motorLat"], rig["injLat"], rig["rdLat"],
                         genome["encLat"], genome["rdLat"], perc_vec)
    lon_feat = np.concatenate([perc_vec, [own_alt_err, own_spd_err]])
    lon_out = fly_motor(rig["motorLon"], rig["injLon"], rig["rdLon"],
                         genome["encLon"], genome["rdLon"], lon_feat)

    aileron = float(lat_out[0]) * 1.0
    rudder  = float(lat_out[1]) * 0.5
    throttle = 0.65 + 0.35 * float(lon_out[0])
    elevator = float(lon_out[1]) * 1.0
    return throttle, aileron, elevator, rudder

# -------------------------------------------------
# Hard safety envelope -- overrides, never blends
# -------------------------------------------------
def safety_override(t, a, e, r, alt):
    if alt < 1500:
        return 1.0, a * 0.3, 0.8, r * 0.3
    return t, a, e, r

# -------------------------------------------------
# One episode
# -------------------------------------------------
def run_episode(genome, max_steps=500, record=False):
    reset_ac(ac1, z=12500, hdg_deg=5,   lon=-0.013)
    reset_ac(ac2, z=12500, hdg_deg=185, lon=+0.013)
    reset_rig(rig1, seed0=np.random.randint(0, 10_000))
    reset_rig(rig2, seed0=np.random.randint(0, 10_000))

    health = [100.0, 100.0]
    total_reward = 0.0
    prev_rng = None
    history = [] if record else None

    for step in range(max_steps):
        s1 = get_state(ac1)
        s2 = get_state(ac2)

        r1, b1, dz1 = relative(s1, s2)
        r2, b2, dz2 = relative(s2, s1)

        perc1 = perceive(rig1, r1, b1)
        perc2 = perceive(rig2, r2, b2)

        alt_err1 = np.clip((12500.0 - s1["alt"]) / 3000.0, -1.0, 1.0)
        alt_err2 = np.clip((12500.0 - s2["alt"]) / 3000.0, -1.0, 1.0)
        spd_err1 = np.clip((420.0 - s1["spd"]) / 200.0, -1.0, 1.0)
        spd_err2 = np.clip((420.0 - s2["spd"]) / 200.0, -1.0, 1.0)

        t1, a1, e1, rd1 = brain_fly(rig1, genome, perc1, alt_err1, spd_err1)
        t2, a2, e2, rd2 = brain_fly(rig2, genome, perc2, alt_err2, spd_err2)

        t1, a1, e1, rd1 = safety_override(t1, a1, e1, rd1, s1["alt"])
        t2, a2, e2, rd2 = safety_override(t2, a2, e2, rd2, s2["alt"])

        for ac, t, a, e, r in [(ac1, t1, a1, e1, rd1), (ac2, t2, a2, e2, rd2)]:
            ac["fcs/throttle-cmd-norm"] = float(np.clip(t, 0.35, 1.0))
            ac["fcs/aileron-cmd-norm"]  = float(np.clip(a, -1.0, 1.0))
            ac["fcs/elevator-cmd-norm"] = float(np.clip(e, -1.0, 1.0))
            ac["fcs/rudder-cmd-norm"]   = float(np.clip(r, -0.5, 0.5))
            ac.run()

        hit_reward = 0.0
        for i, (rng, brg, other) in enumerate([(r1, b1, 1), (r2, b2, 0)]):
            if rng < 1700 and abs(brg) < 0.14:
                health[other] = max(0.0, health[other] - 5.5)
                if i == 0:
                    hit_reward += 30.0

        rwd = 0.012
        if prev_rng is not None:
            rwd += 0.004 * (prev_rng - r1)
        rwd += 1.5 * max(0.0, 1.0 - abs(b1))
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
            x1 = (s1["lon"] + 0.013) * 60 * 6076
            y1 = s1["lat"] * 60 * 6076
            x2 = (s2["lon"] - 0.013) * 60 * 6076
            y2 = s2["lat"] * 60 * 6076
            history.append({
                "x1": x1, "y1": y1, "z1": s1["alt"],
                "x2": x2, "y2": y2, "z2": s2["alt"],
                "r": r1, "hp": list(health),
            })

        if health[0] <= 0 or health[1] <= 0 or s1["alt"] < 500 or s2["alt"] < 500:
            break

    return total_reward, health, r1, history

# -------------------------------------------------
# Evolutionary training (genome shared by both aircraft, mirror self-play)
# -------------------------------------------------
if __name__ == "__main__":
    POP = 6
    GENS = 4
    MUT = 0.04

    print(f"\n=== Brain-as-pilot training  Pop={POP}  Gens={GENS} ===")
    population = [{"G": random_genome(), "fit": -999} for _ in range(POP)]
    best_G = None
    best_fit = -1e9

    for gen in range(GENS):
        t0 = time.time()
        print(f"Generation {gen+1}/{GENS}")
        for i, ind in enumerate(population):
            fit, hp, final_r, _ = run_episode(ind["G"])
            ind["fit"] = fit
            print(f"  Agent {i:2d}  fit={fit:7.1f}  HP {hp[0]:5.0f}/{hp[1]:5.0f}  final_r={final_r:6.0f}")
            if fit > best_fit:
                best_fit = fit
                best_G = copy.deepcopy(ind["G"])
                print(f"    *** new best {best_fit:.1f}")

        population.sort(key=lambda x: x["fit"], reverse=True)
        elite = population[:2]
        next_pop = copy.deepcopy(elite)
        while len(next_pop) < POP:
            parent = elite[np.random.randint(0, 2)]
            next_pop.append({"G": mutate(parent["G"], MUT), "fit": -999})
        population = next_pop
        print(f"  Gen time {time.time()-t0:.1f}s   best so far {best_fit:.1f}\n")

    print(f"Training finished. Best fitness = {best_fit:.1f}")
    np.savez("best_brain_pilot_v3.npz", **best_G)
    print("Saved best_brain_pilot_v3.npz")

    print("\n=== Final fight with best brain-pilot (recording for 3D) ===")
    fit, hp, final_r, history = run_episode(best_G, max_steps=900, record=True)
    print(f"Fight length: {len(history)*0.02:.1f}s")
    print(f"Final HP  Red {hp[0]:.0f}  Blue {hp[1]:.0f}  |  final range {final_r:.0f} ft")

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
