#!/usr/bin/env python3
"""
Basic evolutionary training of FlyBrain readout for 1v1 dogfight.
Aircraft are created once and reset between episodes.
"""

import time
import numpy as np
from flybrain import FlyBrain
import jsbsim
import copy

print("Loading brains (once)...")
brain1 = FlyBrain(device="cpu", batch=1)
brain2 = FlyBrain(device="cpu", batch=1)
loom_L = brain1.cells(["LC4", "LPLC2"], side="L")
loom_R = brain1.cells(["LC4", "LPLC2"], side="R")
desc1 = brain1.cells(["descending_neuron"])
desc2 = brain2.cells(["descending_neuron"])
print("Brains ready")

n_read = min(40, len(desc1))

def make_ac():
    fdm = jsbsim.FGFDMExec(None)
    fdm.load_model("f16")
    fdm.set_dt(0.02)
    return fdm

ac1 = make_ac()
ac2 = make_ac()
print("Aircraft ready")

def reset_ac(fdm, z=11000, heading=0, lon=0.0):
    fdm["ic/h-sl-ft"] = z
    fdm["ic/long-gc-deg"] = lon
    fdm["ic/lat-gc-deg"] = 0.0
    fdm["ic/psi-true-deg"] = heading
    fdm["ic/vc-kts"] = 400
    fdm["ic/phi-deg"] = 0
    fdm["ic/theta-deg"] = 1.5
    fdm.run_ic()

def get_state(fdm):
    return {
        "alt": fdm["position/h-sl-ft"],
        "speed": fdm["velocities/vc-kts"],
        "heading": fdm["attitude/psi-rad"],
        "pitch": fdm["attitude/pitch-rad"],
        "roll": fdm["attitude/roll-rad"],
        "lon": fdm["position/long-gc-deg"],
        "lat": fdm["position/lat-gc-deg"],
    }

def relative(own, enemy):
    dx = (enemy["lon"] - own["lon"]) * 60 * 6076
    dy = (enemy["lat"] - own["lat"]) * 60 * 6076
    rng = np.sqrt(dx*dx + dy*dy) + 1.0
    bearing = np.arctan2(dx, dy) - own["heading"]
    bearing = (bearing + np.pi) % (2*np.pi) - np.pi
    return rng, bearing

def run_episode(W1, W2, max_steps=600):
    reset_ac(ac1, z=11000, heading=0,   lon=-0.006)
    reset_ac(ac2, z=11000, heading=180, lon=+0.006)

    health = [100.0, 100.0]
    total_reward = 0.0
    prev_range = None

    for step in range(max_steps):
        s1 = get_state(ac1)
        s2 = get_state(ac2)
        r1, b1 = relative(s1, s2)
        r2, b2 = relative(s2, s1)

        inject1 = []
        if r1 < 14000:
            stren = float(np.clip(1.1 - r1/16000, 0.2, 0.9))
            if b1 > 0.05:
                inject1 = [(loom_R, stren)]
            elif b1 < -0.05:
                inject1 = [(loom_L, stren)]
            else:
                inject1 = [(loom_L, stren*0.5), (loom_R, stren*0.5)]

        inject2 = []
        if r2 < 14000:
            stren = float(np.clip(1.1 - r2/16000, 0.2, 0.9))
            if b2 > 0.05:
                inject2 = [(loom_R, stren)]
            elif b2 < -0.05:
                inject2 = [(loom_L, stren)]
            else:
                inject2 = [(loom_L, stren*0.5), (loom_R, stren*0.5)]

        def controls(brain, desc, W, inject, state, bearing):
            fired = brain.step(inject=inject)
            act = np.zeros(n_read)
            for i, nid in enumerate(desc[:n_read]):
                if nid in fired:
                    act[i] = 1.0
            raw = W @ act

            elev = -1.0 * state["pitch"] + 0.00008 * (11000 - state["alt"])
            ail  = -1.5 * state["roll"] + 1.0 * np.clip(bearing, -1, 1)
            thr  = 0.60

            thr  += 0.22 * np.tanh(raw[0])
            ail  += 0.30 * np.tanh(raw[1])
            elev += 0.25 * np.tanh(raw[2])
            rud   = 0.15 * np.tanh(raw[3])

            return (np.clip(thr, 0.3, 1.0),
                    np.clip(ail, -1, 1),
                    np.clip(elev, -1, 1),
                    np.clip(rud, -0.5, 0.5))

        thr1, ail1, elev1, rud1 = controls(brain1, desc1, W1, inject1, s1, b1)
        thr2, ail2, elev2, rud2 = controls(brain2, desc2, W2, inject2, s2, b2)

        for ac, thr, ail, elev, rud in [(ac1, thr1, ail1, elev1, rud1),
                                       (ac2, thr2, ail2, elev2, rud2)]:
            ac["fcs/throttle-cmd-norm"] = thr
            ac["fcs/aileron-cmd-norm"]  = ail
            ac["fcs/elevator-cmd-norm"] = elev
            ac["fcs/rudder-cmd-norm"]   = rud
            ac.run()

        hit_reward = 0.0
        for i, (r, b, other) in enumerate([(r1, b1, 1), (r2, b2, 0)]):
            if r < 2200 and abs(b) < 0.18:
                health[other] = max(0, health[other] - 6.0)
                if i == 0:
                    hit_reward += 10.0

        reward = 0.015
        if prev_range is not None:
            reward += 0.002 * (prev_range - r1)
        reward += 1.0 * max(0, 1.0 - abs(b1))
        reward += hit_reward

        if s1["alt"] < 700:
            reward -= 25
        if health[0] <= 0:
            reward -= 40
        if health[1] <= 0:
            reward += 70

        total_reward += reward
        prev_range = r1

        if health[0] <= 0 or health[1] <= 0 or s1["alt"] < 500 or s2["alt"] < 500:
            break

    return total_reward, health, r1

# Evolutionary training
POP_SIZE = 8
GENERATIONS = 6
MUTATION_STD = 0.035

print(f"\nStarting training  |  Pop={POP_SIZE}  Gens={GENERATIONS}")

population = []
for _ in range(POP_SIZE):
    population.append({
        "W1": np.random.randn(4, n_read) * 0.03,
        "W2": np.random.randn(4, n_read) * 0.03,
        "fitness": -999
    })

best_overall = None
best_fitness = -1e9

for gen in range(GENERATIONS):
    t0 = time.time()
    print(f"=== Generation {gen+1}/{GENERATIONS} ===")

    for i, ind in enumerate(population):
        fit, hp, final_r = run_episode(ind["W1"], ind["W2"])
        ind["fitness"] = fit
        print(f"  Agent {i:2d}  fit={fit:7.1f}  HP {hp[0]:5.0f}/{hp[1]:5.0f}  rng={final_r:6.0f}")

        if fit > best_fitness:
            best_fitness = fit
            best_overall = copy.deepcopy(ind)

    population.sort(key=lambda x: x["fitness"], reverse=True)
    print(f"  Best this gen: {population[0]['fitness']:.1f}  |  Overall best: {best_fitness:.1f}")
    print(f"  Time: {time.time()-t0:.1f}s\n")

    elite = population[:2]
    next_pop = copy.deepcopy(elite)
    while len(next_pop) < POP_SIZE:
        parent = elite[np.random.randint(0, len(elite))]
        child = {
            "W1": parent["W1"] + np.random.randn(*parent["W1"].shape) * MUTATION_STD,
            "W2": parent["W2"] + np.random.randn(*parent["W2"].shape) * MUTATION_STD,
            "fitness": -999
        }
        next_pop.append(child)
    population = next_pop

print("Training finished.")
print(f"Best fitness: {best_fitness:.1f}")
np.savez("best_readout.npz", W1=best_overall["W1"], W2=best_overall["W2"])
print("Saved best_readout.npz")
