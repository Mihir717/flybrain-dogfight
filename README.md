# FlyBrain Dogfight

Using the complete adult male fruit-fly central nervous system (MaleCNS connectome via the `flybrain` package) as a frozen spiking reservoir to control F-16s in 1v1 dogfights.

## Goal

Make multiple independent fly brains learn real pilot-like dogfighting:
- Stay alive
- Close range
- Get nose-on
- Score gun hits
- Eventually energy management and advanced tactics

## Architecture (v2)

1. **Frozen connectome** (166k neurons) – never trained
2. **Classical pure-pursuit baseline** that is deliberately strong so range decreases
3. **FlyBrain residual readout** – small linear layer that only adds a correction
4. **Rich sensory injection** into looming cells (range, bearing)
5. **Dopamine-style reward** – large positive signal on every gun hit + shaping for closing and nose-on
6. **Evolutionary training** of the residual only
7. **Pre-recorded 3D visualization**

## Setup

```bash
python -m venv flyjets
source flyjets/bin/activate
pip install -r requirements.txt
```

First `FlyBrain()` call downloads ~260 MB of connectome data.

## Run the current best version

```bash
python fly_dogfight_v2.py
```

This will:
- Train the residual readout for a few generations with strong hit rewards
- Run a final fight while recording trajectories
- Open a 3D animation of the fight

## Design principles taken from real dogfight AI work

- Curriculum / staged difficulty (we start with “just close and shoot”)
- Strong classical baseline + learned residual (common in successful AlphaDogfight-style systems)
- Dense shaping + sparse high-value hit reward
- Body-frame / relative geometry instead of brittle global headings
- Energy and aspect awareness (partially present, to be expanded)

## Current limitations

- Geometry is still approximate (flat-Earth lon/lat). Further body-frame work will help.
- Only guns, no missiles yet.
- Single residual matrix shared for simplicity; can be per-agent later.
- 3D viewer is matplotlib (functional but not as polished as Tacview).

## v3 — the brain is the pilot

`fly_dogfight_v3.py` removes the classical pursuit controller and the
residual-correction framing entirely. Instead, each aircraft is flown by
**4 frozen fly connectomes with specialized roles**:

- **2 perception brains** ("left eye" / "right eye") — real photoreceptors
  (`brain.visual` / `brain.azimuth`) are stimulated with a noisy,
  range-attenuated light bump at the enemy's true bearing. The brain isn't
  handed the bearing directly; it has to represent "where is it" in the
  spiking activity of its own `visual_projection` population (which
  includes the LC4/LPLC2 looming detectors), on its own side.
- **2 motor brains** (lateral / longitudinal) — receive the perception
  brains' activity through an evolved encoding matrix onto `ascending_neuron`
  cells (the real sensory-to-brain relay population), and their `vnc_motor`
  spiking activity (the fly's literal final motor output) is read through an
  evolved matrix straight into flight-control commands. Lateral → aileron +
  rudder. Longitudinal → throttle + elevator, plus its own altitude/speed
  proprioception.

Only the encode/readout matrices are trained (evolutionary strategy, same
dopamine-style reward as v2); the connectomes themselves stay frozen. The
only non-brain code left is a hard ground/stall safety envelope — it
*overrides*, it never blends with the brain's output, so the brain is
genuinely making the dogfighting decisions.

```bash
python fly_dogfight_v3.py
```

Cost: 8 connectome instances per fight (~6 ms/step each) instead of 2, so
expect roughly 4x the wall-clock time of v2 per episode. Expect noticeably
worse early performance than v2 too — real, decoded perception is a harder
problem than being handed ground-truth bearing.

## Next priorities

1. True body-frame relative state + closing-rate features
2. Longer curriculum (intercept → guns merge → defensive energy fight)
3. Population-based / self-play training
4. Tacview or better 3D export
5. Missile dynamics and BVR stages

## License

- Code: MIT
- Connectome: MaleCNS v1.0 (CC BY 4.0) – HHMI Janelia, Cambridge, Google Research
