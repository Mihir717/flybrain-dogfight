# FlyBrain Dogfight

Using the complete adult male fruit-fly central nervous system connectome (`flybrain` / MaleCNS) as a frozen spiking reservoir to control simulated fighter jets in 1v1 dogfights.

## Goal

- Multiple independent fly brains
- Learn to stay alive and kill the opponent
- Proper radar-like sensing, energy management, and pilot-like tactics
- Strong dopamine / hit-based reward for the readout
- Live or pre-recorded 3D visualization

## Current Status (Sep 2026)

- Full 166k-neuron connectome runs on CPU (Apple M4 tested)
- JSBSim F-16 dynamics
- Evolutionary training of a linear readout on top of the frozen connectome
- Hit-based ("dopamine") reward shaping
- Geometry / bearing sign issues still being stabilized (planes sometimes separate instead of closing)
- 2D and 3D matplotlib visualization prototypes

## Setup (Mac / Linux)

```bash
python -m venv flyjets
source flyjets/bin/activate   # or Windows equivalent
pip install flybrain jsbsim numpy matplotlib
```

First run of `FlyBrain` downloads ~260 MB of connectome data.

## Key Ideas

- Connectome stays **frozen** (true biological wiring)
- Only a small linear (or MLP) readout is trained
- Sensory injection into looming / feature-detector neurons (LC4, LPLC2, etc.)
- Descending neurons read out to throttle / aileron / elevator / rudder
- Reward: survival + closing range + nose-on + large bonus for gun hits

## Next Steps

1. Lock correct body-frame relative geometry and pure-pursuit / PN baseline that reliably closes.
2. Curriculum: intercept → guns-only merge → energy fight → full tactics.
3. Stronger residual learning by the flybrain on top of the classical baseline.
4. Better 3D visualization (or Tacview export).
5. Multi-agent self-play and population-based training.

## License

- Code: MIT
- Connectome data: MaleCNS v1.0 (CC BY 4.0) from HHMI Janelia / Cambridge / Google Research
