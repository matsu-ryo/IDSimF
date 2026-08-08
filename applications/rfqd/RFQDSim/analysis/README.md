# RFQDSim analysis

This script reads an RFQDSim HDF5 trajectory and produces:

- `rfqd_summary.txt`: transmission, loss counts, flight time, exit energy,
  beam size, angular spread, and geometric emittance;
- `rfqd_particle_summary.csv`: one row per particle;
- `rfqd_overview.png`: longitudinal motion, kinetic energy, and transverse
  trajectories;
- `rfqd_phase_space.png`: initial and transmitted `x-x'` and `y-y'` phase
  spaces;
- `rfqd_termination_counts.png`: termination classification.

Create a Python virtual environment from the IDSimF repository root:

```bash
python3 -m venv .venv-rfqd
source .venv-rfqd/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy h5py matplotlib
```

Analyze the supplied ideal example run:

```bash
python applications/rfqd/RFQDSim/analysis/analyze_rfqd.py \
  rfqd-ideal_trajectories.h5 \
  --config applications/rfqd/RFQDSim/example/idealRFQD.json \
  --output-dir rfqd-ideal-analysis
```

For a single-species trajectory, `--mass-amu 100` can be used instead of
`--config`. The current RFQDSim trajectory does not store ion mass, so one of
these two options is required to calculate kinetic energy.
