# RFQDSim ROOT tools

These tools keep the IDSimF HDF5 trajectory as the primary simulation output
and create a separate ROOT analysis file.

## Python dependencies

Activate the RFQD virtual environment and install `uproot` in addition to the
analysis dependencies:

```bash
source .venv-rfqd/bin/activate
python -m pip install -r applications/rfqd/RFQDSim/root_tools/requirements.txt
```

No PyROOT installation is required for the conversion itself. ROOT 6 is needed
to run the geometry macro.

## Convert HDF5 to ROOT

From the IDSimF repository root:

```bash
python applications/rfqd/RFQDSim/root_tools/convert_rfqd_hdf5_to_root.py \
  rfqd-ideal_trajectories.h5 \
  --config applications/rfqd/RFQDSim/example/idealRFQD.json \
  --output rfqd-ideal.root
```

The ROOT file contains three genuine TTrees:

- `trajectory`: one entry per particle and saved physical trajectory sample;
- `particle_summary`: one entry per particle;
- `run_config`: one entry containing the RFQD and run parameters.

By default, the converter removes repeated frozen samples after a particle has
terminated. It retains the first terminal sample and replaces its exported
frame time with the exact splat time. The original frame time remains in
`export_time_s`. Use `--keep-frozen-samples` only if an exact frame-by-frame
copy of the HDF5 representation is needed. The corresponding branch flags are
`is_active_sample`, `is_terminal_sample`, and `is_frozen_sample`.

Inspect the file with ROOT:

```bash
root -l rfqd-ideal.root
```

Examples:

```cpp
trajectory->Print();
particle_summary->Scan("global_index:termination_code:flight_time_s:kinetic_energy_final_eV");
trajectory->Draw("x_m*1e3:z_m*1e3", "termination_code<=1", "colz");
particle_summary->Draw("yprime_final_rad*1e3:y_final_m*1e3", "transmitted");
```

## Draw the analytic RFQD geometry

Interactive ROOT session:

```bash
root -l
```

```cpp
.L applications/rfqd/RFQDSim/root_tools/draw_rfqd_geometry.C+
draw_rfqd_geometry("rfqd-ideal.root", "rfqd-geometry.png", 12);
```

The left panel shows the four ideal hyperbolic equipotential surfaces extruded
along `z`, with up to 12 particle tracks. It can be rotated interactively. The
right panel shows the exact analytic `x-y` cross section, the `r0` termination
aperture, and initial/final transmitted particle positions.

This is the geometry of the analytic field model. It is not a mechanical model
of cylindrical rods. A mechanical view requires additional rod radius,
center-position, segmentation, and gap parameters.

## Output sampling for geometry plots

The supplied example uses `trajectory_write_interval = 100`, corresponding to
500 ns between saved frames and only two samples per 1 MHz RF period. For a
smooth trajectory visualization, make a dedicated run with
`trajectory_write_interval = 10` (50 ns, 20 samples per RF period). Keep a
larger interval for high-statistics transmission scans to control file size.
