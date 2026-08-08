# RFQDSim

`RFQDSim` is an initial idealized RFQ decelerator model built on IDSimF. It
tracks ions through an analytic quadrupole RF field, a linear axial DC
potential, optional space charge, and hard-sphere collisions with a uniform
background gas.

## Coordinate and voltage convention

- The beam travels in the positive `z` direction.
- `r0_m` is the distance from the beam axis to the inner rod surface.
- `V_rf` is the **peak voltage difference between the two opposing rod pairs**.
- At the positive RF peak, the x rods are at `+V_rf/2` and the y rods are at
  `-V_rf/2`.

The analytic RF potential and field are

```text
Phi_rf = V_rf/(2 r0^2) (x^2 - y^2) cos(omega t + phase)
E_x    = -V_rf x/r0^2 cos(omega t + phase)
E_y    = +V_rf y/r0^2 cos(omega t + phase)
```

With this convention, the magnitude of the Mathieu parameter is

```text
|q| = |2 Q V_rf / (m r0^2 omega^2)|.
```

The DC potential varies linearly from `dc_potential_start_V` at
`rfqd_z_min_m` to `dc_potential_end_V` at `rfqd_z_max_m`. For a positive ion,
increasing the potential in the positive `z` direction produces deceleration.

## Output classification

Every particle has an integer `termination code` in the HDF5 trajectory:

| Code | Meaning |
| ---: | --- |
| 0 | Still active when the simulation time ended, or not yet born |
| 1 | Transmitted through `rfqd_z_max_m` |
| 2 | Returned through `rfqd_z_min_m` |
| 3 | Reached the radial aperture `r0_m` |
| 4 | Entered a non-finite numerical state |

The HDF5 optional dataset `RFQD termination counts` stores counts in this same
order. The local pressure is also recorded for every particle and output
timestep.

## Run the example

From the repository root after configuring `build-macos`:

```bash
cmake --build build-macos --target RFQDSim -j 8
./build-macos/applications/rfqd/RFQDSim/RFQDSim \
  ./build-macos/applications/rfqd/RFQDSim/example/idealRFQD.json \
  rfqd-ideal -n 8
```

Or run its smoke test:

```bash
ctest --test-dir build-macos -R app_rfqd_RFQDSim_ideal --output-on-failure
```

## Non-uniform pressure extension

The hard-sphere model is already constructed with a position-dependent
pressure callback, even though the current callback returns a constant. A
later implementation can replace that callback with an analytic `p(z)` or an
interpolated pressure map without changing the trajectory integrator or the
collision model itself.
