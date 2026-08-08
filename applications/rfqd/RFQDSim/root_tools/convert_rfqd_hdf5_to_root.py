#!/usr/bin/env python3
"""Convert an RFQDSim HDF5 trajectory to ROOT TTrees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import uproot


AMU_KG = 1.66053906660e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19

FLOAT_ATTRIBUTE_BRANCHES = {
    "velocity x": "vx_m_per_s",
    "velocity y": "vy_m_per_s",
    "velocity z": "vz_m_per_s",
    "pressure Pa": "pressure_Pa",
    "electric field x": "electric_field_x_V_per_m",
    "electric field y": "electric_field_y_V_per_m",
    "electric field z": "electric_field_z_V_per_m",
    "space charge x": "space_charge_field_x_V_per_m",
    "space charge y": "space_charge_field_y_V_per_m",
    "space charge z": "space_charge_field_z_V_per_m",
}


def decode_names(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values]


def scalar_attribute(attributes, name: str) -> int:
    return int(np.asarray(attributes[name]).reshape(-1)[0])


def read_hdf5_trajectory(path: Path) -> dict:
    with h5py.File(path, "r") as h5_file:
        trajectory = h5_file["particle_trajectory"]
        n_steps = scalar_attribute(trajectory.attrs, "number of timesteps")
        times = np.asarray(trajectory["times"], dtype=float).reshape(-1)[:n_steps]
        float_names = decode_names(trajectory.attrs.get("attributes names", []))
        int_names = decode_names(trajectory.attrs.get("integer attributes names", []))
        global_index_column = int_names.index("global index") if "global index" in int_names else None

        frames = []
        largest_global_index = -1
        for step_index in range(n_steps):
            frame = trajectory["timesteps"][str(step_index)]
            positions = np.asarray(frame.get("positions", np.empty((0, 3))), dtype=float)
            float_attributes = np.asarray(
                frame.get("particle_attributes_float", np.empty((len(positions), 0))),
                dtype=float,
            )
            int_attributes = np.asarray(
                frame.get("particle_attributes_integer", np.empty((len(positions), 0))),
                dtype=np.int32,
            )
            if global_index_column is None:
                global_indices = np.arange(len(positions), dtype=np.int32)
            else:
                global_indices = int_attributes[:, global_index_column]
            if len(global_indices):
                largest_global_index = max(largest_global_index, int(np.max(global_indices)))
            frames.append((positions, float_attributes, int_attributes, global_indices))

        if largest_global_index < 0:
            raise ValueError("The HDF5 trajectory contains no particles")

        n_particles = largest_global_index + 1
        positions = np.full((n_steps, n_particles, 3), np.nan, dtype=np.float64)
        float_attributes = np.full(
            (n_steps, n_particles, len(float_names)), np.nan, dtype=np.float64)
        int_attributes = np.full(
            (n_steps, n_particles, len(int_names)), -1, dtype=np.int32)

        for step_index, (frame_positions, frame_float, frame_int, global_indices) in enumerate(frames):
            positions[step_index, global_indices, :] = frame_positions
            if len(float_names):
                float_attributes[step_index, global_indices, :] = frame_float
            if len(int_names):
                int_attributes[step_index, global_indices, :] = frame_int

        start_splat = {}
        if "start_splat" in trajectory:
            group = trajectory["start_splat"]
            datasets = {
                "start_times": "particle start times",
                "end_times": "particle splat times",
                "start_positions": "particle start locations",
                "end_positions": "particle splat locations",
            }
            for output_name, dataset_name in datasets.items():
                if dataset_name in group:
                    start_splat[output_name] = np.asarray(group[dataset_name], dtype=float)

    return {
        "times": times,
        "positions": positions,
        "float_names": float_names,
        "float_attributes": float_attributes,
        "int_names": int_names,
        "int_attributes": int_attributes,
        "start_splat": start_splat,
    }


def float_attribute(data: dict, name: str) -> np.ndarray:
    if name not in data["float_names"]:
        return np.full(data["positions"].shape[:2], np.nan, dtype=np.float64)
    column = data["float_names"].index(name)
    return data["float_attributes"][:, :, column]


def integer_attribute(data: dict, name: str) -> np.ndarray:
    if name not in data["int_names"]:
        raise KeyError(f"Required integer particle attribute is missing: {name}")
    column = data["int_names"].index(name)
    return data["int_attributes"][:, :, column]


def read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def particle_species(config: dict, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray(config["n_ions"], dtype=np.int32)
    masses_amu = np.asarray(config["ion_masses"], dtype=np.float64)
    charges_e = np.asarray(config["ion_charges"], dtype=np.float64)

    if len(counts) != len(masses_amu) or len(counts) != len(charges_e):
        raise ValueError("n_ions, ion_masses, and ion_charges must have the same length")
    if int(np.sum(counts)) != n_particles:
        raise ValueError("The configuration particle count does not match the HDF5 trajectory")
    if len(counts) > 1 and float(config.get("ion_time_of_birth_range_s", 0.0)) > 0.0:
        raise ValueError(
            "Global-index species assignment is ambiguous for multiple species with a nonzero "
            "birth-time spread"
        )
    return np.repeat(masses_amu, counts), np.repeat(charges_e, counts)


def final_integer_values(history: np.ndarray, default: int = 0) -> np.ndarray:
    result = np.full(history.shape[1], default, dtype=np.int32)
    for particle_index in range(history.shape[1]):
        valid = history[:, particle_index] >= 0
        if np.any(valid):
            result[particle_index] = history[valid, particle_index][-1]
    return result


def first_and_last_vectors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_particles = values.shape[1]
    first = np.full((n_particles, values.shape[2]), np.nan, dtype=np.float64)
    last = np.full_like(first, np.nan)
    last_indices = np.full(n_particles, -1, dtype=np.int32)
    for particle_index in range(n_particles):
        valid_indices = np.flatnonzero(np.isfinite(values[:, particle_index, 0]))
        if len(valid_indices):
            first[particle_index] = values[valid_indices[0], particle_index]
            last[particle_index] = values[valid_indices[-1], particle_index]
            last_indices[particle_index] = valid_indices[-1]
    return first, last, last_indices


def start_splat_array(data: dict, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in data["start_splat"]:
        return np.full(shape, np.nan, dtype=np.float64)
    values = np.asarray(data["start_splat"][name], dtype=np.float64).reshape(shape)
    return values


def make_physical_sample_mask(
    positions: np.ndarray,
    termination_history: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep all active samples and only the first frozen terminal sample."""
    mask = np.isfinite(positions[:, :, 0])
    terminal_sample = np.zeros(mask.shape, dtype=np.bool_)
    for particle_index in range(mask.shape[1]):
        terminal_frames = np.flatnonzero(termination_history[:, particle_index] > 0)
        if len(terminal_frames):
            first_terminal_frame = terminal_frames[0]
            mask[first_terminal_frame + 1:, particle_index] = False
            terminal_sample[first_terminal_frame, particle_index] = True
    return mask, terminal_sample


def flatten(values: np.ndarray, mask: np.ndarray, dtype=None) -> np.ndarray:
    result = np.asarray(values)[mask]
    return result.astype(dtype, copy=False) if dtype is not None else result


def write_tree(root_file, name: str, branches: dict[str, np.ndarray]) -> None:
    lengths = {len(values) for values in branches.values()}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent branch lengths for TTree {name}: {lengths}")
    tree = root_file.mktree(name, {branch: values.dtype for branch, values in branches.items()})
    tree.extend(branches)


def config_tree_branches(config: dict) -> dict[str, np.ndarray]:
    float_keys = [
        "dt", "space_charge_factor", "frequency_rf", "V_rf", "rf_phase_rad", "r0_m",
        "rfqd_z_min_m", "rfqd_z_max_m", "dc_potential_start_V", "dc_potential_end_V",
        "collision_gas_mass_amu", "collision_gas_diameter_angstrom",
        "background_gas_temperature_K", "background_gas_pressure_Pa",
    ]
    int_keys = ["sim_time_steps", "trajectory_write_interval"]
    branches = {
        key: np.asarray([float(config.get(key, 0.0))], dtype=np.float64)
        for key in float_keys
    }
    branches.update({
        key: np.asarray([int(config.get(key, 0))], dtype=np.int32)
        for key in int_keys
    })
    branches["n_particles"] = np.asarray([int(np.sum(config["n_ions"]))], dtype=np.int32)
    return branches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="RFQDSim *_trajectories.h5 file")
    parser.add_argument("--config", type=Path, required=True, help="RFQDSim JSON configuration")
    parser.add_argument("--output", "-o", type=Path, required=True, help="output ROOT file")
    parser.add_argument(
        "--keep-frozen-samples",
        action="store_true",
        help="keep repeated post-termination samples from the HDF5 frames",
    )
    args = parser.parse_args()

    if not args.trajectory.is_file():
        parser.error(f"trajectory file not found: {args.trajectory}")
    if not args.config.is_file():
        parser.error(f"configuration file not found: {args.config}")

    config = read_config(args.config)
    data = read_hdf5_trajectory(args.trajectory)
    times = data["times"]
    positions = data["positions"]
    n_steps, n_particles, _ = positions.shape
    masses_amu, charges_e = particle_species(config, n_particles)

    velocities = np.stack([
        float_attribute(data, "velocity x"),
        float_attribute(data, "velocity y"),
        float_attribute(data, "velocity z"),
    ], axis=2)
    termination_history = integer_attribute(data, "termination code")
    termination_codes = final_integer_values(termination_history)

    physical_sample_mask, terminal_sample = make_physical_sample_mask(
        positions, termination_history)
    if args.keep_frozen_samples:
        sample_mask = np.isfinite(positions[:, :, 0])
    else:
        sample_mask = physical_sample_mask
    frozen_sample = (termination_history > 0) & ~terminal_sample

    export_time_grid = np.broadcast_to(times[:, np.newaxis], (n_steps, n_particles)).copy()
    physical_time_grid = export_time_grid.copy()
    start_times = start_splat_array(data, "start_times", (n_particles,))
    end_times = start_splat_array(data, "end_times", (n_particles,))
    for particle_index in range(n_particles):
        terminal_frames = np.flatnonzero(terminal_sample[:, particle_index] & sample_mask[:, particle_index])
        if len(terminal_frames) and np.isfinite(end_times[particle_index]):
            physical_time_grid[terminal_frames[0], particle_index] = end_times[particle_index]

    mass_grid = np.broadcast_to(masses_amu[np.newaxis, :], (n_steps, n_particles))
    charge_grid = np.broadcast_to(charges_e[np.newaxis, :], (n_steps, n_particles))
    speed_squared = np.sum(velocities**2, axis=2)
    kinetic_energy_ev = 0.5 * mass_grid * AMU_KG * speed_squared / ELEMENTARY_CHARGE_C
    radius = np.hypot(positions[:, :, 0], positions[:, :, 1])
    xprime = np.divide(
        velocities[:, :, 0], velocities[:, :, 2],
        out=np.full((n_steps, n_particles), np.nan),
        where=np.abs(velocities[:, :, 2]) > 0.0,
    )
    yprime = np.divide(
        velocities[:, :, 1], velocities[:, :, 2],
        out=np.full((n_steps, n_particles), np.nan),
        where=np.abs(velocities[:, :, 2]) > 0.0,
    )

    frequency_rf = float(config["frequency_rf"])
    rf_phase_offset = float(config.get("rf_phase_rad", 0.0))
    rf_phase = np.mod(2.0 * np.pi * frequency_rf * physical_time_grid + rf_phase_offset, 2.0 * np.pi)
    z_min = float(config["rfqd_z_min_m"])
    z_max = float(config["rfqd_z_max_m"])
    dc_start = float(config["dc_potential_start_V"])
    dc_end = float(config["dc_potential_end_V"])
    dc_potential = dc_start + (dc_end - dc_start) * (positions[:, :, 2] - z_min) / (z_max - z_min)

    particle_index_grid = np.broadcast_to(
        np.arange(n_particles, dtype=np.int32)[np.newaxis, :], (n_steps, n_particles))
    step_index_grid = np.broadcast_to(
        np.arange(n_steps, dtype=np.int32)[:, np.newaxis], (n_steps, n_particles))

    trajectory_branches = {
        "global_index": flatten(particle_index_grid, sample_mask, np.int32),
        "timestep_index": flatten(step_index_grid, sample_mask, np.int32),
        "time_s": flatten(physical_time_grid, sample_mask, np.float64),
        "export_time_s": flatten(export_time_grid, sample_mask, np.float64),
        "x_m": flatten(positions[:, :, 0], sample_mask, np.float64),
        "y_m": flatten(positions[:, :, 1], sample_mask, np.float64),
        "z_m": flatten(positions[:, :, 2], sample_mask, np.float64),
        "r_m": flatten(radius, sample_mask, np.float64),
        "mass_amu": flatten(mass_grid, sample_mask, np.float64),
        "charge_e": flatten(charge_grid, sample_mask, np.float64),
        "kinetic_energy_eV": flatten(kinetic_energy_ev, sample_mask, np.float64),
        "xprime_rad": flatten(xprime, sample_mask, np.float64),
        "yprime_rad": flatten(yprime, sample_mask, np.float64),
        "rf_phase_rad": flatten(rf_phase, sample_mask, np.float64),
        "dc_potential_V": flatten(dc_potential, sample_mask, np.float64),
        "termination_code": flatten(termination_history, sample_mask, np.int32),
        "is_active_sample": flatten(termination_history == 0, sample_mask, np.bool_),
        "is_terminal_sample": flatten(terminal_sample, sample_mask, np.bool_),
        "is_frozen_sample": flatten(frozen_sample, sample_mask, np.bool_),
    }
    for hdf5_name, root_name in FLOAT_ATTRIBUTE_BRANCHES.items():
        trajectory_branches[root_name] = flatten(
            float_attribute(data, hdf5_name), sample_mask, np.float64)

    first_positions, final_positions, last_indices = first_and_last_vectors(positions)
    first_velocities, final_velocities, _ = first_and_last_vectors(velocities)
    hdf5_start_positions = start_splat_array(data, "start_positions", (n_particles, 3))
    hdf5_end_positions = start_splat_array(data, "end_positions", (n_particles, 3))
    valid_start_positions = np.all(np.isfinite(hdf5_start_positions), axis=1)
    first_positions[valid_start_positions] = hdf5_start_positions[valid_start_positions]
    terminated = termination_codes > 0
    valid_end_positions = terminated & np.all(np.isfinite(hdf5_end_positions), axis=1)
    final_positions[valid_end_positions] = hdf5_end_positions[valid_end_positions]

    final_energy = np.asarray([
        kinetic_energy_ev[last_indices[index], index] if last_indices[index] >= 0 else np.nan
        for index in range(n_particles)
    ], dtype=np.float64)
    final_xprime = np.divide(
        final_velocities[:, 0], final_velocities[:, 2],
        out=np.full(n_particles, np.nan), where=np.abs(final_velocities[:, 2]) > 0.0,
    )
    final_yprime = np.divide(
        final_velocities[:, 1], final_velocities[:, 2],
        out=np.full(n_particles, np.nan), where=np.abs(final_velocities[:, 2]) > 0.0,
    )
    summary_end_times = end_times.copy()
    summary_end_times[~terminated] = np.nan

    particle_summary_branches = {
        "global_index": np.arange(n_particles, dtype=np.int32),
        "mass_amu": masses_amu.astype(np.float64),
        "charge_e": charges_e.astype(np.float64),
        "termination_code": termination_codes.astype(np.int32),
        "transmitted": (termination_codes == 1).astype(np.bool_),
        "start_time_s": start_times.astype(np.float64),
        "end_time_s": summary_end_times.astype(np.float64),
        "flight_time_s": (summary_end_times - start_times).astype(np.float64),
        "x_start_m": first_positions[:, 0].astype(np.float64),
        "y_start_m": first_positions[:, 1].astype(np.float64),
        "z_start_m": first_positions[:, 2].astype(np.float64),
        "vx_start_m_per_s": first_velocities[:, 0].astype(np.float64),
        "vy_start_m_per_s": first_velocities[:, 1].astype(np.float64),
        "vz_start_m_per_s": first_velocities[:, 2].astype(np.float64),
        "x_final_m": final_positions[:, 0].astype(np.float64),
        "y_final_m": final_positions[:, 1].astype(np.float64),
        "z_final_m": final_positions[:, 2].astype(np.float64),
        "vx_final_m_per_s": final_velocities[:, 0].astype(np.float64),
        "vy_final_m_per_s": final_velocities[:, 1].astype(np.float64),
        "vz_final_m_per_s": final_velocities[:, 2].astype(np.float64),
        "kinetic_energy_final_eV": final_energy,
        "xprime_final_rad": final_xprime,
        "yprime_final_rad": final_yprime,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(args.output, compression=uproot.ZLIB(4)) as root_file:
        write_tree(root_file, "trajectory", trajectory_branches)
        write_tree(root_file, "particle_summary", particle_summary_branches)
        write_tree(root_file, "run_config", config_tree_branches(config))

    print(f"wrote: {args.output.resolve()}")
    print(f"trajectory entries: {len(trajectory_branches['global_index'])}")
    print(f"particle_summary entries: {n_particles}")
    print("TTrees: trajectory, particle_summary, run_config")


if __name__ == "__main__":
    main()
