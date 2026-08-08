#!/usr/bin/env python3
"""Analyze an RFQDSim HDF5 trajectory and create summary plots/tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


AMU_KG = 1.66053906660e-27
ELEMENTARY_CHARGE_C = 1.602176634e-19
TERMINATION_LABELS = {
    0: "active_or_timeout",
    1: "transmitted",
    2: "entrance_loss",
    3: "radial_loss",
    4: "invalid_state",
}


def decode_names(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in values]


def scalar_attribute(attributes, name: str) -> int:
    return int(np.asarray(attributes[name]).reshape(-1)[0])


def read_trajectory(path: Path) -> dict:
    with h5py.File(path, "r") as h5_file:
        trajectory = h5_file["particle_trajectory"]
        n_steps = scalar_attribute(trajectory.attrs, "number of timesteps")
        times = np.asarray(trajectory["times"]).reshape(-1)[:n_steps]

        float_names = decode_names(trajectory.attrs.get("attributes names", []))
        int_names = decode_names(trajectory.attrs.get("integer attributes names", []))
        global_index_column = int_names.index("global index") if "global index" in int_names else None

        raw_frames = []
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
                dtype=int,
            )

            if global_index_column is None:
                global_indices = np.arange(len(positions), dtype=int)
            else:
                global_indices = int_attributes[:, global_index_column]
            if len(global_indices):
                largest_global_index = max(largest_global_index, int(np.max(global_indices)))
            raw_frames.append((positions, float_attributes, int_attributes, global_indices))

        if largest_global_index < 0:
            raise ValueError("The trajectory contains no particles")

        n_particles = largest_global_index + 1
        positions = np.full((n_steps, n_particles, 3), np.nan)
        float_attributes = np.full((n_steps, n_particles, len(float_names)), np.nan)
        int_attributes = np.full((n_steps, n_particles, len(int_names)), -1, dtype=int)

        for step_index, (frame_positions, frame_float, frame_int, global_indices) in enumerate(raw_frames):
            positions[step_index, global_indices, :] = frame_positions
            if len(float_names):
                float_attributes[step_index, global_indices, :] = frame_float
            if len(int_names):
                int_attributes[step_index, global_indices, :] = frame_int

        start_splat = {}
        if "start_splat" in trajectory:
            group = trajectory["start_splat"]
            dataset_mapping = {
                "start_times": "particle start times",
                "end_times": "particle splat times",
                "start_positions": "particle start locations",
                "end_positions": "particle splat locations",
            }
            for output_name, dataset_name in dataset_mapping.items():
                if dataset_name in group:
                    start_splat[output_name] = np.asarray(group[dataset_name]).squeeze()

        optional = {}
        if "optional_datasets" in trajectory:
            optional = {
                name: np.asarray(dataset)
                for name, dataset in trajectory["optional_datasets"].items()
            }

    return {
        "times": times,
        "positions": positions,
        "float_names": float_names,
        "float_attributes": float_attributes,
        "int_names": int_names,
        "int_attributes": int_attributes,
        "start_splat": start_splat,
        "optional": optional,
    }


def attribute(data: dict, name: str, integer: bool = False) -> np.ndarray:
    names_key = "int_names" if integer else "float_names"
    values_key = "int_attributes" if integer else "float_attributes"
    if name not in data[names_key]:
        raise KeyError(f"Required particle attribute is missing: {name}")
    return data[values_key][:, :, data[names_key].index(name)]


def resolve_masses_amu(data: dict, args, n_particles: int) -> np.ndarray:
    if "mass amu" in data["float_names"]:
        stored_masses = attribute(data, "mass amu")
        masses = np.full(n_particles, np.nan)
        for particle_index in range(n_particles):
            finite = stored_masses[:, particle_index][np.isfinite(stored_masses[:, particle_index])]
            if len(finite):
                masses[particle_index] = finite[-1]
        if np.all(np.isfinite(masses)):
            return masses

    if args.mass_amu is not None:
        return np.full(n_particles, args.mass_amu, dtype=float)

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        counts = np.asarray(config["n_ions"], dtype=int)
        group_masses = np.asarray(config["ion_masses"], dtype=float)
        if len(counts) != len(group_masses) or int(np.sum(counts)) != n_particles:
            raise ValueError("n_ions/ion_masses in the config do not match the HDF5 particle count")
        if len(group_masses) > 1 and float(config.get("ion_time_of_birth_range_s", 0.0)) > 0.0:
            raise ValueError(
                "Automatic mass assignment is ambiguous for multiple species with nonzero birth-time spread; "
                "store mass as an HDF5 attribute instead"
            )
        return np.repeat(group_masses, counts)

    raise ValueError("Specify --config or --mass-amu because this trajectory does not store ion mass")


def last_finite_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_particles = values.shape[1]
    result = np.full((n_particles,) + values.shape[2:], np.nan)
    last_indices = np.full(n_particles, -1, dtype=int)
    for particle_index in range(n_particles):
        finite_indices = np.flatnonzero(np.isfinite(values[:, particle_index, 0]))
        if len(finite_indices):
            last_indices[particle_index] = finite_indices[-1]
            result[particle_index] = values[finite_indices[-1], particle_index]
    return result, last_indices


def first_finite_values(values: np.ndarray) -> np.ndarray:
    n_particles = values.shape[1]
    result = np.full((n_particles,) + values.shape[2:], np.nan)
    for particle_index in range(n_particles):
        finite_indices = np.flatnonzero(np.isfinite(values[:, particle_index, 0]))
        if len(finite_indices):
            result[particle_index] = values[finite_indices[0], particle_index]
    return result


def geometric_emittance(position: np.ndarray, slope: np.ndarray) -> float:
    valid = np.isfinite(position) & np.isfinite(slope)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    covariance = np.cov(np.vstack((position[valid], slope[valid])), ddof=1)
    return float(np.sqrt(max(np.linalg.det(covariance), 0.0)))


def selected_particle_indices(n_particles: int, maximum: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_particles - 1, min(maximum, n_particles), dtype=int))


def save_overview(data: dict, energy_ev: np.ndarray, output_path: Path, max_tracks: int) -> None:
    times_us = data["times"] * 1e6
    positions = data["positions"]
    indices = selected_particle_indices(positions.shape[1], max_tracks)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    for particle_index in indices:
        axes[0, 0].plot(times_us, positions[:, particle_index, 2] * 1e3, lw=0.9, alpha=0.75)
    axes[0, 0].set(xlabel="time [µs]", ylabel="z [mm]", title="Longitudinal trajectories")

    median_energy = np.nanmedian(energy_ev, axis=1)
    lower_energy = np.nanpercentile(energy_ev, 16, axis=1)
    upper_energy = np.nanpercentile(energy_ev, 84, axis=1)
    axes[0, 1].fill_between(times_us, lower_energy, upper_energy, alpha=0.25, label="16–84%")
    axes[0, 1].plot(times_us, median_energy, lw=1.5, label="median")
    axes[0, 1].set(xlabel="time [µs]", ylabel="kinetic energy [eV]", title="Kinetic energy")
    axes[0, 1].legend()

    for particle_index in indices:
        axes[1, 0].plot(
            positions[:, particle_index, 2] * 1e3,
            positions[:, particle_index, 0] * 1e3,
            lw=0.8,
            alpha=0.7,
        )
    axes[1, 0].set(xlabel="z [mm]", ylabel="x [mm]", title="Transverse motion")

    radius = np.hypot(positions[:, :, 0], positions[:, :, 1])
    for particle_index in indices:
        axes[1, 1].plot(
            positions[:, particle_index, 2] * 1e3,
            radius[:, particle_index] * 1e3,
            lw=0.8,
            alpha=0.7,
        )
    axes[1, 1].set(xlabel="z [mm]", ylabel="r [mm]", title="Radial amplitude")

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_phase_space(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    final_positions: np.ndarray,
    final_velocities: np.ndarray,
    transmitted: np.ndarray,
    output_path: Path,
) -> None:
    initial_vz = initial_velocities[:, 2]
    final_vz = final_velocities[:, 2]
    initial_xp = np.divide(initial_velocities[:, 0], initial_vz,
                           out=np.full_like(initial_vz, np.nan), where=np.abs(initial_vz) > 0.0)
    initial_yp = np.divide(initial_velocities[:, 1], initial_vz,
                           out=np.full_like(initial_vz, np.nan), where=np.abs(initial_vz) > 0.0)
    final_xp = np.divide(final_velocities[:, 0], final_vz,
                         out=np.full_like(final_vz, np.nan), where=np.abs(final_vz) > 0.0)
    final_yp = np.divide(final_velocities[:, 1], final_vz,
                         out=np.full_like(final_vz, np.nan), where=np.abs(final_vz) > 0.0)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].scatter(initial_positions[:, 0] * 1e3, initial_xp * 1e3,
                    s=18, alpha=0.5, label="initial")
    axes[0].scatter(final_positions[transmitted, 0] * 1e3, final_xp[transmitted] * 1e3,
                    s=22, alpha=0.8, label="transmitted")
    axes[0].set(xlabel="x [mm]", ylabel="x′ [mrad]", title="Horizontal phase space")

    axes[1].scatter(initial_positions[:, 1] * 1e3, initial_yp * 1e3,
                    s=18, alpha=0.5, label="initial")
    axes[1].scatter(final_positions[transmitted, 1] * 1e3, final_yp[transmitted] * 1e3,
                    s=22, alpha=0.8, label="transmitted")
    axes[1].set(xlabel="y [mm]", ylabel="y′ [mrad]", title="Vertical phase space")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_termination_plot(termination_codes: np.ndarray, output_path: Path) -> None:
    codes = list(TERMINATION_LABELS)
    counts = [int(np.count_nonzero(termination_codes == code)) for code in codes]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([TERMINATION_LABELS[code] for code in codes], counts)
    axis.bar_label(bars)
    axis.set(ylabel="particles", title="RFQD termination classification")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path, help="RFQDSim *_trajectories.h5 file")
    parser.add_argument("--config", type=Path, help="RFQDSim JSON configuration file")
    parser.add_argument("--mass-amu", type=float, help="single ion mass in u (alternative to --config)")
    parser.add_argument("--output-dir", type=Path, help="output directory")
    parser.add_argument("--max-tracks", type=int, default=12, help="maximum trajectories drawn")
    args = parser.parse_args()

    if args.mass_amu is not None and args.mass_amu <= 0.0:
        parser.error("--mass-amu must be positive")
    if args.max_tracks <= 0:
        parser.error("--max-tracks must be positive")
    if not args.trajectory.is_file():
        parser.error(f"trajectory file not found: {args.trajectory}")

    output_dir = args.output_dir or Path(f"{args.trajectory.stem}_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_trajectory(args.trajectory)
    n_particles = data["positions"].shape[1]
    masses_amu = resolve_masses_amu(data, args, n_particles)

    velocities = np.stack(
        [attribute(data, "velocity x"), attribute(data, "velocity y"), attribute(data, "velocity z")],
        axis=2,
    )
    energy_ev = (
        0.5 * masses_amu[np.newaxis, :] * AMU_KG * np.sum(velocities**2, axis=2)
        / ELEMENTARY_CHARGE_C
    )

    termination_history = attribute(data, "termination code", integer=True)
    termination_codes = np.zeros(n_particles, dtype=int)
    for particle_index in range(n_particles):
        valid = termination_history[:, particle_index] >= 0
        if np.any(valid):
            termination_codes[particle_index] = termination_history[valid, particle_index][-1]

    initial_positions = first_finite_values(data["positions"])
    initial_velocities = first_finite_values(velocities)
    final_positions, last_indices = last_finite_values(data["positions"])
    final_velocities, _ = last_finite_values(velocities)
    final_energy = np.array([
        energy_ev[last_indices[index], index] if last_indices[index] >= 0 else np.nan
        for index in range(n_particles)
    ])

    transmitted = termination_codes == 1
    final_vz = final_velocities[:, 2]
    final_xp = np.divide(final_velocities[:, 0], final_vz,
                         out=np.full(n_particles, np.nan), where=np.abs(final_vz) > 0.0)
    final_yp = np.divide(final_velocities[:, 1], final_vz,
                         out=np.full(n_particles, np.nan), where=np.abs(final_vz) > 0.0)

    start_times = np.asarray(data["start_splat"].get("start_times", np.zeros(n_particles))).reshape(-1)
    end_times = np.asarray(data["start_splat"].get("end_times", np.full(n_particles, np.nan))).reshape(-1)
    if len(start_times) != n_particles:
        start_times = np.full(n_particles, np.nan)
    if len(end_times) != n_particles:
        end_times = np.full(n_particles, np.nan)
    flight_times = end_times - start_times

    counts = {code: int(np.count_nonzero(termination_codes == code)) for code in TERMINATION_LABELS}
    transmission_efficiency = counts[1] / n_particles

    summary_lines = [
        f"trajectory: {args.trajectory}",
        f"particles: {n_particles}",
        f"transmission efficiency: {transmission_efficiency:.6f} ({counts[1]}/{n_particles})",
    ]
    summary_lines.extend(
        f"termination {code} ({TERMINATION_LABELS[code]}): {counts[code]}"
        for code in TERMINATION_LABELS
    )
    if np.any(transmitted):
        summary_lines.extend([
            f"transmitted flight time [us], mean ± std: "
            f"{np.nanmean(flight_times[transmitted]) * 1e6:.6g} ± "
            f"{np.nanstd(flight_times[transmitted], ddof=1) * 1e6:.6g}",
            f"transmitted kinetic energy [eV], mean ± std: "
            f"{np.nanmean(final_energy[transmitted]):.6g} ± "
            f"{np.nanstd(final_energy[transmitted], ddof=1):.6g}",
            f"transmitted sigma_x / sigma_y [mm]: "
            f"{np.nanstd(final_positions[transmitted, 0], ddof=1) * 1e3:.6g} / "
            f"{np.nanstd(final_positions[transmitted, 1], ddof=1) * 1e3:.6g}",
            f"transmitted sigma_x' / sigma_y' [mrad]: "
            f"{np.nanstd(final_xp[transmitted], ddof=1) * 1e3:.6g} / "
            f"{np.nanstd(final_yp[transmitted], ddof=1) * 1e3:.6g}",
            f"transmitted geometric emittance x / y [mm mrad]: "
            f"{geometric_emittance(final_positions[transmitted, 0], final_xp[transmitted]) * 1e6:.6g} / "
            f"{geometric_emittance(final_positions[transmitted, 1], final_yp[transmitted]) * 1e6:.6g}",
        ])

    summary_path = output_dir / "rfqd_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    with (output_dir / "rfqd_particle_summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "global_index", "termination_code", "termination_label", "start_time_s", "end_time_s",
            "flight_time_s", "mass_amu", "x_final_m", "y_final_m", "z_final_m",
            "vx_final_m_per_s", "vy_final_m_per_s", "vz_final_m_per_s",
            "kinetic_energy_final_eV", "xprime_final_rad", "yprime_final_rad",
        ])
        for particle_index in range(n_particles):
            code = int(termination_codes[particle_index])
            writer.writerow([
                particle_index, code, TERMINATION_LABELS.get(code, "unknown"),
                start_times[particle_index], end_times[particle_index], flight_times[particle_index],
                masses_amu[particle_index], *final_positions[particle_index], *final_velocities[particle_index],
                final_energy[particle_index], final_xp[particle_index], final_yp[particle_index],
            ])

    save_overview(data, energy_ev, output_dir / "rfqd_overview.png", args.max_tracks)
    save_phase_space(
        initial_positions, initial_velocities, final_positions, final_velocities,
        transmitted, output_dir / "rfqd_phase_space.png",
    )
    save_termination_plot(termination_codes, output_dir / "rfqd_termination_counts.png")

    print("\n".join(summary_lines))
    print(f"outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
