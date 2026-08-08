/***************************
 Ion Dynamics Simulation Framework (IDSimF)

 Copyright 2020 - Physical and Theoretical Chemistry /
 Institute of Pure and Applied Mass Spectrometry
 of the University of Wuppertal, Germany

 IDSimF is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 IDSimF is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with IDSimF.  If not, see <https://www.gnu.org/licenses/>.

 ------------
 RFQDSim.cpp

 Idealized RFQ decelerator trajectory simulation with hard-sphere collisions.

 Coordinate and voltage conventions:
   * the beam axis is z;
   * r0 is the radial aperture (distance from the axis to a rod surface);
   * V_rf is the peak voltage difference between the two opposing rod pairs.

 At the positive RF peak, the x rods are at +V_rf/2 and the y rods are at
 -V_rf/2.  The ideal quadrupole potential is therefore

   Phi_rf = V_rf/(2*r0^2) * (x^2-y^2) * cos(omega*t + phase).

 ****************************/

#include "Core_particle.hpp"
#include "FileIO_trajectoryHDF5Writer.hpp"
#include "PSim_particleStartSplatTracker.hpp"
#include "Integration_parallelVerletIntegrator.hpp"
#include "CollisionModel_HardSphere.hpp"
#include "AppUtils_simulationConfiguration.hpp"
#include "AppUtils_ionDefinitionReading.hpp"
#include "AppUtils_logging.hpp"
#include "AppUtils_stopwatch.hpp"
#include "AppUtils_signalHandler.hpp"
#include "AppUtils_commandlineParser.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

enum IonDataRecordMode {
    FULL,
    SIMPLE
};

enum IonTerminationCode {
    ACTIVE_OR_TIMEOUT = 0,
    TRANSMITTED = 1,
    ENTRANCE_LOSS = 2,
    RADIAL_LOSS = 3,
    INVALID_STATE = 4
};

namespace {

constexpr const char* TERMINATION_CODE_ATTRIBUTE = "termination code";

void requirePositive(double value, const std::string& parameterName) {
    if (!std::isfinite(value) || !(value > 0.0)) {
        throw std::invalid_argument(parameterName + " must be finite and positive");
    }
}

void requireNonNegative(double value, const std::string& parameterName) {
    if (!std::isfinite(value) || !(value >= 0.0)) {
        throw std::invalid_argument(parameterName + " must be finite and non-negative");
    }
}

void requireFinite(double value, const std::string& parameterName) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(parameterName + " must be finite");
    }
}

} // namespace

int main(int argc, const char* argv[]) {

    try {
        AppUtils::CommandlineParser cmdLineParser(
                argc, argv, "RFQDSim",
                "Idealized RFQ decelerator trajectory simulation", true);
        AppUtils::logger_ptr logger = cmdLineParser.logger();
        AppUtils::simConf_ptr simConf = cmdLineParser.simulationConfiguration();

        // Basic simulation parameters
        const unsigned int timeSteps = simConf->unsignedIntParameter("sim_time_steps");
        const unsigned int trajectoryWriteInterval =
                simConf->unsignedIntParameter("trajectory_write_interval");
        const double dt = simConf->doubleParameter("dt");

        if (timeSteps == 0) {
            throw std::invalid_argument("sim_time_steps must be greater than zero");
        }
        if (trajectoryWriteInterval == 0) {
            throw std::invalid_argument("trajectory_write_interval must be greater than zero");
        }
        requirePositive(dt, "dt");

        // Physical and geometrical parameters
        const double spaceChargeFactor = simConf->doubleParameter("space_charge_factor");
        const double collisionGasMassAmu = simConf->doubleParameter("collision_gas_mass_amu");
        const double collisionGasDiameterM =
                simConf->doubleParameter("collision_gas_diameter_angstrom") * 1e-10;
        const double backgroundGasTemperatureK =
                simConf->doubleParameter("background_gas_temperature_K");
        const double backgroundGasPressurePa =
                simConf->doubleParameter("background_gas_pressure_Pa");

        const double vRf = simConf->doubleParameter("V_rf");
        const double frequencyRf = simConf->doubleParameter("frequency_rf");
        const double omegaRf = 2.0 * M_PI * frequencyRf;
        const double rfPhaseRad = simConf->isParameter("rf_phase_rad")
                ? simConf->doubleParameter("rf_phase_rad") : 0.0;

        const double r0M = simConf->doubleParameter("r0_m");
        const double rfqdZMinM = simConf->doubleParameter("rfqd_z_min_m");
        const double rfqdZMaxM = simConf->doubleParameter("rfqd_z_max_m");
        const double dcPotentialStartV = simConf->doubleParameter("dc_potential_start_V");
        const double dcPotentialEndV = simConf->doubleParameter("dc_potential_end_V");

        requireNonNegative(spaceChargeFactor, "space_charge_factor");
        requirePositive(collisionGasMassAmu, "collision_gas_mass_amu");
        requirePositive(collisionGasDiameterM, "collision_gas_diameter_angstrom");
        requirePositive(backgroundGasTemperatureK, "background_gas_temperature_K");
        requireNonNegative(backgroundGasPressurePa, "background_gas_pressure_Pa");
        requireNonNegative(vRf, "V_rf");
        requirePositive(frequencyRf, "frequency_rf");
        requireFinite(rfPhaseRad, "rf_phase_rad");
        requirePositive(r0M, "r0_m");
        requireFinite(rfqdZMinM, "rfqd_z_min_m");
        requireFinite(rfqdZMaxM, "rfqd_z_max_m");
        requireFinite(dcPotentialStartV, "dc_potential_start_V");
        requireFinite(dcPotentialEndV, "dc_potential_end_V");
        if (!(rfqdZMaxM > rfqdZMinM)) {
            throw std::invalid_argument("rfqd_z_max_m must be greater than rfqd_z_min_m");
        }

        const double inverseR0Squared = 1.0 / (r0M * r0M);
        const double axialPotentialSlope =
                (dcPotentialEndV - dcPotentialStartV) / (rfqdZMaxM - rfqdZMinM);
        const double axialElectricField = -axialPotentialSlope;

        // Particle trajectory output mode
        const std::string ionRecordModeString = simConf->stringParameter("record_mode");
        IonDataRecordMode ionRecordMode;
        if (ionRecordModeString == "full") {
            ionRecordMode = FULL;
        }
        else if (ionRecordModeString == "simple") {
            ionRecordMode = SIMPLE;
        }
        else {
            throw std::invalid_argument("record_mode must be either 'full' or 'simple'");
        }

        // Initialize ions
        std::vector<std::unique_ptr<Core::Particle>> particles;
        std::vector<Core::Particle*> particlePtrs;
        AppUtils::readIonDefinition(particles, particlePtrs, *simConf);

        if (particlePtrs.empty()) {
            throw std::invalid_argument("ion definition contains no particles");
        }

        std::size_t particleIndex = 0;
        for (Core::Particle* particle : particlePtrs) {
            particle->setIndex(particleIndex++);
            particle->setIntegerAttribute(TERMINATION_CODE_ATTRIBUTE, ACTIVE_OR_TIMEOUT);

            const Core::Vector position = particle->getLocation();
            const double radiusSquared =
                    position.x() * position.x() + position.y() * position.y();
            if (radiusSquared >= r0M * r0M ||
                    position.z() <= rfqdZMinM || position.z() >= rfqdZMaxM) {
                throw std::invalid_argument(
                        "all initial ion positions must be strictly inside the RFQD aperture");
            }
        }
        const std::size_t totalParticleCount = particlePtrs.size();

        // A position-dependent function is used even for the initial uniform-pressure
        // model.  Replacing this lambda with p(x,y,z) is the only collision-model change
        // needed for a future non-uniform gas-pressure implementation.
        auto backgroundGasPressureFunction =
                [backgroundGasPressurePa](Core::Vector& /*position*/) -> double {
                    return backgroundGasPressurePa;
                };
        auto backgroundGasVelocityFunction =
                [](Core::Vector& /*position*/) -> Core::Vector {
                    return Core::Vector(0.0, 0.0, 0.0);
                };

        CollisionModel::HardSphereModel hardSphereModel(
                backgroundGasPressureFunction,
                backgroundGasVelocityFunction,
                backgroundGasTemperatureK,
                collisionGasMassAmu,
                collisionGasDiameterM);

        // Ideal quadrupole RF field plus a uniform axial field generated by the
        // configured linear DC potential ramp.
        auto electricFieldFunction =
                [vRf, omegaRf, rfPhaseRad, inverseR0Squared, axialElectricField]
                (const Core::Vector& position, double time) -> Core::Vector {
                    const double rfFactor = vRf * inverseR0Squared
                            * std::cos(omegaRf * time + rfPhaseRad);
                    return Core::Vector(
                            -rfFactor * position.x(),
                            rfFactor * position.y(),
                            axialElectricField);
                };

        auto accelerationFunction =
                [spaceChargeFactor, ionRecordMode, &electricFieldFunction]
                (Core::Particle* particle, std::size_t /*particleIndex*/,
                        SpaceCharge::FieldCalculator& spaceChargeFieldCalculator,
                        double time, unsigned int /*timestep*/) -> Core::Vector {
                    const Core::Vector electricField =
                            electricFieldFunction(particle->getLocation(), time);

                    Core::Vector spaceChargeField(0.0, 0.0, 0.0);
                    if (spaceChargeFactor > 0.0) {
                        spaceChargeField =
                                spaceChargeFieldCalculator.getEFieldFromSpaceCharge(*particle)
                                * spaceChargeFactor;
                    }

                    if (ionRecordMode == FULL) {
                        particle->setFloatAttribute("field x", electricField.x());
                        particle->setFloatAttribute("field y", electricField.y());
                        particle->setFloatAttribute("field z", electricField.z());
                        particle->setFloatAttribute("space charge x", spaceChargeField.x());
                        particle->setFloatAttribute("space charge y", spaceChargeField.y());
                        particle->setFloatAttribute("space charge z", spaceChargeField.z());
                    }

                    return (electricField + spaceChargeField)
                            * particle->getCharge() / particle->getMass();
                };

        // Prepare trajectory writer
        auto hdf5Writer = std::make_unique<FileIO::TrajectoryHDF5Writer>(
                cmdLineParser.trajectoriesResultName());

        FileIO::partAttribTransformFctType particleAttributeTransformFunctionSimple =
                [&backgroundGasPressureFunction](Core::Particle* particle) -> std::vector<double> {
                    return {
                            particle->getVelocity().x(),
                            particle->getVelocity().y(),
                            particle->getVelocity().z(),
                            backgroundGasPressureFunction(particle->getLocation())
                    };
                };

        FileIO::partAttribTransformFctType particleAttributeTransformFunctionFull =
                [&backgroundGasPressureFunction](Core::Particle* particle) -> std::vector<double> {
                    return {
                            particle->getVelocity().x(),
                            particle->getVelocity().y(),
                            particle->getVelocity().z(),
                            backgroundGasPressureFunction(particle->getLocation()),
                            particle->getFloatAttribute("field x"),
                            particle->getFloatAttribute("field y"),
                            particle->getFloatAttribute("field z"),
                            particle->getFloatAttribute("space charge x"),
                            particle->getFloatAttribute("space charge y"),
                            particle->getFloatAttribute("space charge z")
                    };
                };

        if (ionRecordMode == FULL) {
            hdf5Writer->setParticleAttributes(
                    {"velocity x", "velocity y", "velocity z", "pressure Pa",
                     "electric field x", "electric field y", "electric field z",
                     "space charge x", "space charge y", "space charge z"},
                    particleAttributeTransformFunctionFull);
        }
        else {
            hdf5Writer->setParticleAttributes(
                    {"velocity x", "velocity y", "velocity z", "pressure Pa"},
                    particleAttributeTransformFunctionSimple);
        }

        FileIO::partAttribTransformFctTypeInteger integerParticleAttributeTransformFunction =
                [](Core::Particle* particle) -> std::vector<int> {
                    return {
                            particle->getIntegerAttribute("global index"),
                            particle->getIntegerAttribute(TERMINATION_CODE_ATTRIBUTE)
                    };
                };
        hdf5Writer->setParticleAttributes(
                {"global index", TERMINATION_CODE_ATTRIBUTE},
                integerParticleAttributeTransformFunction);

        hdf5Writer->writeTrajectoryAttribute(
                "RFQD parameter names",
                {"V_rf pair-to-pair peak [V]", "frequency_rf [Hz]", "rf_phase_rad [rad]",
                 "r0 [m]", "z_min [m]", "z_max [m]",
                 "dc_potential_start [V]", "dc_potential_end [V]"});
        hdf5Writer->writeTrajectoryAttribute(
                "RFQD parameter values",
                {vRf, frequencyRf, rfPhaseRad, r0M, rfqdZMinM, rfqdZMaxM,
                 dcPotentialStartV, dcPotentialEndV});
        hdf5Writer->writeTrajectoryAttribute(
                "termination code names",
                {"0 active_or_timeout", "1 transmitted", "2 entrance_loss",
                 "3 radial_loss", "4 invalid_state"});

        // Report the Mathieu |q| range implied by the explicit voltage convention.
        double minimumMathieuQ = std::numeric_limits<double>::max();
        double maximumMathieuQ = 0.0;
        for (const Core::Particle* particle : particlePtrs) {
            const double mathieuQ = std::abs(
                    2.0 * particle->getCharge() * vRf
                    / (particle->getMass() * r0M * r0M * omegaRf * omegaRf));
            minimumMathieuQ = std::min(minimumMathieuQ, mathieuQ);
            maximumMathieuQ = std::max(maximumMathieuQ, mathieuQ);
        }

        logger->info("RFQD convention: beam axis z; V_rf is pair-to-pair peak voltage");
        logger->info("RFQD aperture r0={:.6g} m; z=[{:.6g}, {:.6g}] m",
                     r0M, rfqdZMinM, rfqdZMaxM);
        logger->info("axial DC potential: {:.6g} V -> {:.6g} V; Ez={:.6g} V/m",
                     dcPotentialStartV, dcPotentialEndV, axialElectricField);
        logger->info("Mathieu |q| range for initialized ions: {:.6g} to {:.6g}",
                     minimumMathieuQ, maximumMathieuQ);

        // Particle start/splat tracker and termination counters
        ParticleSimulation::ParticleStartSplatTracker startSplatTracker;
        auto particleStartMonitoringFunction =
                [&startSplatTracker](Core::Particle* particle, double time) {
                    startSplatTracker.particleStart(particle, time);
                };

        std::atomic<std::size_t> ionsInactive{0};
        std::atomic<std::size_t> ionsTransmitted{0};
        std::atomic<std::size_t> ionsEntranceLoss{0};
        std::atomic<std::size_t> ionsRadialLoss{0};
        std::atomic<std::size_t> ionsInvalid{0};

        auto otherActionsFunction =
                [r0Squared = r0M * r0M, rfqdZMinM, rfqdZMaxM, dt,
                 &ionsInactive, &ionsTransmitted, &ionsEntranceLoss,
                 &ionsRadialLoss, &ionsInvalid, &startSplatTracker]
                (Core::Particle* particle, std::size_t /*particleIndex*/,
                        double time, unsigned int /*timestep*/) {
                    const Core::Vector position = particle->getLocation();
                    const double radiusSquared =
                            position.x() * position.x() + position.y() * position.y();

                    IonTerminationCode terminationCode = ACTIVE_OR_TIMEOUT;
                    std::atomic<std::size_t>* categoryCounter = nullptr;

                    if (!std::isfinite(position.x()) || !std::isfinite(position.y()) ||
                            !std::isfinite(position.z())) {
                        terminationCode = INVALID_STATE;
                        categoryCounter = &ionsInvalid;
                    }
                    else if (position.z() >= rfqdZMaxM) {
                        terminationCode = TRANSMITTED;
                        categoryCounter = &ionsTransmitted;
                    }
                    else if (position.z() <= rfqdZMinM) {
                        terminationCode = ENTRANCE_LOSS;
                        categoryCounter = &ionsEntranceLoss;
                    }
                    else if (radiusSquared >= r0Squared) {
                        terminationCode = RADIAL_LOSS;
                        categoryCounter = &ionsRadialLoss;
                    }

                    if (terminationCode != ACTIVE_OR_TIMEOUT) {
                        const double endOfStepTime = time + dt;
                        particle->setIntegerAttribute(
                                TERMINATION_CODE_ATTRIBUTE, terminationCode);
                        startSplatTracker.particleSplat(particle, endOfStepTime);
                        particle->setActive(false);
                        particle->setSplatTime(endOfStepTime);
                        categoryCounter->fetch_add(1, std::memory_order_relaxed);
                        ionsInactive.fetch_add(1, std::memory_order_relaxed);
                    }
                };

        auto postTimestepFunction =
                [trajectoryWriteInterval, ionRecordMode, totalParticleCount,
                 &electricFieldFunction, &ionsInactive, &ionsTransmitted,
                 &ionsEntranceLoss, &ionsRadialLoss, &ionsInvalid,
                 &hdf5Writer, &startSplatTracker, &logger]
                (Integration::AbstractTimeIntegrator* integrator,
                        std::vector<Core::Particle*>& activeAndInactiveParticles,
                        double time, unsigned int timestep, bool lastTimestep) {
                    if (ionsInactive.load(std::memory_order_relaxed) >= totalParticleCount) {
                        integrator->setTerminationState();
                    }

                    if (timestep == 0 && ionRecordMode == FULL) {
                        for (Core::Particle* particle : activeAndInactiveParticles) {
                            const Core::Vector electricField =
                                    electricFieldFunction(particle->getLocation(), time);
                            particle->setFloatAttribute("field x", electricField.x());
                            particle->setFloatAttribute("field y", electricField.y());
                            particle->setFloatAttribute("field z", electricField.z());
                            particle->setFloatAttribute("space charge x", 0.0);
                            particle->setFloatAttribute("space charge y", 0.0);
                            particle->setFloatAttribute("space charge z", 0.0);
                        }
                    }

                    if (lastTimestep) {
                        hdf5Writer->writeTimestep(activeAndInactiveParticles, time);
                        hdf5Writer->writeStartSplatData(startSplatTracker);

                        const std::size_t inactive = ionsInactive.load(std::memory_order_relaxed);
                        const std::size_t notTerminated = totalParticleCount - inactive;
                        hdf5Writer->writeNumericListDataset(
                                "RFQD termination counts",
                                std::vector<double>{
                                        static_cast<double>(notTerminated),
                                        static_cast<double>(ionsTransmitted.load(std::memory_order_relaxed)),
                                        static_cast<double>(ionsEntranceLoss.load(std::memory_order_relaxed)),
                                        static_cast<double>(ionsRadialLoss.load(std::memory_order_relaxed)),
                                        static_cast<double>(ionsInvalid.load(std::memory_order_relaxed))
                                });
                        hdf5Writer->finalizeTrajectory();

                        logger->info(
                                "finished ts:{} time:{:.6e}; transmitted:{} entrance_loss:{} "
                                "radial_loss:{} invalid:{} not_terminated:{}",
                                timestep, time,
                                ionsTransmitted.load(std::memory_order_relaxed),
                                ionsEntranceLoss.load(std::memory_order_relaxed),
                                ionsRadialLoss.load(std::memory_order_relaxed),
                                ionsInvalid.load(std::memory_order_relaxed), notTerminated);
                    }
                    else if (timestep % trajectoryWriteInterval == 0) {
                        logger->info(
                                "ts:{} time:{:.6e} born:{} inactive:{} transmitted:{}",
                                timestep, time, activeAndInactiveParticles.size(),
                                ionsInactive.load(std::memory_order_relaxed),
                                ionsTransmitted.load(std::memory_order_relaxed));
                        hdf5Writer->writeTimestep(activeAndInactiveParticles, time);
                    }
                };

        AppUtils::Stopwatch stopWatch;
        stopWatch.start();

        Integration::ParallelVerletIntegrator verletIntegrator(
                particlePtrs,
                accelerationFunction,
                postTimestepFunction,
                otherActionsFunction,
                particleStartMonitoringFunction,
                &hardSphereModel);
        AppUtils::SignalHandler::setReceiver(verletIntegrator);
        verletIntegrator.run(timeSteps, dt);

        stopWatch.stop();
        logger->info("CPU time: {} s", stopWatch.elapsedSecondsCPU());
        logger->info("Finished in {} seconds (wall clock time)",
                     stopWatch.elapsedSecondsWall());

        return EXIT_SUCCESS;
    }
    catch (AppUtils::TerminatedWhileCommandlineParsing& terminatedMessage) {
        return terminatedMessage.returnCode();
    }
    catch (const std::invalid_argument& exception) {
        std::cout << exception.what() << std::endl;
        return EXIT_FAILURE;
    }
}
