#![feature(portable_simd)]

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

use anyhow::anyhow;
use indicatif::{ProgressBar, ProgressStyle};
use polars::prelude::*;
use signal_hook::consts::{SIGINFO, SIGUSR1};
use signal_hook::iterator::Signals;

mod brain;
mod drug;
mod program;

use brain::Brain;
use drug::{Drug, Logger, PIDController, PKModel};
use program::{ControlLaw, Program};

#[derive(Clone, Debug, Default)]
struct SharedState {
    tick: u64,
    day: u64,
    phase: String,
    concentrations: Vec<f32>, // No names, just raw values.
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        anyhow::bail!("usage: brain <protocol.yaml>");
    }

    let program_path = &args[1];
    eprintln!("\x1b[1;34m-> Loading protocol:\x1b[0m {}", program_path);
    let program: Program = serde_yaml::from_reader(std::fs::File::open(program_path)?)?;

    eprintln!("\x1b[1;34m-> Loading drug database:\x1b[0m ./data/drugs.parquet");
    let drug_df = ParquetReader::new(File::open("./data/drugs.parquet")?).finish()?;

    let mut logger = Logger::new();
    let mut runtime_state = program.state.clone();

    // Create a stable, sorted order for all drugs used in the simulation.
    let mut drug_order: Vec<String> = program.drugs.keys().cloned().collect();
    drug_order.sort();

    let mut pk_models: HashMap<String, PKModel> =
        program.drugs.iter().map(|(name, params)| (name.clone(), PKModel::new(params))).collect();

    let drugs: HashMap<String, Drug> =
        drug_order.iter().map(|name| (name.clone(), Drug::load(name, &drug_df).unwrap())).collect();

    eprintln!(
        "\x1b[1;34m-> Initializing brain with {} synaptic units...\x1b[0m",
        program.globals.num_synaptic_units
    );
    let mut brain = Brain::new(&program, &drugs, &drug_order)?;
    eprintln!("\x1b[32m-> Initialization complete. Starting simulation.\x1b[0m");
    eprintln!(
        "\x1b[36m   (Press Ctrl+T on macOS or `kill -USR1 {}` on Linux for status)\x1b[0m",
        std::process::id()
    );

    logger.log_header(&runtime_state, &program.sensors)?;

    // --- SETUP FOR ZERO-ALLOCATION LOOP & SIGNAL HANDLING ---
    let start_time = Instant::now();
    let initial_concentrations = drug_order.iter().map(|_| 0.0).collect::<Vec<_>>();
    let shared_state = Arc::new(Mutex::new(SharedState {
        concentrations: initial_concentrations,
        ..Default::default()
    }));

    let signal_state = Arc::clone(&shared_state);
    let signal_drug_order = drug_order.clone(); // Give the signal handler its own copy of names.
    let mut signals = Signals::new(&[SIGINFO, SIGUSR1])?;

    thread::spawn(move || {
        for _ in signals.forever() {
            let state = signal_state.lock().unwrap();
            let elapsed_ms = start_time.elapsed().as_millis();

            eprint!("\x1b[1;33m");
            eprint!("elapsed: {}ms\tphase: {}\tday: {}\ttick: {}",
                elapsed_ms, state.phase, state.day, state.tick
            );
            for (drug, conc) in state.concentrations.iter().zip(signal_drug_order.iter()) {
                eprint!("\t{}: {:.4}nM", drug, conc);
            }
            eprint!("\x1b[0m\n");
        }
    });

    let mut tick_counter: u64 = 0;
    let mut sensor_cache: HashMap<String, f32> = HashMap::new();

    // --- PRE-ALLOCATE THE VECTOR FOR DRUG CONCENTRATIONS ---
    // This is the single most important performance fix.
    let mut concentrations_ordered: Vec<f32> = vec![0.0; drug_order.len()];

    for (phase_idx, phase) in program.phases.iter().enumerate() {
        eprintln!(
            "\n\x1b[1;35m--- Phase {}/{} [{}]: {} days ---\x1b[0m",
            phase_idx + 1,
            program.phases.len(),
            phase.description,
            phase.duration_days
        );

        {
            let mut state = shared_state.lock().unwrap();
            state.phase = phase.description.clone();
        }

        let mut controllers: HashMap<String, PIDController> = HashMap::new();
        for modulator in &phase.modulators {
            if let ControlLaw::Pid { target, .. } = &modulator.law {
                let target_val = if target == "stabilize" {
                    *sensor_cache.get(modulator.sensor.as_ref().unwrap()).unwrap_or(&0.0)
                } else {
                    target.parse()?
                };
                controllers.insert(
                    modulator.actuator.clone(),
                    PIDController::new(target_val, &modulator.law),
                );
            }
        }

        let pb = ProgressBar::new(phase.duration_days);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} days ({eta})")?
            .progress_chars("#>-"));

        for day in 1..=phase.duration_days {
            {
                let mut state = shared_state.lock().unwrap();
                state.day = day;
            }

            for modulator in &phase.modulators {
                let actuator_def = program
                    .actuators
                    .iter()
                    .find(|a| a.name == modulator.actuator)
                    .ok_or_else(|| anyhow!("actuator '{}' not found", modulator.actuator))?;
                let current_val = *runtime_state.get(&actuator_def.variable).unwrap_or(&0.0);
                let new_val = match &modulator.law {
                    ControlLaw::Fixed { value } => *value,
                    ControlLaw::Pid { .. } => {
                        let sensor_def = program
                            .sensors
                            .iter()
                            .find(|s| s.name == *modulator.sensor.as_ref().unwrap())
                            .ok_or_else(|| {
                                anyhow!("sensor '{}' not found", modulator.sensor.as_ref().unwrap())
                            })?;
                        let sensor_val = brain.get_sensor_value(&sensor_def.name)?;
                        let controller = controllers.get_mut(&modulator.actuator).unwrap();
                        current_val + controller.calculate_adjustment(sensor_val)
                    }
                };
                runtime_state.insert(actuator_def.variable.clone(), new_val.clamp(0.0, 150.0));
            }

            for (drug_name, pk_model) in &mut pk_models {
                if let Some(dose) = runtime_state.get(&format!("{}_dose_mg", drug_name)) {
                    pk_model.add_dose(*dose / brain::TICKS_PER_DAY as f32);
                }
            }

            for _ in 0..brain::TICKS_PER_DAY {
                tick_counter += 1;

                // Update PK models and populate the pre-allocated vector. This is fast.
                for (i, drug_name) in drug_order.iter().enumerate() {
                    let pk_model = pk_models.get_mut(drug_name).unwrap();
                    pk_model.tick();
                    concentrations_ordered[i] = pk_model.concentration_nm;
                }

                brain.tick(&concentrations_ordered);

                if tick_counter % 1000 == 0 {
                    let mut state = shared_state.lock().unwrap();
                    state.tick = tick_counter;
                    // This is now a very fast copy of a few f32 values.
                    state.concentrations.copy_from_slice(&concentrations_ordered);
                }
                if tick_counter >= program.globals.log_interval_ticks {
                    if tick_counter % program.globals.log_interval_ticks == 0 {
                        let sensor_values: Vec<f32> = program
                            .sensors
                            .iter()
                            .map(|s| brain.get_sensor_value(&s.name).unwrap_or(0.0))
                            .collect();
                        logger.log_data(
                            tick_counter,
                            day,
                            &phase.description,
                            &runtime_state,
                            &sensor_values,
                        )?;
                    }
                }
            }
            pb.inc(1);
        }
        pb.finish_with_message("Phase complete.");

        logger.flush()?;
        for sensor in &program.sensors {
            sensor_cache.insert(sensor.name.clone(), brain.get_sensor_value(&sensor.name)?);
        }
    }

    eprintln!("\n\x1b[1;32mSimulation complete.\x1b[0m");
    Ok(())
}
