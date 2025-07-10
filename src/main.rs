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
use program::{ActuatorDef, ControlLaw, Phase, Program};

#[derive(Clone, Debug, Default)]
struct SharedState {
    tick: u64,
    day: u64,
    phase: String,
    concentrations: Vec<f32>,
}

fn phase_introduces_drugs(phase: &Phase, actuators: &[ActuatorDef]) -> bool {
    let drug_actuator_vars: Vec<_> =
        actuators.iter().filter(|a| a.variable.ends_with("_dose_mg")).map(|a| &a.name).collect();

    for modulator in &phase.modulators {
        if drug_actuator_vars.contains(&&modulator.actuator) {
            if let ControlLaw::Fixed { value } = &modulator.law {
                if *value > 0.0 {
                    return true;
                }
            }
        }
    }
    false
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

    let mut drug_order: Vec<String> = program.drugs.keys().cloned().collect();
    drug_order.sort();

    let mut pk_models: HashMap<String, PKModel> =
        program.drugs.iter().map(|(name, params)| (name.clone(), PKModel::new(params))).collect();

    let drugs: HashMap<String, Drug> =
        drug_order.iter().map(|name| (name.clone(), Drug::load(name, &drug_df).unwrap())).collect();

    eprintln!(
        "\x1b[1;34m-> Initializing brain with {} synapses...\x1b[0m",
        program.globals.synapses
    );
    let mut brain = Brain::new(&program, &drugs, &drug_order)?;
    eprintln!("\x1b[32m-> Initialization complete. Starting simulation.\x1b[0m");
    eprintln!(
        "\x1b[36m   (Press Ctrl+T on macOS or `kill -USR1 {}` on Linux for status)\x1b[0m",
        std::process::id()
    );

    logger.log_header(&runtime_state, &program.sensors)?;

    let start_time = Instant::now();
    let initial_concentrations = drug_order.iter().map(|_| 0.0).collect::<Vec<_>>();
    let shared_state = Arc::new(Mutex::new(SharedState {
        concentrations: initial_concentrations,
        ..Default::default()
    }));

    let signal_state = Arc::clone(&shared_state);
    let signal_drug_order = drug_order.clone();
    let mut signals = Signals::new(&[SIGINFO, SIGUSR1])?;

    thread::spawn(move || {
        for _ in signals.forever() {
            let state = signal_state.lock().unwrap();
            let elapsed_ms = start_time.elapsed().as_millis();

            eprint!("\x1b[1;33m");
            eprint!(
                "elapsed: {}ms\tphase: {}\tday: {}\ttick: {}",
                elapsed_ms, state.phase, state.day, state.tick
            );
            for (drug, conc) in signal_drug_order.iter().zip(state.concentrations.iter()) {
                eprint!("\t{}: {:.4}nM", drug, conc);
            }
            eprint!("\x1b[0m\n");
        }
    });

    let mut tick_counter: u64 = 0;
    let mut concentrations_ordered: Vec<f32> = vec![0.0; drug_order.len()];
    let mut doses_ordered: Vec<f32> = vec![0.0; drug_order.len()];

    let mut sensor_cache: HashMap<String, f32> = HashMap::new();
    for sensor in &program.sensors {
        sensor_cache.insert(sensor.name.clone(), brain.get_sensor_value(&sensor.name)?);
    }

    let mut setpoints_captured = false;

    for (phase_idx, phase) in program.phases.iter().enumerate() {
        eprintln!(
            "\n\x1b[1;35m[{}/{}] {}: {} days\x1b[0m",
            phase_idx + 1,
            program.phases.len(),
            phase.description,
            phase.duration_days
        );

        if !setpoints_captured && phase_introduces_drugs(phase, &program.actuators) {
            eprintln!(
                "\x1b[1;34m-> First drug-taking phase detected. Capturing homeostatic setpoints...\x1b[0m"
            );
            brain.capture_activity_setpoints();
            setpoints_captured = true;
        }

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

                if let Some(current_val_ref) = runtime_state.get_mut(&actuator_def.variable) {
                    let new_val = match &modulator.law {
                        ControlLaw::Fixed { value } => *value,
                        ControlLaw::Pid { .. } => {
                            let sensor_name = modulator.sensor.as_ref().unwrap();
                            let sensor_val = *sensor_cache.get(sensor_name).ok_or_else(|| {
                                anyhow!("Sensor '{}' not found in cache", sensor_name)
                            })?;
                            let controller = controllers.get_mut(&modulator.actuator).unwrap();
                            *current_val_ref + controller.calculate_adjustment(sensor_val)
                        }
                    };
                    *current_val_ref = new_val;
                }
            }

            for (i, drug_name) in drug_order.iter().enumerate() {
                let dose_key = format!("{}_dose_mg", drug_name);
                doses_ordered[i] =
                    *runtime_state.get(&dose_key).unwrap_or(&0.0) / program.globals.tick_freq as f32;
            }

            let initial_tick_of_day = tick_counter;

            for _ in 0..program.globals.tick_freq {
                tick_counter += 1;

                for (i, drug_name) in drug_order.iter().enumerate() {
                    let pk_model = pk_models.get_mut(drug_name).unwrap();
                    pk_model.add_dose(doses_ordered[i]);
                    pk_model.tick();
                    concentrations_ordered[i] = pk_model.concentration_nm;
                }

                brain.tick(&concentrations_ordered, &runtime_state);

                if tick_counter % 1000 == 0 {
                    let mut state = shared_state.lock().unwrap();
                    state.tick = tick_counter;
                    state.concentrations.copy_from_slice(&concentrations_ordered);
                }
            }

            let logs_at_start_of_day = initial_tick_of_day / program.globals.log_freq;
            let logs_at_end_of_day = tick_counter / program.globals.log_freq;

            if logs_at_end_of_day > logs_at_start_of_day {
                let sensor_values: Vec<f32> = program
                    .sensors
                    .iter()
                    .map(|s| sensor_cache.get(&s.name).map(|v| *v).unwrap_or(0f32))
                    .collect();

                for (sensor, &value) in program.sensors.iter().zip(&sensor_values) {
                    sensor_cache.insert(sensor.name.clone(), value);
                }

                logger.log_data(
                    tick_counter,
                    day,
                    &phase.description,
                    &runtime_state,
                    &sensor_values,
                )?;
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
