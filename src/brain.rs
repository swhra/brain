use std::collections::HashMap;
use std::simd::{f32x8, num::SimdFloat as _};
use std::str::FromStr;

use anyhow::{anyhow, Result};

use crate::drug::{Drug, Receptor};
use crate::program::{self, Program};

pub const SIMD_WIDTH: usize = 8;

#[derive(Debug, Clone, Copy, Default)]
struct ReceptorTickParams {
    total_drug_binding: f32,
    total_drug_activity_contrib: f32,
}

#[derive(Clone, Debug)]
enum Instruction {
    UpdateReceptor {
        activity_ptr: *mut f32,
        density_ptr: *mut f32,
        free_fraction_ptr: *mut f32,
        activity_setpoint_ptr: *mut f32,
        parent_level_ptr: *const f32,
        feedback_accumulator_ptr: *mut f32,
        nt_affinity_ki_recip: f32,
        plasticity_rate: f32,
        feedback_strength: f32,
        drug_params: Vec<(usize, f32, f32)>, // pk_idx, ki_recip, ia
    },
    UpdateSystem {
        level_ptr: *mut f32,
        feedback_accumulator_ptr: *const f32,
        synthesis_rate_variable: String,
        reuptake_eff: f32,
    },
}

#[derive(Debug)]
pub struct Brain {
    pub num_units: usize,
    receptor_update_plan: Vec<Instruction>,
    system_update_plan: Vec<Instruction>,
    metric_plans: HashMap<String, MetricPlan>,
    receptor_tick_params: Vec<ReceptorTickParams>,
}

impl Brain {
    pub fn new(
        protocol: &Program,
        drugs: &HashMap<String, Drug>,
        drug_order: &[String],
    ) -> Result<Self> {
        let num_units = (protocol.globals.synapses / SIMD_WIDTH) * SIMD_WIDTH;
        if num_units == 0 {
            return Err(anyhow!("synapses must be >= {}", SIMD_WIDTH));
        }

        let mut property_map: HashMap<String, usize> = HashMap::new();
        let mut num_properties = 0;
        for component in &protocol.components {
            property_map.insert(component.name.clone(), num_properties);
            match component.component_type.as_str() {
                "region" => {}
                "system" => num_properties += 2,
                "receptor" => num_properties += 4, // activity, density, free_fraction, activity_setpoint
                _ => anyhow::bail!("Unknown component type: {}", component.component_type),
            }
        }

        let mut properties = vec![0.0f32; num_properties * num_units];
        let base_ptr = properties.as_mut_ptr();

        for component in &protocol.components {
            let base_idx = *property_map.get(&component.name).unwrap();
            let start = base_idx * num_units;
            if component.component_type == "system" {
                properties[start..start + num_units]
                    .fill(*component.params.get("level").unwrap_or(&1.0));
                properties[start + num_units..start + 2 * num_units].fill(1.0);
            } else if component.component_type == "receptor" {
                properties[start + num_units..start + 2 * num_units].fill(1.0);
                properties[start + 3 * num_units..start + 4 * num_units].fill(0.0);
            }
        }

        let pk_model_indices: HashMap<_, _> =
            drug_order.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let mut receptor_update_plan = Vec::new();
        let mut system_update_plan = Vec::new();

        let receptor_components: Vec<_> =
            protocol.components.iter().filter(|c| c.component_type == "receptor").collect();
        for component in &receptor_components {
            let params = &component.params;
            let base_idx = *property_map.get(&component.name).unwrap();
            let parent_base_idx = *property_map.get(component.parent.as_ref().unwrap()).unwrap();

            let mut drug_params = vec![];
            let target_name = component.target.as_ref().unwrap_or(&component.name);
            if let Ok(receptor) = Receptor::from_str(target_name) {
                for (drug_name, drug) in drugs {
                    if let Some(target) = drug.targets.get(&receptor) {
                        drug_params.push((
                            *pk_model_indices.get(drug_name).unwrap(),
                            1.0 / target.affinity_nm,
                            target.intrinsic_activity,
                        ));
                    }
                }
            }

            let nt_affinity_ki =
                *params.get("affinity").unwrap_or(&1000.0);

            receptor_update_plan.push(Instruction::UpdateReceptor {
                activity_ptr: unsafe { base_ptr.add(base_idx * num_units) },
                density_ptr: unsafe { base_ptr.add((base_idx + 1) * num_units) },
                free_fraction_ptr: unsafe { base_ptr.add((base_idx + 2) * num_units) },
                activity_setpoint_ptr: unsafe { base_ptr.add((base_idx + 3) * num_units) },
                parent_level_ptr: unsafe { base_ptr.add(parent_base_idx * num_units) },
                feedback_accumulator_ptr: if component.receptor_type
                    == Some(program::ReceptorDef::Presynaptic)
                {
                    unsafe { base_ptr.add((parent_base_idx + 1) * num_units) }
                } else {
                    std::ptr::null_mut()
                },
                nt_affinity_ki_recip: 1.0 / nt_affinity_ki,
                plasticity_rate: *params.get("plasticity").unwrap_or(&1e-9),
                feedback_strength: *params.get("feedback").unwrap_or(&1.0),
                drug_params,
            });
        }

        let system_components: Vec<_> =
            protocol.components.iter().filter(|c| c.component_type == "system").collect();
        for component in system_components {
            let base_idx = *property_map.get(&component.name).unwrap();
            let synthesis_rate_variable =
                format!("{}_synthesis_rate", component.name.replace('.', "_"));

            system_update_plan.push(Instruction::UpdateSystem {
                level_ptr: unsafe { base_ptr.add(base_idx * num_units) },
                feedback_accumulator_ptr: unsafe { base_ptr.add((base_idx + 1) * num_units) },
                synthesis_rate_variable,
                reuptake_eff: *component.params.get("reuptake").unwrap_or(&0.02),
            });
        }

        let mut metric_plans = HashMap::new();
        for sensor in &protocol.sensors {
            metric_plans.insert(
                sensor.name.clone(),
                MetricPlan::new(&sensor.metric, &property_map, base_ptr, num_units)?,
            );
        }

        let receptor_tick_params = vec![ReceptorTickParams::default(); receptor_update_plan.len()];

        Ok(Self {
            num_units,
            receptor_update_plan,
            system_update_plan,
            metric_plans,
            receptor_tick_params,
        })
    }

    pub fn capture_activity_setpoints(&mut self) {
        for instruction in &self.receptor_update_plan {
            if let Instruction::UpdateReceptor { activity_ptr, activity_setpoint_ptr, .. } =
                instruction
            {
                unsafe {
                    let activity_slice = std::slice::from_raw_parts(*activity_ptr, self.num_units);
                    let setpoint_slice =
                        std::slice::from_raw_parts_mut(*activity_setpoint_ptr, self.num_units);
                    setpoint_slice.copy_from_slice(activity_slice);
                }
            }
        }
    }

    #[inline]
    pub fn tick(&mut self, drug_concentrations: &[f32], state: &HashMap<String, f32>) {
        let ones = f32x8::splat(1.0);
        let zeros = f32x8::splat(0.0);

        for instruction in &self.system_update_plan {
            if let Instruction::UpdateSystem { feedback_accumulator_ptr, .. } = instruction {
                let acc_slice = unsafe {
                    std::slice::from_raw_parts_mut(*feedback_accumulator_ptr as *mut _, self.num_units)
                };
                acc_slice.fill(1.0);
            }
        }

        for (idx, inst) in self.receptor_update_plan.iter().enumerate() {
            if let Instruction::UpdateReceptor { drug_params, .. } = inst {
                let mut params = ReceptorTickParams::default();
                for (pk_idx, ki_recip, ia) in drug_params {
                    let concentration = drug_concentrations[*pk_idx];
                    let drug_binding = concentration * *ki_recip;
                    params.total_drug_binding += drug_binding;
                    params.total_drug_activity_contrib += drug_binding * *ia;
                }
                self.receptor_tick_params[idx] = params;
            }
        }

        unsafe {
            for (instruction, tick_params) in
                self.receptor_update_plan.iter().zip(self.receptor_tick_params.iter())
            {
                if let Instruction::UpdateReceptor {
                    activity_ptr,
                    density_ptr,
                    free_fraction_ptr,
                    activity_setpoint_ptr,
                    parent_level_ptr,
                    feedback_accumulator_ptr,
                    nt_affinity_ki_recip,
                    plasticity_rate,
                    feedback_strength,
                    ..
                } = instruction
                {
                    let total_drug_binding = f32x8::splat(tick_params.total_drug_binding);
                    let total_drug_activity_contrib =
                        f32x8::splat(tick_params.total_drug_activity_contrib);
                    let nt_affinity_ki_recip_simd = f32x8::splat(*nt_affinity_ki_recip);

                    for i in (0..self.num_units).step_by(SIMD_WIDTH) {
                        let nt_levels =
                            f32x8::from_slice(std::slice::from_raw_parts(parent_level_ptr.add(i), SIMD_WIDTH));
                        let densities =
                            f32x8::from_slice(std::slice::from_raw_parts(density_ptr.add(i), SIMD_WIDTH));
                        let activity_setpoints =
                            f32x8::from_slice(std::slice::from_raw_parts(activity_setpoint_ptr.add(i), SIMD_WIDTH));

                        let nt_binding = nt_levels * nt_affinity_ki_recip_simd;
                        let denominator = ones + nt_binding + total_drug_binding;
                        let recip_denominator = ones / denominator;

                        let activities = (nt_binding + total_drug_activity_contrib)
                            * recip_denominator
                            * densities;
                        activities.copy_to_slice(std::slice::from_raw_parts_mut(activity_ptr.add(i), SIMD_WIDTH));

                        let free_fraction = recip_denominator;
                        free_fraction.copy_to_slice(std::slice::from_raw_parts_mut(free_fraction_ptr.add(i), SIMD_WIDTH));

                        let mut new_densities = densities
                            - (activities - activity_setpoints) * f32x8::splat(*plasticity_rate);
                        new_densities = new_densities.simd_clamp(f32x8::splat(0.2), f32x8::splat(3.0));
                        new_densities.copy_to_slice(std::slice::from_raw_parts_mut(density_ptr.add(i), SIMD_WIDTH));

                        if !feedback_accumulator_ptr.is_null() {
                            let mut accumulator = f32x8::from_slice(std::slice::from_raw_parts(feedback_accumulator_ptr.add(i), SIMD_WIDTH));
                            accumulator *= ones - (activities - activity_setpoints) * f32x8::splat(*feedback_strength);
                            accumulator.copy_to_slice(std::slice::from_raw_parts_mut(feedback_accumulator_ptr.add(i), SIMD_WIDTH));
                        }
                    }
                }
            }

            for instruction in &self.system_update_plan {
                if let Instruction::UpdateSystem {
                    level_ptr,
                    feedback_accumulator_ptr,
                    synthesis_rate_variable,
                    reuptake_eff,
                } = instruction
                {
                    let synthesis_rate = *state.get(synthesis_rate_variable).unwrap_or(&0.1);
                    for i in (0..self.num_units).step_by(SIMD_WIDTH) {
                        let mut levels =
                            f32x8::from_slice(std::slice::from_raw_parts(level_ptr.add(i), SIMD_WIDTH));
                        let feedback = f32x8::from_slice(std::slice::from_raw_parts(feedback_accumulator_ptr.add(i), SIMD_WIDTH));
                        levels += f32x8::splat(synthesis_rate) * feedback
                            - levels * f32x8::splat(*reuptake_eff);
                        levels = levels.simd_max(zeros);
                        levels.copy_to_slice(std::slice::from_raw_parts_mut(level_ptr.add(i), SIMD_WIDTH));
                    }
                }
            }
        }
    }

    #[inline]
    pub fn get_sensor_value(&self, sensor_name: &str) -> Result<f32> {
        let plan = self
            .metric_plans
            .get(sensor_name)
            .ok_or_else(|| anyhow!("sensor '{}' not found", sensor_name))?;
        let sum = plan.calculate_sum()?;
        let mean = sum / plan.count;
        if plan.subtract_from_one {
            Ok(1.0 - mean)
        } else {
            Ok(mean)
        }
    }
}

#[derive(Debug)]
struct MetricPlan {
    subtract_from_one: bool,
    prop1_ptr: *const f32,
    prop2_ptr: *const f32,
    count: f32,
}

impl MetricPlan {
    fn new(
        metric_str: &str,
        property_map: &HashMap<String, usize>,
        base_ptr: *mut f32,
        num_units: usize,
    ) -> Result<Self> {
        let (subtract_from_one, metric_to_parse) =
            if let Some(rest) = metric_str.strip_prefix("1.0 -") {
                (true, rest.trim())
            } else {
                (false, metric_str)
            };
        let (agg, inner_path) = metric_to_parse
            .split_once('(')
            .and_then(|(a, p)| p.strip_suffix(')').map(|p_inner| (a, p_inner)))
            .ok_or_else(|| anyhow!("Invalid metric format: '{}'", metric_to_parse))?;
        if agg != "mean" {
            return Err(anyhow!("Unsupported aggregator: {}", agg));
        }

        let (prop1_ptr, prop2_ptr) = if inner_path.contains('*') {
            let (path1_str, path2_str) = inner_path
                .split_once('*')
                .ok_or_else(|| anyhow!("Invalid multiplication metric: {}", inner_path))?;
            (
                utils::parse_metric_path(path1_str.trim(), property_map, base_ptr, num_units)?,
                utils::parse_metric_path(path2_str.trim(), property_map, base_ptr, num_units)?,
            )
        } else {
            (
                utils::parse_metric_path(inner_path, property_map, base_ptr, num_units)?,
                std::ptr::null(),
            )
        };

        Ok(Self { subtract_from_one, prop1_ptr, prop2_ptr, count: num_units as f32 })
    }

    #[inline]
    fn calculate_sum(&self) -> Result<f32> {
        let sum: f32 = unsafe {
            let count = self.count as usize;
            if self.prop2_ptr.is_null() {
                let data_slice = std::slice::from_raw_parts(self.prop1_ptr, count);
                data_slice
                    .chunks_exact(SIMD_WIDTH)
                    .map(f32x8::from_slice)
                    .sum::<f32x8>()
                    .reduce_sum()
            } else {
                let slice1 = std::slice::from_raw_parts(self.prop1_ptr, count);
                let slice2 = std::slice::from_raw_parts(self.prop2_ptr, count);
                slice1
                    .chunks_exact(SIMD_WIDTH)
                    .zip(slice2.chunks_exact(SIMD_WIDTH))
                    .map(|(a, b)| f32x8::from_slice(a) * f32x8::from_slice(b))
                    .sum::<f32x8>()
                    .reduce_sum()
            }
        };
        Ok(sum)
    }
}

mod utils {
    use super::*;
    pub(super) fn parse_metric_path(
        path_str: &str,
        property_map: &HashMap<String, usize>,
        base_ptr: *const f32,
        num_units: usize,
    ) -> Result<*const f32> {
        let (component_path, prop_name) = path_str
            .split_once("::")
            .ok_or_else(|| anyhow!("Metric path missing '::' in '{}'", path_str))?;
        let base_idx = *property_map
            .get(component_path)
            .ok_or_else(|| anyhow!("Component path '{}' not found", component_path))?;
        let prop_offset = match prop_name {
            "activity" => 0,
            "density" => 1,
            "free_fraction" => 2,
            _ => return Err(anyhow!("Unknown property: {}", prop_name)),
        };
        Ok(unsafe { base_ptr.add((base_idx + prop_offset) * num_units) })
    }
}