use std::collections::HashMap;
use std::simd::{f32x8, num::SimdFloat as _};
use std::str::FromStr;

use anyhow::{Result, anyhow};

use crate::drug::{Drug, Receptor};
use crate::program::{self, Program};

pub const SIMD_WIDTH: usize = 8;
pub const TICKS_PER_DAY: u64 = 86400;

#[derive(Debug)]
pub struct Brain {
    pub num_units: usize,
    properties: Vec<f32>,
    receptor_update_plan: Vec<Instruction>,
    system_update_plan: Vec<Instruction>,
    metric_plans: HashMap<String, MetricPlan>,
}

impl Brain {
    pub fn new(
        protocol: &Program,
        drugs: &HashMap<String, Drug>,
        drug_order: &[String],
    ) -> Result<Self> {
        let num_units = (protocol.globals.num_synaptic_units / SIMD_WIDTH) * SIMD_WIDTH;
        if num_units == 0 {
            return Err(anyhow!("num_synaptic_units must be >= {}", SIMD_WIDTH));
        }

        let mut property_map: HashMap<String, usize> = HashMap::new();
        let mut num_properties = 0;
        for component in &protocol.components {
            property_map.insert(component.name.clone(), num_properties);
            match component.component_type.as_str() {
                "region" => {}
                "system" => num_properties += 2,
                "receptor" => num_properties += 3,
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
            }
        }

        let pk_model_indices: HashMap<_, _> =
            drug_order.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let mut receptor_update_plan = Vec::new();
        let mut system_update_plan = Vec::new();

        let receptor_components: Vec<_> =
            protocol.components.iter().filter(|c| c.component_type == "receptor").collect();
        for component in receptor_components {
            let params = &component.params;
            let base_idx = *property_map.get(&component.name).unwrap();
            let parent_base_idx = *property_map.get(component.parent.as_ref().unwrap()).unwrap();

            let mut drug_effects = vec![];
            if let Ok(receptor) = Receptor::from_str(&component.name) {
                for (drug_name, drug) in drugs {
                    if let Some(target) = drug.targets.get(&receptor) {
                        drug_effects.push((
                            *pk_model_indices.get(drug_name).unwrap(),
                            target.affinity_nm,
                            target.intrinsic_activity,
                        ));
                    }
                }
            }

            receptor_update_plan.push(Instruction::UpdateReceptor {
                activity_ptr: unsafe { base_ptr.add(base_idx * num_units) },
                density_ptr: unsafe { base_ptr.add((base_idx + 1) * num_units) },
                free_fraction_ptr: unsafe { base_ptr.add((base_idx + 2) * num_units) },
                parent_level_ptr: unsafe { base_ptr.add(parent_base_idx * num_units) },
                feedback_accumulator_ptr: if component.receptor_type
                    == Some(program::ReceptorDef::Presynaptic)
                {
                    unsafe { base_ptr.add((parent_base_idx + 1) * num_units) }
                } else {
                    std::ptr::null_mut()
                },
                nt_affinity_ki: *params.get("affinity").unwrap_or(&1000.0),
                plasticity_rate: *params.get("plasticity").unwrap_or(&1e-9),
                feedback_strength: *params.get("feedback").unwrap_or(&1.0),
                drug_effects,
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

        Ok(Self { num_units, properties, receptor_update_plan, system_update_plan, metric_plans })
    }

    #[inline(always)]
    pub fn tick(&mut self, drug_concentrations: &[f32], state: &HashMap<String, f32>) {
        let ones = f32x8::splat(1.0);
        let point_fives = f32x8::splat(0.5);
        let zeros = f32x8::splat(0.0);

        // PASS 0: INITIALIZE (Once per tick)
        // Reset all feedback accumulators in one pass. This is fast.
        for instruction in &self.system_update_plan {
            if let Instruction::UpdateSystem { feedback_accumulator_ptr, .. } = instruction {
                let acc_slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        *feedback_accumulator_ptr as *mut _,
                        self.num_units,
                    )
                };
                acc_slice.fill(1.0);
            }
        }

        // This `unsafe` block is safe because we've designed the passes to avoid data races.
        // Pass 1 reads system levels and writes to receptor properties.
        // Pass 2 reads receptor properties and writes to system levels.
        unsafe {
            // PASS 1: RECEPTORS (The "Up" Pass)
            for instruction in &self.receptor_update_plan {
                if let Instruction::UpdateReceptor {
                    activity_ptr,
                    density_ptr,
                    free_fraction_ptr,
                    parent_level_ptr,
                    feedback_accumulator_ptr,
                    nt_affinity_ki,
                    plasticity_rate,
                    feedback_strength,
                    drug_effects,
                } = instruction
                {
                    for i in (0..self.num_units).step_by(SIMD_WIDTH) {
                        let nt_levels = f32x8::from_slice(std::slice::from_raw_parts(
                            parent_level_ptr.add(i),
                            SIMD_WIDTH,
                        ));
                        let mut densities = f32x8::from_slice(std::slice::from_raw_parts(
                            density_ptr.add(i),
                            SIMD_WIDTH,
                        ));

                        let nt_binding = nt_levels / f32x8::splat(*nt_affinity_ki);

                        let mut total_drug_binding = zeros;
                        let mut total_drug_activity_contrib = zeros;
                        for (pk_idx, ki, ia) in drug_effects {
                            let drug_binding = f32x8::splat(drug_concentrations[*pk_idx] / *ki);
                            total_drug_binding += drug_binding;
                            total_drug_activity_contrib += drug_binding * f32x8::splat(*ia);
                        }

                        let denominator = ones + nt_binding + total_drug_binding;
                        let activities =
                            (nt_binding + total_drug_activity_contrib) / denominator * densities;
                        activities.copy_to_slice(std::slice::from_raw_parts_mut(
                            activity_ptr.add(i),
                            SIMD_WIDTH,
                        ));

                        let free_fraction = ones / denominator;
                        free_fraction.copy_to_slice(std::slice::from_raw_parts_mut(
                            free_fraction_ptr.add(i),
                            SIMD_WIDTH,
                        ));

                        densities -=
                            (activities - point_fives * densities) * f32x8::splat(*plasticity_rate);
                        densities = densities.simd_clamp(f32x8::splat(0.2), f32x8::splat(3.0));
                        densities.copy_to_slice(std::slice::from_raw_parts_mut(
                            density_ptr.add(i),
                            SIMD_WIDTH,
                        ));

                        if !feedback_accumulator_ptr.is_null() {
                            let mut accumulator = f32x8::from_slice(std::slice::from_raw_parts(
                                feedback_accumulator_ptr.add(i),
                                SIMD_WIDTH,
                            ));
                            accumulator *= ones
                                - (activities - point_fives * densities)
                                    * f32x8::splat(*feedback_strength);
                            accumulator.copy_to_slice(std::slice::from_raw_parts_mut(
                                feedback_accumulator_ptr.add(i),
                                SIMD_WIDTH,
                            ));
                        }
                    }
                }
            }

            // PASS 2: SYSTEMS (The "Down" Pass)
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
                        let mut levels = f32x8::from_slice(std::slice::from_raw_parts(
                            level_ptr.add(i),
                            SIMD_WIDTH,
                        ));
                        let feedback = f32x8::from_slice(std::slice::from_raw_parts(
                            feedback_accumulator_ptr.add(i),
                            SIMD_WIDTH,
                        ));
                        levels += f32x8::splat(synthesis_rate) * feedback
                            - levels * f32x8::splat(*reuptake_eff);
                        levels = levels.simd_max(zeros);
                        levels.copy_to_slice(std::slice::from_raw_parts_mut(
                            level_ptr.add(i),
                            SIMD_WIDTH,
                        ));
                    }
                }
            }
        }
    }

    pub fn get_sensor_value(&self, sensor_name: &str) -> Result<f32> {
        let plan = self
            .metric_plans
            .get(sensor_name)
            .ok_or_else(|| anyhow!("Sensor '{}' not found", sensor_name))?;
        let sum = plan.calculate_sum(&self.properties)?;
        let mean = sum / plan.count;
        if plan.subtract_from_one {
            Ok(1.0 - mean)
        } else {
            Ok(mean)
        }
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

#[derive(Clone, Debug)]
enum Instruction {
    UpdateReceptor {
        activity_ptr: *mut f32,
        density_ptr: *mut f32,
        drug_effects: Vec<(usize, f32, f32)>,
        feedback_accumulator_ptr: *mut f32,
        feedback_strength: f32,
        free_fraction_ptr: *mut f32,
        nt_affinity_ki: f32,
        parent_level_ptr: *const f32,
        plasticity_rate: f32,
    },
    UpdateSystem {
        feedback_accumulator_ptr: *const f32,
        level_ptr: *mut f32,
        reuptake_eff: f32,
        synthesis_rate_variable: String,
    },
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

    fn calculate_sum(&self, _properties: &[f32]) -> Result<f32> {
        let sum: f32 = unsafe {
            if self.prop2_ptr.is_null() {
                let data_slice = std::slice::from_raw_parts(self.prop1_ptr, self.count as usize);
                data_slice
                    .chunks_exact(SIMD_WIDTH)
                    .map(f32x8::from_slice)
                    .sum::<f32x8>()
                    .reduce_sum()
            } else {
                let slice1 = std::slice::from_raw_parts(self.prop1_ptr, self.count as usize);
                let slice2 = std::slice::from_raw_parts(self.prop2_ptr, self.count as usize);
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
