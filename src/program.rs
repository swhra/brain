use serde::Deserialize;
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
pub struct ActuatorDef {
    pub name: String,
    pub variable: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ComponentDef {
    pub name: String,
    #[serde(rename = "type")]
    pub component_type: String,
    pub parent: Option<String>,
    #[serde(rename = "receptor_type")]
    pub receptor_type: Option<ReceptorDef>,
    #[serde(default)]
    pub params: HashMap<String, f32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlLaw {
    Fixed { value: f32 },
    Pid { target: String, kp: f32, ki: f32, kd: f32, step: f32 },
}

#[derive(Clone, Debug, Deserialize)]
pub struct DrugParams {
    pub half_life_hours: f32,
    pub absorption_half_life_hours: f32,
    pub concentration_per_mg: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Modulator {
    pub actuator: String,
    pub sensor: Option<String>,
    pub law: ControlLaw,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Phase {
    pub description: String,
    pub duration_days: u64,
    pub modulators: Vec<Modulator>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Program {
    pub actuators: Vec<ActuatorDef>,
    pub components: Vec<ComponentDef>,
    pub drugs: HashMap<String, DrugParams>,
    pub globals: ProgramGlobals,
    pub phases: Vec<Phase>,
    pub sensors: Vec<SensorDef>,
    pub state: HashMap<String, f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProgramGlobals {
    pub num_synaptic_units: usize,
    pub log_interval_ticks: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceptorDef {
    Presynaptic,
    Postsynaptic,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SensorDef {
    pub name: String,
    pub metric: String,
}