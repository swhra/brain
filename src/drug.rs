use std::collections::HashMap;
use std::io::{BufWriter, Write, stdout};
use std::str::FromStr;

use anyhow::{Result, anyhow};
use polars::prelude::*;
use strum_macros::EnumString;

use crate::program::{ControlLaw, DrugParams};

#[derive(Clone, Debug)]
pub struct Drug {
    pub targets: HashMap<Receptor, DrugTarget>,
}

impl Drug {
    pub fn load(drug_name: &str, df: &DataFrame) -> Result<Self> {
        let filtered = df.clone().lazy().filter(col("name").eq(lit(drug_name))).collect()?;

        if filtered.is_empty() {
            return Err(anyhow!("Drug '{}' not found in database.", drug_name));
        }

        let mut targets = HashMap::new();
        let target_series = filtered.column("target")?.str()?;

        let ki_series = filtered.column("ki").ok().and_then(|s| s.f32().ok());
        let kd_series = filtered.column("kd").ok().and_then(|s| s.f32().ok());
        let ic50_series = filtered.column("ic50").ok().and_then(|s| s.f32().ok());
        let ec50_series = filtered.column("ec50").ok().and_then(|s| s.f32().ok());
        let ac50_series = filtered.column("ac50").ok().and_then(|s| s.f32().ok());
        let potency_series = filtered.column("potency").ok().and_then(|s| s.f32().ok());

        let efficacy_series = filtered.column("efficacy").ok().and_then(|s| s.f32().ok());
        let emax_series = filtered.column("emax").ok().and_then(|s| s.f32().ok());

        for i in 0..filtered.height() {
            if let Some(target_str) = target_series.get(i) {
                if let Ok(receptor) = Receptor::from_str(target_str) {
                    let best_affinity_nm = ki_series
                        .as_ref()
                        .and_then(|s| s.get(i))
                        .or_else(|| kd_series.as_ref().and_then(|s| s.get(i)))
                        .or_else(|| ic50_series.as_ref().and_then(|s| s.get(i)))
                        .or_else(|| ec50_series.as_ref().and_then(|s| s.get(i)))
                        .or_else(|| ac50_series.as_ref().and_then(|s| s.get(i)))
                        .or_else(|| potency_series.as_ref().and_then(|s| s.get(i)));

                    if let Some(affinity_nm) = best_affinity_nm {
                        let intrinsic_activity = efficacy_series
                            .as_ref()
                            .and_then(|s| s.get(i))
                            .or_else(|| emax_series.as_ref().and_then(|s| s.get(i)))
                            .unwrap_or(0.0);

                        if affinity_nm > 0.0 {
                            targets
                                .insert(receptor, DrugTarget { affinity_nm, intrinsic_activity });
                        }
                    }
                }
            }
        }
        Ok(Drug { targets })
    }
}

#[derive(Clone, Debug, Copy)]
pub struct DrugTarget {
    pub intrinsic_activity: f32,
    pub affinity_nm: f32,
}

#[derive(Clone, Debug)]
pub struct PKModel {
    pub concentration_nm: f32,
    absorption_multiplier: f32,
    elimination_multiplier: f32,
    conc_per_mg: f32,
    dose_queue: f32,
}

impl PKModel {
    pub fn new(p: &DrugParams) -> Self {
        let absorption_rate = 0.693 / (p.absorption_half_life_hours * 3600.0);
        let elimination_rate = 0.693 / (p.half_life_hours * 3600.0);

        Self {
            concentration_nm: 0.0,
            absorption_multiplier: 1.0 - (-absorption_rate).exp(),
            elimination_multiplier: (-elimination_rate).exp(),
            conc_per_mg: p.concentration_per_mg,
            dose_queue: 0.0,
        }
    }

    #[inline(always)]
    pub fn add_dose(&mut self, dose_mg: f32) {
        self.dose_queue += dose_mg;
    }

    #[inline(always)]
    pub fn tick(&mut self) {
        let to_absorb = self.dose_queue * self.absorption_multiplier;
        self.dose_queue -= to_absorb;
        self.concentration_nm += to_absorb * self.conc_per_mg;
        self.concentration_nm *= self.elimination_multiplier;
    }
}

pub struct Logger {
    writer: BufWriter<std::io::Stdout>,
    sorted_state_keys: Vec<String>,
}

impl Logger {
    pub fn new() -> Self {
        Self { writer: BufWriter::new(stdout()), sorted_state_keys: Vec::new() }
    }

    pub fn log_header(
        &mut self,
        state: &HashMap<String, f32>,
        sensors: &[crate::program::SensorDef],
    ) -> std::io::Result<()> {
        write!(self.writer, "tick,day,phase")?;
        self.sorted_state_keys = state.keys().cloned().collect();
        self.sorted_state_keys.sort();

        for key in &self.sorted_state_keys {
            write!(self.writer, ",{}", key)?;
        }
        for sensor in sensors {
            write!(self.writer, ",{}", sensor.name)?;
        }
        writeln!(self.writer)
    }

    pub fn log_data(
        &mut self,
        tick: u64,
        day: u64,
        phase: &str,
        state: &HashMap<String, f32>,
        sensor_vals: &[f32],
    ) -> std::io::Result<()> {
        write!(self.writer, "{},{},{}", tick, day, phase)?;

        // --- OPTIMIZATION 3: Use the pre-sorted key list ---
        for key in &self.sorted_state_keys {
            write!(self.writer, ",{:.2}", state.get(key).unwrap_or(&0.0))?;
        }
        for val in sensor_vals {
            write!(self.writer, ",{:.4}", val)?;
        }
        writeln!(self.writer)
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

#[derive(Debug)]
pub struct PIDController {
    pub target: f32,
    kp: f32,
    ki: f32,
    kd: f32,
    step: f32,
    integral_error: f32,
    previous_error: f32,
}
impl PIDController {
    pub fn new(target: f32, law: &ControlLaw) -> Self {
        if let ControlLaw::Pid { kp, ki, kd, step, .. } = law {
            Self {
                target,
                kp: *kp,
                ki: *ki,
                kd: *kd,
                step: *step,
                integral_error: 0.0,
                previous_error: 0.0,
            }
        } else {
            panic!()
        }
    }
    pub fn calculate_adjustment(&mut self, current: f32) -> f32 {
        let err = current - self.target;
        self.integral_error = (self.integral_error * 0.95) + err;
        self.integral_error = self.integral_error.clamp(-5.0, 5.0);
        let deriv = err - self.previous_error;
        self.previous_error = err;
        -self.step + (self.kp * err + self.ki * self.integral_error + self.kd * deriv)
    }
}

#[derive(Clone, Copy, Debug, EnumString, Eq, Hash, PartialEq)]
pub enum Receptor {
    #[strum(to_string = "GPCR.Monoamine.Dopamine.D1")]
    GpcrMonoamineDopamineD1,
    #[strum(to_string = "GPCR.Monoamine.Dopamine.D2")]
    GpcrMonoamineDopamineD2,
    #[strum(to_string = "GPCR.Monoamine.Dopamine.D3")]
    GpcrMonoamineDopamineD3,
    #[strum(to_string = "GPCR.Monoamine.Dopamine.D4")]
    GpcrMonoamineDopamineD4,
    #[strum(to_string = "GPCR.Monoamine.Dopamine.D5")]
    GpcrMonoamineDopamineD5,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT1A")]
    GpcrMonoamineSerotonin5HT1A,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT1B")]
    GpcrMonoamineSerotonin5HT1B,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT1D")]
    GpcrMonoamineSerotonin5HT1D,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT1F")]
    GpcrMonoamineSerotonin5HT1F,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT2A")]
    GpcrMonoamineSerotonin5HT2A,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT2B")]
    GpcrMonoamineSerotonin5HT2B,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT2C")]
    GpcrMonoamineSerotonin5HT2C,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT3")]
    GpcrMonoamineSerotonin5HT3,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT4")]
    GpcrMonoamineSerotonin5HT4,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT6")]
    GpcrMonoamineSerotonin5HT6,
    #[strum(to_string = "GPCR.Monoamine.Serotonin.5HT7")]
    GpcrMonoamineSerotonin5HT7,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A1A")]
    GpcrMonoamineAdrenergicA1A,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A1B")]
    GpcrMonoamineAdrenergicA1B,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A1D")]
    GpcrMonoamineAdrenergicA1D,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A2A")]
    GpcrMonoamineAdrenergicA2A,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A2B")]
    GpcrMonoamineAdrenergicA2B,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.A2C")]
    GpcrMonoamineAdrenergicA2C,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.B1")]
    GpcrMonoamineAdrenergicB1,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.B2")]
    GpcrMonoamineAdrenergicB2,
    #[strum(to_string = "GPCR.Monoamine.Adrenergic.B3")]
    GpcrMonoamineAdrenergicB3,
    #[strum(to_string = "GPCR.Monoamine.Muscarinic.M1")]
    GpcrMonoamineMuscarinicM1,
    #[strum(to_string = "GPCR.Monoamine.Muscarinic.M2")]
    GpcrMonoamineMuscarinicM2,
    #[strum(to_string = "GPCR.Monoamine.Muscarinic.M3")]
    GpcrMonoamineMuscarinicM3,
    #[strum(to_string = "GPCR.Monoamine.Muscarinic.M4")]
    GpcrMonoamineMuscarinicM4,
    #[strum(to_string = "GPCR.Monoamine.Muscarinic.M5")]
    GpcrMonoamineMuscarinicM5,
    #[strum(to_string = "GPCR.Monoamine.Histamine.H1")]
    GpcrMonoamineHistamineH1,
    #[strum(to_string = "GPCR.Monoamine.Histamine.H2")]
    GpcrMonoamineHistamineH2,
    #[strum(to_string = "GPCR.Monoamine.Histamine.H3")]
    GpcrMonoamineHistamineH3,
    #[strum(to_string = "GPCR.Monoamine.Histamine.H4")]
    GpcrMonoamineHistamineH4,
    #[strum(to_string = "GPCR.Peptide.Opioid.MOR")]
    GpcrPeptideOpioidMOR,
    #[strum(to_string = "GPCR.Peptide.Opioid.KOR")]
    GpcrPeptideOpioidKOR,
    #[strum(to_string = "GPCR.Peptide.Opioid.DOR")]
    GpcrPeptideOpioidDOR,
    #[strum(to_string = "GPCR.Peptide.Opioid.NOP")]
    GpcrPeptideOpioidNOP,
    #[strum(to_string = "GPCR.Peptide.Angiotensin.AT1")]
    GpcrPeptideAngiotensinAT1,
    #[strum(to_string = "GPCR.Peptide.Angiotensin.AT2")]
    GpcrPeptideAngiotensinAT2,
    #[strum(to_string = "GPCR.Peptide.Chemokine.CXCR4")]
    GpcrPeptideChemokineCXCR4,
    #[strum(to_string = "GPCR.Peptide.Chemokine.CCR5")]
    GpcrPeptideChemokineCCR5,
    #[strum(to_string = "GPCR.Lipid.Cannabinoid.CB1")]
    GpcrLipidCannabinoidCB1,
    #[strum(to_string = "GPCR.Lipid.Cannabinoid.CB2")]
    GpcrLipidCannabinoidCB2,
    #[strum(to_string = "Kinase.ABL1")]
    KinaseABL1,
    #[strum(to_string = "Kinase.ALK")]
    KinaseALK,
    #[strum(to_string = "Kinase.BTK")]
    KinaseBTK,
    #[strum(to_string = "Kinase.EGFR")]
    KinaseEGFR,
    #[strum(to_string = "Kinase.ERBB2")]
    KinaseERBB2,
    #[strum(to_string = "Kinase.FGFR1")]
    KinaseFGFR1,
    #[strum(to_string = "Kinase.FLT3")]
    KinaseFLT3,
    #[strum(to_string = "Kinase.JAK1")]
    KinaseJAK1,
    #[strum(to_string = "Kinase.JAK2")]
    KinaseJAK2,
    #[strum(to_string = "Kinase.JAK3")]
    KinaseJAK3,
    #[strum(to_string = "Kinase.MET")]
    KinaseMET,
    #[strum(to_string = "Kinase.SRC")]
    KinaseSRC,
    #[strum(to_string = "Kinase.SYK")]
    KinaseSYK,
    #[strum(to_string = "Kinase.VEGFR2")]
    KinaseVEGFR2,
    #[strum(to_string = "Kinase.AKT1")]
    KinaseAKT1,
    #[strum(to_string = "Kinase.AURKA")]
    KinaseAURKA,
    #[strum(to_string = "Kinase.AURKB")]
    KinaseAURKB,
    #[strum(to_string = "Kinase.BRAF")]
    KinaseBRAF,
    #[strum(to_string = "Kinase.CDK1")]
    KinaseCDK1,
    #[strum(to_string = "Kinase.CDK2")]
    KinaseCDK2,
    #[strum(to_string = "Kinase.CDK4")]
    KinaseCDK4,
    #[strum(to_string = "Kinase.CDK6")]
    KinaseCDK6,
    #[strum(to_string = "Kinase.CHK1")]
    KinaseCHK1,
    #[strum(to_string = "Kinase.GSK3B")]
    KinaseGSK3B,
    #[strum(to_string = "Kinase.p38a")]
    Kinasep38a,
    #[strum(to_string = "Kinase.MEK1")]
    KinaseMEK1,
    #[strum(to_string = "Kinase.mTOR")]
    KinaseMTOR,
    #[strum(to_string = "Kinase.PIM1")]
    KinasePIM1,
    #[strum(to_string = "Kinase.PKC")]
    KinasePKC,
    #[strum(to_string = "Kinase.PLK1")]
    KinasePLK1,
    #[strum(to_string = "Kinase.RAF1")]
    KinaseRAF1,
    #[strum(to_string = "Kinase.ROCK1")]
    KinaseROCK1,
    #[strum(to_string = "Kinase.ROCK2")]
    KinaseROCK2,
    #[strum(to_string = "Kinase.PI3Ka")]
    KinasePI3Ka,
    #[strum(to_string = "Kinase.PI3Kd")]
    KinasePI3Kd,
    #[strum(to_string = "Kinase")]
    Kinase,
    #[strum(to_string = "Transporter.SERT")]
    TransporterSERT,
    #[strum(to_string = "Transporter.DAT")]
    TransporterDAT,
    #[strum(to_string = "Transporter.NET")]
    TransporterNET,
    #[strum(to_string = "Transporter.SGLT2")]
    TransporterSGLT2,
    #[strum(to_string = "Transporter.Pgp")]
    TransporterPgp,
    #[strum(to_string = "Transporter.BCRP")]
    TransporterBCRP,
    #[strum(to_string = "Transporter.CFTR")]
    TransporterCFTR,
    #[strum(to_string = "Enzyme.HDAC1")]
    EnzymeHDAC1,
    #[strum(to_string = "Enzyme.HDAC2")]
    EnzymeHDAC2,
    #[strum(to_string = "Enzyme.HDAC3")]
    EnzymeHDAC3,
    #[strum(to_string = "Enzyme.HDAC6")]
    EnzymeHDAC6,
    #[strum(to_string = "Enzyme.HDAC8")]
    EnzymeHDAC8,
    #[strum(to_string = "Enzyme.HDAC")]
    EnzymeHDAC,
    #[strum(to_string = "Enzyme.SIRT1")]
    EnzymeSIRT1,
    #[strum(to_string = "Enzyme.SIRT2")]
    EnzymeSIRT2,
    #[strum(to_string = "Enzyme.CYP1A2")]
    EnzymeCYP1A2,
    #[strum(to_string = "Enzyme.CYP2C9")]
    EnzymeCYP2C9,
    #[strum(to_string = "Enzyme.CYP2C19")]
    EnzymeCYP2C19,
    #[strum(to_string = "Enzyme.CYP2D6")]
    EnzymeCYP2D6,
    #[strum(to_string = "Enzyme.CYP3A4")]
    EnzymeCYP3A4,
    #[strum(to_string = "Enzyme.CYP")]
    EnzymeCYP,
    #[strum(to_string = "Enzyme.COX1")]
    EnzymeCOX1,
    #[strum(to_string = "Enzyme.COX2")]
    EnzymeCOX2,
    #[strum(to_string = "Enzyme.5LOX")]
    Enzyme5LOX,
    #[strum(to_string = "Enzyme.ACE")]
    EnzymeACE,
    #[strum(to_string = "Enzyme.BACE1")]
    EnzymeBACE1,
    #[strum(to_string = "Enzyme.Caspase3")]
    EnzymeCaspase3,
    #[strum(to_string = "Enzyme.Caspase9")]
    EnzymeCaspase9,
    #[strum(to_string = "Enzyme.Thrombin")]
    EnzymeThrombin,
    #[strum(to_string = "Enzyme.PDE4")]
    EnzymePDE4,
    #[strum(to_string = "Enzyme.PDE5")]
    EnzymePDE5,
    #[strum(to_string = "Enzyme.AChE")]
    EnzymeAChE,
    #[strum(to_string = "Enzyme.FAAH")]
    EnzymeFAAH,
    #[strum(to_string = "Enzyme.PARP1")]
    EnzymePARP1,
    #[strum(to_string = "Enzyme.LSD1")]
    EnzymeLSD1,
    #[strum(to_string = "Enzyme.EZH2")]
    EnzymeEZH2,
    #[strum(to_string = "Enzyme.IDO1")]
    EnzymeIDO1,
    #[strum(to_string = "Enzyme.CA")]
    EnzymeCA,
    #[strum(to_string = "IonChannel.hERG")]
    IonChannelHERG,
    #[strum(to_string = "IonChannel.Nav1.5")]
    IonChannelNav1_5,
    #[strum(to_string = "IonChannel.Nav1.7")]
    IonChannelNav1_7,
    #[strum(to_string = "IonChannel.Cav1.2")]
    IonChannelCav1_2,
    #[strum(to_string = "NuclearReceptor.AR")]
    NuclearReceptorAR,
    #[strum(to_string = "NuclearReceptor.ERa")]
    NuclearReceptorERa,
    #[strum(to_string = "NuclearReceptor.ERb")]
    NuclearReceptorERb,
    #[strum(to_string = "NuclearReceptor.GR")]
    NuclearReceptorGR,
    #[strum(to_string = "NuclearReceptor.PR")]
    NuclearReceptorPR,
    #[strum(to_string = "NuclearReceptor.PPARg")]
    NuclearReceptorPPARg,
    #[strum(to_string = "NuclearReceptor.RAR")]
    NuclearReceptorRAR,
    #[strum(to_string = "NuclearReceptor.RXR")]
    NuclearReceptorRXR,
    #[strum(to_string = "Epigenetic.BRD4")]
    EpigeneticBRD4,
    #[strum(to_string = "Epigenetic.BRD")]
    EpigeneticBRD,
    #[strum(to_string = "Chaperone.HSP90")]
    ChaperoneHSP90,
    #[strum(to_string = "Structural.Tubulin")]
    StructuralTubulin,
    #[strum(to_string = "PPI.MDM2-p53")]
    PpiMDM2p53,
    #[strum(to_string = "PPI.BCL2")]
    PpiBCL2,
    #[strum(to_string = "PPI.BCL-XL")]
    PpiBCLXL,
    #[strum(to_string = "PPI.MCL1")]
    PpiMCL1,
}
