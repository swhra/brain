import sys
from pathlib import Path
import polars as pl
from polars import lit

# Inputs
CHEMBL_DB_PATH = Path("/Users/samra/brain/data/chembl_35.db")
BINDINGDB_TSV_PATH = Path("/Users/samra/brain/data/BindingDB_All.tsv")

# Intermediates
CACHE_DIR = Path("/tmp/brain_cache")
CHEMBL_CACHE_PATH = CACHE_DIR / "chembl_raw.parquet"
BINDINGDB_CACHE_PATH = CACHE_DIR / "bindingdb_raw.parquet"

# Outputs
DRUG_DB_PATH = Path("data/drugs.parquet")

# Expanded list to include kinetic, functional, and logarithmic data types
ALL_PIVOT_COLUMNS = [
    "AC50", "EC50", "IC50", "Kb", "Kd", "Ki",
    "Potency", "Efficacy", "Emax", "Smax", "Inhibition",
    "k_on", "k_off", "t1/2",
    "pIC50", "pEC50", "pKi", "pKd",
    "Agonist effect", "Effect", "LD50", "TD50", "CC50",
]

# Expanded to include kon and koff for BindingDB
REQUIRED_BINDINGDB_COLS = {
    "Ligand SMILES", "BindingDB Ligand Name", "Target Name",
    "Kd (nM)", "Ki (nM)", "IC50 (nM)", "EC50 (nM)",
    "kon (M-1-s-1)", "koff (s-1)",
    "pH", "Target Source Organism According to Curator or DataSource"
}


# ==============================================================================
#  CONCISE, HARDCODED TARGET MAPPING
# ==============================================================================
# This list is ordered from MOST specific to LEAST specific for accurate matching.
# Namespace convention is simplified to: CLASS.SYMBOL
TARGET_MAPPING = [
    # --- GPCRs (G-Protein Coupled Receptors) ---
    ("GPCR.Monoamine.Dopamine.D1", r"dopamine receptor d1|d\(?1\)\s"),
    ("GPCR.Monoamine.Dopamine.D2", r"dopamine receptor d2|d\(?2\)\s"),
    ("GPCR.Monoamine.Dopamine.D3", r"dopamine receptor d3|d\(?3\)\s"),
    ("GPCR.Monoamine.Dopamine.D4", r"dopamine receptor d4|d\(?4\)\s"),
    ("GPCR.Monoamine.Dopamine.D5", r"dopamine receptor d5|d\(?5\)\s"),
    ("GPCR.Monoamine.Serotonin.5HT1A", r"5-?ht-?1a|serotonin 1a"),
    ("GPCR.Monoamine.Serotonin.5HT1B", r"5-?ht-?1b|serotonin 1b"),
    ("GPCR.Monoamine.Serotonin.5HT1D", r"5-?ht-?1d|serotonin 1d"),
    ("GPCR.Monoamine.Serotonin.5HT1F", r"5-?ht-?1f|serotonin 1f"),
    ("GPCR.Monoamine.Serotonin.5HT2A", r"5-?ht-?2a|serotonin 2a"),
    ("GPCR.Monoamine.Serotonin.5HT2B", r"5-?ht-?2b|serotonin 2b"),
    ("GPCR.Monoamine.Serotonin.5HT2C", r"5-?ht-?2c|serotonin 2c"),
    ("GPCR.Monoamine.Serotonin.5HT3", r"5-?ht-?3|serotonin 3"),
    ("GPCR.Monoamine.Serotonin.5HT4", r"5-?ht-?4|serotonin 4"),
    ("GPCR.Monoamine.Serotonin.5HT6", r"5-?ht-?6|serotonin 6"),
    ("GPCR.Monoamine.Serotonin.5HT7", r"5-?ht-?7|serotonin 7"),
    ("GPCR.Monoamine.Adrenergic.A1A", r"alpha-?1a adrenergic|adra1a"),
    ("GPCR.Monoamine.Adrenergic.A1B", r"alpha-?1b adrenergic|adra1b"),
    ("GPCR.Monoamine.Adrenergic.A1D", r"alpha-?1d adrenergic|adra1d"),
    ("GPCR.Monoamine.Adrenergic.A2A", r"alpha-?2a adrenergic|adra2a"),
    ("GPCR.Monoamine.Adrenergic.A2B", r"alpha-?2b adrenergic|adra2b"),
    ("GPCR.Monoamine.Adrenergic.A2C", r"alpha-?2c adrenergic|adra2c"),
    ("GPCR.Monoamine.Adrenergic.B1", r"beta-?1 adrenergic"),
    ("GPCR.Monoamine.Adrenergic.B2", r"beta-?2 adrenergic"),
    ("GPCR.Monoamine.Adrenergic.B3", r"beta-?3 adrenergic"),
    ("GPCR.Monoamine.Muscarinic.M1", r"muscarinic .*m1|\bm1 receptor"),
    ("GPCR.Monoamine.Muscarinic.M2", r"muscarinic .*m2|\bm2 receptor"),
    ("GPCR.Monoamine.Muscarinic.M3", r"muscarinic .*m3|\bm3 receptor"),
    ("GPCR.Monoamine.Muscarinic.M4", r"muscarinic .*m4|\bm4 receptor"),
    ("GPCR.Monoamine.Muscarinic.M5", r"muscarinic .*m5|\bm5 receptor"),
    ("GPCR.Monoamine.Histamine.H1", r"histamine .*h1|\bh1 receptor"),
    ("GPCR.Monoamine.Histamine.H2", r"histamine .*h2|\bh2 receptor"),
    ("GPCR.Monoamine.Histamine.H3", r"histamine .*h3|\bh3 receptor"),
    ("GPCR.Monoamine.Histamine.H4", r"histamine .*h4|\bh4 receptor"),
    # Peptide Family
    ("GPCR.Peptide.Opioid.MOR", r"mu.?opioid|mu.?type|\bmor\b"),
    ("GPCR.Peptide.Opioid.KOR", r"kappa.?opioid|kappa.?type|\bkor\b"),
    ("GPCR.Peptide.Opioid.DOR", r"delta.?opioid|delta.?type|\bdor\b"),
    ("GPCR.Peptide.Opioid.NOP", r"nociceptin receptor|nop\b"),
    ("GPCR.Peptide.Angiotensin.AT1", r"angiotensin .* type-?1|at1 receptor"),
    ("GPCR.Peptide.Angiotensin.AT2", r"angiotensin .* type-?2|at2 receptor"),
    ("GPCR.Peptide.Chemokine.CXCR4", r"c-x-c chemokine receptor type 4|cxcr4"),
    ("GPCR.Peptide.Chemokine.CCR5", r"c-c chemokine receptor type 5|ccr5"),
    # Lipid Family
    ("GPCR.Lipid.Cannabinoid.CB1", r"cannabinoid receptor 1|\bcb1\b"),
    ("GPCR.Lipid.Cannabinoid.CB2", r"cannabinoid receptor 2|\bcb2\b"),

    # --- Kinases ---
    ("Kinase.ABL1", r"\babl1?\b|tyrosine.?protein kinase abl"),
    ("Kinase.ALK", r"\balk\b|anaplastic lymphoma kinase"),
    ("Kinase.BTK", r"\bbtk\b|tyrosine.?protein kinase btk"),
    ("Kinase.EGFR", r"epidermal growth factor receptor|\begfr\b|erbb1"),
    ("Kinase.ERBB2", r"erbb-?2|her2\b"),
    ("Kinase.FGFR1", r"fibroblast growth factor receptor 1|\bfgfr1\b"),
    ("Kinase.FLT3", r"\bflt-?3\b"),
    ("Kinase.JAK1", r"janus kinase 1|\bjak1\b"),
    ("Kinase.JAK2", r"janus kinase 2|\bjak2\b"),
    ("Kinase.JAK3", r"janus kinase 3|\bjak3\b"),
    ("Kinase.MET", r"hepatocyte growth factor receptor|\bmet\b"),
    ("Kinase.SRC", r"\bsrc\b|proto-oncogene tyrosine-protein kinase src"),
    ("Kinase.SYK", r"\bsyk\b|tyrosine-protein kinase syk"),
    ("Kinase.VEGFR2", r"vascular endothelial growth factor receptor 2|vegfr2|kdr"),
    ("Kinase.AKT1", r"rac-alpha .* kinase|\bakt1\b"),
    ("Kinase.AURKA", r"aurora kinase a\b"),
    ("Kinase.AURKB", r"aurora kinase b\b"),
    ("Kinase.BRAF", r"serine/threonine-protein kinase b-raf|\bbraf\b"),
    ("Kinase.CDK1", r"cyclin dependent kinase 1\b|\bcdk1\b"),
    ("Kinase.CDK2", r"cyclin dependent kinase 2\b|\bcdk2\b"),
    ("Kinase.CDK4", r"cyclin dependent kinase 4\b|\bcdk4\b"),
    ("Kinase.CDK6", r"cyclin dependent kinase 6\b|\bcdk6\b"),
    ("Kinase.CHK1", r"\bchk1\b"),
    ("Kinase.GSK3B", r"glycogen synthase kinase-?3 beta|gsk-?3b"),
    ("Kinase.p38a", r"map kinase p38 alpha|mapk14"),
    ("Kinase.MEK1", r"map kinase kinase 1|\bmek1\b|map2k1"),
    ("Kinase.mTOR", r"\bmtor\b|serine/threonine-protein kinase mtor"),
    ("Kinase.PIM1", r"\bpim-?1\b"),
    ("Kinase.PKC", r"protein kinase c\b"),
    ("Kinase.PLK1", r"polo-like kinase 1|\bplk1\b"),
    ("Kinase.RAF1", r"raf proto-oncogene .* kinase|\bc-?raf\b|raf1"),
    ("Kinase.ROCK1", r"rho-associated .* kinase 1|rock1"),
    ("Kinase.ROCK2", r"rho-associated .* kinase 2|rock2"),
    ("Kinase.PI3Ka", r"pi3k alpha|phosphatidylinositol 4,5-bisphosphate 3-kinase .* alpha"),
    ("Kinase.PI3Kd", r"pi3k delta|phosphatidylinositol 4,5-bisphosphate 3-kinase .* delta"),
    ("Kinase", r"kinase"),

    # --- Transporters ---
    ("Transporter.SERT", r"serotonin transporter|sert\b|slc6a4"),
    ("Transporter.DAT", r"dopamine transporter|dat\b|slc6a2"),
    ("Transporter.NET", r"norepinephrine transporter|net\b|slc6a3"),
    ("Transporter.SGLT2", r"sodium/glucose cotransporter 2|sglt2"),
    ("Transporter.Pgp", r"p-glycoprotein 1|\bmdr1\b|abc(b|g)1\b"),
    ("Transporter.BCRP", r"atp-binding cassette sub-family g member 2|abcg2|bcrp"),
    ("Transporter.CFTR", r"cystic fibrosis transmembrane conductance regulator|cftr"),

    # --- Enzymes ---
    ("Enzyme.HDAC1", r"histone deacetylase 1\b|\bhdac1\b"),
    ("Enzyme.HDAC2", r"histone deacetylase 2\b|\bhdac2\b"),
    ("Enzyme.HDAC3", r"histone deacetylase 3\b|\bhdac3\b"),
    ("Enzyme.HDAC6", r"histone deacetylase 6\b|\bhdac6\b"),
    ("Enzyme.HDAC8", r"histone deacetylase 8\b|\bhdac8\b"),
    ("Enzyme.HDAC", r"histone deacetylase|\bhdac\d"),
    ("Enzyme.SIRT1", r"sirtuin-?1\b"),
    ("Enzyme.SIRT2", r"sirtuin-?2\b"),
    ("Enzyme.CYP1A2", r"cytochrome p450 1a2|cyp1a2"),
    ("Enzyme.CYP2C9", r"cytochrome p450 2c9|cyp2c9"),
    ("Enzyme.CYP2C19", r"cytochrome p450 2c19|cyp2c19"),
    ("Enzyme.CYP2D6", r"cytochrome p450 2d6|cyp2d6"),
    ("Enzyme.CYP3A4", r"cytochrome p450 3a4|cyp3a4"),
    ("Enzyme.CYP", r"cytochrome p450|cyp\w"),
    ("Enzyme.COX1", r"cyclooxygenase-?1|prostaglandin g/h synthase 1|cox-?1"),
    ("Enzyme.COX2", r"cyclooxygenase-?2|prostaglandin g/h synthase 2|cox-?2"),
    ("Enzyme.5LOX", r"arachidonate 5-lipoxygenase|5-lox"),
    ("Enzyme.ACE", r"\bace\b|angiotensin-converting enzyme"),
    ("Enzyme.BACE1", r"beta-secretase 1|bace-?1"),
    ("Enzyme.Caspase3", r"\bcaspase-?3\b"),
    ("Enzyme.Caspase9", r"\bcaspase-?9\b"),
    ("Enzyme.Thrombin", r"thrombin|coagulation factor 2"),
    ("Enzyme.PDE4", r"phosphodiesterase 4|\bpde4\b"),
    ("Enzyme.PDE5", r"phosphodiesterase 5|\bpde5\b"),
    ("Enzyme.AChE", r"acetylcholinesterase|ache\b"),
    ("Enzyme.FAAH", r"fatty-acid amide hydrolase|\bfaah\b"),
    ("Enzyme.PARP1", r"poly.?\[adp-ribose\].?polymerase-?1|parp-?1\b"),
    ("Enzyme.LSD1", r"lysine-specific histone demethylase 1|\blsd1\b"),
    ("Enzyme.EZH2", r"histone-lysine n-methyltransferase ezh2|\bezh2\b"),
    ("Enzyme.IDO1", r"indoleamine 2,3-dioxygenase 1|\bido1?\b"),
    ("Enzyme.CA", r"carbonic anhydrase"),

    # --- Ion Channels ---
    ("IonChannel.hERG", r"herg|kcn.*?h2"),
    ("IonChannel.Nav1.5", r"sodium channel.*type 5.*alpha|scn5a"),
    ("IonChannel.Nav1.7", r"sodium channel.*type 9.*alpha|scn9a"),
    ("IonChannel.Cav1.2", r"voltage-dependent l-type calcium channel.*alpha-1c|cacna1c"),

    # --- Nuclear Receptors ---
    ("NuclearReceptor.AR", r"androgen receptor"),
    ("NuclearReceptor.ERa", r"estrogen receptor alpha|esr1"),
    ("NuclearReceptor.ERb", r"estrogen receptor beta|esr2"),
    ("NuclearReceptor.GR", r"glucocorticoid receptor"),
    ("NuclearReceptor.PR", r"progesterone receptor"),
    ("NuclearReceptor.PPARg", r"peroxisome proliferator-activated receptor gamma|pparg"),
    ("NuclearReceptor.RAR", r"retinoic acid receptor"),
    ("NuclearReceptor.RXR", r"retinoid x receptor"),

    # --- Other important protein classes ---
    ("Epigenetic.BRD4", r"bromodomain-containing protein 4|brd4"),
    ("Epigenetic.BRD", r"bromodomain"),
    ("Chaperone.HSP90", r"heat shock protein .*90|hsp90"),
    ("Structural.Tubulin", r"tubulin"),
    ("PPI.MDM2-p53", r"mdm2.*p53|cellular tumor antigen p53.*e3 ubiquitin-protein ligase mdm2"),
    ("PPI.BCL2", r"\bbcl-?2\b|apoptosis regulator bcl-2"),
    ("PPI.BCL-XL", r"\bbcl-?xl\b|bcl2-like protein 1"),
    ("PPI.MCL1", r"\bmcl-?1\b"),
]

def normalize_target_names(lf: pl.LazyFrame, target_col: str = "target") -> pl.LazyFrame:
    """ Normalizes protein target names into a concise, hierarchical namespace. """
    if target_col not in lf.collect_schema():
        raise ValueError(f"Column '{target_col}' not found in LazyFrame.")

    cleaned_stem_expr = (
        pl.col(target_col)
        .str.split("/")
        .list.first()
        .str.to_lowercase()
        .str.replace_all(r"\[.*?\]|\(.*?\)|, mitochondrial|, cytoplasmic", " ")
        .str.replace_all(r"[,/'-]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )

    mapping_expr = pl.lit(None, dtype=pl.String)
    for namespace, pattern in reversed(TARGET_MAPPING):
        mapping_expr = (
            pl.when(cleaned_stem_expr.str.contains(pattern))
            .then(pl.lit(namespace))
            .otherwise(mapping_expr)
        )
    return lf.with_columns(target=mapping_expr)

def calculate_cns_mpo(lf: pl.LazyFrame) -> pl.LazyFrame:
    """ Calculates the CNS MPO score. """
    if "pchembl_value" not in lf.collect_schema():
        return lf.with_columns(cns_mpo=lit(None, dtype=pl.UInt8))
    mpo_expr = (
        (pl.col("alogp") <= 5.0).cast(pl.UInt8)
        + (pl.col("mw_freebase") <= 500.0).cast(pl.UInt8)
        + (pl.col("hbd") <= 5.0).cast(pl.UInt8)
        + (pl.col("pchembl_value") >= 6.5).cast(pl.UInt8)
        + (pl.col("psa") <= 120.0).cast(pl.UInt8)
    )
    return lf.with_columns(cns_mpo=mpo_expr.fill_null(0))

def load_chembl_data(db_path: Path, cache_path: Path) -> pl.LazyFrame:
    """ Loads and caches raw data from the ChEMBL SQLite database. """
    if cache_path.exists():
        print(f"-> Found precomputed ChEMBL data: \x1b[3m{cache_path}\x1b[0m")
        return pl.scan_parquet(cache_path)
    print(f"\x1b[33m-> Cached ChEMBL data not found or not valid.\x1b[0m")
    print(f"-> Computing ChEMBL data from: {db_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL database not found at \x1b[3m{db_path}\x1b[0m")
    query = """
    SELECT
        CAST(md.pref_name AS VARCHAR) AS name, CAST(td.pref_name AS VARCHAR) AS target, CAST(td.target_type AS VARCHAR) AS target_type, CAST(td.organism AS VARCHAR) AS organism,
        CAST(cs.canonical_smiles AS VARCHAR) AS smiles,
        CAST(a.assay_type AS VARCHAR) AS assay_type, CAST(a.confidence_score AS INT) AS confidence_score,
        CAST(act.standard_type AS VARCHAR) AS standard_type, CAST(act.standard_value AS FLOAT) AS standard_value, CAST(act.standard_relation AS VARCHAR) AS standard_relation, CAST(act.standard_units AS VARCHAR) AS standard_units,
        CAST(md.withdrawn_flag AS SMALLINT) AS withdrawn, CAST(md.oral AS SMALLINT) AS oral, CAST(md.parenteral AS SMALLINT) AS parenteral, CAST(md.topical AS SMALLINT) AS topical
    FROM molecule_dictionary AS md
    INNER JOIN compound_structures AS cs ON md.molregno = cs.molregno
    INNER JOIN activities AS act ON md.molregno = act.molregno
    INNER JOIN assays AS a ON act.assay_id = a.assay_id
    INNER JOIN target_dictionary AS td ON a.tid = td.tid
    WHERE
        md.pref_name IS NOT NULL
        AND md.structure_type = 'MOL'
        AND act.standard_value IS NOT NULL
        AND td.organism = 'Homo sapiens'
        AND a.confidence_score >= 4;
    """
    df = pl.read_database_uri(query, f"sqlite://{db_path.resolve()}", engine='connectorx')
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"-> Caching {len(df):,} ChEMBL rows at: {cache_path}")
    return df.lazy()

def load_bindingdb_data(tsv_path: Path, cache_path: Path) -> pl.LazyFrame:
    """ Loads and caches raw data from the BindingDB TSV file. """
    if cache_path.exists() and REQUIRED_BINDINGDB_COLS.issubset(set(pl.read_parquet_schema(cache_path))):
        print(f"-> Found precomputed BindingDB data: \x1b[3m{cache_path}\x1b[0m")
        return pl.scan_parquet(cache_path)

    print(f"\x1b[33m-> Cached BindingDB data not found or not valid.\x1b[0m")
    print(f"Performing one-time extraction from: \x1b[3m{tsv_path}\x1b[0m")

    if not tsv_path.exists():
        raise FileNotFoundError(f"BindingDB TSV not found: {tsv_path}")
    df = pl.read_csv(
        tsv_path, separator="\t", columns=list(REQUIRED_BINDINGDB_COLS),
        has_header=True, ignore_errors=True, low_memory=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"-> Caching {len(df):,} BindingDB rows at: \x1b[3m{cache_path}\x1b[0m")
    df.write_parquet(cache_path)
    return df.lazy()

def main():
    try:
        # Step 1: Load data sources
        bindingdb_lf = load_bindingdb_data(BINDINGDB_TSV_PATH, BINDINGDB_CACHE_PATH)
        chembl_lf = load_chembl_data(CHEMBL_DB_PATH, CHEMBL_CACHE_PATH)

        # Step 2: Clean and prepare data sources
        print("-> Cleaning and preparing data sources.")
        chembl_lf = chembl_lf.filter(pl.col("withdrawn") == 0).select(
            pl.col("name").str.to_lowercase(),
            pl.col("smiles"),
            pl.col("target").str.to_lowercase(),
            pl.exclude("name", "smiles", "target", "withdrawn")
        )

        bindingdb_renames = {
            "Ligand SMILES": "smiles", "BindingDB Ligand Name": "name", "Target Name": "target",
            "Target Source Organism According to Curator or DataSource": "organism",
            "Ki (nM)": "Ki_raw", "IC50 (nM)": "IC50_raw", "Kd (nM)": "Kd_raw", "EC50 (nM)": "EC50_raw",
            "kon (M-1-s-1)": "kon_raw", "koff (s-1)": "koff_raw"
        }
        bindingdb_lf = (
            bindingdb_lf.rename(bindingdb_renames)
            .filter(pl.col("organism") == "Homo sapiens")
            .with_columns(
                pl.col("name").str.split("::").list.first().str.to_lowercase(),
                pl.col("target").str.to_lowercase(),
            )
            .unpivot(
                index=["name", "smiles", "target"],
                on=["EC50_raw", "IC50_raw", "Kd_raw", "Ki_raw", "kon_raw", "koff_raw"],
                variable_name="standard_type", value_name="raw_value",
            )
            .filter(pl.col("raw_value").is_not_null() & (pl.col("raw_value") != ""))
            .with_columns(
                standard_type=pl.col("standard_type").str.replace("_raw", "").str.replace("on", "_on").str.replace("off", "_off"),
                standard_relation=pl.col("raw_value").str.extract(r"([<>=~])", 1).fill_null("="),
                standard_value=pl.col("raw_value").str.replace_all(r"[<>=~]", "").cast(pl.Float32, strict=False),
            )
            .filter(pl.col("standard_value").is_not_null())
            .with_columns( # Assign units based on type for BindingDB
                standard_units=pl.when(pl.col("standard_type") == "k_on").then(lit("M-1s-1"))
                               .when(pl.col("standard_type") == "k_off").then(lit("s-1"))
                               .otherwise(lit("nM"))
            )
            .select("name", "smiles", "target", "standard_type", "standard_value", "standard_relation", "standard_units")
        )

        # Step 3: Combine, normalize, and pre-calculate
        with pl.StringCache():
            print("-> Combining ChEMBL and BindingDB data.")
            bindingdb_lf = bindingdb_lf.with_columns(confidence_score=lit(5, dtype=pl.Int32)) # Assign default confidence to BindingDB
            lf = pl.concat([chembl_lf, bindingdb_lf], how="diagonal_relaxed")

            name_map = lf.group_by("smiles").agg(pl.col("name").sort_by(pl.col("name").str.len_chars()).first())
            lf = lf.drop("name").join(name_map, on="smiles", how="left").filter(pl.col("name").is_not_null())

            lf = normalize_target_names(lf)

            print("-> Normalizing units and creating value range columns.")
            lf = lf.with_columns(
                # Create a single, normalized value column with standard units
                normalized_value=pl.when(pl.col("standard_units").str.contains(r"(?i)^M$")).then(pl.col("standard_value"))
                                 .when(pl.col("standard_units").str.contains(r"(?i)mM")).then(pl.col("standard_value") / 1e3)
                                 .when(pl.col("standard_units").str.contains(r"(?i)uM|µM")).then(pl.col("standard_value") / 1e6)
                                 .when(pl.col("standard_units").str.contains(r"(?i)nM")).then(pl.col("standard_value") / 1e9)
                                 .when(pl.col("standard_units").str.contains(r"(?i)pM")).then(pl.col("standard_value") / 1e12)
                                 .when(pl.col("standard_type").str.starts_with("p")).then(10.0**(-pl.col("standard_value"))) # Convert pKi, pIC50 etc to Molar
                                 .when(pl.col("standard_units").str.contains(r"/min")).then(pl.col("standard_value") / 60.0)
                                 .when(pl.col("standard_units") == "%").then(pl.col("standard_value") / 100.0)
                                 .otherwise(pl.col("standard_value")), # For s-1, M-1s-1, or unitless
                # Standardize the type name itself (e.g., pKi -> Ki)
                standard_type=pl.when(pl.col("standard_type").str.to_lowercase() == "pic50").then(lit("IC50"))
                               .when(pl.col("standard_type").str.to_lowercase() == "pec50").then(lit("EC50"))
                               .when(pl.col("standard_type").str.to_lowercase() == "pki").then(lit("Ki"))
                               .when(pl.col("standard_type").str.to_lowercase() == "pkd").then(lit("Kd"))
                               .otherwise(pl.col("standard_type"))
            )

            lf = (
                lf.filter(pl.col("standard_type").is_in(ALL_PIVOT_COLUMNS) & pl.col("normalized_value").is_not_null())
                .with_columns(
                    val=pl.when(pl.col("standard_relation") == "=").then(pl.col("normalized_value")),
                    min_val=pl.when(pl.col("standard_relation").is_in([">", ">="])).then(pl.col("normalized_value")),
                    max_val=pl.when(pl.col("standard_relation").is_in(["<", "<="])).then(pl.col("normalized_value")),
                )
            )
            lf = calculate_cns_mpo(lf)

            print("-> Defining hierarchical aggregation logic.")
            group_keys = ["name", "target"]
            # Define all columns we want in the final output
            pivot_types = ["AC50", "EC50", "Emax", "IC50", "Kb", "Kd", "Ki", "Potency", "Efficacy", "Smax", "k_on", "k_off"]

            schema = lf.collect_schema()
            first_cols = [c for c in ["smiles", "oral", "parenteral", "topical", "assay_type"] if c in schema]
            aggs = []

            for T in pivot_types:
                type_filter = pl.col("standard_type") == T
                exact_filter = type_filter & pl.col("val").is_not_null()
                upper_bound_filter = type_filter & pl.col("max_val").is_not_null()
                lower_bound_filter = type_filter & pl.col("min_val").is_not_null()

                sum_weights_exact = pl.col("confidence_score").filter(exact_filter).sum()
                sum_weights_upper = pl.col("confidence_score").filter(upper_bound_filter).sum()
                sum_weights_lower = pl.col("confidence_score").filter(lower_bound_filter).sum()

                w_mean_exact = (pl.when(sum_weights_exact > 0).then((pl.col("val") * pl.col("confidence_score")).filter(exact_filter).sum() / sum_weights_exact))
                w_mean_upper = (pl.when(sum_weights_upper > 0).then((pl.col("max_val") * pl.col("confidence_score")).filter(upper_bound_filter).sum() / sum_weights_upper))
                w_mean_lower = (pl.when(sum_weights_lower > 0).then((pl.col("min_val") * pl.col("confidence_score")).filter(lower_bound_filter).sum() / sum_weights_lower))

                final_expr = (
                    pl.when(w_mean_exact.is_not_null())
                    .then(w_mean_exact)
                    .otherwise(
                        pl.when(w_mean_upper.is_not_null() & w_mean_lower.is_not_null())
                        .then((w_mean_upper + w_mean_lower) / 2)
                        .otherwise(pl.coalesce(w_mean_upper, w_mean_lower))
                    )
                    .cast(pl.Float32)
                    .alias(T.lower())
                )
                aggs.append(final_expr)

            aggs.extend([pl.col(c).first() for c in first_cols])

            print("-> Performing final aggregation.")
            df = (
                lf.group_by(group_keys, maintain_order=True)
                .agg(aggs)
                .with_columns([pl.col(c).cast(pl.Categorical) for c in ["assay_type"] if c in schema])
                .filter(pl.col("name").is_not_null() & pl.col("target").is_not_null())
                .sort(["name", "target"])
                .drop("confidence_score")
                .collect()
            )

            # Final step: save the data
            DRUG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(DRUG_DB_PATH, compression="zstd", compression_level=6, statistics=True)
            print(f"\x1b[32m-> Wrote {len(df):,} rows to: \x1b[3m{DRUG_DB_PATH}\x1b[0m")

    except Exception as e:
        print(f"❌ {e}", file=sys.stderr)
        raise e

if __name__ == "__main__":
    main()