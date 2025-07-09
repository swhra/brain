import sys
import polars as pl
from pathlib import Path

CHEMBL_DB_PATH = Path("/Users/samra/brain/data/chembl_35.db")
BINDINGDB_TSV_PATH = Path("/Users/samra/brain/data/BindingDB_All.tsv")
CACHE_DIR = Path("/tmp/brain_cache/")
CHEMBL_CACHE_PATH = CACHE_DIR / "chembl_raw.parquet"
BINDINGDB_CACHE_PATH = CACHE_DIR / "bindingdb_raw.parquet"
DRUG_DB_PATH = Path("data/drugs.parquet")

ALL_PIVOT_COLUMNS = [
    "AC50",
    "EC50",
    "IC50",
    "Kb",
    "Kd",
    "Ki",
    "Potency",
    "Efficacy",
    "Emax",
    "Smax",
    "k_on",
    "k_off",
    "t1/2",
    "Effect",
    "CC50",
    "Inhibition",
]

REQUIRED_BINDINGDB_COLS = {
    "Ligand SMILES",
    "Ligand InChI Key",
    "BindingDB Ligand Name",
    "Target Name",
    "Ki (nM)",
    "IC50 (nM)",
    "Kd (nM)",
    "EC50 (nM)",
    "kon (M-1-s-1)",
    "koff (s-1)",
    "Target Source Organism According to Curator or DataSource",
}

# --- Target Mapping Definition (Unchanged) ---
TARGET_MAPPING = [
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
    ("GPCR.Peptide.Opioid.MOR", r"mu.?opioid|mu.?type|\bmor\b"),
    ("GPCR.Peptide.Opioid.KOR", r"kappa.?opioid|kappa.?type|\bkor\b"),
    ("GPCR.Peptide.Opioid.DOR", r"delta.?opioid|delta.?type|\bdor\b"),
    ("GPCR.Peptide.Opioid.NOP", r"nociceptin receptor|nop\b"),
    ("GPCR.Peptide.Angiotensin.AT1", r"angiotensin .* type-?1|at1 receptor"),
    ("GPCR.Peptide.Angiotensin.AT2", r"angiotensin .* type-?2|at2 receptor"),
    ("GPCR.Peptide.Chemokine.CXCR4", r"c-x-c chemokine receptor type 4|cxcr4"),
    ("GPCR.Peptide.Chemokine.CCR5", r"c-c chemokine receptor type 5|ccr5"),
    ("GPCR.Lipid.Cannabinoid.CB1", r"cannabinoid receptor 1|\bcb1\b"),
    ("GPCR.Lipid.Cannabinoid.CB2", r"cannabinoid receptor 2|\bcb2\b"),
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
    ("Transporter.SERT", r"serotonin transporter|sert\b|slc6a4"),
    ("Transporter.DAT", r"dopamine transporter|dat\b|slc6a2"),
    ("Transporter.NET", r"norepinephrine transporter|net\b|slc6a3"),
    ("Transporter.SGLT2", r"sodium/glucose cotransporter 2|sglt2"),
    ("Transporter.Pgp", r"p-glycoprotein 1|\bmdr1\b|abc(b|g)1\b"),
    ("Transporter.BCRP", r"atp-binding cassette sub-family g member 2|abcg2|bcrp"),
    ("Transporter.CFTR", r"cystic fibrosis transmembrane conductance regulator|cftr"),
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
    ("IonChannel.hERG", r"herg|kcn.*?h2"),
    ("IonChannel.Nav1.5", r"sodium channel.*type 5.*alpha|scn5a"),
    ("IonChannel.Nav1.7", r"sodium channel.*type 9.*alpha|scn9a"),
    ("IonChannel.Cav1.2", r"voltage-dependent l-type calcium channel.*alpha-1c|cacna1c"),
    ("NuclearReceptor.AR", r"androgen receptor"),
    ("NuclearReceptor.ERa", r"estrogen receptor alpha|esr1"),
    ("NuclearReceptor.ERb", r"estrogen receptor beta|esr2"),
    ("NuclearReceptor.GR", r"glucocorticoid receptor"),
    ("NuclearReceptor.PR", r"progesterone receptor"),
    ("NuclearReceptor.PPARg", r"peroxisome proliferator-activated receptor gamma|pparg"),
    ("NuclearReceptor.RAR", r"retinoic acid receptor"),
    ("NuclearReceptor.RXR", r"retinoid x receptor"),
    ("Epigenetic.BRD4", r"bromodomain-containing protein 4|brd4"),
    ("Epigenetic.BRD", r"bromodomain"),
    ("Chaperone.HSP90", r"heat shock protein .*90|hsp90"),
    ("Structural.Tubulin", r"tubulin"),
    ("PPI.MDM2-p53", r"mdm2.*p53|cellular tumor antigen p53.*e3 ubiquitin-protein ligase mdm2"),
    ("PPI.BCL2", r"\bbcl-?2\b|apoptosis regulator bcl-2"),
    ("PPI.BCL-XL", r"\bbcl-?xl\b|bcl2-like protein 1"),
    ("PPI.MCL1", r"\bmcl-?1\b"),
]


def normalize_molecule_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    if "name" not in lf.collect_schema():
        raise ValueError("Column 'name' not found in LazyFrame.")
    id_pattern = r"^(CHEMBL|MLS|SMR|US|WO|cid_|GNF-Pf-)\S*|^\d+$"
    split_names_expr = pl.col("name").str.split("::")
    good_parts_expr = split_names_expr.list.eval(
        pl.element().filter(pl.element().str.contains(id_pattern).not_() & (pl.element() != ""))
    )
    best_name_expr = (
        pl.when(good_parts_expr.list.len() > 0)
        .then(good_parts_expr.list.last())
        .otherwise(split_names_expr.list.first())
        .str.strip_chars()
        .str.to_lowercase()
    )
    return lf.with_columns(name=best_name_expr)


def normalize_target_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    if "target" not in lf.collect_schema():
        raise ValueError("Column 'target' not found in LazyFrame.")
    cleaned_stem_expr = (
        pl.col("target")
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
        mapping_expr = pl.when(cleaned_stem_expr.str.contains(pattern)).then(pl.lit(namespace)).otherwise(mapping_expr)
    return lf.with_columns(target=mapping_expr)


def load_chembl_data(db_path: Path, cache_path: Path) -> pl.LazyFrame:
    if cache_path.exists():
        print(f"-> Found cached ChEMBL data: {cache_path}")
        return pl.scan_parquet(cache_path)
    print(f"-> No cache found. Loading from ChEMBL DB: {db_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"ChEMBL database not found at '{db_path}'")
    chembl_query = """
    SELECT
        md.pref_name, cs.standard_inchi_key AS inchi_key, td.pref_name AS target,
        act.standard_type AS property, act.standard_units AS units,
        act.standard_relation AS relation, CAST(act.standard_value AS FLOAT) AS value,
        a.confidence_score, CAST(md.withdrawn_flag AS SMALLINT) AS withdrawn
    FROM molecule_dictionary md
    JOIN compound_structures cs ON md.molregno = cs.molregno
    JOIN activities act ON md.molregno = act.molregno
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN target_dictionary td ON a.tid = td.tid
    WHERE td.organism = 'Homo sapiens' AND a.confidence_score >= 3
      AND cs.standard_inchi_key IS NOT NULL AND act.standard_type IS NOT NULL
      AND act.standard_value IS NOT NULL;
    """
    df = pl.read_database_uri(chembl_query, f"sqlite:///{db_path.resolve()}", engine="connectorx")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"-> Cached {len(df):,} rows from ChEMBL.")
    return df.lazy()


def load_bindingdb_data(tsv_path: Path, cache_path: Path) -> pl.LazyFrame:
    if cache_path.exists():
        print(f"-> Found cached BindingDB data: {cache_path}")
        return pl.scan_parquet(cache_path)
    print(f"-> No cache found. Loading from BindingDB TSV: {tsv_path}")
    if not tsv_path.exists():
        raise FileNotFoundError(f"BindingDB TSV not found at '{tsv_path}'")
    bindingdb_renames = {
        "Ligand InChI Key": "inchi_key",
        "BindingDB Ligand Name": "name",
        "Target Name": "target",
        "Ki (nM)": "Ki",
        "IC50 (nM)": "IC50",
        "Kd (nM)": "Kd",
        "EC50 (nM)": "EC50",
        "kon (M-1-s-1)": "k_on",
        "koff (s-1)": "k_off",
        "Target Source Organism According to Curator or DataSource": "organism",
    }
    df = (
        pl.read_csv(
            tsv_path,
            separator="\t",
            has_header=True,
            low_memory=True,
            columns=list(bindingdb_renames.keys()),
            ignore_errors=True,
        )
        .filter(pl.col("organism") == "Homo sapiens")
        .rename(bindingdb_renames)
        .drop("organism")
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    print(f"-> Cached {len(df):,} rows from BindingDB.")
    return df.lazy()


def main():
    try:
        chembl_lf = load_chembl_data(CHEMBL_DB_PATH, CHEMBL_CACHE_PATH)
        bindingdb_lf = load_bindingdb_data(BINDINGDB_TSV_PATH, BINDINGDB_CACHE_PATH)

        bindingdb_std_lf = (
            bindingdb_lf.unpivot(
                index=["name", "inchi_key", "target"],
                on=["Ki", "IC50", "Kd", "EC50", "k_on", "k_off"],
                variable_name="property",
                value_name="raw_value",
            )
            .filter(pl.col("raw_value").is_not_null() & (pl.col("raw_value") != ""))
            .with_columns(
                relation=pl.col("raw_value").str.extract(r"([<>=~])", 1).fill_null("="),
                value=pl.col("raw_value").str.replace_all(r"[<>=~]", "").cast(pl.Float64, strict=False),
                units=pl.when(pl.col("property").is_in(["k_on", "k_off"])).then(pl.lit("1/s")).otherwise(pl.lit("nM")),
                confidence_score=pl.lit(5, dtype=pl.Int32),
            )
            .select("name", "inchi_key", "target", "property", "value", "relation", "units", "confidence_score")
        )
        chembl_std_lf = (
            chembl_lf.filter(pl.col("withdrawn") == 0)
            .rename({"pref_name": "name"})
            .select("name", "inchi_key", "target", "property", "value", "relation", "units", "confidence_score")
        )

        combined_lf = pl.concat([chembl_std_lf, bindingdb_std_lf], how="diagonal_relaxed")
        combined_lf = normalize_molecule_names(combined_lf).pipe(normalize_target_names)
        combined_lf = combined_lf.filter(pl.col("name").is_not_null() & pl.col("target").is_not_null())

        # Normalize to standard units: nM for affinities, unitless ratio for efficacies
        value_nm_expr = (
            pl.when(pl.col("units").str.contains(r"(?i)uM|µM"))
            .then(pl.col("value") * 1e3)
            .when(pl.col("units").str.contains(r"(?i)pM"))
            .then(pl.col("value") / 1e3)
            .when(pl.col("property").str.starts_with("p"))
            .then(10.0 ** (-pl.col("value")) * 1e9)
            .when(pl.col("units").str.contains(r"(?i)^M$"))
            .then(pl.col("value") * 1e9)  # Molar to nM
            .when(pl.col("units").str.contains(r"(?i)mM"))
            .then(pl.col("value") * 1e6)  # mM to nM
            .otherwise(pl.col("value"))
        )
        efficacy_expr = pl.when(pl.col("units") == "%").then(pl.col("value") / 100.0).otherwise(pl.col("value"))

        affinity_types = ["AC50", "EC50", "IC50", "Kb", "Kd", "Ki", "Potency"]
        efficacy_types = ["Efficacy", "Emax", "Smax", "Inhibition", "Effect"]

        combined_lf = (
            combined_lf.with_columns(
                value_normalized=pl.when(pl.col("property").is_in(affinity_types))
                .then(value_nm_expr)
                .when(pl.col("property").is_in(efficacy_types))
                .then(efficacy_expr)
                .otherwise(pl.col("value"))
            )
            .with_columns(
                property=pl.when(pl.col("property").str.starts_with("p"))
                .then(pl.col("property").str.slice(1, None))
                .otherwise(pl.col("property"))
            )
            .filter(pl.col("value_normalized").is_not_null())
        )

        best_rows_lf = (
            combined_lf.filter(pl.col("property").is_in(ALL_PIVOT_COLUMNS))
            .with_columns(is_exact=(pl.col("relation") == "="))
            .sort("confidence_score", "is_exact", descending=True)
            .unique(subset=["inchi_key", "target", "property"], keep="first")
        )

        final_df = (
            best_rows_lf.collect()
            .pivot(values="value_normalized", index=["name", "inchi_key", "target"], on="property")
            .rename({col: col.lower().replace(" ", "_") for col in ALL_PIVOT_COLUMNS})
        )

        # Consolidate names: for each inchi_key, pick the shortest name as the canonical one
        best_name_map = final_df.group_by("inchi_key").agg(
            pl.col("name").sort_by(pl.col("name").str.len_chars()).first().alias("best_name")
        )
        final_df = final_df.join(best_name_map, on="inchi_key").drop("name").rename({"best_name": "name"})

        DRUG_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_df.write_parquet(DRUG_DB_PATH, compression="zstd")
        print(f"\n\x1b[32m-> Wrote {len(final_df):,} rows to: \x1b[3m{DRUG_DB_PATH}\x1b[0m")
        print(final_df.head())
    except Exception as e:
        print(f"❌ {type(e).__name__}: {e}", file=sys.stderr)
        raise e


if __name__ == "__main__":
    main()
