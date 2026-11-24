//! Fetch FDA drug labels (SPL) from the DailyMed web services (NLM).
//!
//! Only selected clinically important sections are kept, each truncated at a
//! sentence boundary, so the chunk is a short excerpt and not a full label copy.

use crate::text::{clean, dedupe_sentences, truncate_words};
use crate::Result;
use plainline_core::schema::{Source, SourceKind, Split, CANARY};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::LazyLock;

pub const API: &str = "https://dailymed.nlm.nih.gov/dailymed/services/v2";
const HL7: &str = "urn:hl7-org:v3";

/// LOINC section codes used in SPL labels: code -> (heading, word cap).
pub fn section_spec(code: &str) -> Option<(&'static str, usize)> {
    match code {
        "34066-1" => Some(("Boxed warning", 220)),
        "34067-9" => Some(("Indications and usage", 110)),
        "34068-7" => Some(("Dosage and administration", 220)),
        "34070-3" => Some(("Contraindications", 130)),
        "43685-7" => Some(("Warnings and precautions", 260)),
        "34071-1" => Some(("Warnings", 260)),
        _ => None,
    }
}

pub const ORDER: [&str; 6] = ["34066-1", "34067-9", "34068-7", "34070-3", "43685-7", "34071-1"];

pub const REPACKAGERS: &[&str] = &[
    "repack", "bryant", "a-s medication", "proficient rx", "nucare", "preferred pharm", "quality care", "direct_rx",
    "direct rx", "henry schein", "medsource", "rebel dist", "northwind", "asclemed", "enovachem", "lake erie",
    "denton", "unit dose", "pd-rx", "aphena",
];

/// One entry of the DailyMed label list.
#[derive(Debug, Clone, Deserialize)]
pub struct SplRow {
    pub title: String,
    pub setid: String,
    #[serde(default)]
    pub spl_version: Option<Value>,
}

static SEE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)\[\s*see[^\]]*\]").unwrap());
static NUMREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*\)").unwrap());
static FULLPI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)See full prescribing information for complete boxed warning\.?").unwrap());
static WARN_HEAD: LazyLock<fancy_regex::Regex> =
    LazyLock::new(|| fancy_regex::Regex::new(r"^WARNING:\s+[A-Z0-9 ,/&;()\-]{4,}?\s+(?=[A-Z][a-z])").unwrap());
static SPACE_PUNCT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+([.,;:])").unwrap());
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());
static BRACKET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s*\[.*?\]").unwrap());

/// First non-repackager Rx label whose title starts with the drug name.
pub async fn find_setid(client: &reqwest::Client, api: &str, drug_name: &str) -> Result<Option<SplRow>> {
    let r = client
        .get(format!("{api}/spls.json"))
        .query(&[("drug_name", drug_name), ("pagesize", "50")])
        .send()
        .await?
        .error_for_status()?;
    let body: Value = r.json().await?;
    let rows: Vec<SplRow> = body
        .get("data")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| serde_json::from_value(v.clone()).ok()).collect())
        .unwrap_or_default();
    let first = drug_name.split_whitespace().next().unwrap_or("").to_lowercase();
    Ok(rows.into_iter().find(|row| {
        let title = row.title.to_lowercase();
        title.starts_with(&first) && !REPACKAGERS.iter().any(|p| title.contains(p))
    }))
}

/// Remove cross-reference markers and highlight boilerplate from SPL text.
fn tidy(text: &str) -> String {
    let t = SEE.replace_all(text, "");
    let t = NUMREF.replace_all(&t, "");
    let t = FULLPI.replace_all(&t, "");
    let t = match WARN_HEAD.find(&*t) {
        Ok(Some(m)) => t[m.end()..].to_string(),
        _ => t.into_owned(),
    };
    let t = SPACE_PUNCT.replace_all(&t, "$1");
    WS.replace_all(&t, " ").trim().to_string()
}

/// Return {loinc_code: text} for the sections of interest.
pub fn parse_spl(xml: &str) -> Result<HashMap<String, String>> {
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let doc = roxmltree::Document::parse_with_options(xml, opts)?;
    let mut found: HashMap<String, String> = HashMap::new();
    let is = |n: &roxmltree::Node, name: &str| {
        n.is_element() && n.tag_name().name() == name && n.tag_name().namespace() == Some(HL7)
    };
    for sec in doc.descendants().filter(|n| is(n, "section")) {
        let Some(code) = sec.children().find(|c| is(c, "code")).and_then(|c| c.attribute("code")) else {
            continue;
        };
        if section_spec(code).is_none() {
            continue;
        }
        // Text of every child element except the title; the section's own text and the
        // tails of its direct children are left out, as ElementTree's itertext does.
        let pieces: Vec<String> = sec
            .children()
            .filter(|c| c.is_element() && !is(c, "title"))
            .map(|c| c.descendants().filter(|d| d.is_text()).filter_map(|d| d.text()).collect::<Vec<_>>().join(" "))
            .collect();
        let txt = tidy(&clean(&pieces.join(" ")));
        // Highlights sections reuse the same codes as the full text; keep the longest.
        if txt.chars().count() > found.get(code).map_or(0, |t| t.chars().count()) {
            found.insert(code.to_string(), txt);
        }
    }
    Ok(found)
}

/// Python's `str.title()`.
fn title_case(s: &str) -> String {
    let mut out = String::new();
    let mut prev_cased = false;
    for c in s.chars() {
        let cased = c.is_lowercase() || c.is_uppercase();
        if cased {
            if prev_cased {
                out.extend(c.to_lowercase());
            } else {
                out.extend(c.to_uppercase());
            }
        } else {
            out.push(c);
        }
        prev_cased = cased;
    }
    out
}

pub fn build_source(drug_name: &str, row: &SplRow, sections: &HashMap<String, String>, fetched: &str) -> Source {
    let name = title_case(&BRACKET.replace_all(&row.title, ""));
    let mut parts = vec![format!("{name} (prescribing information excerpt)")];
    for code in ORDER {
        if code == "34071-1" && sections.contains_key("43685-7") {
            continue; // prefer the PLR-format section when both exist
        }
        if let Some(text) = sections.get(code) {
            let (heading, cap) = section_spec(code).expect("code from ORDER");
            let body = truncate_words(&dedupe_sentences(text), cap);
            parts.push(format!("{}\n{body}", heading.to_uppercase()));
        }
    }
    let sid = format!("dm-{}", drug_name.to_lowercase().replace([' ', '/'], "-"));
    let version = match &row.spl_version {
        None => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) => "None".to_string(),
        Some(other) => other.to_string(),
    };
    Source {
        source_id: sid,
        kind: SourceKind::DailymedLabel,
        title: row.title.clone(),
        text: parts.join("\n\n"),
        origin_url: format!("https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={}", row.setid),
        license_note: "DailyMed label text (manufacturer-authored FDA labeling). Short excerpt; see DATA_LICENSES.md."
            .to_string(),
        fetched: fetched.to_string(),
        version: format!("spl_version={version};setid={}", row.setid),
        split: Split::Dev,
        canary: CANARY.to_string(),
    }
}

pub async fn fetch_label(client: &reqwest::Client, api: &str, drug_name: &str, fetched: &str) -> Result<Option<Source>> {
    let Some(row) = find_setid(client, api, drug_name).await? else {
        return Ok(None);
    };
    let r = client.get(format!("{api}/spls/{}.xml", row.setid)).send().await?.error_for_status()?;
    let xml = r.text().await?;
    let sections = parse_spl(&xml)?;
    if sections.is_empty() {
        return Ok(None);
    }
    Ok(Some(build_source(drug_name, &row, &sections, fetched)))
}
