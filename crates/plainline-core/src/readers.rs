//! Reader specs and cue conditions.

use crate::error::{Error, Result};
use crate::readability::Band;
use serde::Deserialize;
use std::path::Path;

const DEFAULT_YAML: &str = include_str!("../../../plainline/data/readers.yaml");

/// One way of asking: neutral, an explicit reader, or an implicit cue.
#[derive(Debug, Clone, PartialEq)]
pub struct Condition {
    pub id: String,
    /// "neutral", "reader" or "cue"
    pub kind: String,
    pub instruction: String,
    pub band: Option<Band>,
    pub group: String,
}

#[derive(Deserialize)]
struct Neutral {
    instruction: String,
    #[serde(default)]
    band: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Reader {
    id: String,
    instruction: String,
    #[serde(default)]
    band: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Cue {
    id: String,
    text: String,
    #[serde(default)]
    group: String,
}

#[derive(Deserialize)]
struct Config {
    neutral: Neutral,
    readers: Vec<Reader>,
    cues: Vec<Cue>,
    cue_task: String,
}

fn band(b: &Option<Vec<Option<f64>>>) -> Result<Option<Band>> {
    match b {
        None => Ok(None),
        Some(v) if v.len() == 2 => Ok(Some((v[0], v[1]))),
        Some(_) => Err(Error::data("a band must have exactly two entries")),
    }
}

fn load(path: Option<&Path>) -> Result<Config> {
    let text = match path {
        Some(p) => std::fs::read_to_string(p).map_err(|e| Error::io(p, e))?,
        None => DEFAULT_YAML.to_string(),
    };
    serde_yaml::from_str(&text).map_err(|e| Error::data(format!("readers config: {e}")))
}

/// Neutral first, then the readers (mode "adapt") or the cues (mode "cue").
pub fn conditions(mode: &str, path: Option<&Path>) -> Result<Vec<Condition>> {
    let cfg = load(path)?;
    let n = &cfg.neutral;
    let mut out = vec![Condition {
        id: "neutral".into(),
        kind: "neutral".into(),
        instruction: n.instruction.clone(),
        band: band(&n.band)?,
        group: String::new(),
    }];
    match mode {
        "adapt" => {
            for r in &cfg.readers {
                out.push(Condition {
                    id: r.id.clone(),
                    kind: "reader".into(),
                    instruction: r.instruction.clone(),
                    band: band(&r.band)?,
                    group: String::new(),
                });
            }
        }
        "cue" => {
            for c in &cfg.cues {
                out.push(Condition {
                    id: c.id.clone(),
                    kind: "cue".into(),
                    instruction: format!("{} {}", c.text, cfg.cue_task),
                    band: None,
                    group: c.group.clone(),
                });
            }
        }
        other => return Err(Error::data(format!("unknown mode {other:?}; use 'adapt' or 'cue'"))),
    }
    Ok(out)
}
