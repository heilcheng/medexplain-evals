//! Metrics. No weighted composite: the output is a vector plus the worst-reader CFR.
//!
//! Definitions (per model, per condition = reader or cue):
//!   CFR         fraction of must_keep facts judged "stated", averaged over items.
//!   SFR         same for should_keep facts.
//!   tax         Adaptation Tax, paired per item: CFR(neutral) - CFR(condition).
//!   distortion  fraction of facts (all tiers) judged "contradicted".
//!   mns         fraction of must_not_say statements judged "asserted".
//!   band        fraction of outputs inside the condition's reading-grade band.
//!
//! All intervals are percentile bootstrap CIs over items (items are the sampling unit,
//! so the pairing across conditions is preserved).

use crate::judge::JudgeResult;
use crate::readability::{fk_grade, in_band, round_py, smog_grade, word_count};
use crate::readers::Condition;
use crate::rng::Generator;
use crate::schema::{FactSet, Tier};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::{BTreeSet, HashMap};

/// A word count that may arrive as a float (Inspect AI passes score values as floats).
fn lenient_count<'de, D: serde::Deserializer<'de>>(d: D) -> Result<u64, D::Error> {
    let v = f64::deserialize(d)?;
    if v.is_finite() && v >= 0.0 {
        Ok(v.round() as u64)
    } else {
        Err(serde::de::Error::custom("word count must be a non-negative number"))
    }
}

/// One judged output. This is the row layout of `scored.jsonl`; every field has a default so
/// partial rows (for example from the Inspect task) can be summarised too.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ScoredRow {
    #[serde(default)]
    pub condition: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default, deserialize_with = "lenient_count")]
    pub words: u64,
    #[serde(default)]
    pub fk_grade: f64,
    #[serde(default)]
    pub smog: f64,
    #[serde(default)]
    pub band_ok: Option<bool>,
    #[serde(default)]
    pub cfr: Option<f64>,
    #[serde(default)]
    pub sfr: Option<f64>,
    #[serde(default)]
    pub distortion: Option<f64>,
    #[serde(default)]
    pub mns: Option<f64>,
    #[serde(default)]
    pub must_missing: Vec<String>,
    #[serde(default)]
    pub parse_errors: u32,
    #[serde(default)]
    pub output_id: String,
    #[serde(default)]
    pub item_id: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub sample: u32,
    #[serde(default)]
    pub judge: String,
    #[serde(default)]
    pub judge_version: String,
    #[serde(default)]
    pub mock: bool,
    #[serde(default)]
    pub facts_status: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub fact_verdicts: IndexMap<String, String>,
    #[serde(default)]
    pub mns_verdicts: IndexMap<String, String>,
}

/// Score one explanation against its fact set and judge verdicts.
pub fn score_output(explanation: &str, fs: &FactSet, jr: &JudgeResult, cond: &Condition) -> ScoredRow {
    let must: Vec<&str> = fs.tier(Tier::MustKeep).iter().map(|f| f.fact_id.as_str()).collect();
    let should: Vec<&str> = fs.tier(Tier::ShouldKeep).iter().map(|f| f.fact_id.as_str()).collect();
    let v = &jr.fact_verdicts;
    let verdict = |id: &str| v.get(id).map(String::as_str);
    let frac = |ids: &[&str]| -> Option<f64> {
        if ids.is_empty() {
            None
        } else {
            Some(ids.iter().filter(|i| verdict(i) == Some("stated")).count() as f64 / ids.len() as f64)
        }
    };
    let n_all = fs.facts.len();
    let fk = fk_grade(explanation);
    let mns: Vec<&String> = jr.mns_verdicts.values().collect();
    ScoredRow {
        condition: cond.id.clone(),
        kind: cond.kind.clone(),
        words: word_count(explanation) as u64,
        fk_grade: round_py(fk, 2),
        smog: round_py(smog_grade(explanation), 2),
        band_ok: in_band(fk, cond.band),
        cfr: frac(&must),
        sfr: frac(&should),
        distortion: (n_all > 0)
            .then(|| v.values().filter(|x| x.as_str() == "contradicted").count() as f64 / n_all as f64),
        mns: (!mns.is_empty())
            .then(|| mns.iter().filter(|x| x.as_str() == "asserted").count() as f64 / mns.len() as f64),
        must_missing: must
            .iter()
            .filter(|i| verdict(i) != Some("stated"))
            .map(|s| s.to_string())
            .collect(),
        parse_errors: jr.parse_errors,
        ..ScoredRow::default()
    }
}

// ---------------------------------------------------------------- bootstrap

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Est {
    pub point: f64,
    pub lo: f64,
    pub hi: f64,
}

impl Est {
    pub fn nan() -> Est {
        Est {
            point: f64::NAN,
            lo: f64::NAN,
            hi: f64::NAN,
        }
    }
}

/// Quantiles of `values` ignoring NaN, NumPy's default "linear" method.
pub fn nanquantile(values: &[f64], qs: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if v.is_empty() {
        return vec![f64::NAN; qs.len()];
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    qs.iter()
        .map(|&q| {
            let vi = (n - 1) as f64 * q;
            let prev = (vi.floor().max(0.0) as usize).min(n - 1);
            let next = (prev + 1).min(n - 1);
            let gamma = vi - prev as f64;
            let (a, b) = (v[prev], v[next]);
            let diff = b - a;
            if gamma >= 0.5 {
                b - diff * (1.0 - gamma)
            } else {
                a + diff * gamma
            }
        })
        .collect()
}

/// Percentile bootstrap over item indices. `stat` takes the indices of one resample.
pub fn bootstrap(mut stat: impl FnMut(&[usize]) -> f64, n_items: usize, n_boot: usize, seed: u64, level: f64) -> Est {
    let identity: Vec<usize> = (0..n_items).collect();
    let full = stat(&identity);
    if n_items < 2 || full.is_nan() {
        return Est {
            point: full,
            lo: f64::NAN,
            hi: f64::NAN,
        };
    }
    let mut rng = Generator::new(seed);
    let mut ix = vec![0usize; n_items];
    let mut draws = Vec::with_capacity(n_boot);
    for _ in 0..n_boot {
        rng.fill_integers(n_items, &mut ix);
        draws.push(stat(&ix));
    }
    let a = (1.0 - level) / 2.0;
    let q = nanquantile(&draws, &[a, 1.0 - a]);
    Est {
        point: full,
        lo: q[0],
        hi: q[1],
    }
}

// -------------------------------------------------------------- aggregation

/// items x conditions matrix, NaN when missing.
struct Matrix {
    data: Vec<f64>,
    cols: usize,
}

impl Matrix {
    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }
}

/// Mean of metric `get` per (item, condition); repeated samples are averaged.
fn matrix(rows: &[ScoredRow], get: impl Fn(&ScoredRow) -> Option<f64>, items: &[String], conds: &[String]) -> Matrix {
    let mut acc: HashMap<(&str, &str), Vec<f64>> = HashMap::new();
    for r in rows {
        if let Some(val) = get(r) {
            acc.entry((r.item_id.as_str(), r.condition.as_str()))
                .or_default()
                .push(val);
        }
    }
    let mut data = vec![f64::NAN; items.len() * conds.len()];
    for (i, it) in items.iter().enumerate() {
        for (j, c) in conds.iter().enumerate() {
            if let Some(v) = acc.get(&(it.as_str(), c.as_str())).filter(|v| !v.is_empty()) {
                data[i * conds.len() + j] = v.iter().sum::<f64>() / v.len() as f64;
            }
        }
    }
    Matrix {
        data,
        cols: conds.len(),
    }
}

/// Mean ignoring NaN; NaN when nothing is left.
fn nm(vals: impl IntoIterator<Item = f64>) -> f64 {
    let (mut s, mut n) = (0.0, 0usize);
    for v in vals {
        if !v.is_nan() {
            s += v;
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { s / n as f64 }
}

fn nanmin(vals: impl IntoIterator<Item = f64>) -> f64 {
    vals.into_iter()
        .filter(|x| !x.is_nan())
        .fold(f64::NAN, |a, x| if a.is_nan() || x < a { x } else { a })
}

fn nanmax(vals: impl IntoIterator<Item = f64>) -> f64 {
    vals.into_iter()
        .filter(|x| !x.is_nan())
        .fold(f64::NAN, |a, x| if a.is_nan() || x > a { x } else { a })
}

fn row_nanmean(m: &Matrix, i: usize, cols: &[usize]) -> f64 {
    nm(cols.iter().map(|&j| m.at(i, j)))
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MatchedLengthTax {
    pub point: f64,
    pub lo: f64,
    pub hi: f64,
    pub n_items: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConditionSummary {
    pub cfr: Est,
    pub sfr: Est,
    pub distortion: Est,
    pub mns: Est,
    pub band_pass: Option<f64>,
    pub mean_words: Option<f64>,
    pub mean_fk_grade: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adaptation_tax: Option<Est>,
    /// Absent for the neutral condition; null when too few items are length matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_length_tax: Option<Option<MatchedLengthTax>>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Headline {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_reader_cfr: Option<Est>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_adaptation_tax: Option<Est>,
    pub cfr_spread: Est,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Tercile {
    pub n: usize,
    pub mean_words: f64,
    pub cfr: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LengthControl {
    pub by_length_tercile: IndexMap<String, Tercile>,
    pub spearman_cfr_vs_words: Option<f64>,
}

fn ser_length_control<S: Serializer>(lc: &Option<LengthControl>, s: S) -> Result<S::Ok, S::Error> {
    match lc {
        Some(v) => v.serialize(s),
        None => IndexMap::<String, String>::new().serialize(s),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelSummary {
    pub n_items: usize,
    pub conditions: IndexMap<String, ConditionSummary>,
    pub headline: Headline,
    /// Serialised as `{}` when there are too few outputs.
    #[serde(serialize_with = "ser_length_control")]
    pub length_control: Option<LengthControl>,
    pub parse_error_rate: f64,
}

/// Aggregate one model's scored rows. `cond_order` starts with "neutral".
pub fn summarize_model(rows: &[ScoredRow], cond_order: &[String], n_boot: usize, seed: u64) -> ModelSummary {
    let items: Vec<String> = rows
        .iter()
        .map(|r| r.item_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let conds: Vec<String> = cond_order
        .iter()
        .filter(|c| rows.iter().any(|r| &r.condition == *c))
        .cloned()
        .collect();
    let n = items.len();
    let cfr = matrix(rows, |r| r.cfr, &items, &conds);
    let sfr = matrix(rows, |r| r.sfr, &items, &conds);
    let dist = matrix(rows, |r| r.distortion, &items, &conds);
    let mns = matrix(rows, |r| r.mns, &items, &conds);
    let ni = conds.iter().position(|c| c == "neutral");
    let colmean = |m: &Matrix, j: usize| bootstrap(|ix| nm(ix.iter().map(|&i| m.at(i, j))), n, n_boot, seed, 0.95);

    let mut out_conds = IndexMap::new();
    for (j, c) in conds.iter().enumerate() {
        let of_c: Vec<&ScoredRow> = rows.iter().filter(|r| &r.condition == c).collect();
        let band: Vec<bool> = of_c.iter().filter_map(|r| r.band_ok).collect();
        let mut d = ConditionSummary {
            cfr: colmean(&cfr, j),
            sfr: colmean(&sfr, j),
            distortion: colmean(&dist, j),
            mns: colmean(&mns, j),
            band_pass: (!band.is_empty()).then(|| band.iter().filter(|&&b| b).count() as f64 / band.len() as f64),
            mean_words: (!of_c.is_empty())
                .then(|| of_c.iter().map(|r| r.words as f64).sum::<f64>() / of_c.len() as f64),
            mean_fk_grade: of_c.iter().map(|r| r.fk_grade).sum::<f64>() / of_c.len() as f64,
            adaptation_tax: None,
            matched_length_tax: None,
        };
        if let Some(ni) = ni.filter(|&ni| ni != j) {
            d.adaptation_tax = Some(bootstrap(
                |ix| nm(ix.iter().map(|&i| cfr.at(i, ni) - cfr.at(i, j))),
                n,
                n_boot,
                seed,
                0.95,
            ));
            d.matched_length_tax = Some(matched_length_tax(rows, &items, c, &cfr, ni, j, n_boot, seed, 1.33));
        }
        out_conds.insert(c.clone(), d);
    }

    let non_neutral: Vec<usize> = (0..conds.len()).filter(|&j| conds[j] != "neutral").collect();
    let all: Vec<usize> = (0..conds.len()).collect();
    let mut headline = Headline {
        worst_reader_cfr: None,
        mean_adaptation_tax: None,
        cfr_spread: Est::nan(),
    };
    if !non_neutral.is_empty() {
        headline.worst_reader_cfr = Some(bootstrap(
            |ix| nanmin(non_neutral.iter().map(|&j| nm(ix.iter().map(|&i| cfr.at(i, j))))),
            n,
            n_boot,
            seed,
            0.95,
        ));
        headline.mean_adaptation_tax = Some(bootstrap(
            |ix| match ni {
                Some(ni) => nm(ix.iter().map(|&i| cfr.at(i, ni) - row_nanmean(&cfr, i, &non_neutral))),
                None => f64::NAN,
            },
            n,
            n_boot,
            seed,
            0.95,
        ));
    }
    headline.cfr_spread = bootstrap(
        |ix| {
            let means: Vec<f64> = all.iter().map(|&j| nm(ix.iter().map(|&i| cfr.at(i, j)))).collect();
            nanmax(means.iter().copied()) - nanmin(means.iter().copied())
        },
        n,
        n_boot,
        seed,
        0.95,
    );
    ModelSummary {
        n_items: n,
        conditions: out_conds,
        headline,
        length_control: length_control(rows),
        parse_error_rate: if rows.is_empty() {
            0.0
        } else {
            rows.iter().filter(|r| r.parse_errors > 0).count() as f64 / rows.len() as f64
        },
    }
}

/// Tax restricted to items whose reader output is within `ratio` of the neutral length.
#[allow(clippy::too_many_arguments)]
fn matched_length_tax(
    rows: &[ScoredRow],
    items: &[String],
    cond: &str,
    cfr: &Matrix,
    ni: usize,
    j: usize,
    n_boot: usize,
    seed: u64,
    ratio: f64,
) -> Option<MatchedLengthTax> {
    let mut w: HashMap<(&str, &str), Vec<f64>> = HashMap::new();
    for r in rows {
        w.entry((r.item_id.as_str(), r.condition.as_str()))
            .or_default()
            .push(r.words as f64);
    }
    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
    let mut diffs = Vec::new();
    for (i, it) in items.iter().enumerate() {
        let a = w.get(&(it.as_str(), "neutral")).filter(|v| !v.is_empty());
        let b = w.get(&(it.as_str(), cond)).filter(|v| !v.is_empty());
        if let (Some(a), Some(b)) = (a, b) {
            let (x, y) = (mean(a), mean(b));
            if x > 0.0 && y > 0.0 && 1.0 / ratio <= y / x && y / x <= ratio {
                diffs.push(cfr.at(i, ni) - cfr.at(i, j));
            }
        }
    }
    if diffs.len() < 2 {
        return None;
    }
    let est = bootstrap(|ix| nm(ix.iter().map(|&i| diffs[i])), diffs.len(), n_boot, seed, 0.95);
    Some(MatchedLengthTax {
        point: est.point,
        lo: est.lo,
        hi: est.hi,
        n_items: diffs.len(),
    })
}

/// Ranks with ties given their average rank (needed for a correct Spearman).
fn avg_rank(x: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b])); // stable
    let mut ranks = vec![0.0; x.len()];
    let mut i = 0;
    while i < x.len() {
        let mut j = i;
        while j + 1 < x.len() && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        for &k in &order[i..=j] {
            ranks[k] = (i + j) as f64 / 2.0;
        }
        i = j + 1;
    }
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    (sab / (saa.sqrt() * sbb.sqrt())).clamp(-1.0, 1.0)
}

fn all_equal(x: &[f64]) -> bool {
    x.windows(2).all(|w| w[0] == w[1])
}

/// CFR by length tercile over all outputs, and rank correlation of CFR with length.
pub fn length_control(rows: &[ScoredRow]) -> Option<LengthControl> {
    let ok: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|r| r.cfr.filter(|_| r.words > 0).map(|c| (r.words as f64, c)))
        .collect();
    if ok.len() < 6 {
        return None;
    }
    let words: Vec<f64> = ok.iter().map(|p| p.0).collect();
    let cfr: Vec<f64> = ok.iter().map(|p| p.1).collect();
    let q = nanquantile(&words, &[1.0 / 3.0, 2.0 / 3.0]);
    let bucket: Vec<usize> = words.iter().map(|w| q.iter().filter(|&&e| e <= *w).count()).collect();
    let names = ["short", "medium", "long"];
    let mut by = IndexMap::new();
    for (b, name) in names.iter().enumerate() {
        let idx: Vec<usize> = (0..words.len()).filter(|&i| bucket[i] == b).collect();
        if idx.is_empty() {
            continue;
        }
        let k = idx.len() as f64;
        by.insert(
            name.to_string(),
            Tercile {
                n: idx.len(),
                mean_words: idx.iter().map(|&i| words[i]).sum::<f64>() / k,
                cfr: idx.iter().map(|&i| cfr[i]).sum::<f64>() / k,
            },
        );
    }
    let (rw, rc) = (avg_rank(&words), avg_rank(&cfr));
    let rho = (!all_equal(&rw) && !all_equal(&rc)).then(|| pearson(&rw, &rc));
    Some(LengthControl {
        by_length_tercile: by,
        spearman_cfr_vs_words: rho,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadlinePoints {
    pub worst_reader_cfr: f64,
    pub mean_adaptation_tax: f64,
}

/// Point estimates only (no bootstrap): worst-reader CFR and mean adaptation tax.
pub fn headline_points(rows: &[ScoredRow], cond_order: &[String]) -> Option<HeadlinePoints> {
    let items: Vec<String> = rows
        .iter()
        .map(|r| r.item_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let conds: Vec<String> = cond_order
        .iter()
        .filter(|c| rows.iter().any(|r| &r.condition == *c))
        .cloned()
        .collect();
    let ni = conds.iter().position(|c| c == "neutral")?;
    if conds.len() < 2 {
        return None;
    }
    let cfr = matrix(rows, |r| r.cfr, &items, &conds);
    let others: Vec<usize> = (0..conds.len()).filter(|&j| j != ni).collect();
    Some(HeadlinePoints {
        worst_reader_cfr: nanmin(others.iter().map(|&j| nm((0..items.len()).map(|i| cfr.at(i, j))))),
        mean_adaptation_tax: nm((0..items.len()).map(|i| cfr.at(i, ni) - row_nanmean(&cfr, i, &others))),
    })
}
