use crate::{ChatClient, ClientError};
use plainline_core::judge::{JudgeResult, parse_verdicts};
use plainline_core::prompts::judge_messages;
use plainline_core::schema::FactSet;

type Pairs = Vec<(String, String)>;

/// Check every fact and must-not-say statement against one explanation.
///
/// `per_fact == false` sends one call listing all items (cheap). `per_fact == true` sends
/// one call per item, which avoids cross-item interference at higher cost.
pub async fn judge_explanation(
    client: &dyn ChatClient,
    explanation: &str,
    fs: &FactSet,
    per_fact: bool,
) -> Result<JudgeResult, ClientError> {
    let facts: Pairs = fs.facts.iter().map(|f| (f.fact_id.clone(), f.text.clone())).collect();
    let stmts: Pairs = fs
        .must_not_say
        .iter()
        .enumerate()
        .map(|(i, t)| (format!("n{}", i + 1), t.clone()))
        .collect();
    let groups: Vec<(Pairs, Pairs)> = if per_fact {
        facts
            .iter()
            .map(|f| (vec![f.clone()], vec![]))
            .chain(stmts.iter().map(|s| (vec![], vec![s.clone()])))
            .collect()
    } else {
        vec![(facts, stmts)]
    };
    let mut res = JudgeResult::default();
    for (fgrp, sgrp) in groups {
        let raw = client
            .complete(&judge_messages(explanation, &fgrp, &sgrp), 0.0, 800)
            .await?;
        let fids: Vec<String> = fgrp.iter().map(|p| p.0.clone()).collect();
        let sids: Vec<String> = sgrp.iter().map(|p| p.0.clone()).collect();
        let (fv, sv, err) = parse_verdicts(&raw, &fids, &sids);
        res.fact_verdicts.extend(fv);
        res.mns_verdicts.extend(sv);
        res.parse_errors += err;
        res.raw.push(raw);
    }
    Ok(res)
}
