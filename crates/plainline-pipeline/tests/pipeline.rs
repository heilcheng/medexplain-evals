use plainline_client::ScriptedClient;
use plainline_client::stub::StubServer;
use plainline_core::schema::{DRAFT_STATUS, Fact, FactSet, Source, Status, Tier, load_items, validate_factset};
use plainline_pipeline::draft::{llm_draft, rule_draft};
use plainline_pipeline::text::{clean, dedupe_sentences, sentences, truncate_words};
use plainline_pipeline::{dailymed, medlineplus};
use std::path::Path;

const SPL: &str = r#"<?xml version="1.0"?>
<document xmlns="urn:hl7-org:v3"><component><structuredBody>
<component><section><code code="34066-1"/><title>Boxed</title><text>WARNING: BLEEDING RISK Testdrug can cause major or fatal bleeding [see Warnings and Precautions (5.1)]. Monitor INR regularly in all patients.</text></section></component>
<component><section><code code="34070-3"/><text>Do not use in pregnancy. Contraindicated with severe liver disease.</text></section></component>
<component><section><code code="34068-7"/><text>Do not exceed 10 mg daily. Start at 2 mg once daily.</text></section></component>
<component><section><code code="99999-9"/><text>ignored</text></section></component>
</structuredBody></component></document>"#;

const TOPICS: &str = r#"<?xml version="1.0"?><health-topics>
<health-topic title="Sepsis" language="English" url="https://medlineplus.gov/sepsis.html">
<full-summary>&lt;p&gt;Sepsis is a medical emergency. Call 911 if you have a fast heartbeat and confusion.&lt;/p&gt;</full-summary></health-topic>
<health-topic title="Sepsis" language="Spanish" url="x"><full-summary>hola</full-summary></health-topic>
</health-topics>"#;

fn row() -> dailymed::SplRow {
    serde_json::from_value(serde_json::json!({"title": "TESTDRUG TABLET", "setid": "abc", "spl_version": 2})).unwrap()
}

fn src() -> Source {
    dailymed::build_source("testdrug", &row(), &dailymed::parse_spl(SPL).unwrap(), "2026-01-01")
}

#[test]
fn parse_spl_selects_sections_and_strips_refs() {
    let s = dailymed::parse_spl(SPL).unwrap();
    let mut keys: Vec<_> = s.keys().cloned().collect();
    keys.sort();
    assert_eq!(keys, ["34066-1", "34068-7", "34070-3"]);
    assert!(!s["34066-1"].contains("[see") && !s["34066-1"].contains("(5.1)"));
    assert!(s["34066-1"].starts_with("Testdrug can cause"));
}

#[test]
fn build_source_marks_provenance() {
    let s = src();
    assert!(s.version.contains("setid=abc") && s.version.starts_with("spl_version=2;"));
    assert!(s.text.contains("BOXED WARNING"));
    assert!(s.origin_url.ends_with("setid=abc"));
    assert_eq!(s.source_id, "dm-testdrug");
    assert!(s.text.starts_with("Testdrug Tablet (prescribing information excerpt)"));
}

#[test]
fn parse_topics_english_only() {
    let t = medlineplus::parse_topics(TOPICS).unwrap();
    assert_eq!(t.keys().collect::<Vec<_>>(), ["sepsis"]);
    assert!(!t["sepsis"].summary.contains("<p>"));
}

#[test]
fn latest_topics_file_is_the_newest_listed() {
    let page = "a mplus_topics_2026-01-02.xml b mplus_topics_2026-03-04.xml c mplus_topics_2026-01-02.xml";
    assert_eq!(
        medlineplus::latest_topics_name(page).unwrap(),
        "mplus_topics_2026-03-04.xml"
    );
    assert!(medlineplus::latest_topics_name("nothing").is_err());
}

#[test]
fn sentences_and_truncate() {
    assert_eq!(sentences("One two. Three four five. Six.").len(), 3);
    assert_eq!(truncate_words("One two. Three four five. Six seven.", 4), "One two.");
    assert!(truncate_words("A very long single sentence here.", 2).starts_with("A very"));
}

#[test]
fn rule_draft_is_marked_draft_and_tiers() {
    let fs = rule_draft(&src());
    assert_eq!(fs.status, Status::Draft);
    assert_eq!(fs.generator, "rule-v0");
    assert!(!fs.is_validated());
    let musts: String = fs
        .tier(Tier::MustKeep)
        .iter()
        .map(|f| f.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        musts.contains("fatal bleeding")
            && musts.contains("Do not use in pregnancy")
            && musts.contains("Do not exceed 10 mg")
    );
    assert!(!fs.must_not_say.is_empty());
    assert!(validate_factset(&fs, true).is_empty());
}

#[test]
fn rule_draft_topic() {
    let topic = &medlineplus::parse_topics(TOPICS).unwrap()["sepsis"];
    let fs = rule_draft(&medlineplus::build_source(topic, "x.xml", "2026-01-01"));
    assert!(fs.tier(Tier::MustKeep).iter().any(|f| f.text.contains("Call 911")));
}

#[tokio::test]
async fn llm_draft_parses_and_stays_draft() {
    let reply = serde_json::json!({
        "facts": [{"text": "Do not exceed 10 mg", "tier": "must_keep", "section": "dose"},
                  {"text": "bogus", "tier": "critical"},
                  {"text": "Used for X", "tier": "optional"}],
        "must_not_say": ["Take extra"]
    })
    .to_string();
    let fs = llm_draft(&ScriptedClient::new(&[&reply], "m1"), &src()).await.unwrap();
    let tiers: Vec<_> = fs.facts.iter().map(|f| f.tier).collect();
    assert_eq!(tiers, [Tier::MustKeep, Tier::Optional]);
    assert_eq!(fs.status, Status::Draft);
    assert_eq!(fs.generator, "llm:m1");
    assert_eq!(fs.must_not_say, ["Take extra"]);
}

#[test]
fn validate_rejects_fake_review() {
    let mut fs = FactSet {
        source_id: "a".into(),
        facts: vec![Fact {
            fact_id: "f1".into(),
            text: "t".into(),
            tier: Tier::MustKeep,
            section: String::new(),
        }],
        must_not_say: vec![],
        status: Status::ClinicianReviewed,
        generator: "g".into(),
        reviewers: vec!["only one".into()],
    };
    assert!(validate_factset(&fs, true).join(" ").contains("fewer than 2 reviewers"));
    fs.reviewers.push("second".into());
    assert!(validate_factset(&fs, true).is_empty() && fs.is_validated());
    let empty = FactSet {
        facts: vec![],
        status: Status::Draft,
        reviewers: vec![],
        ..fs.clone()
    };
    assert!(validate_factset(&empty, true).join(" ").contains("no must_keep"));
    assert!(validate_factset(&empty, false).is_empty());
    let dup = FactSet {
        facts: vec![fs.facts[0].clone(), fs.facts[0].clone()],
        ..fs
    };
    assert!(validate_factset(&dup, true).join(" ").contains("duplicate"));
    assert_eq!(DRAFT_STATUS, "draft, needs clinician review");
}

#[test]
fn dedupe_sentences_and_aphena_filtered() {
    assert_eq!(
        dedupe_sentences("Bleeding can be fatal. Check INR. Bleeding can be fatal."),
        "Bleeding can be fatal. Check INR."
    );
    assert!(dailymed::REPACKAGERS.contains(&"aphena"));
}

#[test]
fn clean_decodes_entities_like_python() {
    assert_eq!(
        clean("a&nbsp;b &amp c &copy &#8217; &lt;p&gt;x&lt;/p&gt; &notanentity;"),
        "a b & c \u{a9} \u{2019} x \u{ac}anentity;"
    );
    assert_eq!(clean("1 < 2 and <b>bold</b>"), "1 < 2 and bold");
}

#[test]
fn rule_draft_reproduces_the_committed_pilot_facts() {
    // The committed pilot fact sets were written by the rule drafter; it must reproduce them.
    let items = load_items(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"), None).unwrap();
    assert!(items.len() >= 28);
    for it in &items {
        assert_eq!(rule_draft(&it.source), it.facts, "{}", it.source.source_id);
    }
}

#[test]
fn text_helpers_match_python() {
    let g: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/pipeline_text.json"))
            .unwrap(),
    )
    .unwrap();
    let corpus: std::collections::HashMap<String, String> =
        load_items(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data"), None)
            .unwrap()
            .into_iter()
            .map(|i| (i.source.source_id, i.source.text))
            .collect();
    let rows = g.as_array().unwrap();
    assert!(rows.len() > 40);
    for row in rows {
        let text = match row.get("source_id") {
            Some(id) => corpus[id.as_str().unwrap()].clone(),
            None => row["text"].as_str().unwrap().to_string(),
        };
        let strs = |k: &str| -> Vec<String> { serde_json::from_value(row[k].clone()).unwrap() };
        assert_eq!(clean(&text), row["clean"].as_str().unwrap(), "clean {text:?}");
        assert_eq!(sentences(&text), strs("sentences"), "sentences {text:?}");
        assert_eq!(truncate_words(&text, 30), row["truncate_30"].as_str().unwrap());
        assert_eq!(truncate_words(&text, 4), row["truncate_4"].as_str().unwrap());
        assert_eq!(dedupe_sentences(&text), row["dedupe"].as_str().unwrap());
    }
}

#[tokio::test]
async fn fetch_label_against_a_stub_server() {
    let list = serde_json::json!({"data": [
        {"title": "TESTDRUG REPACK TABLET [A-S MEDICATION SOLUTIONS]", "setid": "bad", "spl_version": 1},
        {"title": "TESTDRUG TABLET [ACME]", "setid": "abc", "spl_version": 2},
    ]})
    .to_string();
    let server = StubServer::start(vec![(200, list), (200, SPL.to_string())]).await;
    let client = reqwest::Client::new();
    let s = dailymed::fetch_label(&client, &server.url, "testdrug", "2026-01-01")
        .await
        .unwrap()
        .unwrap();
    assert!(s.origin_url.ends_with("setid=abc"));
    let reqs = server.requests();
    assert_eq!(reqs[0].target, "/spls.json?drug_name=testdrug&pagesize=50");
    assert_eq!(reqs[1].target, "/spls/abc.xml");
}
