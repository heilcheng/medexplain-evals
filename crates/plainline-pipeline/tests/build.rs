use plainline_client::stub::StubServer;
use plainline_core::schema::{Status, load_items};
use plainline_pipeline::build::{Endpoints, build_pilot};

const SPL: &str = r#"<?xml version="1.0"?>
<document xmlns="urn:hl7-org:v3"><component><structuredBody>
<component><section><code code="34066-1"/><text>Testdrug can cause major or fatal bleeding. Monitor INR regularly in all patients.</text></section></component>
<component><section><code code="34070-3"/><text>Do not use in pregnancy. Contraindicated with severe liver disease.</text></section></component>
</structuredBody></component></document>"#;

const TOPICS: &str = r#"<?xml version="1.0"?><health-topics>
<health-topic title="Sepsis" language="English" url="https://medlineplus.gov/sepsis.html">
<full-summary>Sepsis is a medical emergency. Call 911 if you have a fast heartbeat and confusion.</full-summary></health-topic>
</health-topics>"#;

#[tokio::test]
async fn build_pilot_from_stub_servers() {
    let dm = StubServer::start(vec![
        (
            200,
            serde_json::json!({"data": [{"title": "TESTDRUG TABLET", "setid": "abc", "spl_version": 2}]}).to_string(),
        ),
        (200, SPL.to_string()),
        (200, serde_json::json!({"data": []}).to_string()),
    ])
    .await;
    let mp = StubServer::start(vec![
        (200, "<a>mplus_topics_2026-01-01.xml</a>".to_string()),
        (200, TOPICS.to_string()),
    ])
    .await;
    let endpoints = Endpoints {
        dailymed_api: dm.url.clone(),
        medlineplus_index: format!("{}/xml.html", mp.url),
        medlineplus_xml_base: format!("{}/xml", mp.url),
    };
    let dir = std::env::temp_dir().join(format!("plainline-build-{}", std::process::id()));
    let rep = build_pilot(
        &dir,
        &["testdrug", "missingdrug"],
        &["Sepsis", "Nonexistent"],
        &endpoints,
        "2026-01-01",
    )
    .await
    .unwrap();
    assert_eq!(rep.sources, 2);
    assert_eq!(rep.skipped, ["missingdrug", "Nonexistent"]);
    let items = load_items(&dir, None).unwrap();
    assert_eq!(items.len(), 2);
    assert!(
        items
            .iter()
            .all(|i| i.facts.status == Status::Draft && i.facts.generator == "rule-v0")
    );
    assert_eq!(rep.facts, items.iter().map(|i| i.facts.facts.len()).sum::<usize>());
    std::fs::remove_dir_all(dir).unwrap();
}
