use plainline_client::{MockClient, ScriptedClient, judge_explanation};
use plainline_core::fixtures::{item, verdict_json};

#[tokio::test]
async fn batched_is_one_call() {
    let it = item();
    let c = ScriptedClient::new(
        &[&verdict_json(&[
            ("f1", "stated"),
            ("f2", "absent"),
            ("f3", "stated"),
            ("f4", "contradicted"),
            ("n1", "not_asserted"),
        ])],
        "scripted",
    );
    let r = judge_explanation(&c, "some text", &it.facts, false).await.unwrap();
    assert_eq!(c.calls().len(), 1);
    assert_eq!(r.fact_verdicts["f4"], "contradicted");
    assert_eq!(r.parse_errors, 0);
}

#[tokio::test]
async fn per_fact_calls_each_item() {
    let it = item();
    let c = ScriptedClient::new(&[&verdict_json(&[("f1", "stated")])], "scripted");
    let r = judge_explanation(&c, "text", &it.facts, true).await.unwrap();
    assert_eq!(c.calls().len(), it.facts.facts.len() + 1); // 4 facts and 1 statement
    assert!(r.parse_errors > 0); // the scripted reply never contains the other ids
}

#[tokio::test]
async fn judge_prompt_hides_tiers_and_source() {
    let it = item();
    let c = ScriptedClient::new(&[&verdict_json(&[("f1", "stated")])], "scripted");
    judge_explanation(&c, "EXPL", &it.facts, false).await.unwrap();
    let user = &c.calls()[0][1].content;
    assert!(!user.contains("must_keep") && !user.contains("should_keep"));
    assert!(user.contains("EXPL") && user.contains("Can cause liver damage"));
    assert!(!user.contains(&it.source.text));
}

#[tokio::test]
async fn scripted_client_repeats_its_last_reply() {
    use plainline_client::ChatClient;
    let c = ScriptedClient::new(&["a", "b"], "s");
    let got: Vec<String> = vec![
        c.complete(&[], 0.0, 1).await.unwrap(),
        c.complete(&[], 0.0, 1).await.unwrap(),
        c.complete(&[], 0.0, 1).await.unwrap(),
    ];
    assert_eq!(got, ["a", "b", "b"]);
}

#[tokio::test]
async fn mock_judge_marks_covered_facts_stated() {
    let it = item();
    let expl = "Testdrug can cause liver damage. Do not exceed 4 grams per day.";
    let r = judge_explanation(&MockClient, expl, &it.facts, false).await.unwrap();
    assert_eq!(r.fact_verdicts["f1"], "stated");
    assert_eq!(r.fact_verdicts["f2"], "stated");
    assert_eq!(r.fact_verdicts["f4"], "absent");
    assert_eq!(r.mns_verdicts["n1"], "not_asserted");
    assert_eq!(r.parse_errors, 0);
}

#[tokio::test]
async fn mock_writer_echoes_a_share_of_the_source() {
    use plainline_client::ChatClient;
    use plainline_core::prompts::writer_messages;
    let src = "One a. Two b. Three c. Four d. Five e. Six f. Seven g. Eight h.";
    let plain = MockClient
        .complete(&writer_messages(src, "Reader: a physician."), 0.0, 10)
        .await
        .unwrap();
    let lay = MockClient
        .complete(&writer_messages(src, "Reader: sixth-grade adult."), 0.0, 10)
        .await
        .unwrap();
    assert_eq!(plain, "One a. Two b. Three c. Four d.");
    assert_eq!(lay, "One a. Two b.");
}
