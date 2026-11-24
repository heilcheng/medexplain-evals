use plainline_client::stub::StubServer;
use plainline_client::{ChatClient, ClientError, Message, OpenAiCompatClient, OpenAiConfig};
use std::time::Duration;

fn ok(content: &str) -> (u16, String) {
    (
        200,
        serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string(),
    )
}

fn fast() -> OpenAiConfig {
    OpenAiConfig {
        backoff: Duration::from_millis(1),
        ..OpenAiConfig::default()
    }
}

fn user(text: &str) -> Vec<Message> {
    vec![Message::user(text)]
}

#[tokio::test]
async fn request_shape() {
    let server = StubServer::start(vec![ok("hi")]).await;
    // SAFETY: each test uses its own variable name and nothing else reads it concurrently.
    unsafe { std::env::set_var("PLAINLINE_TEST_KEY_SHAPE", "k") };
    let base = format!("{}/v1/", server.url);
    let c = OpenAiCompatClient::with_config(&base, "m", "PLAINLINE_TEST_KEY_SHAPE", fast());
    assert_eq!(c.name(), "m");
    assert_eq!(c.complete(&user("x"), 0.0, 100).await.unwrap(), "hi");
    let req = &server.requests()[0];
    assert_eq!(req.method, "POST");
    assert_eq!(req.target, "/v1/chat/completions");
    assert_eq!(req.header("authorization"), Some("Bearer k"));
    let body: serde_json::Value = serde_json::from_str(&req.body).unwrap();
    assert_eq!(body["model"], "m");
    assert_eq!(body["max_tokens"], 100);
    assert_eq!(body["temperature"], 0.0);
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"], "x");
}

#[tokio::test]
async fn no_key_means_no_authorization_header() {
    let server = StubServer::start(vec![ok("hi")]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    c.complete(&user("x"), 0.0, 10).await.unwrap();
    assert_eq!(server.requests()[0].header("authorization"), None);
}

#[tokio::test]
async fn extra_body_is_merged() {
    let server = StubServer::start(vec![ok("hi")]).await;
    let mut cfg = fast();
    cfg.extra_body.insert("top_p".into(), serde_json::json!(0.5));
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", cfg);
    c.complete(&user("x"), 0.0, 10).await.unwrap();
    let body: serde_json::Value = serde_json::from_str(&server.requests()[0].body).unwrap();
    assert_eq!(body["top_p"], 0.5);
}

#[tokio::test]
async fn retries_on_503() {
    let server = StubServer::start(vec![(503, "{}".into()), ok("ok")]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    assert_eq!(c.complete(&[], 0.0, 10).await.unwrap(), "ok");
    assert_eq!(server.requests().len(), 2);
}

#[tokio::test]
async fn client_errors_fail_at_once() {
    let server = StubServer::start(vec![(401, "bad key".into()), ok("never")]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    let err = c.complete(&user("x"), 0.0, 10).await.unwrap_err();
    assert!(matches!(err, ClientError::Http { status: 401, .. }), "{err}");
    assert!(err.to_string().contains("HTTP 401: bad key"));
    assert_eq!(server.requests().len(), 1);
}

#[tokio::test]
async fn gives_up_after_the_configured_attempts() {
    let server = StubServer::start(vec![(500, "{}".into())]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    let err = c.complete(&user("x"), 0.0, 10).await.unwrap_err();
    assert!(
        err.to_string().starts_with("model call failed after 3 attempts"),
        "{err}"
    );
    assert_eq!(server.requests().len(), 3);
}

#[tokio::test]
async fn null_content_is_an_empty_string_and_a_missing_message_is_retried() {
    let null = (200, r#"{"choices": [{"message": {"content": null}}]}"#.to_string());
    let server = StubServer::start(vec![null]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    assert_eq!(c.complete(&user("x"), 0.0, 10).await.unwrap(), "");

    let server = StubServer::start(vec![(200, r#"{"choices": []}"#.into()), ok("late")]).await;
    let c = OpenAiCompatClient::with_config(&server.url, "m", "PLAINLINE_TEST_KEY_UNSET", fast());
    assert_eq!(c.complete(&user("x"), 0.0, 10).await.unwrap(), "late");
}
