use crate::{ChatClient, ClientError, Message};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::time::Duration;

/// Settings for [`OpenAiCompatClient`].
#[derive(Debug, Clone)]
pub struct OpenAiConfig {
    pub timeout: Duration,
    pub retries: u32,
    /// Wait before retry `k` (0-based) is `backoff * 2^k`.
    pub backoff: Duration,
    /// Extra fields merged into every request body.
    pub extra_body: Map<String, Value>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        OpenAiConfig {
            timeout: Duration::from_secs(120),
            retries: 3,
            backoff: Duration::from_secs(1),
            extra_body: Map::new(),
        }
    }
}

/// POST `{base_url}/chat/completions`. The API key is read from an environment variable.
pub struct OpenAiCompatClient {
    base_url: String,
    model: String,
    api_key: String,
    config: OpenAiConfig,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: &str, model: &str, api_key_env: &str) -> Self {
        Self::with_config(base_url, model, api_key_env, OpenAiConfig::default())
    }

    pub fn with_config(base_url: &str, model: &str, api_key_env: &str, config: OpenAiConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("HTTP client configuration is valid");
        OpenAiCompatClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: std::env::var(api_key_env).unwrap_or_default(),
            config,
            http,
        }
    }

    /// One attempt. `Ok(Err(..))` is a retryable failure, `Err(..)` is final.
    async fn attempt(&self, body: &Value) -> Result<Result<String, String>, ClientError> {
        let mut req = self.http.post(format!("{}/chat/completions", self.base_url)).json(body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => return Ok(Err(e.to_string())),
        };
        let status = resp.status();
        if matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504) {
            return Ok(Err(format!("retryable HTTP {}", status.as_u16())));
        }
        if status.is_client_error() {
            // Auth or request errors will not fix themselves; fail at once.
            let text = resp.text().await.unwrap_or_default();
            let body: String = text.chars().take(200).collect();
            return Err(ClientError::Http { status: status.as_u16(), body });
        }
        if !status.is_success() {
            return Ok(Err(format!("HTTP {}", status.as_u16())));
        }
        let value: Value = match resp.json().await {
            Ok(v) => v,
            Err(e) => return Ok(Err(e.to_string())),
        };
        match value.pointer("/choices/0/message") {
            Some(m) if m.is_object() => Ok(Ok(m.get("content").and_then(Value::as_str).unwrap_or("").to_string())),
            _ => Ok(Err("response has no choices[0].message".to_string())),
        }
    }
}

#[async_trait]
impl ChatClient for OpenAiCompatClient {
    fn name(&self) -> &str {
        &self.model
    }

    async fn complete(&self, messages: &[Message], temperature: f64, max_tokens: u32) -> Result<String, ClientError> {
        let mut body = json!({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });
        for (k, v) in &self.config.extra_body {
            body[k] = v.clone();
        }
        let mut last = String::new();
        for attempt in 0..self.config.retries {
            match self.attempt(&body).await? {
                Ok(text) => return Ok(text),
                Err(e) => last = e,
            }
            if attempt + 1 < self.config.retries {
                tokio::time::sleep(self.config.backoff * 2u32.pow(attempt)).await;
            }
        }
        Err(ClientError::Exhausted { attempts: self.config.retries, last })
    }
}
