//! Chat clients for Plainline. Any OpenAI-compatible endpoint works; a mock is provided
//! for offline runs and a scripted client for tests.

mod judge_call;
mod mock;
mod openai;
mod scripted;
pub mod stub;

pub use judge_call::judge_explanation;
pub use mock::MockClient;
pub use openai::{OpenAiCompatClient, OpenAiConfig};
pub use plainline_core::prompts::Message;
pub use scripted::ScriptedClient;

use async_trait::async_trait;

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    /// A 4xx answer other than 429: retrying will not help.
    #[error("model call failed with HTTP {status}: {body}")]
    Http { status: u16, body: String },
    #[error("model call failed after {attempts} attempts: {last}")]
    Exhausted { attempts: u32, last: String },
    #[error("{0}")]
    Other(String),
}

/// A chat model. `name` identifies it in the output rows.
#[async_trait]
pub trait ChatClient: Send + Sync {
    fn name(&self) -> &str;

    async fn complete(&self, messages: &[Message], temperature: f64, max_tokens: u32) -> Result<String, ClientError>;
}
