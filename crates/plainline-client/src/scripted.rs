use crate::{ChatClient, ClientError, Message};
use async_trait::async_trait;
use std::sync::Mutex;

/// Returns queued replies and records every call. The last reply repeats once the queue
/// holds only one. Used by tests; it measures nothing.
pub struct ScriptedClient {
    name: String,
    replies: Mutex<Vec<String>>,
    calls: Mutex<Vec<Vec<Message>>>,
}

impl ScriptedClient {
    pub fn new(replies: &[&str], name: &str) -> Self {
        ScriptedClient {
            name: name.to_string(),
            replies: Mutex::new(replies.iter().map(|s| s.to_string()).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, messages: &[Message], _temperature: f64, _max_tokens: u32) -> Result<String, ClientError> {
        self.calls.lock().unwrap().push(messages.to_vec());
        let mut r = self.replies.lock().unwrap();
        Ok(if r.len() > 1 { r.remove(0) } else { r.first().cloned().unwrap_or_default() })
    }
}
