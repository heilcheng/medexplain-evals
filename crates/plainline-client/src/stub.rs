//! A tiny scripted HTTP server for tests: each request gets the next canned response and
//! is recorded. It speaks just enough HTTP/1.1 for `reqwest` and is not a general server.

use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One request as the server saw it.
#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub target: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Canned replies in order; the last one repeats. A reply is `(status, body)`.
pub struct StubServer {
    pub url: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl StubServer {
    pub async fn start(replies: Vec<(u16, String)>) -> StubServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let url = format!("http://{}", listener.local_addr().expect("addr"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let log = requests.clone();
        tokio::spawn(async move {
            let mut n = 0usize;
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    return;
                };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head_end, content_len) = loop {
                    let Ok(k) = sock.read(&mut chunk).await else {
                        break (0, 0);
                    };
                    if k == 0 {
                        break (0, 0);
                    }
                    buf.extend_from_slice(&chunk[..k]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                        let len = head
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:").map(|v| v.trim().parse().unwrap_or(0)))
                            .unwrap_or(0usize);
                        break (p + 4, len);
                    }
                };
                if head_end == 0 {
                    continue;
                }
                while buf.len() < head_end + content_len {
                    let Ok(k) = sock.read(&mut chunk).await else { break };
                    if k == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..k]);
                }
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                let mut lines = head.lines();
                let mut first = lines.next().unwrap_or("").split(' ');
                let (method, target) = (
                    first.next().unwrap_or("").to_string(),
                    first.next().unwrap_or("").to_string(),
                );
                let headers = lines
                    .filter_map(|l| {
                        l.split_once(':')
                            .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                    })
                    .collect();
                let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                log.lock().unwrap().push(Recorded {
                    method,
                    target,
                    headers,
                    body,
                });
                let (status, payload) = replies[n.min(replies.len() - 1)].clone();
                n += 1;
                let resp = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        StubServer { url, requests }
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}
