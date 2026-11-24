//! Error type shared by the core crate.

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: std::io::Error },
    #[error("{context}: {source}")]
    Json { context: String, source: serde_json::Error },
    #[error("{0}")]
    Data(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io { path: path.into(), source }
    }
    pub fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Error::Json { context: context.into(), source }
    }
    pub fn data(msg: impl Into<String>) -> Self {
        Error::Data(msg.into())
    }
}
