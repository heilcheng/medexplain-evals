//! Source fetching, chunking and draft fact extraction.

pub mod build;
pub mod dailymed;
pub mod draft;
pub mod entities;
pub mod medlineplus;
pub mod text;

use plainline_client::ClientError;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("XML error: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("{0}")]
    Data(String),
    #[error("model call: {0}")]
    Client(#[from] ClientError),
    #[error(transparent)]
    Core(#[from] plainline_core::Error),
}

pub type Result<T> = std::result::Result<T, PipelineError>;
