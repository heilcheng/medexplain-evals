//! Plainline core: the pure parts of the benchmark.
//!
//! Schema and data loading, reader conditions, prompts, readability, judge verdict
//! parsing, metrics with NumPy-compatible bootstrap, and reporting. Nothing here does
//! network I/O.

pub mod error;
pub mod fixtures;
pub mod judge;
pub mod metrics;
pub mod prompts;
pub mod pyjson;
pub mod readability;
pub mod readers;
pub mod report;
pub mod rng;
pub mod schema;

pub use error::{Error, Result};
