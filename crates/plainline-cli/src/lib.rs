//! The Plainline command line and the resumable runner, as a library so the Python
//! package can call the same code.

pub mod commands;
pub mod runner;

pub use commands::{Output, run_cli, run_cli_captured};
