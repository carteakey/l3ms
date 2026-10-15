//! Optional shellcheck integration for editable shell scripts.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde::Deserialize;

const MAX_OUTPUT_BYTES: usize = 512 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintLevel {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintDiagnostic {
    pub line: u32,
    pub column: u32,
    pub code: u32,
    pub level: LintLevel,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintReport {
    pub available: bool,
    pub status_code: Option<i32>,
    pub diagnostics: Vec<LintDiagnostic>,
}

impl LintReport {
    pub fn summary(&self) -> String {
        if !self.available {
            return "shellcheck unavailable".to_owned();
        }
        if self.diagnostics.is_empty() {
            return "shellcheck passed".to_owned();
        }
        format!("shellcheck found {} warning(s)", self.diagnostics.len())
    }
}

pub fn lint_shell_script(path: impl AsRef<Path>) -> Result<LintReport> {
    let output = {
        #[cfg(test)]
        let _fixture_guard = crate::control::process::executable_fixture_guard();
        Command::new("shellcheck")
            .args(["--format=json1", "--severity=warning"])
            .arg(path.as_ref())
            .output()
    };
    let output = match output {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(LintReport {
                available: false,
                status_code: None,
                diagnostics: Vec::new(),
            })
        }
        Err(error) => return Err(error).context("run shellcheck"),
    };
    if output.stdout.len() > MAX_OUTPUT_BYTES {
        return Ok(LintReport {
            available: true,
            status_code: output.status.code(),
            diagnostics: vec![LintDiagnostic {
                line: 0,
                column: 0,
                code: 0,
                level: LintLevel::Error,
                message: format!("shellcheck output exceeds {MAX_OUTPUT_BYTES} bytes"),
            }],
        });
    }
    let payload: ShellcheckPayload =
        serde_json::from_slice(&output.stdout).context("parse shellcheck JSON output")?;
    let diagnostics = payload
        .comments
        .into_iter()
        .map(|comment| LintDiagnostic {
            line: comment.line,
            column: comment.column,
            code: comment.code,
            level: match comment.level.as_str() {
                "error" => LintLevel::Error,
                "info" => LintLevel::Info,
                _ => LintLevel::Warning,
            },
            message: comment.message,
        })
        .collect();
    Ok(LintReport {
        available: true,
        status_code: output.status.code(),
        diagnostics,
    })
}

#[derive(Debug, Deserialize)]
struct ShellcheckPayload {
    #[serde(default)]
    comments: Vec<ShellcheckComment>,
}

#[derive(Debug, Deserialize)]
struct ShellcheckComment {
    #[serde(default)]
    line: u32,
    #[serde(default)]
    column: u32,
    #[serde(default)]
    code: u32,
    #[serde(default)]
    level: String,
    #[serde(default)]
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_unavailable_and_clean_reports() {
        assert_eq!(
            (LintReport {
                available: false,
                status_code: None,
                diagnostics: vec![]
            })
            .summary(),
            "shellcheck unavailable"
        );
        assert_eq!(
            (LintReport {
                available: true,
                status_code: Some(0),
                diagnostics: vec![]
            })
            .summary(),
            "shellcheck passed"
        );
    }
}
