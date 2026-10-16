//! Status bar: resident model and machine readiness at a glance.
//!
//! Every threshold lives in [`THRESHOLDS`]; views never compare raw numbers
//! themselves. Values turn amber at their warning level and red at their
//! critical level, and always keep the number beside the colour.

use ratatui::{
    style::Style,
    text::{Line, Span},
};

use crate::theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Ok,
    Warn,
    Critical,
}

impl Level {
    pub fn style(self) -> Style {
        match self {
            Self::Ok => theme::text(),
            Self::Warn => theme::warning(),
            Self::Critical => theme::error(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Percent of VRAM in use. A served model fills most of the card by design,
    /// so only the last few percent (OOM risk) warn.
    VramPercent,
    /// Percent of RAM in use (MemTotal − MemAvailable). Above the warning
    /// level SSD-offloaded experts start to re-fault from disk.
    RamPercent,
    CpuPercent,
    /// Percent used of the fullest watched filesystem.
    DiskPercent,
    GpuTemperatureC,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold {
    pub metric: Metric,
    pub warn: f64,
    pub critical: f64,
}

pub const THRESHOLDS: &[Threshold] = &[
    Threshold {
        metric: Metric::VramPercent,
        warn: 97.0,
        critical: 99.5,
    },
    Threshold {
        metric: Metric::RamPercent,
        warn: 90.0,
        critical: 96.0,
    },
    Threshold {
        metric: Metric::CpuPercent,
        warn: 90.0,
        critical: 98.0,
    },
    Threshold {
        metric: Metric::DiskPercent,
        warn: 90.0,
        critical: 97.0,
    },
    Threshold {
        metric: Metric::GpuTemperatureC,
        warn: 80.0,
        critical: 87.0,
    },
];

pub fn level(metric: Metric, value: f64) -> Level {
    let Some(threshold) = THRESHOLDS.iter().find(|entry| entry.metric == metric) else {
        return Level::Ok;
    };
    if value >= threshold.critical {
        Level::Critical
    } else if value >= threshold.warn {
        Level::Warn
    } else {
        Level::Ok
    }
}

/// Any governor other than `performance` makes bench numbers comparable only
/// to runs with the same governor (decode drops 15–25% under `powersave`).
pub fn governor_level(governor: &str) -> Level {
    if governor == "performance" {
        Level::Ok
    } else {
        Level::Warn
    }
}

/// Fixed-width gauge such as `████░░░░`; callers pair it with a number.
pub fn gauge(percent: u64, cells: usize) -> String {
    let filled = ((percent.min(100) as usize) * cells)
        .div_ceil(100)
        .min(cells);
    format!("{}{}", "█".repeat(filled), "░".repeat(cells - filled))
}

/// Plain readings for one status bar render.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatusInputs {
    pub resident_model: Option<String>,
    /// (used MiB, total MiB)
    pub vram_mib: Option<(u64, u64)>,
    /// (used MiB, total MiB)
    pub ram_mib: Option<(u64, u64)>,
    pub cpu_percent: Option<f64>,
    pub governor: Option<String>,
    /// (watched path, used percent) of the fullest watched filesystem.
    pub fullest_disk: Option<(String, u64)>,
    /// Router host and port as shown to the user, e.g. `:8080`.
    pub endpoint: String,
}

fn percent(used: u64, total: u64) -> u64 {
    (used * 100).checked_div(total).unwrap_or(0)
}

/// Segments in drop order: the first ones survive the narrowest terminals.
fn segments(inputs: &StatusInputs) -> Vec<Vec<Span<'static>>> {
    let mut segments = Vec::new();
    segments.push(match &inputs.resident_model {
        Some(model) => vec![Span::styled(format!("● {model}"), theme::title())],
        None => vec![Span::styled("○ no model loaded", theme::dim())],
    });
    if let Some((used, total)) = inputs.vram_mib {
        let value = percent(used, total);
        let style = level(Metric::VramPercent, value as f64).style();
        segments.push(vec![
            Span::styled("VRAM ", theme::dim()),
            Span::styled(gauge(value, 6), style),
            Span::styled(
                format!(" {:.1}/{:.0}G", used as f64 / 1024.0, total as f64 / 1024.0),
                style,
            ),
        ]);
    }
    if let Some((used, total)) = inputs.ram_mib {
        let value = percent(used, total);
        let style = level(Metric::RamPercent, value as f64).style();
        segments.push(vec![
            Span::styled("RAM ", theme::dim()),
            Span::styled(gauge(value, 6), style),
            Span::styled(format!(" {value}%"), style),
        ]);
    }
    if let Some(cpu) = inputs.cpu_percent {
        segments.push(vec![
            Span::styled("CPU ", theme::dim()),
            Span::styled(format!("{cpu:.0}%"), level(Metric::CpuPercent, cpu).style()),
        ]);
    }
    if let Some((path, used)) = &inputs.fullest_disk {
        segments.push(vec![
            Span::styled(format!("{path} "), theme::dim()),
            Span::styled(
                format!("{used}%"),
                level(Metric::DiskPercent, *used as f64).style(),
            ),
        ]);
    }
    if let Some(governor) = &inputs.governor {
        segments.push(vec![Span::styled(
            governor.clone(),
            governor_level(governor).style(),
        )]);
    }
    if !inputs.endpoint.is_empty() {
        segments.push(vec![Span::styled(inputs.endpoint.clone(), theme::dim())]);
    }
    segments.push(vec![Span::styled("? help", theme::dim())]);
    segments
}

/// One status bar line that fits `width`, dropping trailing segments first
/// and clipping the model name only when it alone is too wide.
pub fn status_line(inputs: &StatusInputs, width: u16) -> Line<'static> {
    let width = width as usize;
    let separator = "  ";
    let mut spans = vec![Span::raw(" ")];
    let mut used = 1;
    for (index, segment) in segments(inputs).into_iter().enumerate() {
        let segment_width: usize = segment.iter().map(|span| span.width()).sum();
        let extra = if index == 0 { 0 } else { separator.len() };
        if used + extra + segment_width > width {
            if index == 0 {
                let budget = width.saturating_sub(used);
                let text: String = segment
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
                    .chars()
                    .take(budget)
                    .collect();
                spans.push(Span::styled(text, segment[0].style));
            }
            break;
        }
        if index > 0 {
            spans.push(Span::raw(separator));
        }
        used += extra + segment_width;
        spans.extend(segment);
    }
    Line::from(spans)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(line: &Line) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect()
    }

    fn sample() -> StatusInputs {
        StatusInputs {
            resident_model: Some("qwen38-flash-next-plat".into()),
            vram_mib: Some((11_500, 12_288)),
            ram_mib: Some((47_000, 61_440)),
            cpu_percent: Some(12.4),
            governor: Some("powersave".into()),
            fullest_disk: Some(("/home".into(), 91)),
            endpoint: ":8080".into(),
        }
    }

    #[test]
    fn thresholds_cover_every_metric_and_order_warn_below_critical() {
        for metric in [
            Metric::VramPercent,
            Metric::RamPercent,
            Metric::CpuPercent,
            Metric::DiskPercent,
            Metric::GpuTemperatureC,
        ] {
            let entry = THRESHOLDS
                .iter()
                .find(|entry| entry.metric == metric)
                .unwrap_or_else(|| panic!("{metric:?} has no threshold"));
            assert!(entry.warn < entry.critical, "{metric:?}");
        }
        assert_eq!(level(Metric::RamPercent, 78.0), Level::Ok);
        assert_eq!(level(Metric::RamPercent, 90.0), Level::Warn);
        assert_eq!(level(Metric::RamPercent, 99.0), Level::Critical);
        assert_eq!(level(Metric::VramPercent, 93.6), Level::Ok);
        assert_eq!(level(Metric::DiskPercent, 91.0), Level::Warn);
        assert_eq!(governor_level("performance"), Level::Ok);
        assert_eq!(governor_level("powersave"), Level::Warn);
    }

    #[test]
    fn wide_status_line_shows_every_reading() {
        let line = text(&status_line(&sample(), 160));
        for expected in [
            "● qwen38-flash-next-plat",
            "VRAM",
            "11.2/12G",
            "RAM",
            "76%",
            "CPU 12%",
            "/home 91%",
            "powersave",
            ":8080",
            "? help",
        ] {
            assert!(line.contains(expected), "{expected} missing from {line:?}");
        }
    }

    #[test]
    fn narrow_status_line_drops_trailing_readings_and_fits() {
        for width in [20_u16, 40, 60, 80] {
            let line = status_line(&sample(), width);
            assert!(line.width() <= width as usize, "{width}: {}", text(&line));
            assert!(text(&line).contains("qwen38"), "{width}");
        }
        let line = text(&status_line(&sample(), 60));
        assert!(line.contains("VRAM"));
        assert!(!line.contains("? help"));
        let none = text(&status_line(&StatusInputs::default(), 80));
        assert!(none.contains("○ no model loaded"));
    }
}
