//! Matrix-style boot animation: falling glyph rain that resolves into a
//! flashing `l3ms` logo before the workbench takes over.

use std::env;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{self, Event};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::{Frame, Terminal};

use crate::theme;

type Tui = Terminal<CrosstermBackend<std::io::Stdout>>;

const LOGO: [&str; 5] = [
    "█  █████  █   █  █████",
    "█      █  ██ ██  █    ",
    "█   ████  █ █ █  █████",
    "█      █  █   █      █",
    "█  █████  █   █  █████",
];
const LOGO_WIDTH: u16 = 22;
const LOGO_HEIGHT: u16 = 5;
const RAIN_TICK: Duration = Duration::from_millis(40);
const RAIN_DURATION: Duration = Duration::from_millis(1400);
const FLASH_STEP: Duration = Duration::from_millis(100);
const FLASH_TOGGLES: usize = 6;
const FLASH_HOLD: Duration = Duration::from_millis(500);

const GLYPHS: &[char] = &[
    'ｱ', 'ｲ', 'ｳ', 'ｴ', 'ｵ', 'ｶ', 'ｷ', 'ｸ', 'ｹ', 'ｺ', 'ｻ', 'ｼ', 'ｽ', 'ｾ', 'ｿ', 'ﾀ', 'ﾁ', 'ﾂ', 'ﾃ',
    'ﾄ', 'ﾅ', 'ﾆ', 'ﾇ', 'ﾈ', 'ﾉ', 'ﾊ', 'ﾋ', 'ﾌ', 'ﾍ', 'ﾎ', 'ﾏ', 'ﾐ', 'ﾑ', 'ﾒ', 'ﾓ', 'ﾔ', 'ﾕ', 'ﾖ',
    'ﾗ', 'ﾘ', 'ﾙ', 'ﾚ', 'ﾛ', 'ﾜ', 'ﾝ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'l', 'm',
    's', ':', '=', '+', '*', '<', '>',
];

/// Play the boot animation on an already-initialized terminal. Any key press
/// skips it; `L3MS_NO_BOOT_ANIM=1` disables it entirely.
pub fn play(terminal: &mut Tui) {
    if env::var_os("L3MS_NO_BOOT_ANIM").is_some_and(|value| value == "1") {
        return;
    }
    let Ok(size) = terminal.size() else {
        return;
    };
    let area = Rect::new(0, 0, size.width, size.height);
    if area.width < LOGO_WIDTH + 4 || area.height < LOGO_HEIGHT + 4 {
        return;
    }
    let _ = terminal.hide_cursor();
    if !rain(terminal, area, Rng::from_clock()) {
        flash(terminal, area);
    }
    drain_input();
    let _ = terminal.clear();
    let _ = terminal.show_cursor();
}

struct Rng(u64);

impl Rng {
    fn from_clock() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15);
        Self(nanos ^ (u64::from(std::process::id())).rotate_left(32) | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            0
        } else {
            self.next() % bound
        }
    }
}

struct Column {
    x: u16,
    head: i32,
    speed: i32,
    len: i32,
}

impl Column {
    fn advance(&mut self, height: u16, rng: &mut Rng) {
        self.head += self.speed;
        if self.head - self.len > i32::from(height) {
            self.head = -(rng.below(u64::from(height).max(2)) as i32);
            self.speed = 1 + rng.below(2) as i32;
            self.len = 4 + rng.below((u64::from(height) / 2).max(1)) as i32;
            if self.len > i32::from(height) {
                self.len = i32::from(height);
            }
        }
    }
}

fn rain(terminal: &mut Tui, area: Rect, mut rng: Rng) -> bool {
    let mut columns = Vec::with_capacity(area.width as usize);
    for x in 0..area.width {
        if rng.below(100) >= 80 {
            continue;
        }
        columns.push(Column {
            x,
            head: -(rng.below(u64::from(area.height).max(2)) as i32) - 1,
            speed: 1 + rng.below(2) as i32,
            len: 4 + rng.below((u64::from(area.height) / 2).max(1)) as i32,
        });
    }
    let start = Instant::now();
    loop {
        if key_pressed(RAIN_TICK) {
            return true;
        }
        for column in &mut columns {
            column.advance(area.height, &mut rng);
        }
        if terminal
            .draw(|frame| draw_rain(frame, &columns, &mut rng))
            .is_err()
        {
            return true;
        }
        if start.elapsed() >= RAIN_DURATION {
            return false;
        }
    }
}

fn draw_rain(frame: &mut Frame<'_>, columns: &[Column], rng: &mut Rng) {
    let area = frame.area();
    let buffer = frame.buffer_mut();
    for column in columns {
        if column.x >= area.width {
            continue;
        }
        for d in 0..column.len {
            let y = column.head - d;
            if y >= i32::from(area.height) {
                continue;
            }
            if y < 0 {
                break;
            }
            let glyph = GLYPHS[rng.below(GLYPHS.len() as u64) as usize];
            if let Some(cell) = buffer.cell_mut((column.x, y as u16)) {
                cell.set_char(glyph);
                cell.set_style(trail_style(d as u32));
            }
        }
    }
}

fn trail_style(distance: u32) -> Style {
    let color = match distance {
        0 => Color::Rgb(0xB4, 0xFF, 0xB4),
        1..=2 => theme::GREEN,
        3..=5 => theme::DIM_GREEN,
        6..=9 => Color::Rgb(0x00, 0x64, 0x00),
        _ => Color::Rgb(0x00, 0x32, 0x00),
    };
    if distance == 0 {
        Style::default().fg(color).add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(color)
    }
}

fn flash(terminal: &mut Tui, area: Rect) {
    let origin = (
        area.x + area.width.saturating_sub(LOGO_WIDTH) / 2,
        area.y + area.height.saturating_sub(LOGO_HEIGHT) / 2,
    );
    for toggle in 0..FLASH_TOGGLES {
        let inverted = toggle % 2 == 1;
        if terminal
            .draw(|frame| draw_flash(frame, origin, inverted))
            .is_err()
        {
            return;
        }
        if !wait(FLASH_STEP) {
            return;
        }
    }
    let _ = terminal.draw(|frame| draw_flash(frame, origin, false));
    let _ = wait(FLASH_HOLD);
}

fn draw_flash(frame: &mut Frame<'_>, origin: (u16, u16), inverted: bool) {
    let area = frame.area();
    let buffer = frame.buffer_mut();
    let style = if inverted {
        Style::default().fg(Color::Black).bg(theme::GREEN)
    } else {
        theme::title()
    };
    if inverted {
        buffer.set_style(area, style);
    }
    for (row, line) in LOGO.iter().enumerate() {
        for (col, ch) in line.chars().enumerate() {
            if ch == ' ' {
                continue;
            }
            let x = origin.0 + col as u16;
            let y = origin.1 + row as u16;
            if x >= area.width || y >= area.height {
                continue;
            }
            if let Some(cell) = buffer.cell_mut((x, y)) {
                cell.set_char(ch);
                cell.set_style(style);
            }
        }
    }
}

fn key_pressed(timeout: Duration) -> bool {
    match event::poll(timeout) {
        Ok(true) => matches!(event::read(), Ok(Event::Key(_))),
        _ => false,
    }
}

fn wait(duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return true;
        }
        if key_pressed(remaining) {
            return false;
        }
    }
}

fn drain_input() {
    while let Ok(true) = event::poll(Duration::ZERO) {
        let _ = event::read();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logo_rows_are_aligned() {
        let widths: Vec<usize> = LOGO.iter().map(|row| row.chars().count()).collect();
        assert!(
            widths.iter().all(|&w| w == LOGO_WIDTH as usize),
            "{widths:?}"
        );
    }

    #[test]
    fn rng_is_deterministic_for_a_seed() {
        let mut first = Rng(42);
        let mut second = Rng(42);
        for _ in 0..16 {
            assert_eq!(first.next(), second.next());
        }
    }

    #[test]
    fn rng_below_respects_bound() {
        let mut rng = Rng(7);
        for _ in 0..64 {
            assert!(rng.below(5) < 5);
        }
        assert_eq!(rng.below(0), 0);
    }

    #[test]
    fn trail_fades_from_bright_head_to_dim_tail() {
        let head = trail_style(0).fg.expect("head style has fg");
        let mid = trail_style(3).fg.expect("mid style has fg");
        let tail = trail_style(20).fg.expect("tail style has fg");
        assert_ne!(head, mid);
        assert_ne!(mid, tail);
    }
}
