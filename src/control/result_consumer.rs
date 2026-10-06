//! Transactional, bounded reassembly of native result pages for all clients.
//! This module never submits/replays work or opens artifact paths. Begin at
//! cursor zero; transport reconnects continue from `cursor()` using the same
//! retained assembler. A fresh client must reconstruct from zero.

use serde::Serialize;
use serde_json::{Map, Value};

use super::catalog::Lane;
use super::ipc::validate_request_id;
use super::result_buffer::{
    Availability, MediaArtifact, PageEvent, ResultPage, ResultTerminal, MAX_JOB_EVENT_BYTES,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConsumeError {
    InvalidPage,
    WrongRequest,
    WrongLane,
    InvalidSequence,
    InvalidNativeOutput,
    CapacityExceeded,
    Unknown,
    Expired,
}

impl std::fmt::Display for ConsumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "coordinator result assembly: {self:?}")
    }
}
impl std::error::Error for ConsumeError {}

/// Explicit lane-native output; reasoning is never merged into visible chat.
/// Media references stay relative to their declared root and are not resolved.
#[derive(Clone, Serialize)]
#[serde(tag = "lane", rename_all = "snake_case")]
pub enum NativeOutput {
    Language { content: String, reasoning: String },
    Decisions { answers: Vec<Map<String, Value>> },
    Media { artifacts: Vec<MediaArtifact> },
}

impl std::fmt::Debug for NativeOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Language { content, reasoning } => f
                .debug_struct("Language")
                .field("content_bytes", &content.len())
                .field("reasoning_bytes", &reasoning.len())
                .finish(),
            Self::Decisions { answers } => f
                .debug_struct("Decisions")
                .field("result_count", &answers.len())
                .finish(),
            Self::Media { artifacts } => f
                .debug_struct("Media")
                .field("artifact_count", &artifacts.len())
                .finish(),
        }
    }
}

#[derive(Clone)]
enum Pending {
    Decision {
        id: u64,
        count: u32,
        next: u32,
        json: String,
    },
    Media {
        id: u64,
        count: u32,
        next: u32,
    },
}

/// One request's bounded native output, private ID and redacted Debug. Page
/// rejection leaves every cursor, body, group and terminal field unchanged.
#[derive(Clone)]
pub struct ResultConsumer {
    request_id: String,
    lane: Lane,
    cursor: u64,
    latest: u64,
    terminal: Option<ResultTerminal>,
    encoded_bytes: usize,
    pending: Option<Pending>,
    output: NativeOutput,
}

impl std::fmt::Debug for ResultConsumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResultConsumer")
            .field("lane", &self.lane)
            .field("cursor", &self.cursor)
            .field("terminal", &self.terminal)
            .field("encoded_bytes", &self.encoded_bytes)
            .finish()
    }
}

impl ResultConsumer {
    pub fn new(request_id: &str, lane: Lane) -> Result<Self, ConsumeError> {
        validate_request_id(request_id).map_err(|_| ConsumeError::WrongRequest)?;
        let output = match lane {
            Lane::Language => NativeOutput::Language {
                content: String::new(),
                reasoning: String::new(),
            },
            Lane::Decisions => NativeOutput::Decisions {
                answers: Vec::new(),
            },
            Lane::Media => NativeOutput::Media {
                artifacts: Vec::new(),
            },
        };
        Ok(Self {
            request_id: request_id.to_owned(),
            lane,
            cursor: 0,
            latest: 0,
            terminal: None,
            encoded_bytes: 0,
            pending: None,
            output,
        })
    }

    pub fn cursor(&self) -> u64 {
        self.cursor
    }
    pub fn latest_cursor(&self) -> u64 {
        self.latest
    }
    pub fn terminal(&self) -> Option<ResultTerminal> {
        self.terminal
    }
    pub fn is_complete(&self) -> bool {
        self.terminal.is_some() && self.cursor == self.latest && self.pending.is_none()
    }
    pub fn output(&self) -> &NativeOutput {
        &self.output
    }

    /// Consume exactly the page requested after `cursor()`. Empty polls are
    /// valid; stale, skipped or duplicated events are rejected, not appended
    /// twice. Terminal metadata can arrive before the last paginated event.
    /// Staging briefly duplicates this request's bounded data; the byte cap is
    /// serialized event accounting, not a claim about exact process RSS.
    pub fn consume(&mut self, page: &ResultPage) -> Result<(), ConsumeError> {
        page.validate().map_err(|_| ConsumeError::InvalidPage)?;
        if page.request_id != self.request_id {
            return Err(ConsumeError::WrongRequest);
        }
        match page.availability {
            Availability::Unknown => return Err(ConsumeError::Unknown),
            Availability::Expired => return Err(ConsumeError::Expired),
            Availability::Available => {}
        }
        if page.latest_cursor < self.latest
            || page.next_cursor < self.cursor
            || (self.terminal.is_some()
                && (page.terminal != self.terminal || page.latest_cursor != self.latest))
            || (page.events.is_empty()
                && (page.next_cursor != self.cursor || page.next_cursor != page.latest_cursor))
            || page
                .events
                .first()
                .is_some_and(|item| self.cursor.checked_add(1) != Some(item.cursor))
        {
            return Err(ConsumeError::InvalidSequence);
        }
        let mut staged = self.clone();
        for item in &page.events {
            if staged.cursor.checked_add(1) != Some(item.cursor) {
                return Err(ConsumeError::InvalidSequence);
            }
            let bytes = serde_json::to_vec(item)
                .map_err(|_| ConsumeError::InvalidPage)?
                .len();
            staged.encoded_bytes = staged
                .encoded_bytes
                .checked_add(bytes)
                .filter(|bytes| *bytes <= MAX_JOB_EVENT_BYTES)
                .ok_or(ConsumeError::CapacityExceeded)?;
            staged.append(item.cursor, &item.event)?;
            staged.cursor = item.cursor;
        }
        if staged.cursor != page.next_cursor {
            return Err(ConsumeError::InvalidSequence);
        }
        staged.latest = page.latest_cursor;
        staged.terminal = page.terminal;
        if staged.terminal.is_some() && staged.cursor == staged.latest && staged.pending.is_some() {
            return Err(ConsumeError::InvalidNativeOutput);
        }
        *self = staged;
        Ok(())
    }

    fn append(&mut self, cursor: u64, event: &PageEvent) -> Result<(), ConsumeError> {
        match (event, &mut self.output) {
            (PageEvent::ChatContent { text }, NativeOutput::Language { content, .. }) => {
                content.push_str(text)
            }
            (PageEvent::ChatReasoning { text }, NativeOutput::Language { reasoning, .. }) => {
                reasoning.push_str(text)
            }
            (
                PageEvent::DecisionJsonPart {
                    result_id,
                    part_index,
                    part_count,
                    json,
                },
                NativeOutput::Decisions { answers },
            ) => {
                if self.pending.is_none() {
                    if *part_index != 0 || *result_id != cursor {
                        return Err(ConsumeError::InvalidSequence);
                    }
                    self.pending = Some(Pending::Decision {
                        id: *result_id,
                        count: *part_count,
                        next: 0,
                        json: String::new(),
                    });
                }
                let Some(Pending::Decision {
                    id,
                    count,
                    next,
                    json: assembled,
                }) = &mut self.pending
                else {
                    return Err(ConsumeError::InvalidSequence);
                };
                if *id != *result_id || *count != *part_count || *next != *part_index {
                    return Err(ConsumeError::InvalidSequence);
                }
                if assembled.len().saturating_add(json.len()) > 1024 * 1024 {
                    return Err(ConsumeError::CapacityExceeded);
                }
                assembled.push_str(json);
                *next += 1;
                if *next == *count {
                    let native: Map<String, Value> = serde_json::from_str(assembled)
                        .map_err(|_| ConsumeError::InvalidNativeOutput)?;
                    // Native producer emits canonical object JSON. Re-encoding
                    // rejects duplicate keys, extra whitespace and non-object
                    // values instead of silently normalizing hostile output.
                    if serde_json::to_string(&native)
                        .map_err(|_| ConsumeError::InvalidNativeOutput)?
                        != *assembled
                    {
                        return Err(ConsumeError::InvalidNativeOutput);
                    }
                    answers.push(native);
                    self.pending = None;
                }
            }
            (
                PageEvent::MediaArtifactBatch {
                    result_id,
                    batch_index,
                    batch_count,
                    artifacts: batch,
                },
                NativeOutput::Media { artifacts },
            ) => {
                if self.pending.is_none() {
                    if *batch_index != 0 || *result_id != cursor {
                        return Err(ConsumeError::InvalidSequence);
                    }
                    self.pending = Some(Pending::Media {
                        id: *result_id,
                        count: *batch_count,
                        next: 0,
                    });
                }
                let Some(Pending::Media { id, count, next }) = &mut self.pending else {
                    return Err(ConsumeError::InvalidSequence);
                };
                if *id != *result_id || *count != *batch_count || *next != *batch_index {
                    return Err(ConsumeError::InvalidSequence);
                }
                artifacts.extend(batch.iter().cloned());
                *next += 1;
                if *next == *count {
                    self.pending = None;
                }
            }
            _ => return Err(ConsumeError::WrongLane),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::result_buffer::{ArtifactKind, ResultBuffer, ResultEvent, MIN_PAGE_BYTES};
    use super::*;

    fn reconstruct(buffer: &ResultBuffer, id: &str, lane: Lane) -> ResultConsumer {
        let mut consumer = ResultConsumer::new(id, lane).unwrap();
        for _ in 0..10_000 {
            let page = buffer.page(id, consumer.cursor(), MIN_PAGE_BYTES).unwrap();
            consumer.consume(&page).unwrap();
            if consumer.is_complete() {
                return consumer;
            }
        }
        panic!("bounded result reconstruction did not complete");
    }

    #[test]
    fn all_lanes_reassemble_across_minimum_pages_without_merging_native_fields() {
        let mut buffer = ResultBuffer::new();
        buffer.begin("chat", Lane::Language).unwrap();
        buffer
            .append("chat", ResultEvent::ChatContent("🙂answer".repeat(1000)))
            .unwrap();
        buffer
            .append(
                "chat",
                ResultEvent::ChatReasoning("private-reason".repeat(100)),
            )
            .unwrap();
        buffer.terminal("chat", ResultTerminal::Succeeded).unwrap();
        let chat = reconstruct(&buffer, "chat", Lane::Language);
        let NativeOutput::Language { content, reasoning } = chat.output() else {
            panic!("wrong lane")
        };
        assert_eq!(content, &"🙂answer".repeat(1000));
        assert_eq!(reasoning, &"private-reason".repeat(100));
        assert!(!format!("{chat:?} {:?}", chat.output()).contains("private-reason"));

        buffer.begin("decisions", Lane::Decisions).unwrap();
        let answers = serde_json::json!({"question": "🌍secret-answer".repeat(1000)})
            .as_object()
            .unwrap()
            .clone();
        buffer
            .append("decisions", ResultEvent::DecisionAnswers(answers.clone()))
            .unwrap();
        buffer
            .terminal("decisions", ResultTerminal::Succeeded)
            .unwrap();
        let decisions = reconstruct(&buffer, "decisions", Lane::Decisions);
        let NativeOutput::Decisions { answers: actual } = decisions.output() else {
            panic!("wrong lane")
        };
        assert_eq!(actual, &vec![answers]);

        buffer.begin("media", Lane::Media).unwrap();
        let artifacts: Vec<_> = (0..100)
            .map(|i| MediaArtifact::new(ArtifactKind::LocalJob, format!("job/{i}.wav")).unwrap())
            .collect();
        buffer
            .append("media", ResultEvent::MediaArtifacts(artifacts.clone()))
            .unwrap();
        buffer.terminal("media", ResultTerminal::Succeeded).unwrap();
        let media = reconstruct(&buffer, "media", Lane::Media);
        let NativeOutput::Media { artifacts: actual } = media.output() else {
            panic!("wrong lane")
        };
        assert_eq!(actual, &artifacts);
    }

    #[test]
    fn stale_wrong_lane_and_skipped_pages_never_change_existing_output() {
        let mut buffer = ResultBuffer::new();
        buffer.begin("chat", Lane::Language).unwrap();
        buffer
            .append("chat", ResultEvent::ChatContent("answer".repeat(1000)))
            .unwrap();
        let mut consumer = ResultConsumer::new("chat", Lane::Language).unwrap();
        let first = buffer.page("chat", 0, MIN_PAGE_BYTES).unwrap();
        consumer.consume(&first).unwrap();
        let before = serde_json::to_vec(consumer.output()).unwrap();
        assert_eq!(consumer.consume(&first), Err(ConsumeError::InvalidSequence));
        let mut wrong = buffer
            .page("chat", consumer.cursor(), MIN_PAGE_BYTES)
            .unwrap();
        wrong.request_id = "other".into();
        assert_eq!(consumer.consume(&wrong), Err(ConsumeError::WrongRequest));
        let mut decisions = ResultConsumer::new("chat", Lane::Decisions).unwrap();
        assert_eq!(decisions.consume(&first), Err(ConsumeError::WrongLane));
        let skipped = buffer
            .page("chat", consumer.cursor() + 1, MIN_PAGE_BYTES)
            .unwrap();
        assert_eq!(
            consumer.consume(&skipped),
            Err(ConsumeError::InvalidSequence)
        );
        assert_eq!(serde_json::to_vec(consumer.output()).unwrap(), before);
        assert_eq!(consumer.cursor(), first.next_cursor);
    }

    #[test]
    fn terminal_group_and_cross_page_counts_are_checked_transactionally() {
        use super::super::result_buffer::CursorEvent;
        let mut consumer = ResultConsumer::new("decision", Lane::Decisions).unwrap();
        let first = ResultPage {
            request_id: "decision".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 1,
                event: PageEvent::DecisionJsonPart {
                    result_id: 1,
                    part_index: 0,
                    part_count: 2,
                    json: "{\"question\":".into(),
                },
            }],
            next_cursor: 1,
            latest_cursor: 2,
            terminal: Some(ResultTerminal::Succeeded),
        };
        consumer.consume(&first).unwrap();
        assert!(!consumer.is_complete());
        let mut second = ResultPage {
            request_id: "decision".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 2,
                event: PageEvent::DecisionJsonPart {
                    result_id: 1,
                    part_index: 1,
                    part_count: 2,
                    json: "\"yes\"}".into(),
                },
            }],
            next_cursor: 2,
            latest_cursor: 2,
            terminal: Some(ResultTerminal::Succeeded),
        };
        if let PageEvent::DecisionJsonPart { part_count, .. } = &mut second.events[0].event {
            *part_count = 3;
        }
        assert_eq!(
            consumer.consume(&second),
            Err(ConsumeError::InvalidSequence)
        );
        assert_eq!(consumer.cursor(), 1);
        if let PageEvent::DecisionJsonPart { part_count, .. } = &mut second.events[0].event {
            *part_count = 2;
        }
        consumer.consume(&second).unwrap();
        assert!(consumer.is_complete());
        second.terminal = Some(ResultTerminal::Cancelled);
        assert!(consumer.consume(&second).is_err());

        let mut incomplete = ResultConsumer::new("decision", Lane::Decisions).unwrap();
        let mut truncated = first;
        truncated.latest_cursor = 1;
        assert_eq!(
            incomplete.consume(&truncated),
            Err(ConsumeError::InvalidNativeOutput)
        );
        assert_eq!(incomplete.cursor(), 0);
    }

    #[test]
    fn unavailable_and_invalid_native_json_are_fixed_errors_without_body_echo() {
        use super::super::result_buffer::CursorEvent;
        let mut buffer = ResultBuffer::new();
        let mut consumer = ResultConsumer::new("decision", Lane::Decisions).unwrap();
        assert_eq!(
            consumer.consume(&buffer.page("decision", 0, MIN_PAGE_BYTES).unwrap()),
            Err(ConsumeError::Unknown)
        );
        let page = ResultPage {
            request_id: "decision".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 1,
                event: PageEvent::DecisionJsonPart {
                    result_id: 1,
                    part_index: 0,
                    part_count: 1,
                    json: "{\"secret\":1,\"secret\":2}".into(),
                },
            }],
            next_cursor: 1,
            latest_cursor: 1,
            terminal: Some(ResultTerminal::Succeeded),
        };
        assert_eq!(
            consumer.consume(&page),
            Err(ConsumeError::InvalidNativeOutput)
        );
        assert_eq!(consumer.cursor(), 0);
        buffer.begin("decision", Lane::Decisions).unwrap();
        buffer
            .terminal("decision", ResultTerminal::Interrupted)
            .unwrap();
        buffer.evict_terminal("decision").unwrap();
        assert_eq!(
            consumer.consume(&buffer.page("decision", 0, MIN_PAGE_BYTES).unwrap()),
            Err(ConsumeError::Expired)
        );
    }
}
