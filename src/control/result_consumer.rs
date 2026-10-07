//! Transactional, bounded reassembly of native result pages for all clients.
//! This module never submits/replays work or opens artifact paths. Begin at
//! cursor zero; transport reconnects continue from `cursor()` using the same
//! retained assembler. A fresh client must reconstruct from zero.

use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use serde_json::{Map, Value};

use super::catalog::Lane;
use super::ipc::validate_request_id;
use super::requests::{
    serialize_embedding_payload, EmbeddingPayload, EmbeddingUsage, MAX_EMBEDDING_JSON_BYTES,
};
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
#[derive(Clone)]
pub enum NativeOutput {
    Language {
        content: String,
        reasoning: String,
    },
    LanguageEmbeddings {
        vectors: Vec<Vec<f32>>,
        usage: EmbeddingUsage,
    },
    Decisions {
        answers: Vec<Map<String, Value>>,
    },
    Media {
        artifacts: Vec<MediaArtifact>,
    },
}

impl Serialize for NativeOutput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Language { content, reasoning } => {
                let mut state = serializer.serialize_struct("NativeOutput", 3)?;
                state.serialize_field("lane", "language")?;
                state.serialize_field("content", content)?;
                state.serialize_field("reasoning", reasoning)?;
                state.end()
            }
            Self::LanguageEmbeddings { vectors, usage } => {
                let mut state = serializer.serialize_struct("NativeOutput", 4)?;
                state.serialize_field("lane", "language")?;
                state.serialize_field("task", "embeddings")?;
                state.serialize_field("vectors", vectors)?;
                state.serialize_field("usage", usage)?;
                state.end()
            }
            Self::Decisions { answers } => {
                let mut state = serializer.serialize_struct("NativeOutput", 2)?;
                state.serialize_field("lane", "decisions")?;
                state.serialize_field("answers", answers)?;
                state.end()
            }
            Self::Media { artifacts } => {
                let mut state = serializer.serialize_struct("NativeOutput", 2)?;
                state.serialize_field("lane", "media")?;
                state.serialize_field("artifacts", artifacts)?;
                state.end()
            }
        }
    }
}

impl std::fmt::Debug for NativeOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Language { content, reasoning } => f
                .debug_struct("Language")
                .field("content_bytes", &content.len())
                .field("reasoning_bytes", &reasoning.len())
                .finish(),
            Self::LanguageEmbeddings { vectors, usage } => f
                .debug_struct("LanguageEmbeddings")
                .field("vector_count", &vectors.len())
                .field("dimensions", &vectors.first().map(Vec::len))
                .field("prompt_tokens", &usage.prompt_tokens)
                .field("total_tokens", &usage.total_tokens)
                .field("vectors", &"<redacted>")
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
    Embedding {
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
    language_task: Option<LanguageTask>,
    embedding_complete: bool,
    output: NativeOutput,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LanguageTask {
    Chat,
    Embeddings,
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
            language_task: None,
            embedding_complete: false,
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
        match event {
            PageEvent::ChatContent { .. } | PageEvent::ChatReasoning { .. } => {
                if self.language_task == Some(LanguageTask::Embeddings) {
                    return Err(ConsumeError::WrongLane);
                }
                self.language_task = Some(LanguageTask::Chat);
            }
            PageEvent::EmbeddingJsonPart {
                result_id,
                part_index,
                part_count,
                json,
            } => {
                if self.language_task == Some(LanguageTask::Chat) {
                    return Err(ConsumeError::WrongLane);
                }
                self.language_task = Some(LanguageTask::Embeddings);
                return self.append_embeddings(cursor, *result_id, *part_index, *part_count, json);
            }
            PageEvent::DecisionJsonPart { .. } | PageEvent::MediaArtifactBatch { .. } => {}
        }
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

    fn append_embeddings(
        &mut self,
        cursor: u64,
        result_id: u64,
        part_index: u32,
        part_count: u32,
        fragment: &str,
    ) -> Result<(), ConsumeError> {
        if self.embedding_complete {
            return Err(ConsumeError::InvalidSequence);
        }
        if self.pending.is_none() {
            if part_index != 0 || result_id != cursor {
                return Err(ConsumeError::InvalidSequence);
            }
            if matches!(&self.output, NativeOutput::Language { .. }) {
                self.output = NativeOutput::LanguageEmbeddings {
                    vectors: Vec::new(),
                    usage: EmbeddingUsage {
                        prompt_tokens: 0,
                        total_tokens: 0,
                    },
                };
            }
            if !matches!(&self.output, NativeOutput::LanguageEmbeddings { .. }) {
                return Err(ConsumeError::WrongLane);
            }
            self.pending = Some(Pending::Embedding {
                id: result_id,
                count: part_count,
                next: 0,
                json: String::new(),
            });
        }

        let completed_payload = {
            let Some(Pending::Embedding {
                id,
                count,
                next,
                json,
            }) = &mut self.pending
            else {
                return Err(ConsumeError::InvalidSequence);
            };
            if *id != result_id || *count != part_count || *next != part_index {
                return Err(ConsumeError::InvalidSequence);
            }
            if json.len().saturating_add(fragment.len()) > MAX_EMBEDDING_JSON_BYTES {
                return Err(ConsumeError::CapacityExceeded);
            }
            json.push_str(fragment);
            *next += 1;
            if *next == *count {
                let payload: EmbeddingPayload =
                    serde_json::from_str(json).map_err(|_| ConsumeError::InvalidNativeOutput)?;
                let canonical = serialize_embedding_payload(&payload.vectors, &payload.usage)
                    .map_err(|_| ConsumeError::InvalidNativeOutput)?;
                if canonical != json.as_bytes() {
                    return Err(ConsumeError::InvalidNativeOutput);
                }
                Some(payload)
            } else {
                None
            }
        };

        if let Some(payload) = completed_payload {
            let NativeOutput::LanguageEmbeddings { vectors, usage } = &mut self.output else {
                return Err(ConsumeError::WrongLane);
            };
            *vectors = payload.vectors;
            *usage = payload.usage;
            self.pending = None;
            self.embedding_complete = true;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::requests::EmbeddingUsage;
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
    fn native_output_serialization_keeps_chat_bytes_and_uses_language_embeddings_shape() {
        let chat = NativeOutput::Language {
            content: "answer".into(),
            reasoning: "think".into(),
        };
        assert_eq!(
            serde_json::to_string(&chat).unwrap(),
            r#"{"lane":"language","content":"answer","reasoning":"think"}"#
        );
        let embeddings = NativeOutput::LanguageEmbeddings {
            vectors: vec![vec![0.5]],
            usage: EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        };
        assert_eq!(
            serde_json::to_string(&embeddings).unwrap(),
            r#"{"lane":"language","task":"embeddings","vectors":[[0.5]],"usage":{"prompt_tokens":0,"total_tokens":0}}"#
        );
    }

    #[test]
    fn embeddings_reassemble_as_a_bounded_language_task_and_remain_redacted() {
        let mut buffer = ResultBuffer::new();
        buffer.begin("embedding", Lane::Language).unwrap();
        let vectors = vec![[0.125, -0.5, 0.0].repeat(1024); 2];
        let usage = EmbeddingUsage {
            prompt_tokens: 0,
            total_tokens: 0,
        };
        buffer
            .append(
                "embedding",
                ResultEvent::Embeddings {
                    vectors: vectors.clone(),
                    usage,
                },
            )
            .unwrap();
        buffer
            .terminal("embedding", ResultTerminal::Succeeded)
            .unwrap();
        let consumer = reconstruct(&buffer, "embedding", Lane::Language);
        let NativeOutput::LanguageEmbeddings {
            vectors: actual,
            usage: actual_usage,
        } = consumer.output()
        else {
            panic!("wrong language task")
        };
        assert_eq!(actual, &vectors);
        assert_eq!(*actual_usage, usage);
        assert!(consumer.is_complete());
        let serialized = serde_json::to_string(consumer.output()).unwrap();
        assert!(serialized.starts_with(r#"{"lane":"language","task":"embeddings","vectors":[["#));
        let debug = format!("{:?} {:?}", consumer, consumer.output());
        assert!(!debug.contains("0.125"));
        assert!(!debug.contains("-0.5"));
    }

    #[test]
    fn malformed_embedding_payload_and_chat_mixing_roll_back() {
        use super::super::result_buffer::CursorEvent;

        let payload = serialize_embedding_payload(
            &[vec![0.5]],
            &EmbeddingUsage {
                prompt_tokens: 0,
                total_tokens: 0,
            },
        )
        .unwrap();
        let embedding_event = PageEvent::EmbeddingJsonPart {
            result_id: 2,
            part_index: 0,
            part_count: 1,
            json: String::from_utf8(payload).unwrap(),
        };
        let mut chat_then_embedding = ResultConsumer::new("mixed-chat", Lane::Language).unwrap();
        chat_then_embedding
            .consume(&ResultPage {
                request_id: "mixed-chat".into(),
                availability: Availability::Available,
                events: vec![CursorEvent {
                    cursor: 1,
                    event: PageEvent::ChatContent {
                        text: "answer".into(),
                    },
                }],
                next_cursor: 1,
                latest_cursor: 2,
                terminal: None,
            })
            .unwrap();
        let before = serde_json::to_vec(chat_then_embedding.output()).unwrap();
        let next = ResultPage {
            request_id: "mixed-chat".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 2,
                event: embedding_event.clone(),
            }],
            next_cursor: 2,
            latest_cursor: 2,
            terminal: Some(ResultTerminal::Succeeded),
        };
        assert_eq!(
            chat_then_embedding.consume(&next),
            Err(ConsumeError::WrongLane)
        );
        assert_eq!(chat_then_embedding.cursor(), 1);
        assert_eq!(
            serde_json::to_vec(chat_then_embedding.output()).unwrap(),
            before
        );

        let mut embedding_then_chat =
            ResultConsumer::new("mixed-embedding", Lane::Language).unwrap();
        let PageEvent::EmbeddingJsonPart {
            part_index,
            part_count,
            json,
            ..
        } = embedding_event
        else {
            unreachable!()
        };
        embedding_then_chat
            .consume(&ResultPage {
                request_id: "mixed-embedding".into(),
                availability: Availability::Available,
                events: vec![CursorEvent {
                    cursor: 1,
                    event: PageEvent::EmbeddingJsonPart {
                        result_id: 1,
                        part_index,
                        part_count,
                        json,
                    },
                }],
                next_cursor: 1,
                latest_cursor: 2,
                terminal: None,
            })
            .unwrap();
        let before = serde_json::to_vec(embedding_then_chat.output()).unwrap();
        let chat = ResultPage {
            request_id: "mixed-embedding".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 2,
                event: PageEvent::ChatContent { text: "x".into() },
            }],
            next_cursor: 2,
            latest_cursor: 2,
            terminal: Some(ResultTerminal::Succeeded),
        };
        assert_eq!(
            embedding_then_chat.consume(&chat),
            Err(ConsumeError::WrongLane)
        );
        assert_eq!(embedding_then_chat.cursor(), 1);
        assert_eq!(
            serde_json::to_vec(embedding_then_chat.output()).unwrap(),
            before
        );

        let mut invalid = ResultConsumer::new("bad-embedding", Lane::Language).unwrap();
        let page = ResultPage {
            request_id: "bad-embedding".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 1,
                event: PageEvent::EmbeddingJsonPart {
                    result_id: 1,
                    part_index: 0,
                    part_count: 1,
                    json: r#"{"vectors":[[1.0]],"usage":{"prompt_tokens":0,"total_tokens":1}}"#
                        .into(),
                },
            }],
            next_cursor: 1,
            latest_cursor: 1,
            terminal: Some(ResultTerminal::Succeeded),
        };
        let before = serde_json::to_vec(invalid.output()).unwrap();
        assert_eq!(
            invalid.consume(&page),
            Err(ConsumeError::InvalidNativeOutput)
        );
        assert_eq!(invalid.cursor(), 0);
        assert_eq!(serde_json::to_vec(invalid.output()).unwrap(), before);
    }

    #[test]
    fn embedding_page_part_corruption_rolls_back_cross_page_state() {
        use super::super::result_buffer::CursorEvent;

        let vectors = vec![vec![0.25; 300]];
        let usage = EmbeddingUsage {
            prompt_tokens: 2,
            total_tokens: 2,
        };
        let json =
            String::from_utf8(serialize_embedding_payload(&vectors, &usage).unwrap()).unwrap();
        // This canonical payload contains only JSON punctuation and numbers,
        // so fixed byte chunks are also valid UTF-8 test fragments.
        let parts: Vec<String> = json
            .as_bytes()
            .chunks(512)
            .map(|chunk| std::str::from_utf8(chunk).unwrap().to_owned())
            .collect();
        assert!(parts.len() > 1);
        let part_count = parts.len() as u32;
        let mut consumer = ResultConsumer::new("fragmented-embedding", Lane::Language).unwrap();
        consumer
            .consume(&ResultPage {
                request_id: "fragmented-embedding".into(),
                availability: Availability::Available,
                events: vec![CursorEvent {
                    cursor: 1,
                    event: PageEvent::EmbeddingJsonPart {
                        result_id: 1,
                        part_index: 0,
                        part_count,
                        json: parts[0].clone(),
                    },
                }],
                next_cursor: 1,
                latest_cursor: part_count as u64,
                terminal: None,
            })
            .unwrap();
        let before = serde_json::to_vec(consumer.output()).unwrap();
        let bad_page = ResultPage {
            request_id: "fragmented-embedding".into(),
            availability: Availability::Available,
            events: vec![CursorEvent {
                cursor: 2,
                event: PageEvent::EmbeddingJsonPart {
                    result_id: 1,
                    part_index: 1,
                    part_count: part_count + 1,
                    json: parts[1].clone(),
                },
            }],
            next_cursor: 2,
            latest_cursor: part_count as u64,
            terminal: None,
        };
        assert_eq!(
            consumer.consume(&bad_page),
            Err(ConsumeError::InvalidSequence)
        );
        assert_eq!(consumer.cursor(), 1);
        assert_eq!(serde_json::to_vec(consumer.output()).unwrap(), before);

        for index in 1..parts.len() {
            let cursor = index as u64 + 1;
            let terminal = (index + 1 == parts.len()).then_some(ResultTerminal::Succeeded);
            consumer
                .consume(&ResultPage {
                    request_id: "fragmented-embedding".into(),
                    availability: Availability::Available,
                    events: vec![CursorEvent {
                        cursor,
                        event: PageEvent::EmbeddingJsonPart {
                            result_id: 1,
                            part_index: index as u32,
                            part_count,
                            json: parts[index].clone(),
                        },
                    }],
                    next_cursor: cursor,
                    latest_cursor: part_count as u64,
                    terminal,
                })
                .unwrap();
        }
        assert!(consumer.is_complete());
        assert!(matches!(
            consumer.output(),
            NativeOutput::LanguageEmbeddings { vectors: actual, usage: actual_usage }
                if actual == &vectors && actual_usage == &usage
        ));
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
