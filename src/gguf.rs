//! Bounded GGUF metadata parsing and local model inventory.
//!
//! The parser intentionally stops after the tensor descriptors. Tensor payloads
//! can be many gigabytes and are neither needed nor read by the model browser.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{ErrorKind, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{bail, ensure, Context, Result};

const GGUF_MAGIC: [u8; 4] = *b"GGUF";

const TYPE_UINT8: u32 = 0;
const TYPE_INT8: u32 = 1;
const TYPE_UINT16: u32 = 2;
const TYPE_INT16: u32 = 3;
const TYPE_UINT32: u32 = 4;
const TYPE_INT32: u32 = 5;
const TYPE_FLOAT32: u32 = 6;
const TYPE_BOOL: u32 = 7;
const TYPE_STRING: u32 = 8;
const TYPE_ARRAY: u32 = 9;
const TYPE_UINT64: u32 = 10;
const TYPE_INT64: u32 = 11;
const TYPE_FLOAT64: u32 = 12;

const MAX_METADATA_COUNT: u64 = 100_000;
const MAX_TENSOR_COUNT: u64 = 1_000_000;
const MAX_STRING_BYTES: u64 = 16 * 1024 * 1024;
const MAX_ARRAY_ITEMS: u64 = 50_000_000;
const MAX_COMPLEX_ARRAY_ITEMS: u64 = 2_000_000;
const MAX_ARRAY_NESTING: usize = 8;
const MAX_TENSOR_DIMENSIONS: u32 = 64;
const MAX_TOTAL_TENSOR_DIMENSIONS: u64 = 8_000_000;
const MAX_VARIABLE_BYTES: u64 = 256 * 1024 * 1024;

/// Metadata needed by the L3MS model browser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub name: Option<String>,
    pub architecture: Option<String>,
    pub file_type: Option<u32>,
    pub tokenizer_model: Option<String>,
    /// Best available parameter count: canonical metadata, architecture-specific
    /// metadata, then the count derived from tensor dimensions.
    pub parameter_count: Option<u64>,
    /// Metadata key that supplied `parameter_count`; `None` means it was derived.
    pub parameter_count_key: Option<String>,
    /// Every scalar `*.parameter_count` value found in the metadata.
    pub parameter_counts: BTreeMap<String, u64>,
    pub derived_parameter_count: u64,
}

/// One local GGUF inventory entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufFile {
    pub path: PathBuf,
    pub size: u64,
    pub quantization: String,
    pub modified: Option<SystemTime>,
    pub metadata: Option<GgufMetadata>,
    /// A malformed GGUF remains visible in scans with its parse error attached.
    pub parse_error: Option<String>,
}

#[derive(Debug)]
enum ScalarValue {
    Unsigned(u64),
    Signed(i64),
    Float,
    Bool,
    String(String),
}

impl ScalarValue {
    fn into_string(self, key: &str) -> Result<String> {
        match self {
            Self::String(value) => Ok(value),
            _ => bail!("GGUF metadata {key:?} must be a string"),
        }
    }

    fn into_u64(self, key: &str) -> Result<u64> {
        match self {
            Self::Unsigned(value) => Ok(value),
            Self::Signed(value) if value >= 0 => Ok(value as u64),
            _ => bail!("GGUF metadata {key:?} must be a non-negative integer"),
        }
    }
}

#[derive(Default)]
struct CapturedMetadata {
    name: Option<String>,
    architecture: Option<String>,
    file_type: Option<u32>,
    tokenizer_model: Option<String>,
    parameter_counts: BTreeMap<String, u64>,
}

struct Parser<R> {
    reader: R,
    file_len: u64,
    variable_bytes: u64,
    array_items: u64,
    complex_array_items: u64,
    tensor_dimensions: u64,
}

impl<R: Read + Seek> Parser<R> {
    fn new(reader: R, file_len: u64) -> Self {
        Self {
            reader,
            file_len,
            variable_bytes: 0,
            array_items: 0,
            complex_array_items: 0,
            tensor_dimensions: 0,
        }
    }

    fn parse(mut self) -> Result<GgufMetadata> {
        ensure!(self.read_array::<4>()? == GGUF_MAGIC, "invalid GGUF magic");

        let version = self.read_u32()?;
        ensure!(
            matches!(version, 2 | 3),
            "unsupported GGUF version: {version}"
        );

        let tensor_count = self.read_u64()?;
        ensure!(
            tensor_count <= MAX_TENSOR_COUNT,
            "GGUF tensor count too large: {tensor_count} (maximum {MAX_TENSOR_COUNT})"
        );

        let metadata_count = self.read_u64()?;
        ensure!(
            metadata_count <= MAX_METADATA_COUNT,
            "GGUF metadata count too large: {metadata_count} (maximum {MAX_METADATA_COUNT})"
        );

        let mut captured = CapturedMetadata::default();
        for index in 0..metadata_count {
            let key = self
                .read_string()
                .with_context(|| format!("failed to read GGUF metadata key {index}"))?;
            let value_type = self
                .read_u32()
                .with_context(|| format!("failed to read type for GGUF metadata {key:?}"))?;
            validate_value_type(value_type)
                .with_context(|| format!("invalid type for GGUF metadata {key:?}"))?;

            if is_capture_key(&key) {
                let value = self
                    .read_scalar(value_type)
                    .with_context(|| format!("failed to read GGUF metadata {key:?}"))?;
                capture_value(&mut captured, key, value)?;
            } else {
                self.skip_value(value_type, 0)
                    .with_context(|| format!("failed to skip GGUF metadata {key:?}"))?;
            }
        }

        let mut derived_parameter_count = 0_u64;
        for index in 0..tensor_count {
            self.skip_string()
                .with_context(|| format!("failed to read GGUF tensor name {index}"))?;
            let dimensions = self
                .read_u32()
                .with_context(|| format!("failed to read dimensions for GGUF tensor {index}"))?;
            ensure!(
                dimensions <= MAX_TENSOR_DIMENSIONS,
                "GGUF tensor {index} has too many dimensions: {dimensions} (maximum {MAX_TENSOR_DIMENSIONS})"
            );
            self.tensor_dimensions = self
                .tensor_dimensions
                .checked_add(u64::from(dimensions))
                .context("GGUF cumulative tensor dimension count overflow")?;
            ensure!(
                self.tensor_dimensions <= MAX_TOTAL_TENSOR_DIMENSIONS,
                "GGUF cumulative tensor dimension count too large: {} (maximum {MAX_TOTAL_TENSOR_DIMENSIONS})",
                self.tensor_dimensions
            );

            let mut tensor_elements = 1_u64;
            for dimension in 0..dimensions {
                let size = self.read_u64().with_context(|| {
                    format!("failed to read dimension {dimension} for GGUF tensor {index}")
                })?;
                tensor_elements = tensor_elements
                    .checked_mul(size)
                    .with_context(|| format!("element count overflow for GGUF tensor {index}"))?;
            }
            // Tensor type and offset belong to the descriptor, not its payload.
            self.read_u32()
                .with_context(|| format!("failed to read type for GGUF tensor {index}"))?;
            self.read_u64()
                .with_context(|| format!("failed to read offset for GGUF tensor {index}"))?;
            derived_parameter_count = derived_parameter_count
                .checked_add(tensor_elements)
                .context("GGUF derived parameter count overflow")?;
        }

        let (parameter_count_key, parameter_count) = select_parameter_count(
            captured.architecture.as_deref(),
            &captured.parameter_counts,
            derived_parameter_count,
        );

        Ok(GgufMetadata {
            version,
            tensor_count,
            metadata_count,
            name: captured.name,
            architecture: captured.architecture,
            file_type: captured.file_type,
            tokenizer_model: captured.tokenizer_model,
            parameter_count,
            parameter_count_key,
            parameter_counts: captured.parameter_counts,
            derived_parameter_count,
        })
    }

    fn read_scalar(&mut self, value_type: u32) -> Result<ScalarValue> {
        match value_type {
            TYPE_UINT8 => Ok(ScalarValue::Unsigned(u64::from(self.read_u8()?))),
            TYPE_INT8 => Ok(ScalarValue::Signed(i64::from(self.read_i8()?))),
            TYPE_UINT16 => Ok(ScalarValue::Unsigned(u64::from(self.read_u16()?))),
            TYPE_INT16 => Ok(ScalarValue::Signed(i64::from(self.read_i16()?))),
            TYPE_UINT32 => Ok(ScalarValue::Unsigned(u64::from(self.read_u32()?))),
            TYPE_INT32 => Ok(ScalarValue::Signed(i64::from(self.read_i32()?))),
            TYPE_FLOAT32 => {
                let _ = f32::from_le_bytes(self.read_array::<4>()?);
                Ok(ScalarValue::Float)
            }
            TYPE_BOOL => {
                let _ = self.read_u8()?;
                Ok(ScalarValue::Bool)
            }
            TYPE_STRING => Ok(ScalarValue::String(self.read_string()?)),
            TYPE_UINT64 => Ok(ScalarValue::Unsigned(self.read_u64()?)),
            TYPE_INT64 => Ok(ScalarValue::Signed(self.read_i64()?)),
            TYPE_FLOAT64 => {
                let _ = f64::from_le_bytes(self.read_array::<8>()?);
                Ok(ScalarValue::Float)
            }
            TYPE_ARRAY => bail!("GGUF arrays are not valid for captured scalar metadata"),
            other => bail!("unsupported GGUF value type: {other}"),
        }
    }

    fn skip_value(&mut self, value_type: u32, depth: usize) -> Result<()> {
        if let Some(size) = fixed_type_size(value_type) {
            return self.skip_bytes(size);
        }
        match value_type {
            TYPE_STRING => self.skip_string(),
            TYPE_ARRAY => {
                ensure!(
                    depth < MAX_ARRAY_NESTING,
                    "GGUF array nesting exceeds maximum {MAX_ARRAY_NESTING}"
                );
                let item_type = self.read_u32()?;
                validate_value_type(item_type).context("invalid GGUF array item type")?;
                let count = self.read_u64()?;
                ensure!(
                    count <= MAX_ARRAY_ITEMS,
                    "GGUF array too large: {count} items (maximum {MAX_ARRAY_ITEMS})"
                );
                self.array_items = self
                    .array_items
                    .checked_add(count)
                    .context("GGUF cumulative array item count overflow")?;
                ensure!(
                    self.array_items <= MAX_ARRAY_ITEMS,
                    "GGUF cumulative array item count too large: {} (maximum {MAX_ARRAY_ITEMS})",
                    self.array_items
                );

                if let Some(item_size) = fixed_type_size(item_type) {
                    let bytes = count
                        .checked_mul(item_size)
                        .context("GGUF array byte length overflow")?;
                    return self.skip_bytes(bytes);
                }
                self.complex_array_items = self
                    .complex_array_items
                    .checked_add(count)
                    .context("GGUF cumulative complex array item count overflow")?;
                ensure!(
                    self.complex_array_items <= MAX_COMPLEX_ARRAY_ITEMS,
                    "GGUF complex array too large: {} items (maximum {MAX_COMPLEX_ARRAY_ITEMS})",
                    self.complex_array_items
                );
                for _ in 0..count {
                    self.skip_value(item_type, depth + 1)?;
                }
                Ok(())
            }
            other => bail!("unsupported GGUF value type: {other}"),
        }
    }

    fn read_string(&mut self) -> Result<String> {
        let size = self.read_string_size()?;
        self.charge_variable_bytes(size)?;
        self.ensure_remaining(size)?;
        let size = usize::try_from(size).context("GGUF string length does not fit in memory")?;
        let mut bytes = vec![0_u8; size];
        self.read_exact(&mut bytes)?;
        String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
    }

    fn skip_string(&mut self) -> Result<()> {
        let size = self.read_string_size()?;
        self.skip_bytes(size)
    }

    fn read_string_size(&mut self) -> Result<u64> {
        let size = self.read_u64()?;
        ensure!(
            size <= MAX_STRING_BYTES,
            "GGUF string too large: {size} bytes (maximum {MAX_STRING_BYTES})"
        );
        Ok(size)
    }

    fn skip_bytes(&mut self, size: u64) -> Result<()> {
        self.charge_variable_bytes(size)?;
        let end = self.ensure_remaining(size)?;
        self.reader
            .seek(SeekFrom::Start(end))
            .context("seek within GGUF")?;
        Ok(())
    }

    fn ensure_remaining(&mut self, size: u64) -> Result<u64> {
        let position = self
            .reader
            .stream_position()
            .context("read GGUF position")?;
        let end = position
            .checked_add(size)
            .context("GGUF skip position overflow")?;
        ensure!(
            end <= self.file_len,
            "unexpected end of GGUF: need {size} more bytes"
        );
        Ok(end)
    }

    fn charge_variable_bytes(&mut self, size: u64) -> Result<()> {
        self.variable_bytes = self
            .variable_bytes
            .checked_add(size)
            .context("GGUF variable-length byte count overflow")?;
        ensure!(
            self.variable_bytes <= MAX_VARIABLE_BYTES,
            "GGUF variable-length data too large: {} bytes (maximum {MAX_VARIABLE_BYTES})",
            self.variable_bytes
        );
        Ok(())
    }

    fn read_exact(&mut self, buffer: &mut [u8]) -> Result<()> {
        match self.reader.read_exact(buffer) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                bail!("unexpected end of GGUF")
            }
            Err(error) => Err(error).context("read GGUF"),
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut bytes = [0_u8; N];
        self.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(i8::from_le_bytes(self.read_array::<1>()?))
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(i16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.read_array::<8>()?))
    }
}

/// Parse a GGUF v2/v3 header and tensor descriptors without reading payloads.
pub fn parse_metadata(path: impl AsRef<Path>) -> Result<GgufMetadata> {
    let path = path.as_ref();
    let path_metadata =
        fs::symlink_metadata(path).with_context(|| format!("inspect GGUF {}", path.display()))?;
    ensure!(
        !path_metadata.file_type().is_symlink(),
        "refusing to parse GGUF symlink {}",
        path.display()
    );
    ensure!(
        path_metadata.is_file(),
        "GGUF path is not a regular file: {}",
        path.display()
    );

    let file = File::open(path).with_context(|| format!("open GGUF {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("inspect opened GGUF {}", path.display()))?
        .len();
    Parser::new(file, file_len)
        .parse()
        .with_context(|| format!("parse GGUF {}", path.display()))
}

/// Scan a directory for GGUF files. Directory and file symlinks are ignored.
/// Malformed files are returned with `parse_error` so the inventory remains useful.
pub fn scan_directory(root: impl AsRef<Path>, recursive: bool) -> Result<Vec<GgufFile>> {
    let root = root.as_ref();
    let root_metadata = fs::symlink_metadata(root)
        .with_context(|| format!("inspect GGUF root {}", root.display()))?;
    ensure!(
        !root_metadata.file_type().is_symlink(),
        "refusing to scan symlinked GGUF root {}",
        root.display()
    );
    ensure!(
        root_metadata.is_dir(),
        "GGUF root is not a directory: {}",
        root.display()
    );

    let mut directories = vec![root.to_path_buf()];
    let mut paths = Vec::new();
    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory)
            .with_context(|| format!("read GGUF directory {}", directory.display()))?;
        for entry in entries {
            let entry = entry
                .with_context(|| format!("read entry in GGUF directory {}", directory.display()))?;
            let file_type = entry
                .file_type()
                .with_context(|| format!("inspect GGUF candidate {}", entry.path().display()))?;
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                if recursive {
                    directories.push(entry.path());
                }
                continue;
            }
            if file_type.is_file() && has_gguf_extension(&entry.path()) {
                paths.push(entry.path());
            }
        }
    }
    paths.sort();

    let mut files = Vec::with_capacity(paths.len());
    for path in paths {
        let file_metadata = fs::metadata(&path)
            .with_context(|| format!("inspect GGUF candidate {}", path.display()))?;
        let parsed = parse_metadata(&path);
        let (metadata, parse_error) = match parsed {
            Ok(metadata) => (Some(metadata), None),
            Err(error) => (None, Some(format!("{error:#}"))),
        };
        let quantization = infer_quantization(&path, metadata.as_ref());
        files.push(GgufFile {
            path,
            size: file_metadata.len(),
            quantization,
            modified: file_metadata.modified().ok(),
            metadata,
            parse_error,
        });
    }
    Ok(files)
}

/// Shard position encoded in a split GGUF file name such as
/// `model-00001-of-00033.gguf`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardPosition {
    pub index: u32,
    pub total: u32,
}

/// A display grouping for the inventory: either one standalone GGUF file or a
/// multi-part shard set collapsed into a single entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSet {
    /// Representative file: shard 1 when present, otherwise the lowest index.
    pub primary: GgufFile,
    /// Every member file of the set, ordered by shard index.
    pub members: Vec<GgufFile>,
    /// Summed byte size across all members.
    pub total_size: u64,
    /// Declared shard count for multi-part sets (`None` for standalone files).
    pub shard_total: Option<u32>,
    /// True when a multi-part set is missing shards, has duplicates, or is
    /// otherwise inconsistent with its declared shard count.
    pub incomplete: bool,
}

impl ShardSet {
    /// Newest modification time across all member files.
    pub fn modified(&self) -> Option<SystemTime> {
        self.members.iter().filter_map(|file| file.modified).max()
    }
}

/// Extract the shard position from a split GGUF file name.
pub fn shard_position(path: &Path) -> Option<ShardPosition> {
    shard_parts(path).map(|(_, position)| position)
}

/// Parse a split GGUF file name into its model-name prefix and shard position.
fn shard_parts(path: &Path) -> Option<(&str, ShardPosition)> {
    let stem = path.file_stem()?.to_str()?;
    let (prefix, total_text) = stem.rsplit_once("-of-")?;
    if total_text.is_empty() || !total_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let total = total_text.parse::<u32>().ok()?;
    let index_bytes = prefix
        .bytes()
        .rev()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if index_bytes == 0 {
        return None;
    }
    let index_text = &prefix[prefix.len() - index_bytes..];
    let index = index_text.parse::<u32>().ok()?;
    if index == 0 || total == 0 {
        return None;
    }
    let prefix = &stem[..stem.len() - total_text.len() - "-of-".len() - index_bytes];
    Some((prefix, ShardPosition { index, total }))
}

/// Collapse scanned files into display entries: standalone files pass through,
/// while shard sets sharing a directory, name prefix, and declared total
/// become a single entry anchored at shard 1 (or the lowest present index).
pub fn group_shard_sets(files: Vec<GgufFile>) -> Vec<ShardSet> {
    let mut singles: Vec<GgufFile> = Vec::new();
    let mut sets: BTreeMap<(PathBuf, String, u32), Vec<(u32, GgufFile)>> = BTreeMap::new();
    for file in files {
        match shard_parts(&file.path) {
            None => singles.push(file),
            Some((prefix, position)) => {
                let key = (
                    file.path
                        .parent()
                        .unwrap_or_else(|| Path::new(""))
                        .to_path_buf(),
                    prefix.to_owned(),
                    position.total,
                );
                sets.entry(key).or_default().push((position.index, file));
            }
        }
    }

    let mut entries: Vec<ShardSet> = singles
        .into_iter()
        .map(|file| ShardSet {
            total_size: file.size,
            shard_total: None,
            incomplete: false,
            primary: file.clone(),
            members: vec![file],
        })
        .collect();
    for ((_directory, _prefix, total), mut members) in sets {
        members.sort_by_key(|(index, _)| *index);
        let indices: BTreeSet<u32> = members.iter().map(|(index, _)| *index).collect();
        let complete = members.len() == total as usize
            && indices.len() == members.len()
            && indices.iter().next() == Some(&1)
            && indices.iter().next_back() == Some(&total);
        let primary_index = members
            .iter()
            .position(|(index, _)| *index == 1)
            .unwrap_or(0);
        let primary = members[primary_index].1.clone();
        entries.push(ShardSet {
            total_size: members.iter().map(|(_, file)| file.size).sum(),
            shard_total: Some(total),
            incomplete: !complete,
            primary,
            members: members.into_iter().map(|(_, file)| file).collect(),
        });
    }
    entries.sort_by(|left, right| left.primary.path.cmp(&right.primary.path));
    entries
}

/// Infer a readable quantization label, preferring recognized GGUF file types.
pub fn infer_quantization(path: &Path, metadata: Option<&GgufMetadata>) -> String {
    let file_type = metadata.and_then(|metadata| metadata.file_type);
    if let Some(label) = file_type.and_then(file_type_label) {
        return label.to_owned();
    }

    if let Some(quantization) =
        quantization_from_text(&path.file_name().unwrap_or_default().to_string_lossy())
    {
        return quantization;
    }
    if let Some(quantization) = metadata
        .and_then(|metadata| metadata.name.as_deref())
        .and_then(quantization_from_text)
    {
        return quantization;
    }
    file_type.map_or_else(|| "unknown".to_owned(), |value| format!("ftype:{value}"))
}

fn capture_value(captured: &mut CapturedMetadata, key: String, value: ScalarValue) -> Result<()> {
    match key.as_str() {
        "general.name" => captured.name = Some(value.into_string(&key)?),
        "general.architecture" => {
            captured.architecture = Some(value.into_string(&key)?);
        }
        "general.file_type" => {
            let value = value.into_u64(&key)?;
            captured.file_type = Some(
                u32::try_from(value)
                    .with_context(|| format!("GGUF metadata {key:?} exceeds u32"))?,
            );
        }
        "tokenizer.ggml.model" => {
            captured.tokenizer_model = Some(value.into_string(&key)?);
        }
        _ if is_parameter_count_key(&key) => {
            captured
                .parameter_counts
                .insert(key.clone(), value.into_u64(&key)?);
        }
        _ => unreachable!("capture_value called for an uncaptured key"),
    }
    Ok(())
}

fn select_parameter_count(
    architecture: Option<&str>,
    counts: &BTreeMap<String, u64>,
    derived: u64,
) -> (Option<String>, Option<u64>) {
    if let Some(&count) = counts
        .get("general.parameter_count")
        .filter(|&&count| count > 0)
    {
        return (Some("general.parameter_count".to_owned()), Some(count));
    }
    if let Some(architecture) = architecture {
        let key = format!("{architecture}.parameter_count");
        if let Some(&count) = counts.get(&key).filter(|&&count| count > 0) {
            return (Some(key), Some(count));
        }
    }
    if let Some((key, &count)) = counts.iter().find(|(_, count)| **count > 0) {
        return (Some(key.clone()), Some(count));
    }
    (None, (derived > 0).then_some(derived))
}

fn is_capture_key(key: &str) -> bool {
    matches!(
        key,
        "general.name" | "general.architecture" | "general.file_type" | "tokenizer.ggml.model"
    ) || is_parameter_count_key(key)
}

fn is_parameter_count_key(key: &str) -> bool {
    key == "general.parameter_count" || key.ends_with(".parameter_count")
}

fn validate_value_type(value_type: u32) -> Result<()> {
    ensure!(
        value_type <= TYPE_FLOAT64,
        "unsupported GGUF value type: {value_type}"
    );
    Ok(())
}

fn fixed_type_size(value_type: u32) -> Option<u64> {
    match value_type {
        TYPE_UINT8 | TYPE_INT8 | TYPE_BOOL => Some(1),
        TYPE_UINT16 | TYPE_INT16 => Some(2),
        TYPE_UINT32 | TYPE_INT32 | TYPE_FLOAT32 => Some(4),
        TYPE_UINT64 | TYPE_INT64 | TYPE_FLOAT64 => Some(8),
        _ => None,
    }
}

fn has_gguf_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn file_type_label(file_type: u32) -> Option<&'static str> {
    match file_type {
        0 => Some("F32"),
        1 => Some("F16"),
        2 => Some("Q4_0"),
        3 => Some("Q4_1"),
        7 => Some("Q8_0"),
        8 => Some("Q5_0"),
        9 => Some("Q5_1"),
        10 => Some("Q2_K"),
        11 => Some("Q3_K_S"),
        12 => Some("Q3_K_M"),
        13 => Some("Q3_K_L"),
        14 => Some("Q4_K_S"),
        15 => Some("Q4_K_M"),
        16 => Some("Q5_K_S"),
        17 => Some("Q5_K_M"),
        18 => Some("Q6_K"),
        19 => Some("IQ2_XXS"),
        20 => Some("IQ2_XS"),
        21 => Some("Q2_K_S"),
        22 => Some("IQ3_XS"),
        23 => Some("IQ3_XXS"),
        24 => Some("IQ1_S"),
        25 => Some("IQ4_NL"),
        26 => Some("IQ3_S"),
        27 => Some("IQ3_M"),
        28 => Some("IQ2_S"),
        29 => Some("IQ2_M"),
        30 => Some("IQ4_XS"),
        31 => Some("IQ1_M"),
        32 => Some("BF16"),
        36 => Some("TQ1_0"),
        37 => Some("TQ2_0"),
        38 => Some("MXFP4_MOE"),
        39 => Some("NVFP4"),
        40 => Some("Q1_0"),
        41 => Some("Q2_0"),
        _ => None,
    }
}

fn quantization_from_text(text: &str) -> Option<String> {
    let upper = text.to_ascii_uppercase();
    let bytes = upper.as_bytes();
    for start in 0..bytes.len() {
        if start > 0 && !is_quantization_boundary(bytes[start - 1]) {
            continue;
        }
        let remainder = &upper[start..];
        for marker in ["MXFP4", "BF16", "FP16", "FP32", "F16", "F32"] {
            if remainder.starts_with(marker)
                && remainder
                    .as_bytes()
                    .get(marker.len())
                    .is_none_or(|next| is_quantization_boundary(*next))
            {
                return Some(marker.to_owned());
            }
        }

        let (prefix_len, number_index) =
            if remainder.starts_with("UD-IQ") || remainder.starts_with("UD-TQ") {
                (5, 5)
            } else if remainder.starts_with("UD-Q") {
                (4, 4)
            } else if remainder.starts_with("IQ") || remainder.starts_with("TQ") {
                (2, 2)
            } else if remainder.starts_with('Q') {
                (1, 1)
            } else {
                continue;
            };
        if !remainder
            .as_bytes()
            .get(number_index)
            .is_some_and(u8::is_ascii_digit)
        {
            continue;
        }
        let mut end = prefix_len;
        while remainder
            .as_bytes()
            .get(end)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_')
        {
            end += 1;
        }
        if remainder
            .as_bytes()
            .get(end)
            .is_none_or(|next| is_quantization_boundary(*next))
        {
            return Some(remainder[..end].to_owned());
        }
    }
    None
}

fn is_quantization_boundary(byte: u8) -> bool {
    matches!(byte, b'-' | b'_' | b'.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[derive(Clone)]
    enum TestValue {
        U8(u8),
        I8(i8),
        U16(u16),
        I16(i16),
        U32(u32),
        I32(i32),
        F32(f32),
        Bool(bool),
        String(&'static str),
        Array(u32, Vec<TestValue>),
        U64(u64),
        I64(i64),
        F64(f64),
    }

    impl TestValue {
        fn value_type(&self) -> u32 {
            match self {
                Self::U8(_) => TYPE_UINT8,
                Self::I8(_) => TYPE_INT8,
                Self::U16(_) => TYPE_UINT16,
                Self::I16(_) => TYPE_INT16,
                Self::U32(_) => TYPE_UINT32,
                Self::I32(_) => TYPE_INT32,
                Self::F32(_) => TYPE_FLOAT32,
                Self::Bool(_) => TYPE_BOOL,
                Self::String(_) => TYPE_STRING,
                Self::Array(_, _) => TYPE_ARRAY,
                Self::U64(_) => TYPE_UINT64,
                Self::I64(_) => TYPE_INT64,
                Self::F64(_) => TYPE_FLOAT64,
            }
        }
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_value(bytes: &mut Vec<u8>, value: &TestValue) {
        match value {
            TestValue::U8(value) => bytes.push(*value),
            TestValue::I8(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::U16(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::I16(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::U32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::I32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::F32(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::Bool(value) => bytes.push(u8::from(*value)),
            TestValue::String(value) => push_string(bytes, value),
            TestValue::Array(item_type, values) => {
                bytes.extend_from_slice(&item_type.to_le_bytes());
                bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                for value in values {
                    push_value(bytes, value);
                }
            }
            TestValue::U64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::I64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
            TestValue::F64(value) => bytes.extend_from_slice(&value.to_le_bytes()),
        }
    }

    fn fixture(
        version: u32,
        metadata: &[(&str, TestValue)],
        tensors: &[(&str, &[u64])],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            push_string(&mut bytes, key);
            bytes.extend_from_slice(&value.value_type().to_le_bytes());
            push_value(&mut bytes, value);
        }
        for (name, dimensions) in tensors {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(dimensions.len() as u32).to_le_bytes());
            for dimension in *dimensions {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&0_u64.to_le_bytes());
        }
        bytes
    }

    fn write_fixture(path: &Path, bytes: &[u8]) {
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn parses_v2_scalars_and_stops_before_tensor_payload() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.gguf");
        let mut bytes = fixture(
            2,
            &[
                ("general.name", TestValue::String("Tiny Llama")),
                ("general.architecture", TestValue::String("llama")),
                ("general.file_type", TestValue::U32(15)),
                ("general.parameter_count", TestValue::U64(7_000_000_000)),
                ("tokenizer.ggml.model", TestValue::String("llama")),
                ("test.u8", TestValue::U8(1)),
                ("test.i8", TestValue::I8(-1)),
                ("test.u16", TestValue::U16(2)),
                ("test.i16", TestValue::I16(-2)),
                ("test.u32", TestValue::U32(3)),
                ("test.i32", TestValue::I32(-3)),
                ("test.f32", TestValue::F32(1.25)),
                ("test.bool", TestValue::Bool(true)),
                ("test.u64", TestValue::U64(4)),
                ("test.i64", TestValue::I64(-4)),
                ("test.f64", TestValue::F64(2.5)),
            ],
            &[("weight", &[10, 20])],
        );
        // Deliberately invalid payload bytes: the metadata parser must not read them.
        bytes.extend_from_slice(&[0xff, 0xfe, 0xfd]);
        write_fixture(&path, &bytes);

        let metadata = parse_metadata(&path).unwrap();
        assert_eq!(metadata.version, 2);
        assert_eq!(metadata.tensor_count, 1);
        assert_eq!(metadata.name.as_deref(), Some("Tiny Llama"));
        assert_eq!(metadata.architecture.as_deref(), Some("llama"));
        assert_eq!(metadata.file_type, Some(15));
        assert_eq!(metadata.tokenizer_model.as_deref(), Some("llama"));
        assert_eq!(metadata.parameter_count, Some(7_000_000_000));
        assert_eq!(
            metadata.parameter_count_key.as_deref(),
            Some("general.parameter_count")
        );
        assert_eq!(metadata.derived_parameter_count, 200);
    }

    #[test]
    fn parses_v3_arrays_and_architecture_parameter_count() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("array.gguf");
        write_fixture(
            &path,
            &fixture(
                3,
                &[
                    (
                        "test.fixed",
                        TestValue::Array(TYPE_UINT32, vec![TestValue::U32(1), TestValue::U32(2)]),
                    ),
                    (
                        "test.strings",
                        TestValue::Array(
                            TYPE_STRING,
                            vec![TestValue::String("one"), TestValue::String("two")],
                        ),
                    ),
                    (
                        "test.nested",
                        TestValue::Array(
                            TYPE_ARRAY,
                            vec![TestValue::Array(
                                TYPE_BOOL,
                                vec![TestValue::Bool(true), TestValue::Bool(false)],
                            )],
                        ),
                    ),
                    ("general.architecture", TestValue::String("qwen")),
                    ("qwen.parameter_count", TestValue::U64(42)),
                ],
                &[],
            ),
        );

        let metadata = parse_metadata(&path).unwrap();
        assert_eq!(metadata.version, 3);
        assert_eq!(metadata.parameter_count, Some(42));
        assert_eq!(
            metadata.parameter_count_key.as_deref(),
            Some("qwen.parameter_count")
        );
        assert_eq!(metadata.parameter_counts["qwen.parameter_count"], 42);
    }

    #[test]
    fn derives_parameter_count_when_metadata_has_none() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("derived.gguf");
        write_fixture(&path, &fixture(3, &[], &[("a", &[2, 3]), ("b", &[4])]));
        let metadata = parse_metadata(&path).unwrap();
        assert_eq!(metadata.derived_parameter_count, 10);
        assert_eq!(metadata.parameter_count, Some(10));
        assert_eq!(metadata.parameter_count_key, None);
    }

    #[test]
    fn rejects_truncated_metadata_and_tensor_descriptors() {
        let directory = tempfile::tempdir().unwrap();
        let metadata_path = directory.path().join("metadata.gguf");
        let mut metadata = fixture(3, &[("general.name", TestValue::String("model"))], &[]);
        metadata.pop();
        write_fixture(&metadata_path, &metadata);
        assert!(format!("{:#}", parse_metadata(&metadata_path).unwrap_err())
            .contains("unexpected end of GGUF"));

        let tensor_path = directory.path().join("tensor.gguf");
        let mut tensor = fixture(3, &[], &[("weight", &[4, 8])]);
        tensor.pop();
        write_fixture(&tensor_path, &tensor);
        assert!(format!("{:#}", parse_metadata(&tensor_path).unwrap_err())
            .contains("unexpected end of GGUF"));
    }

    #[test]
    fn rejects_oversized_header_counts_and_lengths() {
        let directory = tempfile::tempdir().unwrap();

        let tensor_path = directory.path().join("tensors.gguf");
        let mut tensors = Vec::from(GGUF_MAGIC);
        tensors.extend_from_slice(&3_u32.to_le_bytes());
        tensors.extend_from_slice(&(MAX_TENSOR_COUNT + 1).to_le_bytes());
        tensors.extend_from_slice(&0_u64.to_le_bytes());
        write_fixture(&tensor_path, &tensors);
        assert!(format!("{:#}", parse_metadata(&tensor_path).unwrap_err())
            .contains("tensor count too large"));

        let metadata_path = directory.path().join("metadata.gguf");
        let mut metadata = Vec::from(GGUF_MAGIC);
        metadata.extend_from_slice(&3_u32.to_le_bytes());
        metadata.extend_from_slice(&0_u64.to_le_bytes());
        metadata.extend_from_slice(&(MAX_METADATA_COUNT + 1).to_le_bytes());
        write_fixture(&metadata_path, &metadata);
        assert!(format!("{:#}", parse_metadata(&metadata_path).unwrap_err())
            .contains("metadata count too large"));

        let string_path = directory.path().join("string.gguf");
        let mut string = Vec::from(GGUF_MAGIC);
        string.extend_from_slice(&3_u32.to_le_bytes());
        string.extend_from_slice(&0_u64.to_le_bytes());
        string.extend_from_slice(&1_u64.to_le_bytes());
        string.extend_from_slice(&(MAX_STRING_BYTES + 1).to_le_bytes());
        write_fixture(&string_path, &string);
        assert!(
            format!("{:#}", parse_metadata(&string_path).unwrap_err()).contains("string too large")
        );
    }

    #[test]
    fn rejects_oversized_arrays_and_skip_budgets_without_allocating() {
        let directory = tempfile::tempdir().unwrap();

        let count_path = directory.path().join("array-count.gguf");
        let mut count = fixture_prefix_one_metadata("test.array", TYPE_ARRAY);
        count.extend_from_slice(&TYPE_UINT8.to_le_bytes());
        count.extend_from_slice(&(MAX_ARRAY_ITEMS + 1).to_le_bytes());
        write_fixture(&count_path, &count);
        assert!(
            format!("{:#}", parse_metadata(&count_path).unwrap_err()).contains("array too large")
        );

        let complex_path = directory.path().join("complex-array-count.gguf");
        let mut complex = fixture_prefix_one_metadata("test.array", TYPE_ARRAY);
        complex.extend_from_slice(&TYPE_STRING.to_le_bytes());
        complex.extend_from_slice(&(MAX_COMPLEX_ARRAY_ITEMS + 1).to_le_bytes());
        write_fixture(&complex_path, &complex);
        assert!(format!("{:#}", parse_metadata(&complex_path).unwrap_err())
            .contains("complex array too large"));

        let budget_path = directory.path().join("array-budget.gguf");
        let mut budget = fixture_prefix_one_metadata("test.array", TYPE_ARRAY);
        budget.extend_from_slice(&TYPE_UINT64.to_le_bytes());
        budget.extend_from_slice(&(MAX_VARIABLE_BYTES / 8 + 1).to_le_bytes());
        write_fixture(&budget_path, &budget);
        assert!(format!("{:#}", parse_metadata(&budget_path).unwrap_err())
            .contains("variable-length data too large"));
    }

    #[test]
    fn rejects_unsupported_versions_and_types() {
        let directory = tempfile::tempdir().unwrap();
        let version_path = directory.path().join("version.gguf");
        write_fixture(&version_path, &fixture(1, &[], &[]));
        assert!(format!("{:#}", parse_metadata(&version_path).unwrap_err())
            .contains("unsupported GGUF version"));

        let type_path = directory.path().join("type.gguf");
        write_fixture(
            &type_path,
            &fixture_prefix_one_metadata("test.unsupported", 99),
        );
        assert!(format!("{:#}", parse_metadata(&type_path).unwrap_err())
            .contains("unsupported GGUF value type"));

        let array_type_path = directory.path().join("array-type.gguf");
        let mut array = fixture_prefix_one_metadata("test.array", TYPE_ARRAY);
        array.extend_from_slice(&99_u32.to_le_bytes());
        array.extend_from_slice(&0_u64.to_le_bytes());
        write_fixture(&array_type_path, &array);
        assert!(
            format!("{:#}", parse_metadata(&array_type_path).unwrap_err())
                .contains("unsupported GGUF value type")
        );
    }

    #[test]
    fn rejects_excessive_array_nesting() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested.gguf");
        let mut bytes = fixture_prefix_one_metadata("test.array", TYPE_ARRAY);
        for _ in 0..=MAX_ARRAY_NESTING {
            bytes.extend_from_slice(&TYPE_ARRAY.to_le_bytes());
            bytes.extend_from_slice(&1_u64.to_le_bytes());
        }
        bytes.extend_from_slice(&TYPE_BOOL.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        write_fixture(&path, &bytes);
        assert!(
            format!("{:#}", parse_metadata(&path).unwrap_err()).contains("array nesting exceeds")
        );
    }

    #[test]
    fn quantization_prefers_known_metadata_then_filename_fallback() {
        let directory = tempfile::tempdir().unwrap();
        let metadata_path = directory.path().join("model-Q6_K.gguf");
        write_fixture(
            &metadata_path,
            &fixture(3, &[("general.file_type", TestValue::U32(15))], &[]),
        );
        let metadata = parse_metadata(&metadata_path).unwrap();
        assert_eq!(
            infer_quantization(&metadata_path, Some(&metadata)),
            "Q4_K_M"
        );
        assert_eq!(
            infer_quantization(Path::new("model-UD-Q4_K_XL-00001-of-00002.gguf"), None),
            "UD-Q4_K_XL"
        );

        let unknown_path = directory.path().join("generic.gguf");
        write_fixture(
            &unknown_path,
            &fixture(3, &[("general.file_type", TestValue::U32(999))], &[]),
        );
        let unknown = parse_metadata(&unknown_path).unwrap();
        assert_eq!(
            infer_quantization(&unknown_path, Some(&unknown)),
            "ftype:999"
        );
    }

    #[test]
    fn directory_scan_is_sorted_recursive_and_keeps_parse_errors() {
        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();
        write_fixture(
            &directory.path().join("z.GGUF"),
            &fixture(3, &[("general.file_type", TestValue::U32(18))], &[]),
        );
        write_fixture(&directory.path().join("ignore.txt"), b"GGUF");
        write_fixture(&nested.join("a.gguf"), &fixture(2, &[], &[]));
        write_fixture(&nested.join("broken.gguf"), b"not a GGUF");

        let top_level = scan_directory(directory.path(), false).unwrap();
        assert_eq!(top_level.len(), 1);
        assert_eq!(top_level[0].quantization, "Q6_K");

        let recursive = scan_directory(directory.path(), true).unwrap();
        assert_eq!(recursive.len(), 3);
        assert!(recursive.windows(2).all(|pair| pair[0].path < pair[1].path));
        let broken = recursive
            .iter()
            .find(|file| file.path.ends_with("broken.gguf"))
            .unwrap();
        assert!(broken.metadata.is_none());
        assert!(broken
            .parse_error
            .as_deref()
            .unwrap()
            .contains("invalid GGUF magic"));
    }

    #[cfg(unix)]
    #[test]
    fn scans_and_direct_parsing_do_not_follow_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let nested = directory.path().join("nested");
        fs::create_dir(&nested).unwrap();
        let model = nested.join("model.gguf");
        write_fixture(&model, &fixture(3, &[], &[]));
        let file_link = directory.path().join("linked.gguf");
        let directory_link = directory.path().join("linked-directory");
        symlink(&model, &file_link).unwrap();
        symlink(&nested, &directory_link).unwrap();

        let files = scan_directory(directory.path(), true).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, model);
        assert!(format!("{:#}", parse_metadata(&file_link).unwrap_err())
            .contains("refusing to parse GGUF symlink"));

        let root_link = directory.path().join("root-link");
        symlink(directory.path(), &root_link).unwrap();
        assert!(
            format!("{:#}", scan_directory(&root_link, true).unwrap_err())
                .contains("refusing to scan symlinked GGUF root")
        );
    }

    fn fixture_prefix_one_metadata(key: &str, value_type: u32) -> Vec<u8> {
        let mut bytes = Vec::from(GGUF_MAGIC);
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        push_string(&mut bytes, key);
        bytes.extend_from_slice(&value_type.to_le_bytes());
        bytes
    }

    fn bare_file(path: &str, size: u64) -> GgufFile {
        GgufFile {
            path: PathBuf::from(path),
            size,
            quantization: "unknown".to_owned(),
            modified: None,
            metadata: None,
            parse_error: None,
        }
    }

    #[test]
    fn shard_position_parses_split_names() {
        let parsed = shard_position(Path::new(
            "/models/quant/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf",
        ))
        .unwrap();
        assert_eq!(parsed.index, 1);
        assert_eq!(parsed.total, 33);
        assert!(shard_position(Path::new("single.gguf")).is_none());
        assert!(shard_position(Path::new("model-of-parts.gguf")).is_none());
        assert!(shard_position(Path::new("model-00003-of-two.gguf")).is_none());
        assert!(shard_position(Path::new("model-00000-of-00003.gguf")).is_none());
        assert!(shard_position(Path::new("model-00002-of-00000.gguf")).is_none());
        assert!(shard_position(Path::new("model--of-00003.gguf")).is_none());
    }

    #[test]
    fn group_shard_sets_collapses_complete_sets() {
        let files = vec![
            bare_file("/m/A-00002-of-00003.gguf", 2),
            bare_file("/m/standalone.gguf", 7),
            bare_file("/m/A-00003-of-00003.gguf", 3),
            bare_file("/m/A-00001-of-00003.gguf", 1),
        ];
        let sets = group_shard_sets(files);
        assert_eq!(sets.len(), 2);
        let set = &sets[0];
        assert_eq!(set.primary.path, PathBuf::from("/m/A-00001-of-00003.gguf"));
        assert_eq!(set.members.len(), 3);
        assert_eq!(set.total_size, 6);
        assert_eq!(set.shard_total, Some(3));
        assert!(!set.incomplete);
        let single = &sets[1];
        assert_eq!(single.members.len(), 1);
        assert_eq!(single.total_size, 7);
        assert_eq!(single.shard_total, None);
        assert!(!single.incomplete);
    }

    #[test]
    fn group_shard_sets_flags_incomplete_and_duplicate_shards() {
        let missing = group_shard_sets(vec![
            bare_file("/m/A-00001-of-00003.gguf", 1),
            bare_file("/m/A-00002-of-00003.gguf", 2),
        ]);
        assert_eq!(missing.len(), 1);
        assert!(missing[0].incomplete);
        assert_eq!(
            missing[0].primary.path,
            PathBuf::from("/m/A-00001-of-00003.gguf")
        );

        let anchored = group_shard_sets(vec![
            bare_file("/m/A-00002-of-00003.gguf", 2),
            bare_file("/m/A-00003-of-00003.gguf", 3),
        ]);
        assert!(anchored[0].incomplete);
        assert_eq!(
            anchored[0].primary.path,
            PathBuf::from("/m/A-00002-of-00003.gguf")
        );

        let mismatched_extension = group_shard_sets(vec![
            bare_file("/m/A-00001-of-00002.gguf", 1),
            bare_file("/m/A-00001-of-00002.GGUF", 9),
            bare_file("/m/A-00002-of-00002.gguf", 2),
        ]);
        assert_eq!(mismatched_extension.len(), 1);
        assert!(mismatched_extension[0].incomplete);

        let stray_lookup = group_shard_sets(vec![
            bare_file("/m/A-00001-of-00002.gguf", 1),
            bare_file("/m/A-00002-of-00002.gguf", 2),
            bare_file("/m/A-00002-of-00002.gguf.bak", 9),
        ]);
        assert_eq!(stray_lookup.len(), 2);
        assert!(!stray_lookup[0].incomplete);
        assert!(!stray_lookup[1].incomplete);
    }

    #[test]
    fn group_shard_sets_separates_directories_and_totals() {
        let files = vec![
            bare_file("/m1/A-00001-of-00002.gguf", 1),
            bare_file("/m2/A-00001-of-00002.gguf", 4),
            bare_file("/m1/A-00001-of-00003.gguf", 2),
            bare_file("/m1/A-00002-of-00003.gguf", 3),
            bare_file("/m1/A-00003-of-00003.gguf", 4),
        ];
        let sets = group_shard_sets(files);
        assert_eq!(sets.len(), 3);
        // Different totals and directories form distinct sets.
        let complete = &sets[1];
        assert_eq!(complete.members.len(), 3);
        assert_eq!(complete.shard_total, Some(3));
        assert!(!complete.incomplete);
        assert!(sets[0].incomplete);
        assert_eq!(sets[0].shard_total, Some(2));
        assert_eq!(
            sets[0].primary.path,
            PathBuf::from("/m1/A-00001-of-00002.gguf")
        );
        assert!(sets[2].incomplete);
        assert_eq!(
            sets[2].primary.path,
            PathBuf::from("/m2/A-00001-of-00002.gguf")
        );
    }
}
