//! Read-only catalog over the native language, media, and decision manifests.
//!
//! The catalog identifies profiles by lane and profile ID, preserves each
//! manifest's runtime and task declarations, and resolves every source path
//! from the repository root supplied by the caller. It never probes model
//! weights, installs, processes, or runtime health. Every loaded entry is
//! therefore reported as [`Readiness::ManifestPresent`].

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, ensure, Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;

const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_PROFILES_PER_MANIFEST: usize = 4096;
const MAX_PROFILE_ID_BYTES: usize = 128;

/// The toolkit lane that owns a profile.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Lane {
    Language,
    Media,
    Decisions,
}

impl Lane {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Language => "language",
            Self::Media => "media",
            Self::Decisions => "decisions",
        }
    }
}

impl fmt::Display for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A profile ID within one lane. IDs use 1–128 ASCII letters, digits, `.`,
/// `_`, or `-`; namespacing is handled by [`ProfileKey`].
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProfileId(String);

impl ProfileId {
    pub fn new(value: impl Into<String>) -> std::result::Result<Self, ProfileKeyError> {
        let value = value.into();
        if value.len() > MAX_PROFILE_ID_BYTES {
            return Err(ProfileKeyError(format!(
                "invalid profile ID {value:?}; maximum length is {MAX_PROFILE_ID_BYTES} bytes"
            )));
        }
        let mut chars = value.chars();
        let valid_first = chars
            .next()
            .is_some_and(|character| character.is_ascii_alphanumeric());
        let valid_rest = chars.all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
        });
        if !valid_first || !valid_rest {
            return Err(ProfileKeyError(format!(
                "invalid profile ID {value:?}; expected ASCII letters, digits, '.', '_' or '-'"
            )));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A stable profile identifier namespaced as `lane:profile_id`.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ProfileKey {
    pub lane: Lane,
    pub profile_id: ProfileId,
}

impl ProfileKey {
    pub fn new(
        lane: Lane,
        profile_id: impl Into<String>,
    ) -> std::result::Result<Self, ProfileKeyError> {
        Ok(Self {
            lane,
            profile_id: ProfileId::new(profile_id)?,
        })
    }
}

impl fmt::Display for ProfileKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.lane, self.profile_id)
    }
}

impl FromStr for ProfileKey {
    type Err = ProfileKeyError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let Some((lane, profile_id)) = value.split_once(':') else {
            return Err(ProfileKeyError(format!(
                "invalid profile key {value:?}; expected language:id, media:id, or decisions:id"
            )));
        };
        let lane = match lane {
            "language" => Lane::Language,
            "media" => Lane::Media,
            "decisions" => Lane::Decisions,
            _ => {
                return Err(ProfileKeyError(format!(
                    "unknown lane {lane:?}; expected language, media, or decisions"
                )));
            }
        };
        Self::new(lane, profile_id)
    }
}

/// Invalid namespaced profile key input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileKeyError(String);

impl fmt::Display for ProfileKeyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ProfileKeyError {}

/// A task tag. Media tags are preserved exactly as declared in its manifest.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Task(String);

impl Task {
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        ensure!(!value.trim().is_empty(), "task must not be empty");
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Task {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A runtime identity is namespaced because different lanes may use the same
/// native runtime label while owning independent runtime lifecycles.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct RuntimeIdentity {
    pub lane: Lane,
    id: String,
}

impl RuntimeIdentity {
    pub fn new(lane: Lane, id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        ensure!(!id.trim().is_empty(), "runtime identity must not be empty");
        Ok(Self { lane, id })
    }

    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl fmt::Display for RuntimeIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.lane, self.id)
    }
}

/// A device declared by a native manifest. `Other` preserves values the
/// catalog does not currently classify.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Device {
    Cpu,
    Cuda,
    Remote,
    Other(String),
}

impl Device {
    fn from_native(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Self::Cpu,
            "cuda" => Self::Cuda,
            "remote" => Self::Remote,
            _ => Self::Other(value.to_owned()),
        }
    }
}

/// What the catalog can establish without contacting or starting a runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Readiness {
    /// A native source file declares the profile. This does not imply that
    /// dependencies or weights are installed or that a runtime is healthy.
    ManifestPresent,
    /// An independent, explicit installation check has passed.
    Installed,
    /// An independent runtime health check has passed.
    Healthy,
}

/// A normalized, read-only view of one profile declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CatalogProfile {
    pub key: ProfileKey,
    pub name: String,
    pub tasks: Vec<Task>,
    pub runtime: RuntimeIdentity,
    /// Absolute path to the native manifest that owns this profile.
    pub source_path: PathBuf,
    /// `None` means the native source does not declare a device.
    pub device: Option<Device>,
    /// Catalog loading only emits `ManifestPresent`; it does not probe installs
    /// or runtime health.
    pub readiness: Readiness,
    /// The language manifest's config-level disabled flag, when present.
    pub disabled: bool,
    /// Manifest-native route fields preserved for the route resolver. This is
    /// crate-visible so consumers use the validated route interface instead
    /// of parsing manifest details a second time.
    pub(crate) native_source: NativeProfileSource,
}

/// Native manifest fields needed to construct a lane-specific dispatch route.
/// Paths remain raw until the route resolver applies bounded relative-path
/// validation and anchors them to the already-canonical manifest directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum NativeProfileSource {
    Language {
        model_id: ProfileId,
    },
    Media {
        script: String,
    },
    Decisions {
        entrypoint: String,
        model: String,
        profile_type: String,
        tier: Option<String>,
    },
}

/// A deterministic catalog built from the three native L3MS manifests.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    profiles: Vec<CatalogProfile>,
    roots: Option<[PathBuf; 3]>,
}

impl Catalog {
    /// Read native manifests from the conventional roots beneath a repository.
    /// This is a convenience for the default layout; callers with configured
    /// toolkit locations should use [`Self::load_from_roots`].
    pub fn load(repo_root: impl AsRef<Path>) -> Result<Self> {
        let repo_root = repo_root.as_ref();
        Self::load_from_roots(
            repo_root,
            repo_root.join("toolkits/l2m2"),
            repo_root.join("toolkits/l3ds"),
        )
    }

    /// Read manifests from explicit language, media, and decisions roots.
    /// The language root must contain `llama-swap.yaml`; the other roots must
    /// contain their lane's native JSON manifest. Each root is canonicalized
    /// independently, so configured external toolkit locations and arbitrary
    /// process working directories are supported. Reads are limited to 1 MiB
    /// and 4,096 profiles per manifest. This method performs filesystem reads
    /// only; it does not invoke engines or inspect runtime health.
    pub fn load_from_roots(
        language_root: impl AsRef<Path>,
        media_root: impl AsRef<Path>,
        decisions_root: impl AsRef<Path>,
    ) -> Result<Self> {
        let language_root = canonicalize_root(language_root.as_ref(), "language")?;
        let media_root = canonicalize_root(media_root.as_ref(), "media")?;
        let decisions_root = canonicalize_root(decisions_root.as_ref(), "decisions")?;
        let mut profiles = Vec::new();
        let mut keys = BTreeSet::new();

        load_language(&language_root, &mut profiles, &mut keys)?;
        load_media(&media_root, &mut profiles, &mut keys)?;
        load_decisions(&decisions_root, &mut profiles, &mut keys)?;

        profiles.sort_by(|left, right| left.key.cmp(&right.key));
        Ok(Self {
            profiles,
            roots: Some([language_root, media_root, decisions_root]),
        })
    }

    /// Re-read native manifests from their original canonical roots. Empty
    /// lanes retain their configured locations; the current working directory
    /// and cached profile contents never supply replacement roots.
    pub fn refresh(&self) -> Result<Self> {
        let roots = self
            .roots
            .as_ref()
            .context("catalog has no loaded manifest roots")?;
        Self::load_from_roots(&roots[0], &roots[1], &roots[2])
    }

    /// Profiles in stable lane and profile-ID order.
    pub fn list(&self) -> &[CatalogProfile] {
        &self.profiles
    }

    /// Find a profile by its lane-qualified key.
    pub fn inspect(&self, key: &ProfileKey) -> Option<&CatalogProfile> {
        self.profiles.iter().find(|profile| &profile.key == key)
    }
}

fn canonicalize_root(path: &Path, lane: &str) -> Result<PathBuf> {
    let root = fs::canonicalize(path)
        .with_context(|| format!("resolve {lane} catalog root {}", path.display()))?;
    ensure!(
        root.is_dir(),
        "{lane} catalog root is not a directory: {}",
        root.display()
    );
    Ok(root)
}

#[derive(Deserialize)]
struct MediaManifest {
    schema_version: u32,
    profiles: Vec<MediaProfile>,
}

#[derive(Deserialize)]
struct MediaProfile {
    id: String,
    name: String,
    runtime: String,
    tasks: Vec<String>,
    script: String,
    #[serde(default)]
    device: Option<String>,
}

#[derive(Deserialize)]
struct DecisionManifest {
    version: String,
    profiles: Vec<DecisionProfile>,
}

#[derive(Deserialize)]
struct DecisionProfile {
    id: String,
    name: String,
    engine: String,
    tier: Option<String>,
    #[serde(rename = "type")]
    profile_type: String,
    model: String,
    entrypoint: String,
    device: Option<String>,
}

fn load_language(
    language_root: &Path,
    profiles: &mut Vec<CatalogProfile>,
    keys: &mut BTreeSet<ProfileKey>,
) -> Result<()> {
    let source_path = language_root.join("llama-swap.yaml");
    let bytes = read_manifest_bytes(&source_path)?;
    let contents = String::from_utf8(bytes)
        .with_context(|| format!("parse language manifest {} as UTF-8", source_path.display()))?;
    let entries = crate::swap_yaml::parse_entries(&contents);
    ensure_profile_count(&source_path, entries.len())?;
    if entries.is_empty() {
        bail!(
            "no language model entries found in {}",
            source_path.display()
        );
    }

    for entry in entries {
        let key = ProfileKey::new(Lane::Language, entry.id)
            .with_context(|| format!("invalid profile ID in {}", source_path.display()))?;
        let native_source = NativeProfileSource::Language {
            model_id: key.profile_id.clone(),
        };
        let tasks = entry
            .tasks
            .into_iter()
            .map(Task::new)
            .collect::<Result<Vec<_>>>()?;
        let profile = CatalogProfile {
            name: key.profile_id.to_string(),
            key,
            tasks,
            runtime: RuntimeIdentity::new(Lane::Language, "llama-swap")?,
            source_path: source_path.clone(),
            device: None,
            readiness: Readiness::ManifestPresent,
            disabled: entry.disabled,
            native_source,
        };
        insert_profile(profiles, keys, profile)?;
    }
    Ok(())
}

fn load_media(
    media_root: &Path,
    profiles: &mut Vec<CatalogProfile>,
    keys: &mut BTreeSet<ProfileKey>,
) -> Result<()> {
    let source_path = media_root.join("media-runtimes.json");
    let manifest: MediaManifest = read_json(&source_path)?;
    ensure_profile_count(&source_path, manifest.profiles.len())?;
    ensure!(
        manifest.schema_version == 1,
        "unsupported media manifest schema {} in {}",
        manifest.schema_version,
        source_path.display()
    );

    for native in manifest.profiles {
        ensure!(
            !native.name.trim().is_empty(),
            "empty profile name in {}",
            source_path.display()
        );
        ensure!(
            !native.tasks.is_empty(),
            "profile {:?} has no task declarations in {}",
            native.id,
            source_path.display()
        );
        ensure!(
            !native.script.trim().is_empty(),
            "profile {:?} has no script path in {}",
            native.id,
            source_path.display()
        );
        let key = ProfileKey::new(Lane::Media, native.id)
            .with_context(|| format!("invalid profile ID in {}", source_path.display()))?;
        let native_source = NativeProfileSource::Media {
            script: native.script,
        };
        let profile = CatalogProfile {
            key,
            name: native.name,
            tasks: native
                .tasks
                .into_iter()
                .map(Task::new)
                .collect::<Result<_>>()?,
            runtime: RuntimeIdentity::new(Lane::Media, native.runtime)
                .with_context(|| format!("invalid runtime in {}", source_path.display()))?,
            source_path: source_path.clone(),
            device: native.device.as_deref().map(Device::from_native),
            readiness: Readiness::ManifestPresent,
            disabled: false,
            native_source,
        };
        insert_profile(profiles, keys, profile)?;
    }
    Ok(())
}

fn load_decisions(
    decisions_root: &Path,
    profiles: &mut Vec<CatalogProfile>,
    keys: &mut BTreeSet<ProfileKey>,
) -> Result<()> {
    let source_path = decisions_root.join("decision-runtimes.json");
    let manifest: DecisionManifest = read_json(&source_path)?;
    ensure_profile_count(&source_path, manifest.profiles.len())?;
    ensure!(
        !manifest.version.trim().is_empty(),
        "empty decision manifest version in {}",
        source_path.display()
    );

    for native in manifest.profiles {
        ensure!(
            !native.name.trim().is_empty(),
            "empty profile name in {}",
            source_path.display()
        );
        ensure!(
            !native.model.trim().is_empty(),
            "profile {:?} has no model identifier in {}",
            native.id,
            source_path.display()
        );
        ensure!(
            !native.profile_type.trim().is_empty(),
            "profile {:?} has no type in {}",
            native.id,
            source_path.display()
        );
        ensure!(
            !native.entrypoint.trim().is_empty(),
            "profile {:?} has no entrypoint in {}",
            native.id,
            source_path.display()
        );
        let key = ProfileKey::new(Lane::Decisions, native.id)
            .with_context(|| format!("invalid profile ID in {}", source_path.display()))?;
        let native_source = NativeProfileSource::Decisions {
            entrypoint: native.entrypoint,
            model: native.model,
            profile_type: native.profile_type,
            tier: native.tier,
        };
        let profile = CatalogProfile {
            key,
            name: native.name,
            tasks: vec![Task::new("decision")?],
            runtime: RuntimeIdentity::new(Lane::Decisions, native.engine)
                .with_context(|| format!("invalid engine in {}", source_path.display()))?,
            source_path: source_path.clone(),
            device: native.device.as_deref().map(Device::from_native),
            readiness: Readiness::ManifestPresent,
            disabled: false,
            native_source,
        };
        insert_profile(profiles, keys, profile)?;
    }
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = read_manifest_bytes(path)?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse manifest {}", path.display()))
}

fn read_manifest_bytes(path: &Path) -> Result<Vec<u8>> {
    let file = fs::File::open(path).with_context(|| format!("read manifest {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(MAX_MANIFEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read manifest {}", path.display()))?;
    ensure!(
        bytes.len() as u64 <= MAX_MANIFEST_BYTES,
        "manifest {} exceeds the {MAX_MANIFEST_BYTES}-byte limit",
        path.display()
    );
    Ok(bytes)
}

fn ensure_profile_count(path: &Path, count: usize) -> Result<()> {
    ensure!(
        count <= MAX_PROFILES_PER_MANIFEST,
        "manifest {} contains {count} profiles; limit is {MAX_PROFILES_PER_MANIFEST}",
        path.display()
    );
    Ok(())
}

fn insert_profile(
    profiles: &mut Vec<CatalogProfile>,
    keys: &mut BTreeSet<ProfileKey>,
    profile: CatalogProfile,
) -> Result<()> {
    if !keys.insert(profile.key.clone()) {
        bail!("duplicate profile ID {}", profile.key);
    }
    profiles.push(profile);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Catalog, Device, Lane, ProfileKey, Readiness, Task};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::str::FromStr;

    use tempfile::TempDir;

    const YAML: &str = "models:\n  shared-id:\n    cmd: |\n      -m /models/not-installed.gguf\n";
    const MEDIA_JSON: &str = r#"{
      "schema_version": 1,
      "profiles": [{
        "id": "shared-id",
        "name": "Fixture media profile",
        "runtime": "ComfyUI",
        "tasks": ["image", "video"],
        "script": "media-models/not-installed.sh"
      }]
    }"#;
    const DECISION_JSON: &str = r#"{
      "version": "1.0",
      "profiles": [{
        "id": "shared-id",
        "name": "Fixture decision profile",
        "engine": "nagi",
        "type": "local",
        "model": "not-installed/model",
        "device": "cuda",
        "entrypoint": "decision-models/not-installed.py"
      }]
    }"#;

    fn fixture_root() -> TempDir {
        let root = tempfile::tempdir().expect("create fixture root");
        write(root.path(), "llama-swap.yaml", YAML);
        write(root.path(), "toolkits/l2m2/media-runtimes.json", MEDIA_JSON);
        write(
            root.path(),
            "toolkits/l3ds/decision-runtimes.json",
            DECISION_JSON,
        );
        root
    }

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("fixture path has parent"))
            .expect("create fixture parent");
        fs::write(path, contents).expect("write fixture file");
    }

    fn key(value: &str) -> ProfileKey {
        ProfileKey::from_str(value).expect("valid namespaced key")
    }

    #[test]
    fn namespaced_keys_round_trip_and_validate() {
        for (text, lane) in [
            ("language:qwen38-flash-next", Lane::Language),
            ("media:breeze-tts-2", Lane::Media),
            ("decisions:nagi-smol", Lane::Decisions),
        ] {
            let parsed = key(text);
            assert_eq!(parsed.lane, lane);
            assert_eq!(parsed.to_string(), text);
        }

        assert!(ProfileKey::from_str("audio:breeze").is_err());
        assert!(ProfileKey::from_str("media:").is_err());
        assert!(ProfileKey::from_str("media:bad/id").is_err());
        assert!(ProfileKey::from_str("media:bad:extra").is_err());
        assert!(ProfileKey::new(Lane::Media, "a".repeat(128)).is_ok());
        assert!(ProfileKey::new(Lane::Media, "a".repeat(129)).is_err());
    }

    #[test]
    fn duplicate_profile_ids_are_distinct_across_lanes() {
        let root = fixture_root();
        let catalog = Catalog::load(root.path()).expect("load fixture catalog");

        assert_eq!(catalog.list().len(), 3);
        for lane in [Lane::Language, Lane::Media, Lane::Decisions] {
            let profile_key = ProfileKey::new(lane, "shared-id").expect("valid fixture ID");
            assert_eq!(catalog.inspect(&profile_key).unwrap().key, profile_key);
        }
        assert_eq!(
            catalog.inspect(&key("media:shared-id")).unwrap().tasks,
            vec![Task::new("image").unwrap(), Task::new("video").unwrap(),]
        );
    }

    #[test]
    fn language_tasks_follow_active_embedding_flags_not_profile_names() {
        let root = fixture_root();
        write(
            root.path(),
            "llama-swap.yaml",
            r#"models:
  "arbitrary-name":
    cmd: |
      llama-server --embedding
  "ordinary-chat":
    cmd: |
      llama-server --port ${PORT}
  # "nomic-embed-text-v1.5":
  #   cmd: |
  #     llama-server --embedding
"#,
        );

        let catalog = Catalog::load(root.path()).expect("load fixture catalog");
        assert_eq!(
            catalog
                .inspect(&key("language:arbitrary-name"))
                .unwrap()
                .tasks,
            vec![Task::new("embeddings").unwrap()]
        );
        assert_eq!(
            catalog
                .inspect(&key("language:ordinary-chat"))
                .unwrap()
                .tasks,
            vec![Task::new("chat").unwrap()]
        );
        assert!(catalog
            .inspect(&key("language:nomic-embed-text-v1.5"))
            .is_none());
    }

    #[test]
    fn duplicate_ids_within_one_lane_are_rejected() {
        let root = fixture_root();
        write(
            root.path(),
            "llama-swap.yaml",
            "models:\n  repeated:\n    cmd: first\n  repeated:\n    cmd: second\n",
        );

        let error = Catalog::load(root.path()).expect_err("duplicate lane IDs must fail");
        assert!(error
            .to_string()
            .contains("duplicate profile ID language:repeated"));
    }

    #[test]
    fn missing_manifest_reports_its_source_path() {
        let root = fixture_root();
        fs::remove_file(root.path().join("toolkits/l3ds/decision-runtimes.json"))
            .expect("remove decision manifest fixture");

        let error = Catalog::load(root.path()).expect_err("missing manifest must fail");
        assert!(error.to_string().contains("decision-runtimes.json"));
    }

    #[test]
    fn refresh_retains_empty_lane_roots_and_reloads_native_changes() {
        let root = fixture_root();
        write(
            root.path(),
            "toolkits/l2m2/media-runtimes.json",
            r#"{"schema_version":1,"profiles":[]}"#,
        );
        let catalog = Catalog::load(root.path()).unwrap();
        assert!(!catalog
            .list()
            .iter()
            .any(|profile| profile.key.lane == Lane::Media));
        write(
            root.path(),
            "toolkits/l2m2/media-runtimes.json",
            r#"{"schema_version":1,"profiles":[{"id":"added","name":"Added","runtime":"audio.cpp","tasks":["tts"],"script":"media-models/added.sh"}]}"#,
        );
        let refreshed = catalog.refresh().unwrap();
        assert!(refreshed
            .inspect(&ProfileKey::new(Lane::Media, "added").unwrap())
            .is_some());
        assert!(catalog
            .inspect(&ProfileKey::new(Lane::Media, "added").unwrap())
            .is_none());
        assert!(Catalog::default().refresh().is_err());
    }

    #[test]
    fn malformed_manifest_reports_its_source_path() {
        let root = fixture_root();
        write(
            root.path(),
            "toolkits/l2m2/media-runtimes.json",
            "{ malformed",
        );

        let error = Catalog::load(root.path()).expect_err("malformed manifest must fail");
        assert!(error.to_string().contains("media-runtimes.json"));
    }

    #[test]
    fn explicit_root_resolves_sources_independently_of_working_directory() {
        let root = fixture_root();
        let root_path = fs::canonicalize(root.path()).expect("canonical fixture root");
        let catalog = Catalog::load(root.path()).expect("load fixture root, not process cwd");

        assert!(catalog.inspect(&key("media:shared-id")).is_some());
        for profile in catalog.list() {
            assert!(profile.source_path.is_absolute());
            assert!(profile.source_path.starts_with(&root_path));
        }
        assert_eq!(
            catalog
                .inspect(&key("media:shared-id"))
                .unwrap()
                .source_path,
            root_path.join("toolkits/l2m2/media-runtimes.json")
        );
        assert_ne!(root_path, PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    }

    #[test]
    fn distinct_external_roots_resolve_each_lane_without_cwd_assumptions() {
        let temp = tempfile::tempdir().expect("create independent roots");
        let language_root = temp.path().join("host-language");
        let media_root = temp.path().join("user-media");
        let decisions_root = temp.path().join("cli-decisions");
        write(&language_root, "llama-swap.yaml", YAML);
        write(&media_root, "media-runtimes.json", MEDIA_JSON);
        write(&decisions_root, "decision-runtimes.json", DECISION_JSON);

        let catalog = Catalog::load_from_roots(&language_root, &media_root, &decisions_root)
            .expect("load each independently selected root");

        let language_path = fs::canonicalize(language_root.join("llama-swap.yaml")).unwrap();
        let media_path = fs::canonicalize(media_root.join("media-runtimes.json")).unwrap();
        let decisions_path =
            fs::canonicalize(decisions_root.join("decision-runtimes.json")).unwrap();
        assert_eq!(
            catalog
                .inspect(&key("language:shared-id"))
                .unwrap()
                .source_path,
            language_path
        );
        assert_eq!(
            catalog
                .inspect(&key("media:shared-id"))
                .unwrap()
                .source_path,
            media_path
        );
        assert_eq!(
            catalog
                .inspect(&key("decisions:shared-id"))
                .unwrap()
                .source_path,
            decisions_path
        );
        assert_ne!(language_root, media_root);
        assert_ne!(media_root, decisions_root);
        assert_ne!(
            fs::canonicalize(temp.path()).unwrap(),
            fs::canonicalize(env!("CARGO_MANIFEST_DIR")).unwrap()
        );
    }

    #[test]
    fn manifest_presence_does_not_claim_installation_or_health_or_write_state() {
        let root = fixture_root();
        let paths = [
            root.path().join("llama-swap.yaml"),
            root.path().join("toolkits/l2m2/media-runtimes.json"),
            root.path().join("toolkits/l3ds/decision-runtimes.json"),
        ];
        let before = paths
            .iter()
            .map(fs::read)
            .collect::<std::io::Result<Vec<_>>>()
            .expect("read fixture source files");

        let catalog = Catalog::load(root.path()).expect("load fixture catalog");

        let after = paths
            .iter()
            .map(fs::read)
            .collect::<std::io::Result<Vec<_>>>()
            .expect("re-read fixture source files");
        assert_eq!(before, after);
        assert!(!root.path().join(".venv").exists());
        assert!(!root.path().join("models").exists());
        assert!(catalog
            .list()
            .iter()
            .all(|profile| profile.readiness == Readiness::ManifestPresent));
        assert_eq!(
            catalog.inspect(&key("decisions:shared-id")).unwrap().device,
            Some(Device::Cuda)
        );
        assert!(!root
            .path()
            .join("toolkits/l3ds/decision-models/not-installed.py")
            .exists());
    }

    #[test]
    fn oversized_manifest_is_rejected_before_parsing() {
        let root = fixture_root();
        let source_path = root.path().join("toolkits/l2m2/media-runtimes.json");
        fs::write(
            &source_path,
            " ".repeat((super::MAX_MANIFEST_BYTES + 1) as usize),
        )
        .expect("write oversized fixture manifest");

        let error = Catalog::load(root.path()).expect_err("oversized manifest must fail");
        assert!(error.to_string().contains("exceeds the 1048576-byte limit"));
        assert!(error.to_string().contains("media-runtimes.json"));
    }

    #[test]
    fn excessive_profile_count_is_rejected() {
        let root = fixture_root();
        let mut yaml = String::from("models:\n");
        for index in 0..=super::MAX_PROFILES_PER_MANIFEST {
            yaml.push_str(&format!("  profile-{index}:\n    cmd: server\n"));
        }
        write(root.path(), "llama-swap.yaml", &yaml);

        let error = Catalog::load(root.path()).expect_err("too many profiles must fail");
        assert!(error.to_string().contains("contains 4097 profiles"));
        assert!(error.to_string().contains("llama-swap.yaml"));
    }

    #[test]
    fn loads_repository_manifests_without_runtime_checks() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let catalog = Catalog::load(&root).expect("load checked-in manifests");

        assert!(!catalog.list().is_empty());
        assert!(catalog
            .list()
            .iter()
            .all(|profile| profile.readiness == Readiness::ManifestPresent));
        assert_eq!(
            catalog.inspect(&key("media:breeze-tts-2")).unwrap().device,
            Some(Device::Cuda)
        );
        assert!(catalog.inspect(&key("decisions:systemone")).is_some());
    }
}
