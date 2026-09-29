//! Read-only resolution from catalog profiles to manifest-native dispatch data.
//!
//! The resolver anchors bounded relative entrypoints to the canonical directory
//! containing each native manifest. It does not check that files exist, follow
//! symlinks, inspect the host, check runtime health, or start/stop a runtime.
//! A future dispatcher must revalidate filesystem containment when opening a
//! resolved path. Returned paths describe declared locations, not readiness.

use std::path::{Component, Path, PathBuf};

use anyhow::{ensure, Context, Result};

use super::catalog::{
    Catalog, NativeProfileSource, ProfileId, ProfileKey, Readiness, RuntimeIdentity, Task,
};

const MAX_NATIVE_PATH_BYTES: usize = 512;
const MAX_NATIVE_PATH_COMPONENT_BYTES: usize = 128;

/// Symbolic endpoint reference for llama-swap's currently configured chat
/// connection. The route resolver deliberately does not choose a URL or auth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaSwapEndpointReference {
    ConfiguredChatConnection,
}

/// Native dispatch data preserved from the source manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NativeRoute {
    /// Model key from `llama-swap.yaml` plus a caller-managed endpoint reference.
    LlamaSwapModel {
        model_id: ProfileId,
        endpoint: LlamaSwapEndpointReference,
    },
    /// Relative media script resolved under the media toolkit root.
    MediaScript { path: PathBuf },
    /// Decision runner entrypoint and its declared model/type/tier.
    DecisionRunner {
        entrypoint: PathBuf,
        model: String,
        profile_type: String,
        tier: Option<String>,
    },
}

/// A route resolved from one catalog profile without runtime side effects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedProfileRoute {
    /// Lane-qualified catalog identity; never inferred from the runtime name.
    pub profile: ProfileKey,
    /// Namespaced runtime declared or implied by the native manifest.
    pub runtime: RuntimeIdentity,
    /// Native dispatch location and associated manifest values.
    pub native_route: NativeRoute,
    /// Original task labels from the native catalog entry, without aliases.
    pub tasks: Vec<Task>,
    /// Catalog readiness only; resolution does not upgrade or probe it.
    pub readiness: Readiness,
}

/// Resolve a profile key into its actual manifest-declared native route.
///
/// Unknown and disabled profiles are rejected. Media scripts and decision
/// entrypoints must be bounded safe relative paths; returned paths are joined
/// to the canonical directory of their owning manifest. Language endpoint
/// selection remains with the chat connection (`LLAMA_SWAP_URL`/UI config); no
/// endpoint URL or credentials are read here. This result is an identity and
/// location contract, not permission to dispatch.
pub fn resolve_profile_route(catalog: &Catalog, key: &ProfileKey) -> Result<ResolvedProfileRoute> {
    let profile = catalog
        .inspect(key)
        .ok_or_else(|| anyhow::anyhow!("unknown profile {key}"))?;
    ensure!(!profile.disabled, "profile {key} is disabled");

    let native_route = match &profile.native_source {
        NativeProfileSource::Language { model_id } => NativeRoute::LlamaSwapModel {
            model_id: model_id.clone(),
            endpoint: LlamaSwapEndpointReference::ConfiguredChatConnection,
        },
        NativeProfileSource::Media { script } => NativeRoute::MediaScript {
            path: resolve_manifest_path(&profile.source_path, script, "media script")?,
        },
        NativeProfileSource::Decisions {
            entrypoint,
            model,
            profile_type,
            tier,
        } => NativeRoute::DecisionRunner {
            entrypoint: resolve_manifest_path(
                &profile.source_path,
                entrypoint,
                "decision entrypoint",
            )?,
            model: model.clone(),
            profile_type: profile_type.clone(),
            tier: tier.clone(),
        },
    };

    Ok(ResolvedProfileRoute {
        profile: profile.key.clone(),
        runtime: profile.runtime.clone(),
        native_route,
        tasks: profile.tasks.clone(),
        readiness: profile.readiness,
    })
}

fn resolve_manifest_path(manifest_path: &Path, raw: &str, label: &str) -> Result<PathBuf> {
    ensure!(
        !raw.is_empty() && raw.len() <= MAX_NATIVE_PATH_BYTES,
        "{label} path must be 1–{MAX_NATIVE_PATH_BYTES} bytes"
    );
    ensure!(
        !raw.contains('\\') && !raw.contains(':') && !raw.chars().any(char::is_control),
        "{label} path contains a forbidden character"
    );
    for component in raw.split('/') {
        ensure!(
            !component.is_empty()
                && component != "."
                && component != ".."
                && component.len() <= MAX_NATIVE_PATH_COMPONENT_BYTES,
            "{label} path contains an invalid component"
        );
    }

    let relative = Path::new(raw);
    ensure!(!relative.is_absolute(), "{label} path must be relative");
    ensure!(
        relative
            .components()
            .all(|component| matches!(component, Component::Normal(_))),
        "{label} path must contain only normal relative components"
    );

    let root = manifest_path
        .parent()
        .with_context(|| format!("manifest path has no parent: {}", manifest_path.display()))?;
    let resolved = root.join(relative);
    ensure!(
        resolved.starts_with(root),
        "{label} path escapes its manifest root"
    );
    Ok(resolved)
}

#[cfg(test)]
mod tests {
    use super::super::catalog::Lane;
    use super::*;
    use std::fs;

    use tempfile::TempDir;

    fn fixture_root() -> TempDir {
        let root = tempfile::tempdir().expect("create fixture root");
        write(
            root.path(),
            "llama-swap.yaml",
            "models:\n  shared-id:\n    cmd: model\n  disabled-model:\n    disabled: true\n    cmd: model\n",
        );
        write(
            root.path(),
            "toolkits/l2m2/media-runtimes.json",
            r#"{
              "schema_version": 1,
              "profiles": [{
                "id": "shared-id",
                "name": "Fixture media profile",
                "runtime": "ComfyUI",
                "tasks": ["tts", "design", "clon"],
                "script": "media-models/not-installed.sh"
              }]
            }"#,
        );
        write(
            root.path(),
            "toolkits/l3ds/decision-runtimes.json",
            r#"{
              "version": "1.0",
              "profiles": [{
                "id": "shared-id",
                "name": "Fixture decision profile",
                "engine": "nagi",
                "type": "local",
                "tier": "smol",
                "model": "org/model-name",
                "entrypoint": "decision-models/not-installed.py"
              }]
            }"#,
        );
        root
    }

    fn write(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().expect("fixture path has parent"))
            .expect("create fixture parent");
        fs::write(path, contents).expect("write fixture file");
    }

    fn key(lane: Lane, profile_id: &str) -> ProfileKey {
        ProfileKey::new(lane, profile_id).expect("valid fixture profile key")
    }

    #[test]
    fn resolves_actual_native_locations_and_keeps_profile_runtime_separate() {
        let root = fixture_root();
        let catalog = Catalog::load(root.path()).expect("load fixture catalog");
        let root_path = fs::canonicalize(root.path()).expect("canonical fixture root");

        let language = resolve_profile_route(&catalog, &key(Lane::Language, "shared-id"))
            .expect("resolve language profile");
        assert_eq!(
            language.native_route,
            NativeRoute::LlamaSwapModel {
                model_id: ProfileId::new("shared-id").unwrap(),
                endpoint: LlamaSwapEndpointReference::ConfiguredChatConnection,
            }
        );
        assert_eq!(language.profile.lane, Lane::Language);
        assert_eq!(language.runtime.lane, Lane::Language);
        assert_eq!(language.runtime.as_str(), "llama-swap");

        let media = resolve_profile_route(&catalog, &key(Lane::Media, "shared-id"))
            .expect("resolve media profile");
        assert_eq!(
            media.native_route,
            NativeRoute::MediaScript {
                path: root_path.join("toolkits/l2m2/media-models/not-installed.sh"),
            }
        );
        assert_eq!(media.profile.lane, Lane::Media);
        assert_eq!(media.runtime.lane, Lane::Media);
        assert_eq!(media.runtime.as_str(), "ComfyUI");
        assert_eq!(
            media.tasks.iter().map(Task::as_str).collect::<Vec<_>>(),
            ["tts", "design", "clon"]
        );
        assert_eq!(media.readiness, Readiness::ManifestPresent);
        assert!(!matches!(&media.native_route, NativeRoute::MediaScript { path } if path.exists()));

        let decision = resolve_profile_route(&catalog, &key(Lane::Decisions, "shared-id"))
            .expect("resolve decision profile");
        assert_eq!(
            decision.native_route,
            NativeRoute::DecisionRunner {
                entrypoint: root_path.join("toolkits/l3ds/decision-models/not-installed.py"),
                model: "org/model-name".into(),
                profile_type: "local".into(),
                tier: Some("smol".into()),
            }
        );
        assert_eq!(decision.profile.lane, Lane::Decisions);
        assert_eq!(decision.runtime.lane, Lane::Decisions);
        assert_eq!(decision.runtime.as_str(), "nagi");
        assert_eq!(decision.readiness, Readiness::ManifestPresent);
    }

    #[test]
    fn hosted_decision_type_and_tier_are_preserved_verbatim() {
        let root = fixture_root();
        let decision_path = root.path().join("toolkits/l3ds/decision-runtimes.json");
        let decision_manifest = fs::read_to_string(&decision_path)
            .expect("read fixture decision manifest")
            .replace("\"type\": \"local\"", "\"type\": \"api\"")
            .replace("\"tier\": \"smol\"", "\"tier\": \"hosted\"");
        fs::write(&decision_path, decision_manifest).expect("write hosted fixture manifest");

        let catalog = Catalog::load(root.path()).expect("load fixture catalog");
        let route = resolve_profile_route(&catalog, &key(Lane::Decisions, "shared-id"))
            .expect("resolve hosted profile");
        assert!(matches!(
            route.native_route,
            NativeRoute::DecisionRunner {
                profile_type,
                tier: Some(tier),
                ..
            } if profile_type == "api" && tier == "hosted"
        ));
        assert_eq!(route.readiness, Readiness::ManifestPresent);
    }

    #[test]
    fn rejects_escaping_and_unbounded_manifest_paths() {
        let too_long = "a".repeat(MAX_NATIVE_PATH_BYTES + 1);
        for (raw_path, expected_error) in [
            ("../escape.sh", "invalid component"),
            ("/absolute/script.sh", "invalid component"),
            ("media-models\\escape.sh", "forbidden character"),
            (too_long.as_str(), "1–512 bytes"),
        ] {
            let error = resolve_manifest_path(
                Path::new("/toolkit/media-runtimes.json"),
                raw_path,
                "media script",
            )
            .expect_err("unsafe route path must fail");
            assert!(error.to_string().contains(expected_error), "{error}");
        }
    }

    #[test]
    fn resolver_rejects_unknown_and_disabled_profiles() {
        let root = fixture_root();
        let catalog = Catalog::load(root.path()).expect("load fixture catalog");

        let disabled = key(Lane::Language, "disabled-model");
        let error = resolve_profile_route(&catalog, &disabled)
            .expect_err("disabled profile must not produce a route");
        assert!(error.to_string().contains("disabled"));

        let unknown = key(Lane::Media, "not-in-catalog");
        let error = resolve_profile_route(&catalog, &unknown)
            .expect_err("unknown profile must not produce a route");
        assert!(error.to_string().contains("unknown profile"));
    }
}
