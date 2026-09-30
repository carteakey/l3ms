//! Dispatch-time verification of manifest-declared native entrypoints.
//!
//! Verification rereads the current manifests through the configured roots,
//! resolves the lane-qualified profile route, and canonicalizes the declared
//! media script or decision runner. It verifies only path containment and
//! basic file properties; it does not claim installation, readiness, runtime
//! health, or GPU availability, and it never executes a file. This check is
//! not a security sandbox or a defense against time-of-check/time-of-use races.
//! A future opener/runner must operate in trusted toolkit directories and
//! retain appropriate filesystem protections while opening the file. A changed
//! runtime identity requires the executor to rebind and reconcile its resources;
//! verification fails closed rather than reusing a stale reservation.

use std::fs;
use std::path::PathBuf;

use anyhow::{ensure, Context, Result};

use super::catalog::{Catalog, Lane, ProfileKey};
use super::routes::{resolve_profile_route, NativeRoute};

/// Kind of manifest-owned executable/script location that was checked.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeEntrypointKind {
    MediaScript,
    DecisionRunner,
}

/// Canonical entrypoint and canonical root that owns it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VerifiedNativePath {
    pub profile: ProfileKey,
    pub kind: NativeEntrypointKind,
    pub entrypoint: PathBuf,
    pub toolkit_root: PathBuf,
}

/// Re-read manifests and verify that the current native media/decision
/// entrypoint resolves to a non-empty regular file inside its owning toolkit
/// root. Internal symlinks are allowed because the canonical target remains
/// within that root; symlinks escaping the root are rejected.
///
/// This operation performs local filesystem reads only. Call it immediately
/// before dispatch, then have the eventual runner open the returned path from
/// a trusted toolkit directory; the returned path is not a sandbox or a
/// guarantee that the file remains unchanged after this call.
pub fn verify_native_entrypoint(catalog: &Catalog, key: &ProfileKey) -> Result<VerifiedNativePath> {
    let cached_profile = catalog
        .inspect(key)
        .with_context(|| format!("unknown profile {key}"))?;
    let cached_runtime = cached_profile.runtime.clone();

    let refreshed = catalog
        .refresh()
        .context("reload native manifests for dispatch-time route verification")?;
    let profile = refreshed
        .inspect(key)
        .with_context(|| format!("current manifest no longer declares {key}"))?;
    ensure!(
        profile.runtime == cached_runtime,
        "runtime identity for {key} changed; executor must rebind and reconcile resources"
    );
    let route = resolve_profile_route(&refreshed, key)
        .with_context(|| format!("revalidate current native route for {key}"))?;

    let (kind, declared_path) = match route.native_route {
        NativeRoute::MediaScript { path } if key.lane == Lane::Media => {
            (NativeEntrypointKind::MediaScript, path)
        }
        NativeRoute::DecisionRunner { entrypoint, .. } if key.lane == Lane::Decisions => {
            (NativeEntrypointKind::DecisionRunner, entrypoint)
        }
        NativeRoute::LlamaSwapModel { .. } => {
            anyhow::bail!("language profile {key} has no local native entrypoint")
        }
        _ => anyhow::bail!("profile {key} has no matching native entrypoint"),
    };

    let manifest_root = profile
        .source_path
        .parent()
        .with_context(|| format!("manifest path has no parent for {key}"))?;
    let toolkit_root = fs::canonicalize(manifest_root)
        .with_context(|| format!("canonicalize toolkit root for {key}"))?;
    ensure!(
        toolkit_root.is_dir(),
        "toolkit root for {key} is not a directory"
    );

    let entrypoint = fs::canonicalize(&declared_path)
        .with_context(|| format!("resolve declared native entrypoint for {key}"))?;
    ensure!(
        entrypoint.starts_with(&toolkit_root),
        "native entrypoint for {key} escapes its toolkit root"
    );
    let metadata = fs::metadata(&entrypoint)
        .with_context(|| format!("inspect declared native entrypoint for {key}"))?;
    ensure!(
        metadata.is_file(),
        "native entrypoint for {key} is not a regular file"
    );
    ensure!(metadata.len() > 0, "native entrypoint for {key} is empty");

    Ok(VerifiedNativePath {
        profile: route.profile,
        kind,
        entrypoint,
        toolkit_root,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    use tempfile::TempDir;

    struct Fixture {
        _parent: TempDir,
        language_root: PathBuf,
        media_root: PathBuf,
        decisions_root: PathBuf,
    }

    impl Fixture {
        fn new(media_script: &str, decision_entrypoint: &str) -> Self {
            let parent = tempfile::tempdir().expect("create fixture parent");
            let language_root = parent.path().join("configured-language");
            let media_root = parent.path().join("external-media-toolkit");
            let decisions_root = parent.path().join("external-decisions-toolkit");
            fs::create_dir_all(&language_root).expect("create language root");
            fs::create_dir_all(&media_root).expect("create media root");
            fs::create_dir_all(&decisions_root).expect("create decisions root");

            fs::write(
                language_root.join("llama-swap.yaml"),
                "models:\n  fixture-model:\n    cmd: model\n",
            )
            .expect("write language manifest");
            fs::write(
                media_root.join("media-runtimes.json"),
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "fixture-media",
                        "name": "Fixture media",
                        "runtime": "fixture-media-runtime",
                        "tasks": ["tts"],
                        "script": media_script,
                    }]
                })
                .to_string(),
            )
            .expect("write media manifest");
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                serde_json::json!({
                    "version": "1.0",
                    "profiles": [{
                        "id": "fixture-decision",
                        "name": "Fixture decision",
                        "engine": "fixture-decision-runtime",
                        "type": "local",
                        "tier": "test",
                        "model": "fixture/model",
                        "entrypoint": decision_entrypoint,
                    }]
                })
                .to_string(),
            )
            .expect("write decisions manifest");

            Self {
                _parent: parent,
                language_root,
                media_root,
                decisions_root,
            }
        }

        fn catalog(&self) -> Catalog {
            Catalog::load_from_roots(&self.language_root, &self.media_root, &self.decisions_root)
                .expect("load configured fixture catalogs")
        }
    }

    fn key(lane: Lane) -> ProfileKey {
        ProfileKey::new(
            lane,
            match lane {
                Lane::Language => "fixture-model",
                Lane::Media => "fixture-media",
                Lane::Decisions => "fixture-decision",
            },
        )
        .expect("valid fixture profile key")
    }

    fn write_nonempty(path: &Path) {
        fs::create_dir_all(path.parent().expect("fixture entrypoint has parent"))
            .expect("create entrypoint parent");
        fs::write(path, "fixture entrypoint\n").expect("write fixture entrypoint");
    }

    #[test]
    fn verifies_regular_media_and_decision_files_under_configured_alternate_roots() {
        let fixture = Fixture::new("native/media.sh", "native/decision.py");
        let media_path = fixture.media_root.join("native/media.sh");
        let decision_path = fixture.decisions_root.join("native/decision.py");
        write_nonempty(&media_path);
        write_nonempty(&decision_path);
        let catalog = fixture.catalog();

        let media = verify_native_entrypoint(&catalog, &key(Lane::Media)).unwrap();
        assert_eq!(media.kind, NativeEntrypointKind::MediaScript);
        assert_eq!(media.entrypoint, fs::canonicalize(media_path).unwrap());
        assert_eq!(
            media.toolkit_root,
            fs::canonicalize(&fixture.media_root).unwrap()
        );

        let decision = verify_native_entrypoint(&catalog, &key(Lane::Decisions)).unwrap();
        assert_eq!(decision.kind, NativeEntrypointKind::DecisionRunner);
        assert_eq!(
            decision.entrypoint,
            fs::canonicalize(decision_path).unwrap()
        );
        assert_eq!(
            decision.toolkit_root,
            fs::canonicalize(&fixture.decisions_root).unwrap()
        );
    }

    #[test]
    fn revalidates_the_current_manifest_route_instead_of_using_a_stale_catalog_path() {
        let fixture = Fixture::new("old/runner.sh", "native/decision.py");
        write_nonempty(&fixture.media_root.join("old/runner.sh"));
        write_nonempty(&fixture.media_root.join("new/runner.sh"));
        let catalog = fixture.catalog();
        let manifest = fixture.media_root.join("media-runtimes.json");
        let updated = serde_json::json!({
            "schema_version": 1,
            "profiles": [{
                "id": "fixture-media",
                "name": "Fixture media",
                "runtime": "fixture-media-runtime",
                "tasks": ["tts"],
                "script": "new/runner.sh",
            }]
        });
        fs::write(manifest, updated.to_string()).expect("update current route");

        let verified = verify_native_entrypoint(&catalog, &key(Lane::Media)).unwrap();
        assert_eq!(
            verified.entrypoint,
            fs::canonicalize(fixture.media_root.join("new/runner.sh")).unwrap()
        );
    }

    #[test]
    fn rejects_runtime_identity_changes_that_invalidate_the_existing_reservation() {
        let fixture = Fixture::new("native/runner.sh", "native/decision.py");
        write_nonempty(&fixture.media_root.join("native/runner.sh"));
        let catalog = fixture.catalog();
        let manifest = fixture.media_root.join("media-runtimes.json");
        let mut current: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
        current["profiles"][0]["runtime"] = serde_json::json!("replacement-runtime");
        fs::write(manifest, current.to_string()).expect("change native runtime identity");

        let error = verify_native_entrypoint(&catalog, &key(Lane::Media))
            .expect_err("changed runtime must force reservation rebind");
        assert!(error
            .to_string()
            .contains("must rebind and reconcile resources"));
    }

    #[cfg(unix)]
    #[test]
    fn allows_internal_symlinks_and_rejects_symlinks_that_escape_the_toolkit_root() {
        use std::os::unix::fs::symlink;

        let fixture = Fixture::new("native/alias.sh", "native/decision.py");
        let actual = fixture.media_root.join("native/actual.sh");
        let alias = fixture.media_root.join("native/alias.sh");
        write_nonempty(&actual);
        symlink(&actual, &alias).expect("create internal symlink");
        let catalog = fixture.catalog();
        let verified = verify_native_entrypoint(&catalog, &key(Lane::Media)).unwrap();
        assert_eq!(verified.entrypoint, fs::canonicalize(actual).unwrap());

        let outside = tempfile::tempdir().expect("create outside directory");
        let outside_target = outside.path().join("outside.sh");
        write_nonempty(&outside_target);
        let escaped = fixture.media_root.join("native/escaped.sh");
        symlink(outside_target, &escaped).expect("create external symlink");
        let manifest = fixture.media_root.join("media-runtimes.json");
        fs::write(
            manifest,
            serde_json::json!({
                "schema_version": 1,
                "profiles": [{
                    "id": "fixture-media",
                    "name": "Fixture media",
                    "runtime": "fixture-media-runtime",
                    "tasks": ["tts"],
                    "script": "native/escaped.sh",
                }]
            })
            .to_string(),
        )
        .expect("update route to escaped symlink");
        let error = verify_native_entrypoint(&catalog, &key(Lane::Media))
            .expect_err("outside symlink target must fail closed");
        assert!(error.to_string().contains("escapes its toolkit root"));
    }

    #[test]
    fn rejects_missing_directory_and_empty_entrypoints() {
        let missing = Fixture::new("bin/missing.sh", "native/decision.py");
        let catalog = missing.catalog();
        assert!(verify_native_entrypoint(&catalog, &key(Lane::Media)).is_err());

        let directory = Fixture::new("bin/runner-dir", "native/decision.py");
        fs::create_dir_all(directory.media_root.join("bin/runner-dir"))
            .expect("create directory entrypoint");
        let catalog = directory.catalog();
        let error = verify_native_entrypoint(&catalog, &key(Lane::Media))
            .expect_err("directory must not be treated as an entrypoint");
        assert!(error.to_string().contains("not a regular file"));

        let empty = Fixture::new("bin/empty.sh", "native/decision.py");
        let empty_path = empty.media_root.join("bin/empty.sh");
        fs::create_dir_all(empty_path.parent().unwrap()).unwrap();
        fs::write(&empty_path, "").unwrap();
        let catalog = empty.catalog();
        let error = verify_native_entrypoint(&catalog, &key(Lane::Media))
            .expect_err("empty file must not be treated as an entrypoint");
        assert!(error.to_string().contains("is empty"));
    }

    #[test]
    fn language_routes_are_not_local_native_entrypoints() {
        let fixture = Fixture::new("native/media.sh", "native/decision.py");
        let catalog = fixture.catalog();
        let error = verify_native_entrypoint(&catalog, &key(Lane::Language))
            .expect_err("llama-swap profiles do not declare local entrypoint files");
        assert!(error.to_string().contains("no local native entrypoint"));
    }
}
