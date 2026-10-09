//! Opt-in, immutable ComfyUI launch declaration for one managed GPU session.
//!
//! This module accepts only the repository's pinned ComfyUI/Python layout,
//! the native Qwen Image 2.1 and Krea 2 workflow profiles, and one explicit
//! physical CUDA UUID. It never installs, downloads, probes, or starts an
//! engine. A process session allocates a new private user/output directory for
//! each history job; collisions are refused and old directories are never
//! reused or deleted here.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use serde::de::{Error as _, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

use super::catalog::{Catalog, Device, Lane, ProfileKey, RuntimeIdentity};
use super::cuda_device::CudaDevice;
use super::gpu_probe::GpuProbeConfig;
use super::managed_service::{LaunchRecipe, VerifiedPythonInterpreter};
use super::routes::{resolve_profile_route, NativeRoute, ResolvedProfileRoute};
use super::service_owner::IsolatedEngineLaunch;

const MAX_PROFILES: usize = 16;
const MAX_SCRIPT_BYTES: u64 = 2 * 1024 * 1024;
const MAX_WORKFLOW_BYTES: u64 = 4 * 1024 * 1024;
const MAX_EXTRA_CONFIG_BYTES: u64 = 64 * 1024;
const MAX_WORKFLOW_NODES: usize = 4096;
const PRIVATE_MODE: u32 = 0o700;
const CONFIG_FILE_MODE: u32 = 0o600;
const RUNTIME_NAME: &str = "ComfyUI";

/// Strict opt-in declaration stored inside the private managed-session JSON.
/// Absence means no managed Comfy admission or launch plan.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManagedComfyLaunchInput {
    pub(crate) profiles: Vec<String>,
    pub(crate) python_executable: PathBuf,
    pub(crate) comfy_checkout: PathBuf,
    #[serde(default)]
    pub(crate) extra_model_paths_config: Option<PathBuf>,
    pub(crate) private_user_directory_root: PathBuf,
    pub(crate) private_output_directory_root: PathBuf,
    pub(crate) port: u16,
}

/// Fixed, redacted configuration failures; no supplied path or file content
/// is retained in errors or debug output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ManagedComfyConfigError {
    InvalidDeclaration,
    UnsupportedWorkflowProfile,
    InvalidNativeRoute,
    NativeInputUnavailable,
    NativeInputChanged,
    InvalidExtraModelPathsConfig,
    InvalidPrivateDirectory,
    ProtectedDirectoryOverlap,
    JobDirectoryAlreadyAllocated,
    InvalidLaunch,
}

impl fmt::Display for ManagedComfyConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDeclaration => "managed Comfy declaration is invalid",
            Self::UnsupportedWorkflowProfile => "managed Comfy workflow profile is not allowlisted",
            Self::InvalidNativeRoute => "managed Comfy native profile route is invalid",
            Self::NativeInputUnavailable => "managed Comfy native launch input is unavailable",
            Self::NativeInputChanged => "managed Comfy native launch input changed",
            Self::InvalidExtraModelPathsConfig => {
                "managed Comfy model-path configuration is invalid"
            }
            Self::InvalidPrivateDirectory => "managed Comfy private directory is invalid",
            Self::ProtectedDirectoryOverlap => {
                "managed Comfy output directories overlap a protected path"
            }
            Self::JobDirectoryAlreadyAllocated => {
                "managed Comfy job output directory already exists"
            }
            Self::InvalidLaunch => "managed Comfy launch plan is invalid",
        })
    }
}

impl Error for ManagedComfyConfigError {}

#[derive(Clone, Copy)]
struct WorkflowProfileSpec {
    id: &'static str,
    script: &'static str,
    workflows: &'static [&'static str],
}

const WORKFLOW_PROFILES: &[WorkflowProfileSpec] = &[
    WorkflowProfileSpec {
        id: "qwen-image-2.1",
        script: "media-models/generate-qwen-image-2.1.sh",
        workflows: &[
            "workflows/qwen-image-2.1/image_qwen_image_2_1_t2i.json",
            "workflows/qwen-image-2.1/image_qwen_image_2_1_image_edit.json",
        ],
    },
    WorkflowProfileSpec {
        id: "krea-2",
        script: "media-models/generate-krea-2.sh",
        workflows: &[
            "workflows/krea-2/image_krea2_turbo_t2i_int8.json",
            "workflows/krea-2/image_krea2_turbo_int8_image_style_reference.json",
        ],
    },
];

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    size: u64,
    mode: u32,
    uid: u32,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl FileIdentity {
    fn from_metadata(metadata: &fs::Metadata) -> Self {
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size: metadata.len(),
            mode: metadata.mode() & 0o7777,
            uid: metadata.uid(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
            changed_seconds: metadata.ctime(),
            changed_nanoseconds: metadata.ctime_nsec(),
        }
    }
}

struct PinnedFile {
    configured_path: PathBuf,
    canonical_path: PathBuf,
    identity: FileIdentity,
    bytes: Option<Vec<u8>>,
}

impl PinnedFile {
    fn capture(
        path: &Path,
        max_bytes: Option<u64>,
        require_private: bool,
        require_executable: bool,
    ) -> Result<Self, ManagedComfyConfigError> {
        if !path.is_absolute() {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }
        let canonical_path = path
            .canonicalize()
            .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        if canonical_path.as_path() != path {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }
        let metadata = fs::metadata(&canonical_path)
            .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        let uid = unsafe { libc::geteuid() };
        if !metadata.is_file()
            || metadata.len() == 0
            || (require_private
                && (metadata.uid() != uid
                    || metadata.mode() & 0o7777 != CONFIG_FILE_MODE
                    || metadata.nlink() != 1))
            || (require_executable && metadata.permissions().mode() & 0o111 == 0)
            || max_bytes.is_some_and(|maximum| metadata.len() > maximum)
        {
            return Err(if require_private {
                ManagedComfyConfigError::InvalidExtraModelPathsConfig
            } else {
                ManagedComfyConfigError::NativeInputUnavailable
            });
        }
        let identity = FileIdentity::from_metadata(&metadata);
        let bytes = if let Some(maximum) = max_bytes {
            Some(read_pinned_file(&canonical_path, &identity, maximum)?)
        } else {
            None
        };
        Ok(Self {
            configured_path: path.to_path_buf(),
            canonical_path,
            identity,
            bytes,
        })
    }

    fn verify(&self) -> Result<(), ManagedComfyConfigError> {
        if self.configured_path.canonicalize().ok().as_deref()
            != Some(self.canonical_path.as_path())
        {
            return Err(ManagedComfyConfigError::NativeInputChanged);
        }
        let metadata = fs::metadata(&self.canonical_path)
            .map_err(|_| ManagedComfyConfigError::NativeInputChanged)?;
        if !metadata.is_file() || FileIdentity::from_metadata(&metadata) != self.identity {
            return Err(ManagedComfyConfigError::NativeInputChanged);
        }
        if let Some(expected) = self.bytes.as_ref() {
            let actual =
                read_pinned_file(&self.canonical_path, &self.identity, expected.len() as u64)
                    .map_err(|_| ManagedComfyConfigError::NativeInputChanged)?;
            if &actual != expected {
                return Err(ManagedComfyConfigError::NativeInputChanged);
            }
        }
        Ok(())
    }
}

fn read_pinned_file(
    path: &Path,
    expected: &FileIdentity,
    maximum: u64,
) -> Result<Vec<u8>, ManagedComfyConfigError> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
    let before = file
        .metadata()
        .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
    if !before.is_file() || FileIdentity::from_metadata(&before) != *expected {
        return Err(ManagedComfyConfigError::NativeInputChanged);
    }
    let mut bytes = Vec::new();
    (&file)
        .take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
    let after = file
        .metadata()
        .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
    if bytes.len() as u64 > maximum || FileIdentity::from_metadata(&after) != *expected {
        return Err(ManagedComfyConfigError::NativeInputChanged);
    }
    Ok(bytes)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DirectoryIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    uid: u32,
}

struct PinnedDirectory {
    configured_path: PathBuf,
    canonical_path: PathBuf,
    identity: DirectoryIdentity,
    private: bool,
}

impl PinnedDirectory {
    fn capture(path: &Path, private: bool) -> Result<Self, ManagedComfyConfigError> {
        if !path.is_absolute() {
            return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
        }
        let canonical_path = path
            .canonicalize()
            .map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
        if canonical_path.as_path() != path {
            return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
        }
        let link_metadata = fs::symlink_metadata(path)
            .map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
        let metadata = fs::metadata(&canonical_path)
            .map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
        let uid = unsafe { libc::geteuid() };
        if !link_metadata.is_dir()
            || link_metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || (private && (metadata.uid() != uid || metadata.mode() & 0o7777 != PRIVATE_MODE))
        {
            return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
        }
        Ok(Self {
            configured_path: path.to_path_buf(),
            canonical_path: canonical_path.clone(),
            identity: DirectoryIdentity {
                device: metadata.dev(),
                inode: metadata.ino(),
                mode: metadata.mode() & 0o7777,
                uid: metadata.uid(),
            },
            private,
        })
    }

    fn verify(&self) -> Result<(), ManagedComfyConfigError> {
        if self.configured_path.canonicalize().ok().as_deref()
            != Some(self.canonical_path.as_path())
        {
            return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
        }
        let link_metadata = fs::symlink_metadata(&self.configured_path)
            .map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
        let metadata = fs::metadata(&self.canonical_path)
            .map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
        let identity = DirectoryIdentity {
            device: metadata.dev(),
            inode: metadata.ino(),
            mode: metadata.mode() & 0o7777,
            uid: metadata.uid(),
        };
        if !link_metadata.is_dir()
            || link_metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || identity != self.identity
            || (self.private && metadata.mode() & 0o7777 != PRIVATE_MODE)
        {
            return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
        }
        Ok(())
    }
}

/// Validated launch declaration. The program, script, workflow templates,
/// optional model-path config, protected roots, and output roots are pinned
/// and rechecked before each job launch.
pub(crate) struct ManagedComfyConfig {
    catalog: Catalog,
    runtime: RuntimeIdentity,
    profiles: BTreeSet<ProfileKey>,
    python: VerifiedPythonInterpreter,
    python_target: PinnedFile,
    main_script: PinnedFile,
    native_scripts: Vec<PinnedFile>,
    workflows: Vec<PinnedFile>,
    extra_model_paths_config: Option<PinnedFile>,
    model_roots: Vec<PinnedDirectory>,
    checkout: PinnedDirectory,
    user_root: PinnedDirectory,
    output_root: PinnedDirectory,
    transport: PinnedDirectory,
    helper_executable: PinnedFile,
    port: u16,
    cuda_device: CudaDevice,
    stop_timeout: Duration,
    protected_roots: Vec<PinnedDirectory>,
    protected_path_declarations: Vec<PathBuf>,
    protected_path_names: Vec<PathBuf>,
}

impl ManagedComfyConfig {
    pub(crate) fn validate(
        input: ManagedComfyLaunchInput,
        catalog: &Catalog,
        gpu: &GpuProbeConfig,
        helper_executable: PathBuf,
        transport_directory: PathBuf,
        stop_timeout: Duration,
    ) -> Result<Self, ManagedComfyConfigError> {
        if input.profiles.is_empty()
            || input.profiles.len() > MAX_PROFILES
            || input.port == 0
            || stop_timeout.is_zero()
            || stop_timeout > Duration::from_secs(300)
            || !helper_executable.is_absolute()
        {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }

        let mut profiles = BTreeSet::new();
        for profile in &input.profiles {
            let key = profile
                .parse::<ProfileKey>()
                .map_err(|_| ManagedComfyConfigError::InvalidDeclaration)?;
            if key.lane != Lane::Media || !profiles.insert(key) {
                return Err(ManagedComfyConfigError::InvalidDeclaration);
            }
        }

        let refreshed = catalog
            .refresh()
            .map_err(|_| ManagedComfyConfigError::InvalidNativeRoute)?;
        let mut toolkit_root = None;
        let mut workflow_paths = Vec::new();
        let mut native_script_paths = Vec::new();
        for profile in &profiles {
            let (route, profile_toolkit_root) = validate_comfy_profile(&refreshed, profile)?;
            if toolkit_root
                .as_ref()
                .is_some_and(|root| root != &profile_toolkit_root)
            {
                return Err(ManagedComfyConfigError::InvalidNativeRoute);
            }
            toolkit_root = Some(profile_toolkit_root.clone());
            let spec = workflow_profile_spec(profile)?;
            native_script_paths.push(profile_toolkit_root.join(spec.script));
            workflow_paths.extend(
                spec.workflows
                    .iter()
                    .map(|path| profile_toolkit_root.join(path)),
            );
            let NativeRoute::MediaScript { path } = route.native_route else {
                return Err(ManagedComfyConfigError::InvalidNativeRoute);
            };
            if path.as_path()
                != native_script_paths
                    .last()
                    .expect("script path just pushed")
                    .as_path()
            {
                return Err(ManagedComfyConfigError::InvalidNativeRoute);
            }
        }
        let toolkit_root = toolkit_root.ok_or(ManagedComfyConfigError::InvalidNativeRoute)?;
        let expected_checkout = toolkit_root.join("vendor/ComfyUI");
        let checkout = PinnedDirectory::capture(&input.comfy_checkout, false)?;
        if checkout.canonical_path
            != expected_checkout
                .canonicalize()
                .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?
        {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }
        let expected_python = toolkit_root.join(".venv-comfy/bin/python");
        if input.python_executable != expected_python {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }
        let python = VerifiedPythonInterpreter::capture(
            &input.python_executable,
            toolkit_root.join(".venv-comfy"),
        )
        .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        let python_target = PinnedFile::capture(python.target_path(), None, false, true)?;
        let main_script_path = checkout.canonical_path.join("main.py");
        let main_script =
            PinnedFile::capture(&main_script_path, Some(MAX_SCRIPT_BYTES), false, false)?;
        let helper_executable = PinnedFile::capture(&helper_executable, None, false, true)?;
        let native_scripts = native_script_paths
            .iter()
            .map(|path| PinnedFile::capture(path, Some(MAX_SCRIPT_BYTES), false, false))
            .collect::<Result<Vec<_>, _>>()?;

        let workflows = workflow_paths
            .iter()
            .map(|path| {
                let pinned = PinnedFile::capture(path, Some(MAX_WORKFLOW_BYTES), false, false)?;
                let bytes = pinned
                    .bytes
                    .as_ref()
                    .ok_or(ManagedComfyConfigError::NativeInputUnavailable)?;
                validate_native_workflow_template(bytes)?;
                Ok(pinned)
            })
            .collect::<Result<Vec<_>, ManagedComfyConfigError>>()?;

        let (extra_model_paths_config, mut model_roots) = match input.extra_model_paths_config {
            Some(path) => {
                let pinned = PinnedFile::capture(&path, Some(MAX_EXTRA_CONFIG_BYTES), true, false)?;
                let bytes = pinned
                    .bytes
                    .as_ref()
                    .ok_or(ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
                let roots = validate_extra_model_paths_config(bytes)?;
                (Some(pinned), roots)
            }
            None => (None, Vec::new()),
        };

        let default_model_root = checkout.canonical_path.join("models");
        model_roots.push(default_model_root.clone());
        let model_roots = model_roots
            .iter()
            .map(|path| PinnedDirectory::capture(path, false))
            .collect::<Result<Vec<_>, _>>()?;
        let user_root = PinnedDirectory::capture(&input.private_user_directory_root, true)?;
        let output_root = PinnedDirectory::capture(&input.private_output_directory_root, true)?;
        let transport = PinnedDirectory::capture(&transport_directory, false)?;
        let cuda_device = CudaDevice::new(gpu.gpu_uuid())
            .map_err(|_| ManagedComfyConfigError::InvalidDeclaration)?;

        let protected_path_declarations = vec![
            toolkit_root.join(".toolkit/comfyui/user"),
            toolkit_root.join("vendor/ComfyUI/user"),
            default_output_directory(),
        ];
        let protected_path_names: Vec<PathBuf> = protected_path_declarations
            .iter()
            .map(|path| normalize_protected_path(path))
            .collect();
        let config = Self {
            catalog: catalog.clone(),
            runtime: RuntimeIdentity::new(Lane::Media, RUNTIME_NAME)
                .map_err(|_| ManagedComfyConfigError::InvalidDeclaration)?,
            profiles,
            python,
            python_target,
            main_script,
            native_scripts,
            workflows,
            extra_model_paths_config,
            model_roots,
            checkout,
            user_root,
            output_root,
            transport,
            helper_executable,
            port: input.port,
            cuda_device,
            stop_timeout,
            protected_roots: Vec::new(),
            protected_path_declarations,
            protected_path_names,
        };
        config.verify_roots_disjoint()?;
        Ok(config)
    }

    pub(crate) fn profiles(&self) -> &BTreeSet<ProfileKey> {
        &self.profiles
    }

    /// Bind the output roots away from history, WAL, local-job, and other
    /// caller-owned private roots before the configuration enters a session.
    pub(crate) fn validate_protected_directories(
        &mut self,
        roots: &[PathBuf],
    ) -> Result<(), ManagedComfyConfigError> {
        let mut pinned = Vec::with_capacity(roots.len());
        for root in roots {
            let directory = PinnedDirectory::capture(root, false)?;
            if paths_overlap(&directory.canonical_path, &self.user_root.canonical_path)
                || paths_overlap(&directory.canonical_path, &self.output_root.canonical_path)
            {
                return Err(ManagedComfyConfigError::ProtectedDirectoryOverlap);
            }
            pinned.push(directory);
        }
        self.protected_roots = pinned;
        self.verify_roots_disjoint()
    }

    /// Allocate fresh mode-0700 Comfy user and output roots for one history
    /// job, then build the fixed native launch recipe. A stale directory with
    /// the same history ID makes this fail closed; existing output is never
    /// reused or removed.
    pub(crate) fn allocate_job_launch(
        &self,
        job_id: u64,
    ) -> Result<(IsolatedEngineLaunch, PathBuf), ManagedComfyConfigError> {
        if job_id == 0 {
            return Err(ManagedComfyConfigError::InvalidDeclaration);
        }
        self.revalidate()?;
        let user_directory = self.user_root.canonical_path.join(format!("job-{job_id}"));
        let output_directory = self
            .output_root
            .canonical_path
            .join(format!("job-{job_id}"));
        create_private_job_directory(&user_directory)?;
        create_private_job_directory(&output_directory)?;
        self.revalidate()?;
        let user_pin = PinnedDirectory::capture(&user_directory, true)?;
        let output_pin = PinnedDirectory::capture(&output_directory, true)?;
        if paths_overlap(&user_pin.canonical_path, &output_pin.canonical_path)
            || self.is_protected(&user_pin.canonical_path)
            || self.is_protected(&output_pin.canonical_path)
        {
            return Err(ManagedComfyConfigError::ProtectedDirectoryOverlap);
        }

        let mut arguments = vec![
            self.main_script.canonical_path.as_os_str().to_owned(),
            "--listen".into(),
            "127.0.0.1".into(),
            "--port".into(),
            self.port.to_string().into(),
            "--disable-auto-launch".into(),
            "--disable-api-nodes".into(),
            "--user-directory".into(),
            user_pin.canonical_path.as_os_str().to_owned(),
            "--output-directory".into(),
            output_pin.canonical_path.as_os_str().to_owned(),
            "--lowvram".into(),
            "--reserve-vram".into(),
            "1.0".into(),
        ];
        if let Some(config) = self.extra_model_paths_config.as_ref() {
            arguments.push("--extra-model-paths-config".into());
            arguments.push(config.canonical_path.as_os_str().to_owned());
        }
        let recipe = LaunchRecipe::new_with_verified_python(
            &self.python,
            &self.checkout.canonical_path,
            arguments,
        )
        .map_err(|_| ManagedComfyConfigError::InvalidLaunch)?;
        let launch = IsolatedEngineLaunch::new(
            self.runtime.clone(),
            recipe,
            &self.helper_executable.canonical_path,
            &self.transport.canonical_path,
            self.port,
            self.stop_timeout,
        )
        .map_err(|_| ManagedComfyConfigError::InvalidLaunch)?
        .with_cuda_device(self.cuda_device.clone());
        Ok((launch, output_pin.canonical_path))
    }

    fn revalidate(&self) -> Result<(), ManagedComfyConfigError> {
        self.python
            .verify()
            .map_err(|_| ManagedComfyConfigError::NativeInputChanged)?;
        self.python_target.verify()?;
        if self.python.target_path() != self.python_target.canonical_path.as_path() {
            return Err(ManagedComfyConfigError::NativeInputChanged);
        }
        self.main_script.verify()?;
        self.helper_executable.verify()?;
        for script in &self.native_scripts {
            script.verify()?;
        }
        for workflow in &self.workflows {
            workflow.verify()?;
            validate_native_workflow_template(
                workflow
                    .bytes
                    .as_deref()
                    .ok_or(ManagedComfyConfigError::NativeInputChanged)?,
            )?;
        }
        if let Some(config) = self.extra_model_paths_config.as_ref() {
            config.verify()?;
        }
        for model_root in &self.model_roots {
            model_root.verify()?;
        }
        self.checkout.verify()?;
        self.user_root.verify()?;
        self.output_root.verify()?;
        self.transport.verify()?;
        for root in &self.protected_roots {
            root.verify()?;
        }
        if self
            .protected_path_declarations
            .iter()
            .map(|path| normalize_protected_path(path))
            .ne(self.protected_path_names.iter().cloned())
        {
            return Err(ManagedComfyConfigError::ProtectedDirectoryOverlap);
        }
        self.verify_roots_disjoint()?;
        let refreshed = self
            .catalog
            .refresh()
            .map_err(|_| ManagedComfyConfigError::InvalidNativeRoute)?;
        for profile in &self.profiles {
            validate_comfy_profile(&refreshed, profile)?;
        }
        Ok(())
    }

    fn verify_roots_disjoint(&self) -> Result<(), ManagedComfyConfigError> {
        for private in [
            &self.user_root.canonical_path,
            &self.output_root.canonical_path,
        ] {
            if paths_overlap(private, &self.checkout.canonical_path)
                || self
                    .model_roots
                    .iter()
                    .any(|model| paths_overlap(private, &model.canonical_path))
                || paths_overlap(private, &self.transport.canonical_path)
                || self
                    .protected_roots
                    .iter()
                    .any(|root| paths_overlap(private, &root.canonical_path))
                || self
                    .protected_path_names
                    .iter()
                    .any(|path| paths_overlap(private, path))
            {
                return Err(ManagedComfyConfigError::ProtectedDirectoryOverlap);
            }
        }
        if paths_overlap(
            &self.user_root.canonical_path,
            &self.output_root.canonical_path,
        ) {
            return Err(ManagedComfyConfigError::ProtectedDirectoryOverlap);
        }
        Ok(())
    }

    fn is_protected(&self, path: &Path) -> bool {
        paths_overlap(path, &self.checkout.canonical_path)
            || self
                .model_roots
                .iter()
                .any(|model| paths_overlap(path, &model.canonical_path))
            || paths_overlap(path, &self.transport.canonical_path)
            || self
                .protected_roots
                .iter()
                .any(|root| paths_overlap(path, &root.canonical_path))
            || self
                .protected_path_names
                .iter()
                .any(|protected| paths_overlap(path, protected))
    }
}

impl fmt::Debug for ManagedComfyConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ManagedComfyConfig([redacted])")
    }
}

/// Validate one profile against the current native manifest and its pinned
/// workflow-template family. The media script is identity evidence only; this
/// function never prepares or executes its script route.
pub(crate) fn validate_comfy_profile(
    catalog: &Catalog,
    profile: &ProfileKey,
) -> Result<(ResolvedProfileRoute, PathBuf), ManagedComfyConfigError> {
    let spec = workflow_profile_spec(profile)?;
    let route = resolve_profile_route(catalog, profile)
        .map_err(|_| ManagedComfyConfigError::InvalidNativeRoute)?;
    if profile.lane != Lane::Media
        || route.runtime.lane != Lane::Media
        || route.runtime.as_str() != RUNTIME_NAME
        || route.device != Some(Device::Cuda)
        || route.tasks.len() != 1
        || route.tasks[0].as_str() != "image"
    {
        return Err(ManagedComfyConfigError::InvalidNativeRoute);
    }
    let source = catalog
        .inspect(profile)
        .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?
        .source_path
        .clone();
    let toolkit_root = source
        .parent()
        .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?
        .canonicalize()
        .map_err(|_| ManagedComfyConfigError::InvalidNativeRoute)?;
    let expected_script = toolkit_root.join(spec.script);
    match &route.native_route {
        NativeRoute::MediaScript { path }
            if path == &expected_script
                && path.canonicalize().ok().is_some_and(|resolved| {
                    resolved.starts_with(&toolkit_root)
                        && fs::metadata(&resolved)
                            .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
                }) => {}
        _ => return Err(ManagedComfyConfigError::InvalidNativeRoute),
    }

    for workflow in spec.workflows {
        let path = toolkit_root.join(workflow);
        let canonical = path
            .canonicalize()
            .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        if !canonical.starts_with(&toolkit_root) {
            return Err(ManagedComfyConfigError::InvalidNativeRoute);
        }
        let metadata = fs::metadata(&canonical)
            .map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > MAX_WORKFLOW_BYTES {
            return Err(ManagedComfyConfigError::NativeInputUnavailable);
        }
        let bytes =
            fs::read(canonical).map_err(|_| ManagedComfyConfigError::NativeInputUnavailable)?;
        validate_native_workflow_template(&bytes)?;
    }
    Ok((route, toolkit_root))
}

/// Validate the Comfy native API prompt graph shape accepted by managed
/// admission. Artifact readiness and native execution validation remain the
/// adapter's responsibility.
pub(crate) fn validate_api_workflow_graph(graph: &Value) -> Result<(), ManagedComfyConfigError> {
    let nodes = graph
        .as_object()
        .filter(|nodes| !nodes.is_empty() && nodes.len() <= MAX_WORKFLOW_NODES)
        .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?;
    for (node_id, node) in nodes {
        if node_id.parse::<u64>().ok().filter(|id| *id > 0).is_none() {
            return Err(ManagedComfyConfigError::InvalidNativeRoute);
        }
        let node = node
            .as_object()
            .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?;
        if node
            .keys()
            .any(|key| !matches!(key.as_str(), "class_type" | "inputs"))
            || node
                .get("class_type")
                .and_then(Value::as_str)
                .is_none_or(|kind| {
                    kind.is_empty()
                        || kind.len() > 128
                        || !kind
                            .bytes()
                            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
                })
            || !node.get("inputs").is_some_and(Value::is_object)
        {
            return Err(ManagedComfyConfigError::InvalidNativeRoute);
        }
    }
    Ok(())
}

fn workflow_profile_spec(
    profile: &ProfileKey,
) -> Result<&'static WorkflowProfileSpec, ManagedComfyConfigError> {
    if profile.lane != Lane::Media {
        return Err(ManagedComfyConfigError::UnsupportedWorkflowProfile);
    }
    WORKFLOW_PROFILES
        .iter()
        .find(|spec| spec.id == profile.profile_id.as_str())
        .ok_or(ManagedComfyConfigError::UnsupportedWorkflowProfile)
}

struct StrictJsonValue(Value);

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrictValueVisitor;

        impl<'de> Visitor<'de> for StrictValueVisitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON value with unique object keys")
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                serde_json::Number::from_f64(value)
                    .map(Value::Number)
                    .map(StrictJsonValue)
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(value)))
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<StrictJsonValue>()? {
                    values.push(value.0);
                }
                Ok(StrictJsonValue(Value::Array(values)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = serde_json::Map::new();
                while let Some((key, value)) = map.next_entry::<String, StrictJsonValue>()? {
                    if values.insert(key, value.0).is_some() {
                        return Err(A::Error::custom("duplicate object key"));
                    }
                }
                Ok(StrictJsonValue(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(StrictValueVisitor)
    }
}

fn parse_strict_json(bytes: &[u8]) -> Result<Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let value = StrictJsonValue::deserialize(&mut deserializer)?.0;
    deserializer.end()?;
    Ok(value)
}

fn validate_native_workflow_template(bytes: &[u8]) -> Result<(), ManagedComfyConfigError> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_WORKFLOW_BYTES {
        return Err(ManagedComfyConfigError::NativeInputUnavailable);
    }
    let value =
        parse_strict_json(bytes).map_err(|_| ManagedComfyConfigError::InvalidNativeRoute)?;
    let root = value
        .as_object()
        .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?;
    let nodes = root
        .get("nodes")
        .and_then(Value::as_array)
        .filter(|nodes| !nodes.is_empty() && nodes.len() <= MAX_WORKFLOW_NODES)
        .ok_or(ManagedComfyConfigError::InvalidNativeRoute)?;
    if !root.get("links").is_some_and(Value::is_array)
        || nodes.iter().any(|node| {
            node.as_object().is_none_or(|node| {
                node.get("id").and_then(Value::as_u64).is_none()
                    || node
                        .get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| kind.trim().is_empty() || kind.len() > 256)
            })
        })
    {
        return Err(ManagedComfyConfigError::InvalidNativeRoute);
    }
    Ok(())
}

fn validate_extra_model_paths_config(
    bytes: &[u8],
) -> Result<Vec<PathBuf>, ManagedComfyConfigError> {
    if bytes.is_empty() || bytes.len() as u64 > MAX_EXTRA_CONFIG_BYTES {
        return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
    }
    // JSON is a strict YAML subset accepted by ComfyUI's safe YAML loader.
    // Restricting this native config to a known object shape excludes YAML
    // tags, aliases, commands, credentials, and unrecognized secret fields.
    let root = parse_strict_json(bytes)
        .map_err(|_| ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
    let root = root
        .as_object()
        .filter(|root| root.len() == 1 && root.contains_key("comfyui"))
        .ok_or(ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
    let comfy = root["comfyui"]
        .as_object()
        .ok_or(ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
    const ALLOWED_KEYS: &[&str] = &[
        "base_path",
        "is_default",
        "checkpoints",
        "clip",
        "clip_vision",
        "configs",
        "controlnet",
        "diffusion_models",
        "embeddings",
        "loras",
        "model_patches",
        "style_models",
        "text_encoders",
        "unet",
        "upscale_models",
        "vae",
    ];
    if comfy
        .keys()
        .any(|key| !ALLOWED_KEYS.contains(&key.as_str()))
    {
        return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
    }
    if comfy
        .get("is_default")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
    }
    let base = comfy
        .get("base_path")
        .and_then(Value::as_str)
        .filter(|path| {
            Path::new(path).is_absolute()
                && !path.contains(['$', '~'])
                && !path.chars().any(char::is_control)
        })
        .ok_or(ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
    let declared_base = Path::new(base);
    let base = declared_base
        .canonicalize()
        .map_err(|_| ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
    if base.as_path() != declared_base || !base.is_dir() {
        return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
    }
    for (key, value) in comfy {
        if matches!(key.as_str(), "base_path" | "is_default") {
            continue;
        }
        // Native utils/extra_config.py calls .split("\n") on each
        // category; JSON arrays are not an executable native contract.
        let paths = value
            .as_str()
            .ok_or(ManagedComfyConfigError::InvalidExtraModelPathsConfig)?;
        let mut has_path = false;
        for path in paths.split('\n').filter(|path| !path.is_empty()) {
            has_path = true;
            if !safe_model_subpath(path) {
                return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
            }
        }
        if !has_path {
            return Err(ManagedComfyConfigError::InvalidExtraModelPathsConfig);
        }
    }
    Ok(vec![base])
}

fn safe_model_subpath(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 512
        && !Path::new(value).is_absolute()
        && !value.contains(['$', '~'])
        && Path::new(value)
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
        && !value.chars().any(char::is_control)
}

fn default_output_directory() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|home| home.is_absolute())
        .map(|home| home.join("media-output/comfyui"))
        .unwrap_or_else(|| PathBuf::from("/nonexistent-home/media-output/comfyui"))
}

fn normalize_protected_path(path: &Path) -> PathBuf {
    let mut cursor = path.to_path_buf();
    let mut missing = Vec::new();
    loop {
        if let Ok(mut canonical) = cursor.canonicalize() {
            for component in missing.iter().rev() {
                canonical.push(component);
            }
            return canonical;
        }
        let Some(name) = cursor.file_name().map(|name| name.to_os_string()) else {
            return path.to_path_buf();
        };
        missing.push(name);
        if !cursor.pop() {
            return path.to_path_buf();
        }
    }
}

fn paths_overlap(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn create_private_job_directory(path: &Path) -> Result<(), ManagedComfyConfigError> {
    fs::DirBuilder::new()
        .mode(PRIVATE_MODE)
        .create(path)
        .map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                ManagedComfyConfigError::JobDirectoryAlreadyAllocated
            } else {
                ManagedComfyConfigError::InvalidPrivateDirectory
            }
        })?;
    let metadata =
        fs::symlink_metadata(path).map_err(|_| ManagedComfyConfigError::InvalidPrivateDirectory)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.uid() != unsafe { libc::geteuid() }
        || metadata.mode() & 0o7777 != PRIVATE_MODE
    {
        return Err(ManagedComfyConfigError::InvalidPrivateDirectory);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::catalog::Catalog;
    use crate::control::gpu_probe::GpuProbeConfig;
    use std::os::unix::fs::{symlink, PermissionsExt};

    const TEST_GPU_UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";

    struct NativeLaunchFixture {
        _root: tempfile::TempDir,
        catalog: Catalog,
        helper: PathBuf,
        transport: PathBuf,
        gpu: GpuProbeConfig,
        user_root: PathBuf,
        output_root: PathBuf,
        protected_root: PathBuf,
        native_script: PathBuf,
        extra_config: PathBuf,
    }

    impl NativeLaunchFixture {
        fn new() -> Self {
            let root = tempfile::tempdir().unwrap();
            let language_root = root.path().join("language");
            let toolkit_root = root.path().join("toolkit");
            let decisions_root = root.path().join("decisions");
            fs::create_dir_all(&language_root).unwrap();
            fs::create_dir_all(&toolkit_root).unwrap();
            fs::create_dir_all(&decisions_root).unwrap();
            fs::write(
                language_root.join("llama-swap.yaml"),
                "models:\n  fixture-model:\n    cmd: fixture\n",
            )
            .unwrap();
            fs::write(
                toolkit_root.join("media-runtimes.json"),
                serde_json::json!({
                    "schema_version": 1,
                    "profiles": [{
                        "id": "qwen-image-2.1",
                        "name": "Fixture Qwen Image",
                        "runtime": "ComfyUI",
                        "device": "cuda",
                        "tasks": ["image"],
                        "script": "media-models/generate-qwen-image-2.1.sh"
                    }]
                })
                .to_string(),
            )
            .unwrap();
            fs::write(
                decisions_root.join("decision-runtimes.json"),
                r#"{"version":"1.0","profiles":[]}"#,
            )
            .unwrap();

            let native_script = toolkit_root.join("media-models/generate-qwen-image-2.1.sh");
            fs::create_dir_all(native_script.parent().unwrap()).unwrap();
            fs::write(&native_script, b"#!/bin/sh\nexit 0\n").unwrap();
            for workflow in [
                "workflows/qwen-image-2.1/image_qwen_image_2_1_t2i.json",
                "workflows/qwen-image-2.1/image_qwen_image_2_1_image_edit.json",
            ] {
                let path = toolkit_root.join(workflow);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(
                    path,
                    br#"{"nodes":[{"id":1,"type":"SaveImage"}],"links":[]}"#,
                )
                .unwrap();
            }

            let checkout = toolkit_root.join("vendor/ComfyUI");
            fs::create_dir_all(checkout.join("models")).unwrap();
            fs::write(checkout.join("main.py"), b"# fixture Comfy entrypoint\n").unwrap();
            let venv = toolkit_root.join(".venv-comfy");
            fs::create_dir_all(venv.join("bin")).unwrap();
            fs::write(venv.join("pyvenv.cfg"), b"home = fixture\n").unwrap();
            let interpreter_target = std::env::current_exe().unwrap().canonicalize().unwrap();
            symlink(interpreter_target, venv.join("bin/python")).unwrap();

            let user_root = root.path().join("private-user");
            let output_root = root.path().join("private-output");
            let transport = root.path().join("transport");
            let protected_root = root.path().join("protected-history");
            let extra_models = root.path().join("extra-models");
            for path in [
                &user_root,
                &output_root,
                &transport,
                &protected_root,
                &extra_models,
            ] {
                fs::create_dir(path).unwrap();
            }
            for path in [&user_root, &output_root] {
                fs::set_permissions(path, fs::Permissions::from_mode(PRIVATE_MODE)).unwrap();
            }
            let extra_config = root.path().join("extra-model-paths.json");
            fs::write(
                &extra_config,
                serde_json::json!({
                    "comfyui": {
                        "base_path": extra_models.to_string_lossy(),
                        "checkpoints": "checkpoints"
                    }
                })
                .to_string(),
            )
            .unwrap();
            fs::set_permissions(&extra_config, fs::Permissions::from_mode(CONFIG_FILE_MODE))
                .unwrap();

            let catalog =
                Catalog::load_from_roots(&language_root, &toolkit_root, &decisions_root).unwrap();
            Self {
                _root: root,
                catalog,
                helper: std::env::current_exe().unwrap().canonicalize().unwrap(),
                transport,
                gpu: GpuProbeConfig::new(TEST_GPU_UUID, 0, 128).unwrap(),
                user_root,
                output_root,
                protected_root,
                native_script,
                extra_config,
            }
        }

        fn validate(&self) -> Result<ManagedComfyConfig, ManagedComfyConfigError> {
            let toolkit_root = self
                .native_script
                .parent()
                .and_then(Path::parent)
                .ok_or(ManagedComfyConfigError::InvalidDeclaration)?;
            ManagedComfyConfig::validate(
                ManagedComfyLaunchInput {
                    profiles: vec!["media:qwen-image-2.1".into()],
                    python_executable: toolkit_root.join(".venv-comfy/bin/python"),
                    comfy_checkout: toolkit_root.join("vendor/ComfyUI"),
                    extra_model_paths_config: Some(self.extra_config.clone()),
                    private_user_directory_root: self.user_root.clone(),
                    private_output_directory_root: self.output_root.clone(),
                    port: 8188,
                },
                &self.catalog,
                &self.gpu,
                self.helper.clone(),
                self.transport.clone(),
                Duration::from_secs(10),
            )
        }
    }

    #[test]
    fn workflow_graph_requires_unique_keys_and_exact_object_nodes() {
        let accepted = parse_strict_json(
            br#"{"1":{"class_type":"SaveImage","inputs":{"filename_prefix":"private"}}}"#,
        )
        .unwrap();
        validate_api_workflow_graph(&accepted).unwrap();

        assert!(parse_strict_json(br#"{"1":{"inputs":{},"inputs":{}}}"#).is_err());
        for rejected in [
            serde_json::json!([]),
            serde_json::json!({}),
            serde_json::json!({"0":{"class_type":"SaveImage","inputs":{}}}),
            serde_json::json!({"1":{"class_type":"SaveImage","inputs":{},"extra":true}}),
            serde_json::json!({"1":{"class_type":"SaveImage","inputs":[]}}),
        ] {
            assert!(validate_api_workflow_graph(&rejected).is_err());
        }
    }

    #[test]
    fn optional_model_path_config_is_strict_and_redacts_error_details() {
        let directory = tempfile::tempdir().unwrap();
        let accepted = serde_json::json!({
            "comfyui": {
                "base_path": directory.path().to_string_lossy(),
                "checkpoints": "checkpoints\nmore/checkpoints",
                "is_default": false
            }
        });
        assert_eq!(
            validate_extra_model_paths_config(accepted.to_string().as_bytes()).unwrap(),
            vec![directory.path().canonicalize().unwrap()]
        );
        for bytes in [
            br#"{"comfyui":{"base_path":"/models","base_path":"/other"}}"#.as_slice(),
            br#"{"comfyui":{"base_path":"/models","api_key":"secret"}}"#.as_slice(),
            br#"{"comfyui":{"base_path":"/models","checkpoints":["../escape"]}}"#.as_slice(),
            br#"{"comfyui":{"base_path":"relative"}}"#.as_slice(),
        ] {
            assert!(validate_extra_model_paths_config(bytes).is_err());
        }
        for category in [
            serde_json::json!(["checkpoints"]),
            serde_json::json!("$MODEL_PATH"),
            serde_json::json!("~/models"),
            serde_json::json!(""),
        ] {
            let mut rejected = accepted.clone();
            rejected["comfyui"]["checkpoints"] = category;
            assert!(validate_extra_model_paths_config(rejected.to_string().as_bytes()).is_err());
        }
    }

    #[test]
    fn private_directory_pins_reject_aliases_and_replaced_inodes() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("private");
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(PRIVATE_MODE)).unwrap();
        let pinned = PinnedDirectory::capture(&path, true).unwrap();
        pinned.verify().unwrap();

        let alias = parent.path().join("alias");
        symlink(&path, &alias).unwrap();
        assert!(matches!(
            PinnedDirectory::capture(&alias, true),
            Err(ManagedComfyConfigError::InvalidPrivateDirectory)
        ));

        let old_path = parent.path().join("old-private");
        fs::rename(&path, &old_path).unwrap();
        fs::create_dir(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(PRIVATE_MODE)).unwrap();
        assert_eq!(
            pinned.verify().unwrap_err(),
            ManagedComfyConfigError::InvalidPrivateDirectory
        );
    }

    #[test]
    fn per_job_private_directory_allocation_never_reuses_existing_work() {
        let parent = tempfile::tempdir().unwrap();
        let path = parent.path().join("job-42");
        create_private_job_directory(&path).unwrap();
        assert_eq!(
            create_private_job_directory(&path).unwrap_err(),
            ManagedComfyConfigError::JobDirectoryAlreadyAllocated
        );
        assert!(path.is_dir());
    }

    #[test]
    fn validated_launch_declaration_allocates_fresh_private_cuda_job_directories() {
        let fixture = NativeLaunchFixture::new();
        let mut config = fixture.validate().unwrap();
        assert_eq!(config.profiles().len(), 1);
        assert_eq!(config.cuda_device.gpu_uuid(), TEST_GPU_UUID);
        config
            .validate_protected_directories(std::slice::from_ref(&fixture.protected_root))
            .unwrap();

        let (_first_launch, first_output) = config.allocate_job_launch(41).unwrap();
        let (_second_launch, second_output) = config.allocate_job_launch(42).unwrap();
        assert_ne!(first_output, second_output);
        assert!(first_output.is_dir() && second_output.is_dir());
        assert_eq!(
            fs::metadata(&first_output).unwrap().permissions().mode() & 0o7777,
            PRIVATE_MODE
        );
        assert_eq!(
            fs::metadata(fixture.user_root.join("job-41"))
                .unwrap()
                .permissions()
                .mode()
                & 0o7777,
            PRIVATE_MODE
        );
        assert_eq!(
            config.allocate_job_launch(41).unwrap_err(),
            ManagedComfyConfigError::JobDirectoryAlreadyAllocated
        );
        assert_eq!(
            config
                .validate_protected_directories(std::slice::from_ref(&fixture.output_root))
                .unwrap_err(),
            ManagedComfyConfigError::ProtectedDirectoryOverlap
        );
    }

    #[test]
    fn launch_revalidation_refuses_native_config_and_parent_directory_drift() {
        let native_fixture = NativeLaunchFixture::new();
        let native_config = native_fixture.validate().unwrap();
        fs::write(
            &native_fixture.native_script,
            b"changed native script bytes\n",
        )
        .unwrap();
        assert_eq!(
            native_config.allocate_job_launch(51).unwrap_err(),
            ManagedComfyConfigError::NativeInputChanged
        );

        let config_fixture = NativeLaunchFixture::new();
        let config = config_fixture.validate().unwrap();
        fs::write(&config_fixture.extra_config, b"changed config bytes\n").unwrap();
        assert_eq!(
            config.allocate_job_launch(52).unwrap_err(),
            ManagedComfyConfigError::NativeInputChanged
        );

        let parent_fixture = NativeLaunchFixture::new();
        let parent_config = parent_fixture.validate().unwrap();
        let old_root = parent_fixture.output_root.with_extension("old");
        fs::rename(&parent_fixture.output_root, &old_root).unwrap();
        fs::create_dir(&parent_fixture.output_root).unwrap();
        fs::set_permissions(
            &parent_fixture.output_root,
            fs::Permissions::from_mode(PRIVATE_MODE),
        )
        .unwrap();
        assert_eq!(
            parent_config.allocate_job_launch(53).unwrap_err(),
            ManagedComfyConfigError::InvalidPrivateDirectory
        );
    }
}
