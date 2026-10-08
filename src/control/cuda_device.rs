//! Typed selection of one physical NVIDIA GPU for a child process.
//!
//! The parser accepts only a complete, lowercase NVIDIA GPU UUID. It does not
//! accept ordinals, abbreviated UUIDs, lists, MIG identifiers, or arbitrary
//! CUDA visibility expressions.

use std::fmt;
use std::process::Command;

/// A canonical physical NVIDIA GPU UUID suitable for a child process's
/// `CUDA_VISIBLE_DEVICES` value.
#[derive(Clone, Eq, PartialEq)]
pub struct CudaDevice {
    gpu_uuid: String,
}

impl CudaDevice {
    /// Parse one complete canonical `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
    /// UUID. Only lowercase hexadecimal digits are accepted so one device has
    /// a single representation throughout configuration and process launch.
    pub fn new(gpu_uuid: &str) -> Result<Self, CudaDeviceError> {
        let bytes = gpu_uuid.as_bytes();
        if bytes.len() != 40 || &bytes[..4] != b"GPU-" {
            return Err(CudaDeviceError::InvalidUuid);
        }
        for (index, byte) in bytes[4..].iter().enumerate() {
            let valid = if matches!(index, 8 | 13 | 18 | 23) {
                *byte == b'-'
            } else {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(byte)
            };
            if !valid {
                return Err(CudaDeviceError::InvalidUuid);
            }
        }
        Ok(Self {
            gpu_uuid: gpu_uuid.to_owned(),
        })
    }

    /// Return the exact canonical UUID selected by this value.
    pub fn gpu_uuid(&self) -> &str {
        &self.gpu_uuid
    }

    /// Set this one explicit physical device on a child command.
    pub fn apply_to(&self, command: &mut Command) {
        command.env("CUDA_VISIBLE_DEVICES", self.gpu_uuid());
    }
}

impl fmt::Debug for CudaDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CudaDevice([redacted])")
    }
}

/// Fixed, redacted CUDA device construction failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaDeviceError {
    InvalidUuid,
}

impl fmt::Display for CudaDeviceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("CUDA device must be one full canonical GPU UUID")
    }
}

impl std::error::Error for CudaDeviceError {}

#[cfg(test)]
mod tests {
    use super::*;

    const UUID: &str = "GPU-01234567-89ab-cdef-0123-456789abcdef";

    #[test]
    fn accepts_only_one_full_lowercase_physical_gpu_uuid() {
        assert_eq!(CudaDevice::new(UUID).unwrap().gpu_uuid(), UUID);

        for invalid in [
            "0",
            "GPU-01234567",
            "GPU-0123456789abcdef0123456789abcdef",
            "GPU-01234567-89AB-cdef-0123-456789abcdef",
            "GPU-01234567-89ab-cdef-0123-456789abcdeg",
            "GPU-01234567-89ab-cdef-0123-456789abcdef,GPU-11234567-89ab-cdef-0123-456789abcdef",
            "MIG-GPU-01234567-89ab-cdef-0123-456789abcdef/1/0",
            "GPU-01234567-89ab-cdef-0123-456789abcdef/0",
        ] {
            assert_eq!(
                CudaDevice::new(invalid),
                Err(CudaDeviceError::InvalidUuid),
                "accepted invalid UUID form"
            );
        }
    }

    #[test]
    fn debug_and_errors_do_not_echo_the_supplied_identifier() {
        let device = CudaDevice::new(UUID).unwrap();
        assert_eq!(format!("{device:?}"), "CudaDevice([redacted])");
        assert_eq!(
            CudaDevice::new("GPU-01234567-89ab-cdef-0123-456789abcdeg")
                .unwrap_err()
                .to_string(),
            "CUDA device must be one full canonical GPU UUID"
        );
    }

    #[test]
    fn applies_only_the_typed_uuid_to_a_command() {
        let device = CudaDevice::new(UUID).unwrap();
        let mut command = Command::new("/bin/true");
        device.apply_to(&mut command);
        let environment = command
            .get_envs()
            .filter_map(|(key, value)| Some((key.to_str()?, value?.to_str()?)))
            .collect::<Vec<_>>();
        assert_eq!(environment, [("CUDA_VISIBLE_DEVICES", UUID)]);
    }
}
