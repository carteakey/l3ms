//! Pinned private engine sockets for native HTTP adapters.
//!
//! The lifecycle owner must establish the isolated helper's identity and hold
//! its host/service ownership before selecting this transport. Filesystem
//! validation is not an isolation, admission, drain or GPU-release proof.
//! Private directory ancestors and same-user writers must remain trusted.

use std::fs::{self, File, OpenOptions};
use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use super::engine_transport::ENGINE_SOCKET_NAME;
use super::local_ipc::private_directory;

pub(crate) struct PinnedEngineSocket {
    directory: File,
    directory_path: PathBuf,
    directory_identity: (u64, u64),
    socket_path: PathBuf,
    socket_identity: (u64, u64),
    uid: u32,
}

impl PinnedEngineSocket {
    pub(crate) fn open(directory: &Path) -> Result<Self, ()> {
        let directory_path = private_directory(directory).map_err(|_| ())?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&directory_path)
            .map_err(|_| ())?;
        let metadata = file.metadata().map_err(|_| ())?;
        let uid = unsafe { libc::geteuid() };
        if !metadata.is_dir() || metadata.uid() != uid || metadata.mode() & 0o7777 != 0o700 {
            return Err(());
        }
        let socket_path = directory_path.join(ENGINE_SOCKET_NAME);
        let socket = fs::symlink_metadata(&socket_path).map_err(|_| ())?;
        if !socket.file_type().is_socket()
            || socket.uid() != uid
            || socket.mode() & 0o7777 != 0o600
            || socket.nlink() != 1
        {
            return Err(());
        }
        let pinned = Self {
            directory: file,
            directory_path,
            directory_identity: (metadata.dev(), metadata.ino()),
            socket_path,
            socket_identity: (socket.dev(), socket.ino()),
            uid,
        };
        pinned.check()?;
        Ok(pinned)
    }

    fn check(&self) -> Result<(), ()> {
        let descriptor = self.directory.metadata().map_err(|_| ())?;
        let directory = fs::symlink_metadata(&self.directory_path).map_err(|_| ())?;
        for entry in [&descriptor, &directory] {
            if !entry.is_dir()
                || entry.uid() != self.uid
                || entry.mode() & 0o7777 != 0o700
                || (entry.dev(), entry.ino()) != self.directory_identity
            {
                return Err(());
            }
        }
        let socket = fs::symlink_metadata(&self.socket_path).map_err(|_| ())?;
        if !socket.file_type().is_socket()
            || socket.uid() != self.uid
            || socket.mode() & 0o7777 != 0o600
            || socket.nlink() != 1
            || (socket.dev(), socket.ino()) != self.socket_identity
        {
            return Err(());
        }
        Ok(())
    }

    /// Never falls back to TCP, proxies or DNS on socket failure. Native URL
    /// paths still select HTTP operations; redirects/retries remain disabled
    /// by each adapter. Revalidation precedes each client construction.
    pub(crate) fn configure(
        &self,
        client: reqwest::ClientBuilder,
    ) -> Result<reqwest::ClientBuilder, ()> {
        self.check()?;
        Ok(client.no_proxy().unix_socket(self.socket_path.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::os::unix::fs::{symlink, PermissionsExt};
    use std::os::unix::net::UnixListener;

    fn endpoint() -> (tempfile::TempDir, UnixListener) {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = directory.path().join(ENGINE_SOCKET_NAME);
        let listener = UnixListener::bind(&path).unwrap();
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
        (directory, listener)
    }

    #[test]
    fn private_http_uses_the_socket_without_tcp_or_dns() {
        let (directory, listener) = endpoint();
        let pinned = PinnedEngineSocket::open(directory.path()).unwrap();
        listener.set_nonblocking(true).unwrap();
        let server = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            let mut peer = loop {
                match listener.accept() {
                    Ok((peer, _)) => break peer,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        assert!(std::time::Instant::now() < deadline);
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(_) => panic!("fixture Unix accept failed"),
                }
            };
            peer.set_read_timeout(Some(std::time::Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            while !request.ends_with(b"\r\n\r\n") {
                let mut byte = [0];
                peer.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
                assert!(request.len() < 4096);
            }
            assert!(request.starts_with(b"GET /health HTTP/1.1\r\n"));
            peer.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                .unwrap();
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            let builder = reqwest::Client::builder()
                .retry(reqwest::retry::never())
                .redirect(reqwest::redirect::Policy::none())
                .timeout(std::time::Duration::from_secs(3));
            let client = pinned.configure(builder).unwrap().build().unwrap();
            let response = client
                .get("http://nonexistent.invalid/health")
                .send()
                .await
                .unwrap();
            assert_eq!(response.bytes().await.unwrap().as_ref(), b"ok");
        });
        server.join().unwrap();
    }

    #[test]
    fn permissions_links_and_replacement_refuse_client_construction() {
        let (directory, _listener) = endpoint();
        let path = directory.path().join(ENGINE_SOCKET_NAME);
        let pinned = PinnedEngineSocket::open(directory.path()).unwrap();
        let alias = directory.path().join("alias.sock");
        fs::hard_link(&path, &alias).unwrap();
        assert!(pinned.configure(reqwest::Client::builder()).is_err());
        fs::remove_file(alias).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o660)).unwrap();
        assert!(pinned.configure(reqwest::Client::builder()).is_err());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::remove_file(&path).unwrap();
        let replacement = UnixListener::bind(&path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(pinned.configure(reqwest::Client::builder()).is_err());
        drop(replacement);
        fs::remove_file(&path).unwrap();
        symlink("absent", &path).unwrap();
        assert!(PinnedEngineSocket::open(directory.path()).is_err());
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o750)).unwrap();
        assert!(PinnedEngineSocket::open(directory.path()).is_err());
    }
}
