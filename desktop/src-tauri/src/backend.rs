//! Backend process management for auto-launching the Python socket server.
//!
//! Spawns `uv run python -m jarvis.interfaces.desktop` when the Tauri app
//! starts, monitors readiness via socket probe, and kills on app exit.

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager};
use tempfile::tempdir;

/// Managed state holding the spawned backend child process.
#[derive(Default)]
pub struct BackendProcess {
    child: Mutex<Option<Child>>,
}

impl BackendProcess {
    /// Kill the child process if it's still running.
    pub fn kill(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(ref mut child) = *guard {
                let _ = child.kill();
                let _ = child.wait(); // reap to avoid zombie
            }
            *guard = None;
        }
    }
}

/// Check if the backend is already running by probing the socket file.
fn is_backend_running() -> bool {
    let sock = socket_path();
    if !sock.exists() {
        return false;
    }
    // Try a TCP-style connect to the Unix socket
    std::os::unix::net::UnixStream::connect(&sock).is_ok()
}

/// Return the socket path (`~/.jarvis/jarvis.sock`).
fn socket_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".jarvis").join("jarvis.sock")
}

/// Locate the `uv` binary by searching common paths then falling back to `which`.
fn find_uv() -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.local/bin/uv"),
        format!("{home}/.cargo/bin/uv"),
        "/opt/homebrew/bin/uv".into(),
        "/usr/local/bin/uv".into(),
    ];
    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    // Fallback: ask the shell
    Command::new("/bin/sh")
        .args(["-c", "which uv"])
        .output()
        .ok()
        .and_then(|out| {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if s.is_empty() {
                None
            } else {
                Some(PathBuf::from(s))
            }
        })
}

/// Find the project root containing `pyproject.toml`.
///
/// Checks `JARVIS_PROJECT_ROOT` env var first, then walks up from the
/// executable's directory.
fn project_root() -> Option<PathBuf> {
    // Env var override
    if let Ok(root) = std::env::var("JARVIS_PROJECT_ROOT") {
        let p = PathBuf::from(&root);
        if p.join("pyproject.toml").exists() {
            return Some(p);
        }
    }

    // Walk up from executable
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(|p| p.to_path_buf());
        while let Some(d) = dir {
            if d.join("pyproject.toml").exists() {
                return Some(d);
            }
            dir = d.parent().map(|p| p.to_path_buf());
        }
    }

    // Common dev location fallback
    let home = std::env::var("HOME").unwrap_or_default();
    let dev_path = PathBuf::from(&home).join("projects").join("jarvis-ai-assistant");
    if dev_path.join("pyproject.toml").exists() {
        return Some(dev_path);
    }

    None
}

/// Spawn the backend process, returning the Child handle.
fn spawn_backend() -> Result<Child, String> {
    let uv = find_uv().ok_or("Could not find `uv` binary")?;
    let root = project_root().ok_or("Could not find JARVIS project root (pyproject.toml)")?;

    // Ensure log directory exists
    let dir = tempdir()?;
    let log_path = dir.path().join("backend.log");

    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open log file: {e}"))?;
    let stderr_file = log_file
        .try_clone()
        .map_err(|e| format!("Failed to clone log file handle: {e}"))?;

    let child = Command::new(uv.as_os_str())
        .args(["run", "python", "-m", "jarvis.interfaces.desktop"])
        .current_dir(&root)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .map_err(|e| format!("Failed to spawn backend: {e}"))?;

    eprintln!(
        "[Backend] Spawned PID {} (uv={}, root={})",
        child.id(),
        uv.display(),
        root.display()
    );

    Ok(child)
}

/// Ensure the backend is running. Spawns if needed, polls for socket readiness,
/// and emits Tauri events on success/failure.
pub async fn ensure_backend_running(app: AppHandle) {
    if is_backend_running() {
        eprintln!("[Backend] Already running, skipping spawn");
        let _ = app.emit("jarvis:backend_ready", ());
        return;
    }

    eprintln!("[Backend] Not running, attempting to spawn...");

    let child = match spawn_backend() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[Backend] Spawn failed: {e}");
            let _ = app.emit("jarvis:backend_error", e);
            return;
        }
    };

    // Store the child in managed state
    if let Some(state) = app.try_state::<BackendProcess>() {
        if let Ok(mut guard) = state.child.lock() {
            *guard = Some(child);
        }
    }

    // Poll for socket readiness (100ms intervals, 15s timeout)
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(15);
    loop {
        if is_backend_running() {
            eprintln!("[Backend] Socket is ready");
            let _ = app.emit("jarvis:backend_ready", ());
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            eprintln!("[Backend] Timed out waiting for socket");
            let _ = app.emit(
                "jarvis:backend_error",
                "Backend started but socket not ready after 15s",
            );
            return;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}
