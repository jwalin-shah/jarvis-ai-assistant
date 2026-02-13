//! JARVIS Desktop Application Library
//!
//! Provides the Tauri application setup and tray functionality.

mod backend;
mod logging;
mod socket;
mod tray;

use serde::{Deserialize, Serialize};
use tauri::Manager;

/// Saved window position and size for persistence across launches.
#[derive(Debug, Serialize, Deserialize)]
struct WindowState {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
}

fn window_state_path() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("/tmp"));
    std::path::PathBuf::from(home)
        .join(".jarvis")
        .join("window-state.json")
}

fn load_window_state() -> Option<WindowState> {
    let path = window_state_path();
    let data = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_window_state(state: &WindowState) {
    let path = window_state_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(state) {
        let _ = std::fs::write(path, json);
    }
}

/// Update the tray tooltip with unread count
#[tauri::command]
fn update_tray_badge(app: tauri::AppHandle, count: u32) {
    if let Some(tray) = app.tray_by_id("main") {
        let tooltip = if count > 0 {
            format!("JARVIS — {} unread", count)
        } else {
            "JARVIS".to_string()
        };
        let _ = tray.set_tooltip(Some(&tooltip));
    }
}

/// List AddressBook source directory UUIDs for direct contact resolution.
/// Returns paths to AddressBook-v22.abcddb files that exist.
#[tauri::command]
fn list_addressbook_sources() -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| String::from("/tmp"));
    let sources_dir = std::path::PathBuf::from(&home)
        .join("Library")
        .join("Application Support")
        .join("AddressBook")
        .join("Sources");

    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&sources_dir) {
        for entry in entries.flatten() {
            if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                let db_path = entry.path().join("AddressBook-v22.abcddb");
                if db_path.exists() {
                    if let Some(path_str) = db_path.to_str() {
                        paths.push(path_str.to_string());
                    }
                }
            }
        }
    }
    paths
}

/// Run the Tauri application
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_global_shortcut::Builder::default().build())
        // Register socket state for persistent connection
        .manage(socket::SocketState::default())
        // Register backend process state for auto-launch
        .manage(backend::BackendProcess::default())
        .invoke_handler(tauri::generate_handler![
            socket::connect_socket,
            socket::disconnect_socket,
            socket::send_message,
            socket::send_batch,
            socket::send_streaming_message,
            socket::is_socket_connected,
            logging::frontend_log,
            update_tray_badge,
            list_addressbook_sources,
        ])
        .setup(|app| {
            // Set up the system tray
            tray::setup_tray(app)?;

            // Auto-launch backend socket server
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(backend::ensure_backend_running(handle));

            // Get the main window
            if let Some(window) = app.get_webview_window("main") {
                // Restore saved window position and size
                if let Some(state) = load_window_state() {
                    let pos: tauri::Position =
                        tauri::PhysicalPosition::new(state.x, state.y).into();
                    let size: tauri::Size =
                        tauri::PhysicalSize::new(state.width, state.height).into();
                    let _ = window.set_position(pos);
                    let _ = window.set_size(size);
                }

                // Ensure window shows after a short delay (fallback if frontend show() doesn't fire)
                let show_window = window.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs(2));
                    if !show_window.is_visible().unwrap_or(true) {
                        let _ = show_window.show();
                        let _ = show_window.set_focus();
                    }
                });

                // Handle close button - hide instead of quit, save window state
                let window_clone = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        // Save window position and size before hiding
                        if let (Ok(pos), Ok(size)) =
                            (window_clone.outer_position(), window_clone.outer_size())
                        {
                            save_window_state(&WindowState {
                                x: pos.x,
                                y: pos.y,
                                width: size.width,
                                height: size.height,
                            });
                        }
                        // Hide the window instead of closing
                        if let Err(e) = window_clone.hide() {
                            eprintln!("[App] Failed to hide window on close request: {}", e);
                        }
                        api.prevent_close();
                    }
                });
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .unwrap_or_else(|e| {
            panic!(
                "Failed to build JARVIS Tauri application - check dependencies/configuration: {e}"
            );
        });

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            // Kill the backend process on app exit
            if let Some(state) = app_handle.try_state::<backend::BackendProcess>() {
                eprintln!("[App] Shutting down backend process");
                state.kill();
            }
        }
    });
}
