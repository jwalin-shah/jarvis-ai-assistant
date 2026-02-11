//! JARVIS Desktop Application Library
//!
//! Provides the Tauri application setup and tray functionality.

mod socket;
mod logging;
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
    std::path::PathBuf::from(home).join(".jarvis").join("window-state.json")
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

/// Run the Tauri application
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_sql::Builder::default().build())
        // Register socket state for persistent connection
        .manage(socket::SocketState::default())
        .invoke_handler(tauri::generate_handler![
            socket::connect_socket,
            socket::disconnect_socket,
            socket::send_message,
            socket::send_batch,
            socket::send_streaming_message,
            socket::is_socket_connected,
            logging::frontend_log,
        ])
        .setup(|app| {
            // Set up the system tray
            tray::setup_tray(app)?;

            // Get the main window
            if let Some(window) = app.get_webview_window("main") {
                // Restore saved window position and size
                if let Some(state) = load_window_state() {
                    let pos: tauri::Position = tauri::PhysicalPosition::new(state.x, state.y).into();
                    let size: tauri::Size = tauri::PhysicalSize::new(state.width, state.height).into();
                    let _ = window.set_position(pos);
                    let _ = window.set_size(size);
                }

                // Handle close button - hide instead of quit, save window state
                let window_clone = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                        // Save window position and size before hiding
                        if let (Ok(pos), Ok(size)) = (window_clone.outer_position(), window_clone.outer_size()) {
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
        .run(tauri::generate_context!())
        .expect("Failed to run JARVIS Tauri application - check that all dependencies are available and the configuration is valid");
}
