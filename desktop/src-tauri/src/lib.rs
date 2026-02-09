//! JARVIS Desktop Application Library
//!
//! Provides the Tauri application setup and tray functionality.

mod socket;
mod logging;
mod tray;

use tauri::Manager;

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
            socket::send_streaming_message,
            socket::is_socket_connected,
            logging::frontend_log,
        ])
        .setup(|app| {
            // Set up the system tray
            tray::setup_tray(app)?;

            // Get the main window
            if let Some(window) = app.get_webview_window("main") {
                // Handle close button - hide instead of quit
                let window_clone = window.clone();
                window.on_window_event(move |event| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = event {
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
