//! System tray (menu bar) functionality for JARVIS.
//!
//! Provides:
//! - Menu bar icon
//! - Left-click to toggle window visibility
//! - Right-click context menu

use tauri::{
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Emitter, Manager,
};

/// Set up the system tray icon and menu
pub fn setup_tray(app: &App) -> Result<(), Box<dyn std::error::Error>> {
    // Create menu items
    let show_item = MenuItem::with_id(app, "show", "Show JARVIS", true, None::<&str>)?;
    let status_item = MenuItem::with_id(app, "status", "Status: Disconnected", false, None::<&str>)?;
    let health_item = MenuItem::with_id(app, "health", "System Health", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;

    // Build the tray menu
    let menu = Menu::with_items(app, &[&show_item, &status_item, &health_item, &separator, &quit_item])?;

    // Create the tray icon
    let _tray = TrayIconBuilder::new()
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "show" => {
                if let Some(window) = app.get_webview_window("main") {
                    if let Err(e) = window.show() {
                        eprintln!("[Tray] Failed to show window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        eprintln!("[Tray] Failed to set focus: {}", e);
                    }
                }
            }
            "health" => {
                if let Some(window) = app.get_webview_window("main") {
                    if let Err(e) = window.show() {
                        eprintln!("[Tray] Failed to show window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        eprintln!("[Tray] Failed to set focus: {}", e);
                    }
                    // Emit event to navigate to health view
                    if let Err(e) = window.emit("navigate", "health") {
                        eprintln!("[Tray] Failed to emit navigate event: {}", e);
                    }
                }
            }
            "quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                // Toggle window visibility on left click
                let app = tray.app_handle();
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_visible().unwrap_or(false) {
                        if let Err(e) = window.hide() {
                            eprintln!("[Tray] Failed to hide window: {}", e);
                        }
                    } else {
                        if let Err(e) = window.show() {
                            eprintln!("[Tray] Failed to show window: {}", e);
                        }
                        if let Err(e) = window.set_focus() {
                            eprintln!("[Tray] Failed to set focus: {}", e);
                        }
                    }
                }
            }
        })
        .build(app)?;

    Ok(())
}
