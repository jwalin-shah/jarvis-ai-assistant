//! Frontend log bridge for sending WebView console logs to the Rust terminal.

/// Forward frontend logs to terminal output.
#[tauri::command]
pub async fn frontend_log(level: String, message: String) -> Result<(), String> {
    match level.as_str() {
        "error" => eprintln!("[frontend:error] {message}"),
        "warn" => eprintln!("[frontend:warn] {message}"),
        "info" => println!("[frontend:info] {message}"),
        "debug" => println!("[frontend:debug] {message}"),
        _ => println!("[frontend:log] {message}"),
    }
    Ok(())
}
