//! Unix socket client for communicating with JARVIS Python daemon
//!
//! Provides Tauri commands for connecting to ~/.jarvis/jarvis.sock and sending JSON-RPC messages.
//! Supports both request/response and streaming patterns.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex, RwLock};

/// Get socket path for JARVIS daemon (~/.jarvis/jarvis.sock)
fn get_socket_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
    format!("{home}/.jarvis/jarvis.sock")
}

/// Request ID counter
static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Pending request state
struct PendingRequest {
    response_tx: mpsc::Sender<Result<serde_json::Value, String>>,
}

/// Shared socket connection state
pub struct SocketState {
    /// Writer half of the connection
    writer: Arc<Mutex<Option<BufWriter<tokio::net::unix::OwnedWriteHalf>>>>,
    /// Pending requests waiting for responses
    pending: Arc<RwLock<HashMap<u64, PendingRequest>>>,
    /// Whether we're connected
    connected: Arc<RwLock<bool>>,
    /// Reader task handle
    reader_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
}

impl Default for SocketState {
    fn default() -> Self {
        Self {
            writer: Arc::new(Mutex::new(None)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(false)),
            reader_task: Arc::new(Mutex::new(None)),
        }
    }
}

/// JSON-RPC request structure
#[derive(Serialize, Debug)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: serde_json::Value,
    id: u64,
}

/// JSON-RPC response/notification structure
#[derive(Deserialize, Debug)]
struct JsonRpcMessage {
    #[serde(rename = "jsonrpc")]
    _jsonrpc: String,
    /// Method name (for notifications)
    #[serde(default)]
    method: Option<String>,
    /// Parameters (for notifications)
    #[serde(default)]
    params: Option<serde_json::Value>,
    /// Result (for responses)
    #[serde(default)]
    result: Option<serde_json::Value>,
    /// Error (for error responses)
    #[serde(default)]
    error: Option<JsonRpcError>,
    /// Request ID (for responses, None for notifications)
    #[serde(default)]
    id: Option<u64>,
}

/// JSON-RPC error structure
#[derive(Deserialize, Debug)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[serde(default, rename = "data")]
    _data: Option<serde_json::Value>,
}

/// Stream token event emitted to frontend
#[derive(Serialize, Clone, Debug)]
pub struct StreamTokenEvent {
    pub token: String,
    pub index: u64,
    pub final_token: bool,
    pub request_id: u64,
}

/// New message notification event
#[derive(Serialize, Clone, Debug)]
pub struct NewMessageEvent {
    pub message_id: i64,
    pub chat_id: String,
    pub sender: Option<String>,
    pub text_preview: Option<String>,
    pub is_from_me: bool,
}

/// Connect to the JARVIS socket server with persistent connection
#[tauri::command]
pub async fn connect_socket(
    app: AppHandle,
    state: State<'_, SocketState>,
) -> Result<bool, String> {
    // Clone Arc references out of state to avoid lifetime issues
    let writer_arc = state.writer.clone();
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let reader_task_arc = state.reader_task.clone();

    // Abort any existing reader task to prevent stale readers
    {
        let mut task = reader_task_arc.lock().await;
        if let Some(handle) = task.take() {
            handle.abort();
        }
    }

    // Close existing writer
    {
        let mut w = writer_arc.lock().await;
        if let Some(ref mut writer) = *w {
            let _ = writer.shutdown().await;
        }
        *w = None;
    }

    // Reset connected flag
    {
        let mut connected = connected_arc.write().await;
        *connected = false;
    }

    // Connect to socket
    let socket_path = get_socket_path();
    println!("[Socket] Connecting to JARVIS socket at: {}", socket_path);
    let stream = UnixStream::connect(&socket_path)
        .await
        .map_err(|e| format!("Failed to connect to socket at {}: {}", socket_path, e))?;

    let (reader, writer) = stream.into_split();

    // Store writer
    {
        let mut w = writer_arc.lock().await;
        *w = Some(BufWriter::new(writer));
    }

    // Mark as connected
    {
        let mut connected = connected_arc.write().await;
        *connected = true;
    }

    // Spawn reader task to handle responses and notifications
    let connected_flag = connected_arc.clone();
    let app_handle = app.clone();

    let reader_handle = tokio::spawn(async move {
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        loop {
            line.clear();
            match buf_reader.read_line(&mut line).await {
                Ok(0) => {
                    // Connection closed
                    break;
                }
                Ok(_) => {
                    // Parse message
                    if let Ok(msg) = serde_json::from_str::<JsonRpcMessage>(&line) {
                        handle_message(msg, &pending_arc, &app_handle).await;
                    } else {
                        eprintln!("[Socket] Failed to parse message: {}", line.trim());
                    }
                }
                Err(e) => {
                    eprintln!("[Socket] Read error: {}", e);
                    break;
                }
            }
        }

        // Mark as disconnected
        let mut connected = connected_flag.write().await;
        *connected = false;

        // Emit disconnected event
        let _ = app_handle.emit("jarvis:disconnected", ());
    });

    // Store reader task handle
    {
        let mut task = reader_task_arc.lock().await;
        *task = Some(reader_handle);
    }

    // Emit connected event
    let _ = app.emit("jarvis:connected", ());

    Ok(true)
}

/// Handle incoming JSON-RPC message (response or notification)
async fn handle_message(
    msg: JsonRpcMessage,
    pending: &Arc<RwLock<HashMap<u64, PendingRequest>>>,
    app: &AppHandle,
) {
    if let Some(id) = msg.id {
        // This is a response to a request
        let pending_req = {
            let mut p = pending.write().await;
            p.remove(&id)
        };

        if let Some(req) = pending_req {
            let result = if let Some(error) = msg.error {
                Err(format!("RPC error {}: {}", error.code, error.message))
            } else {
                Ok(msg.result.unwrap_or(serde_json::Value::Null))
            };
            let _ = req.response_tx.send(result).await;
        }
    } else if let Some(method) = msg.method {
        // This is a notification
        match method.as_str() {
            "stream.token" => {
                if let Some(params) = msg.params {
                    let event = StreamTokenEvent {
                        token: params.get("token").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        index: params.get("index").and_then(|v| v.as_u64()).unwrap_or(0),
                        final_token: params.get("final").and_then(|v| v.as_bool()).unwrap_or(false),
                        request_id: params.get("request_id").and_then(|v| v.as_u64()).unwrap_or(0),
                    };
                    let _ = app.emit("jarvis:stream_token", event);
                }
            }
            "new_message" => {
                if let Some(params) = msg.params {
                    let event = NewMessageEvent {
                        message_id: params.get("message_id").and_then(|v| v.as_i64()).unwrap_or(0),
                        chat_id: params.get("chat_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        sender: params.get("sender").and_then(|v| v.as_str()).map(String::from),
                        text_preview: params.get("text").and_then(|v| v.as_str()).map(String::from),
                        is_from_me: params.get("is_from_me").and_then(|v| v.as_bool()).unwrap_or(false),
                    };
                    let _ = app.emit("jarvis:new_message", event);
                }
            }
            _ => {
                // Unknown notification, emit generic event
                let _ = app.emit(&format!("jarvis:{}", method), msg.params);
            }
        }
    }
}

/// Disconnect from the socket server
#[tauri::command]
pub async fn disconnect_socket(state: State<'_, SocketState>) -> Result<(), String> {
    // Clone Arc references
    let reader_task_arc = state.reader_task.clone();
    let writer_arc = state.writer.clone();
    let pending_arc = state.pending.clone();
    let connected_arc = state.connected.clone();

    // Cancel reader task
    {
        let mut task = reader_task_arc.lock().await;
        if let Some(handle) = task.take() {
            handle.abort();
        }
    }

    // Close writer
    {
        let mut writer = writer_arc.lock().await;
        if let Some(mut w) = writer.take() {
            let _ = w.shutdown().await;
        }
    }

    // Clear pending requests
    {
        let mut pending = pending_arc.write().await;
        pending.clear();
    }

    // Mark as disconnected
    {
        let mut connected = connected_arc.write().await;
        *connected = false;
    }

    Ok(())
}

/// Send a JSON-RPC message and get response
#[tauri::command]
pub async fn send_message(
    method: String,
    params: serde_json::Value,
    state: State<'_, SocketState>,
) -> Result<serde_json::Value, String> {
    // Clone Arc references
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let writer_arc = state.writer.clone();

    // Check if connected
    {
        let connected = connected_arc.read().await;
        if !*connected {
            return Err("Not connected to socket server".to_string());
        }
    }

    let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

    // Create channel for response
    let (tx, mut rx) = mpsc::channel(1);

    // Register pending request
    {
        let mut pending = pending_arc.write().await;
        pending.insert(id, PendingRequest { response_tx: tx });
    }

    // Build and send request
    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params,
        id,
    };

    let request_json = serde_json::to_string(&request)
        .map_err(|e| format!("Failed to serialize request: {}", e))?;

    {
        let mut writer_guard = writer_arc.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            writer
                .write_all(request_json.as_bytes())
                .await
                .map_err(|e| format!("Failed to write request: {}", e))?;
            writer
                .write_all(b"\n")
                .await
                .map_err(|e| format!("Failed to write newline: {}", e))?;
            writer
                .flush()
                .await
                .map_err(|e| format!("Failed to flush: {}", e))?;
        } else {
            return Err("Writer not available".to_string());
        }
    }

    // Wait for response with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(120), rx.recv()).await {
        Ok(Some(result)) => result,
        Ok(None) => Err("Response channel closed".to_string()),
        Err(_) => {
            // Remove pending request on timeout
            let mut pending = pending_arc.write().await;
            pending.remove(&id);
            Err("Request timed out".to_string())
        }
    }
}

/// Send a streaming request (response comes via events)
#[tauri::command]
pub async fn send_streaming_message(
    method: String,
    params: serde_json::Value,
    state: State<'_, SocketState>,
) -> Result<u64, String> {
    // Clone Arc references
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let writer_arc = state.writer.clone();

    // Check if connected
    {
        let connected = connected_arc.read().await;
        if !*connected {
            return Err("Not connected to socket server".to_string());
        }
    }

    let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

    // Create channel for final response
    let (tx, _rx) = mpsc::channel(1);

    // Register pending request
    {
        let mut pending = pending_arc.write().await;
        pending.insert(id, PendingRequest { response_tx: tx });
    }

    // Add stream flag to params
    let mut params_with_stream = params;
    if let Some(obj) = params_with_stream.as_object_mut() {
        obj.insert("stream".to_string(), serde_json::Value::Bool(true));
    }

    // Build and send request
    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params: params_with_stream,
        id,
    };

    let request_json = serde_json::to_string(&request)
        .map_err(|e| format!("Failed to serialize request: {}", e))?;

    {
        let mut writer_guard = writer_arc.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            writer
                .write_all(request_json.as_bytes())
                .await
                .map_err(|e| format!("Failed to write request: {}", e))?;
            writer
                .write_all(b"\n")
                .await
                .map_err(|e| format!("Failed to write newline: {}", e))?;
            writer
                .flush()
                .await
                .map_err(|e| format!("Failed to flush: {}", e))?;
        } else {
            return Err("Writer not available".to_string());
        }
    }

    // Return request ID so frontend can correlate stream tokens
    Ok(id)
}

/// Check if socket server is available
#[tauri::command]
pub async fn is_socket_connected(state: State<'_, SocketState>) -> Result<bool, String> {
    let connected_arc = state.connected.clone();
    let connected = connected_arc.read().await;
    Ok(*connected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_id_increments() {
        let id1 = REQUEST_ID.fetch_add(1, Ordering::SeqCst);
        let id2 = REQUEST_ID.fetch_add(1, Ordering::SeqCst);
        assert!(id2 > id1);
    }
}
