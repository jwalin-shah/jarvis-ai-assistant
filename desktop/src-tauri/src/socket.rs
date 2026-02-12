//! Unix socket client for communicating with JARVIS Python daemon
//!
//! Provides Tauri commands for connecting to ~/.jarvis/jarvis.sock and sending JSON-RPC messages.
//! Supports both request/response and streaming patterns.

use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex, RwLock};

/// Get socket path for JARVIS daemon (~/.jarvis/jarvis.sock)
fn get_socket_path() -> Result<String, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME environment variable not set - cannot determine socket path".to_string())?;
    Ok(format!("{home}/.jarvis/jarvis.sock"))
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
    /// Message processor task handle
    processor_task: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Whether the writer is dead (marked after write failure)
    writer_dead: Arc<RwLock<bool>>,
}

impl Default for SocketState {
    fn default() -> Self {
        Self {
            writer: Arc::new(Mutex::new(None)),
            pending: Arc::new(RwLock::new(HashMap::new())),
            connected: Arc::new(RwLock::new(false)),
            reader_task: Arc::new(Mutex::new(None)),
            processor_task: Arc::new(Mutex::new(None)),
            writer_dead: Arc::new(RwLock::new(false)),
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

/// Check that the socket is connected and the writer isn't dead.
async fn check_connection(
    connected: &Arc<RwLock<bool>>,
    writer_dead: &Arc<RwLock<bool>>,
) -> Result<(), String> {
    {
        let c = connected.read().await;
        if !*c {
            return Err("Not connected to socket server".to_string());
        }
    }
    {
        let wd = writer_dead.read().await;
        if *wd {
            return Err("Writer is dead after previous failure".to_string());
        }
    }
    Ok(())
}

/// Write a JSON string to the socket writer, marking it dead on failure.
async fn write_to_socket(
    writer_arc: &Arc<Mutex<Option<BufWriter<tokio::net::unix::OwnedWriteHalf>>>>,
    writer_dead_arc: &Arc<RwLock<bool>>,
    json: &str,
) -> Result<(), String> {
    let mut writer_guard = writer_arc.lock().await;
    if let Some(writer) = writer_guard.as_mut() {
        let result = async {
            writer.write_all(json.as_bytes()).await?;
            writer.write_all(b"\n").await?;
            writer.flush().await?;
            Ok::<(), std::io::Error>(())
        }.await;

        if let Err(e) = result {
            let mut wd = writer_dead_arc.write().await;
            *wd = true;
            return Err(format!("Failed to write to socket: {} (writer marked dead)", e));
        }
        Ok(())
    } else {
        Err("Writer not available".to_string())
    }
}

/// Connect to the JARVIS socket server with persistent connection
#[tauri::command]
pub async fn connect_socket(
    app: AppHandle,
    state: State<'_, SocketState>,
) -> Result<bool, String> {
    let socket_path = get_socket_path()?;
    println!("[Socket] Connecting to JARVIS socket at: {}", socket_path);
    connect_socket_internal(&app, &state).await
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
                    if let Err(e) = app.emit("jarvis:stream_token", event) {
                        eprintln!("[Socket] Failed to emit stream_token event: {}", e);
                    }
                } else {
                    eprintln!("[Socket] Warning: stream.token notification missing params");
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
                    if let Err(e) = app.emit("jarvis:new_message", event) {
                        eprintln!("[Socket] Failed to emit new_message event: {}", e);
                    }
                } else {
                    eprintln!("[Socket] Warning: new_message notification missing params");
                }
            }
            _ => {
                // Whitelist of allowed notification method names for event emission.
                // Only methods explicitly listed here are forwarded to the frontend.
                const ALLOWED_METHODS: &[&str] = &[
                    "connection.status",
                    "contact.updated",
                    "conversation.updated",
                    "fact.extracted",
                    "message.classified",
                    "message.updated",
                    "prefetch.complete",
                    "reply.ready",
                    "search.result",
                    "status.update",
                    "stream.complete",
                    "stream.error",
                    "watcher.event",
                ];

                if ALLOWED_METHODS.contains(&method.as_str()) {
                    let event_name = format!("jarvis:{}", method);
                    if let Err(e) = app.emit(&event_name, msg.params) {
                        eprintln!("[Socket] Failed to emit event '{}': {}", event_name, e);
                    }
                } else {
                    eprintln!(
                        "[Socket] Warning: method '{}' not in allowed whitelist, ignoring",
                        method
                    );
                }
            }
        }
    }
}

/// Disconnect from the socket server
#[tauri::command]
pub async fn disconnect_socket(state: State<'_, SocketState>) -> Result<(), String> {
    // Clone Arc references
    let reader_task_arc = state.reader_task.clone();
    let processor_task_arc = state.processor_task.clone();
    let writer_arc = state.writer.clone();
    let pending_arc = state.pending.clone();
    let connected_arc = state.connected.clone();
    let writer_dead_arc = state.writer_dead.clone();

    // Cancel reader task
    {
        let mut task = reader_task_arc.lock().await;
        if let Some(handle) = task.take() {
            handle.abort();
        }
    }

    // Cancel message processor task
    {
        let mut task = processor_task_arc.lock().await;
        if let Some(handle) = task.take() {
            handle.abort();
        }
    }

    // Close writer with timeout
    {
        let mut writer = writer_arc.lock().await;
        if let Some(mut w) = writer.take() {
            let shutdown_result = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                w.shutdown()
            ).await;

            if shutdown_result.is_err() {
                eprintln!("[Socket] Warning: writer shutdown timed out after 5 seconds");
            }
        }
    }

    // Fail all pending requests so callers get an error instead of a silent channel close
    {
        let mut pending = pending_arc.write().await;
        for (_id, req) in pending.drain() {
            let _ = req.response_tx.send(Err("Disconnected by client".to_string())).await;
        }
    }

    // Mark as disconnected and reset writer_dead flag
    {
        let mut connected = connected_arc.write().await;
        *connected = false;
    }
    {
        let mut writer_dead = writer_dead_arc.write().await;
        *writer_dead = false;
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
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let writer_arc = state.writer.clone();
    let writer_dead_arc = state.writer_dead.clone();

    check_connection(&connected_arc, &writer_dead_arc).await?;

    let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);
    let (tx, mut rx) = mpsc::channel(1);

    {
        let mut pending = pending_arc.write().await;
        pending.insert(id, PendingRequest { response_tx: tx });
    }

    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params,
        id,
    };

    let request_json = serde_json::to_string(&request)
        .map_err(|e| {
            let pending_arc = pending_arc.clone();
            tokio::spawn(async move {
                match pending_arc.try_write() {
                    Ok(mut pending) => { pending.remove(&id); }
                    Err(_) => {
                        eprintln!(
                            "[Socket] Warning: could not acquire pending lock to clean up \
                             request {} during serialization error",
                            id
                        );
                    }
                }
            });
            format!("Failed to serialize request: {}", e)
        })?;

    if let Err(e) = write_to_socket(&writer_arc, &writer_dead_arc, &request_json).await {
        let mut pending = pending_arc.write().await;
        pending.remove(&id);
        return Err(e);
    }

    // Wait for response with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(30), rx.recv()).await {
        Ok(Some(result)) => result,
        Ok(None) => {
            // Clean up pending request
            let mut pending = pending_arc.write().await;
            pending.remove(&id);
            Err("Response channel closed".to_string())
        },
        Err(_) => {
            // Remove pending request on timeout
            let mut pending = pending_arc.write().await;
            pending.remove(&id);
            Err("Request timed out after 30 seconds".to_string())
        }
    }
}

/// Batch request item
#[derive(Deserialize, Debug)]
pub struct BatchRequestItem {
    method: String,
    params: serde_json::Value,
}

/// Batch response item
#[derive(Serialize, Debug)]
pub struct BatchResponseItem {
    result: Option<serde_json::Value>,
    error: Option<String>,
}

/// Send multiple JSON-RPC messages as a batch.
/// Writes all requests to the socket, then waits for all responses concurrently.
/// This overlaps response wait times for better throughput.
#[tauri::command]
pub async fn send_batch(
    requests: Vec<BatchRequestItem>,
    state: State<'_, SocketState>,
) -> Result<Vec<BatchResponseItem>, String> {
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let writer_arc = state.writer.clone();
    let writer_dead_arc = state.writer_dead.clone();

    check_connection(&connected_arc, &writer_dead_arc).await?;

    // Phase 1a: Serialize all requests and register pending receivers (no lock needed)
    let mut prepared = Vec::with_capacity(requests.len());
    let mut receivers = Vec::with_capacity(requests.len());

    for req in &requests {
        let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: req.method.clone(),
            params: req.params.clone(),
            id,
        };

        match serde_json::to_string(&request) {
            Ok(json) => {
                let (tx, rx) = mpsc::channel(1);
                {
                    let mut pending = pending_arc.write().await;
                    pending.insert(id, PendingRequest { response_tx: tx });
                }
                prepared.push(Some((id, json)));
                receivers.push(Some((id, rx)));
            }
            Err(e) => {
                eprintln!("[Socket] Failed to serialize batch request '{}': {}", req.method, e);
                prepared.push(None);
                receivers.push(None);
            }
        }
    }

    // Phase 1b: Hold writer lock once for all writes + flush (prevents interleaving)
    {
        let mut writer_guard = writer_arc.lock().await;
        if let Some(writer) = writer_guard.as_mut() {
            for (i, item) in prepared.iter().enumerate() {
                if let Some((_id, json)) = item {
                    let result = async {
                        writer.write_all(json.as_bytes()).await?;
                        writer.write_all(b"\n").await?;
                        Ok::<(), std::io::Error>(())
                    }.await;

                    if let Err(e) = result {
                        eprintln!("[Socket] Batch write failed: {} (writer marked dead)", e);
                        let mut writer_dead = writer_dead_arc.write().await;
                        *writer_dead = true;
                        // Clean up pending for this and all remaining requests
                        let mut pending = pending_arc.write().await;
                        for item in prepared.iter().skip(i) {
                            if let Some((id, _)) = item {
                                pending.remove(id);
                            }
                        }
                        // Mark remaining receivers as failed
                        for receiver in receivers.iter_mut().skip(i) {
                            *receiver = None;
                        }
                        break;
                    }
                }
            }

            // Flush all written requests
            let is_dead = *writer_dead_arc.read().await;
            if !is_dead {
                if let Err(e) = writer.flush().await {
                    eprintln!("[Socket] Batch flush failed: {} (writer marked dead)", e);
                    let mut writer_dead = writer_dead_arc.write().await;
                    *writer_dead = true;
                }
            }
        } else {
            // Writer not available - fail all pending requests
            let mut pending = pending_arc.write().await;
            for item in &prepared {
                if let Some((id, _)) = item {
                    pending.remove(id);
                }
            }
            receivers.iter_mut().for_each(|r| *r = None);
        }
    }

    // Phase 2: Wait for all responses concurrently
    let futures: Vec<_> = receivers
        .into_iter()
        .map(|receiver| {
            let pending_arc = pending_arc.clone();
            async move {
                match receiver {
                    Some((id, mut rx)) => {
                        match tokio::time::timeout(
                            std::time::Duration::from_secs(30),
                            rx.recv(),
                        )
                        .await
                        {
                            Ok(Some(Ok(value))) => {
                                BatchResponseItem { result: Some(value), error: None }
                            }
                            Ok(Some(Err(e))) => {
                                BatchResponseItem { result: None, error: Some(e) }
                            }
                            Ok(None) => {
                                let mut pending = pending_arc.write().await;
                                pending.remove(&id);
                                BatchResponseItem {
                                    result: None,
                                    error: Some("Response channel closed".to_string()),
                                }
                            }
                            Err(_) => {
                                let mut pending = pending_arc.write().await;
                                pending.remove(&id);
                                BatchResponseItem {
                                    result: None,
                                    error: Some("Request timed out".to_string()),
                                }
                            }
                        }
                    }
                    None => BatchResponseItem {
                        result: None,
                        error: Some("Failed to send request".to_string()),
                    },
                }
            }
        })
        .collect();

    let results = join_all(futures).await;

    Ok(results)
}

/// Send a streaming request (response comes via events)
#[tauri::command]
pub async fn send_streaming_message(
    method: String,
    params: serde_json::Value,
    state: State<'_, SocketState>,
) -> Result<u64, String> {
    let connected_arc = state.connected.clone();
    let writer_arc = state.writer.clone();
    let writer_dead_arc = state.writer_dead.clone();

    check_connection(&connected_arc, &writer_dead_arc).await?;

    let id = REQUEST_ID.fetch_add(1, Ordering::SeqCst);

    // Streaming requests don't register in pending - tokens arrive as notifications
    // and the frontend correlates them by request_id.

    let mut params_with_stream = params;
    if let Some(obj) = params_with_stream.as_object_mut() {
        obj.insert("stream".to_string(), serde_json::Value::Bool(true));
    }

    let request = JsonRpcRequest {
        jsonrpc: "2.0",
        method,
        params: params_with_stream,
        id,
    };

    let request_json = serde_json::to_string(&request)
        .map_err(|e| format!("Failed to serialize request: {}", e))?;

    write_to_socket(&writer_arc, &writer_dead_arc, &request_json).await?;

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

/// Internal connect logic that can be called by reconnect
async fn connect_socket_internal(
    app: &AppHandle,
    state: &SocketState,
) -> Result<bool, String> {
    // Clone Arc references out of state to avoid lifetime issues
    let writer_arc = state.writer.clone();
    let connected_arc = state.connected.clone();
    let pending_arc = state.pending.clone();
    let reader_task_arc = state.reader_task.clone();
    let processor_task_arc = state.processor_task.clone();
    let writer_dead_arc = state.writer_dead.clone();

    // Abort any existing reader task to prevent stale readers
    {
        let mut task = reader_task_arc.lock().await;
        if let Some(handle) = task.take() {
            handle.abort();
        }
    }

    // Abort any existing processor task
    {
        let mut task = processor_task_arc.lock().await;
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

    // Reset connected flag and writer dead flag
    {
        let mut connected = connected_arc.write().await;
        *connected = false;
    }
    {
        let mut writer_dead = writer_dead_arc.write().await;
        *writer_dead = false;
    }

    // Connect to socket
    let socket_path = get_socket_path()?;
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

    // Create a bounded channel for backpressure (max 500 messages queued)
    const MSG_CHANNEL_CAPACITY: usize = 500;
    let (msg_tx, mut msg_rx) = mpsc::channel::<JsonRpcMessage>(MSG_CHANNEL_CAPACITY);

    // Spawn reader task
    let pending_for_reader = pending_arc.clone();
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
                        // Monitor channel backpressure
                        let queued = MSG_CHANNEL_CAPACITY - msg_tx.capacity();
                        let threshold = MSG_CHANNEL_CAPACITY * 80 / 100;
                        if queued > threshold {
                            eprintln!(
                                "[Socket] Warning: message channel {:.0}% full ({}/{})",
                                (queued as f64 / MSG_CHANNEL_CAPACITY as f64) * 100.0,
                                queued,
                                MSG_CHANNEL_CAPACITY
                            );
                        }

                        // Send to processing channel with backpressure
                        if msg_tx.send(msg).await.is_err() {
                            eprintln!("[Socket] Message processing channel closed");
                            break;
                        }
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

        // Fail all pending requests
        {
            let mut pending = pending_for_reader.write().await;
            for (id, req) in pending.drain() {
                let _ = req.response_tx.send(Err(format!("Connection lost (request ID: {})", id))).await;
            }
        }

        // Emit disconnected event - frontend handles reconnection
        println!("[Socket] Connection lost, notifying frontend for reconnection");
        if let Err(e) = app_handle.emit("jarvis:disconnected", ()) {
            eprintln!("[Socket] Failed to emit disconnected event: {}", e);
        }
    });

    // Spawn message processor task
    let pending_for_processor = pending_arc.clone();
    let app_for_processor = app.clone();
    let processor_handle = tokio::spawn(async move {
        while let Some(msg) = msg_rx.recv().await {
            handle_message(msg, &pending_for_processor, &app_for_processor).await;
        }
    });

    // Store reader task handle
    {
        let mut task = reader_task_arc.lock().await;
        *task = Some(reader_handle);
    }

    // Store processor task handle
    {
        let mut task = processor_task_arc.lock().await;
        *task = Some(processor_handle);
    }

    // Emit connected event
    if let Err(e) = app.emit("jarvis:connected", ()) {
        eprintln!("[Socket] Failed to emit connected event: {}", e);
    }

    Ok(true)
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
