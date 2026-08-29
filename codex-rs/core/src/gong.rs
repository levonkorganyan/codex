//! Gong retrieval harness mode.
//!
//! When `CODEX_GONG_SIDECAR` is set, user turns bypass the model agent loop
//! entirely and run against the M12.22 Gong retrieval sidecar (a long-lived
//! Python process speaking one JSON object per line over stdio; see
//! `semantic_sql_workflow/harness/README.md` in the research repo). Every
//! visible update is translated deterministically from sidecar events into
//! ordinary turn items, so the stock TUI renders them: retrieval stages as
//! dynamic tool calls, the planned IR and ranked results as agent messages.
//!
//! Environment:
//! - `CODEX_GONG_SIDECAR`: whitespace-separated sidecar command line, e.g.
//!   `uv run harness/gong_sidecar.py --model-file ...`.
//! - `CODEX_GONG_CWD`: working directory for the sidecar (defaults to the
//!   current process cwd).

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use serde_json::json;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncWriteExt;
use tokio::io::BufReader;
use tokio::process::Child;
use tokio::process::ChildStdin;
use tokio::process::ChildStdout;
use tokio::process::Command;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::session::TurnInput;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::state::TaskKind;
use crate::tasks::SessionTask;
use crate::tasks::SessionTaskResult;
use codex_protocol::error::CodexErr;
use codex_protocol::items::AgentMessageContent;
use codex_protocol::items::AgentMessageItem;
use codex_protocol::items::McpToolCallItem;
use codex_protocol::items::McpToolCallStatus;
use codex_protocol::items::TurnItem;
use codex_protocol::mcp::CallToolResult;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::request_user_input::RequestUserInputArgs;
use codex_protocol::request_user_input::RequestUserInputQuestion;
use codex_protocol::request_user_input::RequestUserInputQuestionOption;
use codex_protocol::user_input::UserInput;

const SIDECAR_ENV: &str = "CODEX_GONG_SIDECAR";
/// Client-side control token carried at the front of the submitted text; the
/// TUI cannot reach core directly, so the run/debug mode rides the message.
pub const DEBUG_MODE_TOKEN: &str = "[gong:interactive] ";
/// Client-side control token selecting the fast (M2 agentic MCP search) engine.
pub const FAST_SEARCH_TOKEN: &str = "[gong:fast] ";
const SIDECAR_CWD_ENV: &str = "CODEX_GONG_CWD";
const PROTOCOL_VERSION: u64 = 1;

pub(crate) fn enabled() -> bool {
    std::env::var(SIDECAR_ENV).map(|v| !v.trim().is_empty()) == Ok(true)
}

fn stage_label(stage: &str) -> &'static str {
    match stage {
        "planning" => "Planning retrieval",
        "cross_type_entity_repair" => "Reconciling entity types",
        "indexed_entity_resolution" => "Resolving entities",
        "pre_binding_constraints" => "Applying identity & time constraints",
        "binding_resolution" => "Resolving structured bindings",
        "theme_resolution" => "Theme retrieval (BM25)",
        "finalization" => "Building candidate pool",
        "ranking_evidence" => "Hydrating candidate evidence",
        "ranking" => "Reranking candidates",
        "mcp_search" => "Searching Fivetran AI (hybrid)",
        _ => "Retrieval stage",
    }
}

fn fmt_bytes(value: &Value) -> Option<String> {
    let mut bytes = value.as_f64()?;
    if bytes <= 0.0 {
        return None;
    }
    for unit in ["B", "KB", "MB", "GB"] {
        if bytes < 1024.0 {
            return Some(format!("{bytes:.1} {unit}"));
        }
        bytes /= 1024.0;
    }
    Some(format!("{bytes:.1} TB"))
}

fn fmt_seconds(value: &Value) -> Option<String> {
    let ms = value.as_f64()?;
    if ms <= 0.0 {
        return None;
    }
    Some(format!("{:.1}s", ms / 1000.0))
}

fn stage_detail(event: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    if event["stage"] == "mcp_search" {
        if let Some(attempts) = event["search_attempts"].as_u64() {
            let plural = if attempts == 1 { "" } else { "es" };
            parts.push(format!("{attempts} search{plural}"));
        }
    } else if event["stage"] == "ranking" {
        if let (Some(cin), Some(cout)) = (
            event["candidates_in"].as_u64(),
            event["results_out"].as_u64(),
        ) {
            parts.push(format!("{cin} candidates → {cout} results"));
        }
        if event["selection_route"] == "full_llm_ranking" {
            parts.push("full LLM ranking".to_string());
        }
    } else {
        if let Some(jobs) = event["warehouse_jobs"].as_u64()
            && jobs > 0
        {
            let plural = if jobs == 1 { "" } else { "s" };
            parts.push(format!("{jobs} warehouse job{plural}"));
        }
        if let Some(latency) = fmt_seconds(&event["warehouse_latency_ms"]) {
            parts.push(latency);
        }
        if let Some(bytes) = fmt_bytes(&event["bytes_processed"]) {
            parts.push(bytes);
        }
    }
    parts.join(" · ")
}

fn plan_markdown(event: &Value) -> String {
    let mut lines = vec!["**Planned retrieval**".to_string()];
    let list = |key: &str, render: &dyn Fn(&Value) -> String| -> Option<String> {
        let items = event[key].as_array()?;
        if items.is_empty() {
            return None;
        }
        Some(
            items
                .iter()
                .map(render)
                .collect::<Vec<_>>()
                .join(" · "),
        )
    };
    if let Some(text) = list("entities", &|item| {
        format!(
            "{} ({})",
            item["value"].as_str().unwrap_or("?"),
            item["entity"].as_str().unwrap_or("?")
        )
    }) {
        lines.push(format!("- entities: {text}"));
    }
    if let Some(text) = list("time_windows", &|item| {
        format!(
            "{} → {}",
            item["start"].as_str().unwrap_or("…"),
            item["end_exclusive"].as_str().unwrap_or("…")
        )
    }) {
        lines.push(format!("- time: {text}"));
    }
    if let Some(text) = list("bindings", &|item| {
        format!(
            "{} = \"{}\"",
            item["member"].as_str().unwrap_or("?"),
            item["value"].as_str().unwrap_or("?")
        )
    }) {
        lines.push(format!("- bindings: {text}"));
    }
    if let Some(text) = list("themes", &|item| {
        format!("\"{}\"", item["value"].as_str().unwrap_or("?"))
    }) {
        lines.push(format!("- themes: {text}"));
    }
    lines.join("\n")
}

/// Human labels for the deterministic retrieval-signal member names.
fn friendly_signal(member: &str) -> String {
    match member {
        "calls.title" => "call title".to_string(),
        "calls.brief" => "call brief".to_string(),
        "calls.started" => "call date".to_string(),
        "users.full_name" | "users.identity" => "your name".to_string(),
        "users.email_address" => "your email".to_string(),
        "participants.name" => "participant name".to_string(),
        "participants.email_address" => "participant email".to_string(),
        "participants.email_domain" => "participant email domain".to_string(),
        "organizations.identity" => "organization".to_string(),
        "call_topic_occurrences.name" => "call topic".to_string(),
        "trackers.name" | "tracker_occurrences.name" => "tracker".to_string(),
        "outline_sections.section" | "outline_items.text" => "call outline".to_string(),
        "transcript_turns.sentence_text" => "transcript".to_string(),
        "bm25" => "keywords (BM25)".to_string(),
        "declared_index" | "theme_index:bm25" => "theme index".to_string(),
        other => other.replace('_', " ").replace('.', " "),
    }
}

fn results_markdown(event: &Value) -> String {
    let results = event["results"].as_array().cloned().unwrap_or_default();
    let mut summary: Vec<String> = Vec::new();
    if let Some(total) = event["total_candidates"].as_u64() {
        summary.push(format!("{} of {} candidates", results.len(), total));
    }
    if let Some(latency) = fmt_seconds(&event["latency_ms"]) {
        summary.push(latency);
    }
    if let Some(jobs) = event["warehouse_jobs"].as_u64() {
        summary.push(format!("{jobs} warehouse jobs"));
    }
    if let Some(bytes) = fmt_bytes(&event["bytes_processed"]) {
        summary.push(format!("{bytes} scanned"));
    }
    let mut lines = vec![format!("**Results**  ({})", summary.join(" · "))];
    if results.is_empty() {
        lines.push("_No calls returned._".to_string());
    }
    for row in &results {
        let rank = row["rank"].as_u64().unwrap_or(0);
        let title = row["title"].as_str().unwrap_or("(untitled call)");
        let date = row["started"]
            .as_str()
            .map(|s| s.chars().take(10).collect::<String>())
            .unwrap_or_default();
        match row["url"].as_str() {
            Some(url) => lines.push(format!("{rank}. **[{title}]({url})** — {date}")),
            None => lines.push(format!("{rank}. **{title}** — {date}")),
        }
        if let Some(excerpt) = row["brief_excerpt"].as_str()
            && !excerpt.is_empty()
        {
            lines.push(format!("   *Brief:* {excerpt}"));
        }
        let matched = row["matched"]
            .as_array()
            .map(|signals| {
                let mut labels: Vec<String> = signals
                    .iter()
                    .filter_map(Value::as_str)
                    .map(friendly_signal)
                    .collect();
                labels.dedup();
                labels.join(" · ")
            })
            .unwrap_or_default();
        if !matched.is_empty() {
            lines.push(format!("   *Matched on:* {matched}"));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

fn stage_item(tool: &str, arguments: Value) -> McpToolCallItem {
    McpToolCallItem {
        id: Uuid::new_v4().to_string(),
        server: "gong".to_string(),
        tool: tool.to_string(),
        arguments,
        connector_id: None,
        mcp_app_resource_uri: None,
        link_id: None,
        app_name: None,
        action_name: None,
        plugin_id: None,
        read_only_hint: Some(true),
        status: McpToolCallStatus::InProgress,
        result: None,
        error: None,
        duration: None,
    }
}

fn complete_stage_item(mut item: McpToolCallItem, detail: String) -> McpToolCallItem {
    item.status = McpToolCallStatus::Completed;
    item.result = Some(CallToolResult {
        content: vec![json!({"type": "text", "text": detail})],
        structured_content: None,
        is_error: Some(false),
        meta: None,
    });
    item
}

/// One live sidecar process, kept warm across turns.
struct Sidecar {
    child: Child,
    stdin: ChildStdin,
    lines: tokio::io::Lines<BufReader<ChildStdout>>,
}

static SIDECAR: Mutex<Option<Sidecar>> = Mutex::const_new(None);

async fn spawn_sidecar() -> std::io::Result<Sidecar> {
    let command_line = std::env::var(SIDECAR_ENV).unwrap_or_default();
    let mut parts = command_line.split_whitespace();
    let program = parts.next().ok_or_else(|| {
        std::io::Error::other(format!("{SIDECAR_ENV} is empty"))
    })?;
    let mut command = Command::new(program);
    command.args(parts);
    if let Ok(cwd) = std::env::var(SIDECAR_CWD_ENV) {
        command.current_dir(cwd);
    }
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    let stdin = child.stdin.take().expect("sidecar stdin piped");
    let stdout = child.stdout.take().expect("sidecar stdout piped");
    let mut lines = BufReader::new(stdout).lines();
    loop {
        let Some(line) = lines.next_line().await? else {
            return Err(std::io::Error::other(
                "gong sidecar exited before reporting ready",
            ));
        };
        let Ok(event) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if event["event"] == "ready" {
            return Ok(Sidecar {
                child,
                stdin,
                lines,
            });
        }
    }
}

pub(crate) struct GongTask;

impl GongTask {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl SessionTask for GongTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Regular
    }

    fn span_name(&self) -> &'static str {
        "session_task.gong"
    }

    async fn run(
        self: Arc<Self>,
        sess: Arc<Session>,
        ctx: Arc<TurnContext>,
        input: Vec<TurnInput>,
        cancellation_token: CancellationToken,
    ) -> SessionTaskResult {
        let event = EventMsg::TurnStarted(TurnStartedEvent {
            turn_id: ctx.sub_id.clone(),
            trace_id: ctx.trace_id.clone(),
            started_at: ctx.turn_timing_state.started_at_unix_secs().await,
            model_context_window: ctx.model_context_window(),
            collaboration_mode_kind: ctx.mode(),
        });
        sess.send_event(ctx.as_ref(), event).await;

        let mut question = extract_question(&input);
        let mut mode = "run";
        let mut search = "deep";
        loop {
            if let Some(rest) = question.strip_prefix(DEBUG_MODE_TOKEN) {
                question = rest.trim_start().to_string();
                mode = "interactive";
            } else if let Some(rest) = question.strip_prefix(FAST_SEARCH_TOKEN) {
                question = rest.trim_start().to_string();
                search = "fast";
            } else {
                break;
            }
        }
        if question.is_empty() {
            let text = "Ask a Gong retrieval question in natural language.".to_string();
            emit_agent_message(&sess, &ctx, &text).await;
            return Ok(Some(text));
        }

        let mut guard = SIDECAR.lock().await;
        if guard.is_none() || guard.as_mut().is_some_and(sidecar_exited) {
            *guard = None;
            match spawn_sidecar().await {
                Ok(sidecar) => *guard = Some(sidecar),
                Err(err) => {
                    let text = format!("Gong sidecar failed to start: {err}");
                    emit_agent_message(&sess, &ctx, &text).await;
                    return Ok(Some(text));
                }
            }
        }
        let sidecar = guard.as_mut().expect("sidecar just ensured");

        let turn_id = Uuid::new_v4().to_string();
        let ask = json!({
            "v": PROTOCOL_VERSION,
            "op": "ask",
            "id": turn_id,
            "question": question,
            "mode": mode,
            "search": search,
        });
        if let Err(err) = write_line(&mut sidecar.stdin, &ask).await {
            *guard = None;
            let text = format!("Gong sidecar is unreachable: {err}");
            emit_agent_message(&sess, &ctx, &text).await;
            return Ok(Some(text));
        }

        let mut open_stage: Option<(String, McpToolCallItem)> = None;
        let mut cancelled = false;
        // Terminal text is already emitted as an AgentMessage turn item, so the
        // task returns no final message: returning it again would make clients
        // that print the final message (for example `codex exec`) show it twice.
        let _terminal_text: Option<String> = loop {
            let line = tokio::select! {
                line = sidecar.lines.next_line() => line,
                _ = cancellation_token.cancelled(), if !cancelled => {
                    cancelled = true;
                    let cancel = json!({
                        "v": PROTOCOL_VERSION,
                        "op": "cancel",
                        "id": turn_id,
                    });
                    let _ = write_line(&mut sidecar.stdin, &cancel).await;
                    continue;
                }
            };
            let line = match line {
                Ok(Some(line)) => line,
                Ok(None) | Err(_) => {
                    *guard = None;
                    if cancelled {
                        return Err(CodexErr::TurnAborted);
                    }
                    let text = "Gong sidecar exited unexpectedly.".to_string();
                    emit_agent_message(&sess, &ctx, &text).await;
                    return Ok(Some(text));
                }
            };
            let Ok(event) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if event["id"] != json!(turn_id.clone()) {
                continue;
            }
            match event["event"].as_str().unwrap_or_default() {
                "stage_begin" => {
                    let stage = event["stage"].as_str().unwrap_or_default().to_string();
                    let item = stage_item(stage_label(&stage), json!({"stage": stage}));
                    sess.emit_turn_item_started(
                        ctx.as_ref(),
                        &TurnItem::McpToolCall(item.clone()),
                    )
                    .await;
                    open_stage = Some((stage, item));
                }
                "stage_end" => {
                    if let Some((_, item)) = open_stage.take() {
                        let mut completed = complete_stage_item(item, stage_detail(&event));
                        if let Some(ms) = event["wall_latency_ms"].as_f64() {
                            completed.duration =
                                Some(Duration::from_millis(ms.max(0.0) as u64));
                        }
                        sess.emit_turn_item_completed(
                            ctx.as_ref(),
                            TurnItem::McpToolCall(completed),
                        )
                        .await;
                    }
                }
                "plan" => {
                    emit_agent_message(&sess, &ctx, &plan_markdown(&event)).await;
                }
                "pool" => {
                    if let Some(count) = event["candidate_count"].as_u64() {
                        emit_completed_tool(
                            &sess,
                            &ctx,
                            "Candidate pool",
                            &format!("{count} calls"),
                        )
                        .await;
                    }
                }
                "choice" => {
                    let selected = request_choice_from_user(&sess, &ctx, &event).await;
                    let reply = json!({
                        "v": PROTOCOL_VERSION,
                        "op": "choose",
                        "id": turn_id,
                        "choice_id": event["choice_id"],
                        "selected": selected,
                    });
                    if write_line(&mut sidecar.stdin, &reply).await.is_err() {
                        *guard = None;
                        let text = "Gong sidecar is unreachable.".to_string();
                        emit_agent_message(&sess, &ctx, &text).await;
                        return Ok(Some(text));
                    }
                }
                "results" => {
                    let text = results_markdown(&event);
                    emit_agent_message(&sess, &ctx, &text).await;
                    break Some(text);
                }
                "abstain" => {
                    let text = format!(
                        "**Need more to go on.** {}",
                        event["message"].as_str().unwrap_or_default()
                    );
                    emit_agent_message(&sess, &ctx, &text).await;
                    break Some(text);
                }
                "error" => {
                    if cancelled {
                        return Err(CodexErr::TurnAborted);
                    }
                    let text = format!(
                        "Retrieval failed: {}",
                        event["message"].as_str().unwrap_or("unknown error")
                    );
                    emit_agent_message(&sess, &ctx, &text).await;
                    break Some(text);
                }
                _ => {}
            }
        };
        if cancelled {
            return Err(CodexErr::TurnAborted);
        }
        Ok(None)
    }
}

/// Present a sidecar choice through codex's option picker and map the answer
/// back to an option index. None keeps the workflow's recommended default.
async fn request_choice_from_user(
    sess: &Session,
    ctx: &TurnContext,
    event: &Value,
) -> Option<usize> {
    let options: Vec<(String, String)> = event["options"]
        .as_array()
        .map(|options| {
            options
                .iter()
                .map(|option| {
                    let label = option["label"].as_str().unwrap_or("(option)").to_string();
                    let description =
                        option["description"].as_str().unwrap_or_default().to_string();
                    (label, description)
                })
                .collect()
        })
        .unwrap_or_default();
    if options.is_empty() {
        return None;
    }
    let question_id = event["choice_id"].as_str().unwrap_or("gong-choice").to_string();
    let args = RequestUserInputArgs {
        questions: vec![RequestUserInputQuestion {
            id: question_id.clone(),
            header: "Disambiguate".to_string(),
            question: event["prompt"]
                .as_str()
                .unwrap_or("Which did you mean?")
                .to_string(),
            is_other: false,
            is_secret: false,
            options: Some(
                options
                    .iter()
                    .map(|(label, description)| RequestUserInputQuestionOption {
                        label: label.clone(),
                        description: description.clone(),
                    })
                    .collect(),
            ),
        }],
        is_blocking: true,
        auto_resolution_ms: None,
    };
    let response = sess
        .request_user_input(ctx, Uuid::new_v4().to_string(), args)
        .await?;
    let answer = response.answers.get(&question_id)?;
    let chosen = answer.answers.first()?;
    options.iter().position(|(label, _)| label == chosen)
}

fn sidecar_exited(sidecar: &mut Sidecar) -> bool {
    matches!(sidecar.child.try_wait(), Ok(Some(_)) | Err(_))
}

fn extract_question(input: &[TurnInput]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for item in input {
        if let TurnInput::UserInput { content, .. } = item {
            for user_input in content {
                if let UserInput::Text { text, .. } = user_input {
                    parts.push(text.clone());
                }
            }
        }
    }
    parts.join("\n").trim().to_string()
}

async fn write_line(stdin: &mut ChildStdin, value: &Value) -> std::io::Result<()> {
    stdin.write_all(value.to_string().as_bytes()).await?;
    stdin.write_all(b"\n").await?;
    stdin.flush().await
}

async fn emit_agent_message(sess: &Session, ctx: &TurnContext, text: &str) {
    let item = TurnItem::AgentMessage(AgentMessageItem {
        id: Uuid::new_v4().to_string(),
        content: vec![AgentMessageContent::Text {
            text: text.to_string(),
        }],
        phase: None,
        memory_citation: None,
        delivery: None,
    });
    sess.emit_turn_item_started(ctx, &item).await;
    sess.emit_turn_item_completed(ctx, item).await;
}

async fn emit_completed_tool(sess: &Session, ctx: &TurnContext, tool: &str, detail: &str) {
    let item = stage_item(tool, json!({}));
    sess.emit_turn_item_started(ctx, &TurnItem::McpToolCall(item.clone()))
        .await;
    let completed = complete_stage_item(item, detail.to_string());
    sess.emit_turn_item_completed(ctx, TurnItem::McpToolCall(completed))
        .await;
}
