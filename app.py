import json
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import build_agent
from src.db_utils import DB_PATH, get_schema
from src.llm import get_llm, get_provider_label, is_local_provider

load_dotenv()

st.set_page_config(
    page_title="Assistente Virtual de Dados - FRANQ",
    page_icon="💹",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    background-attachment: fixed;
}
.main .block-container {
    animation: fadeUp 0.5s ease-out;
    padding-top: 2rem;
    max-width: 860px;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:translateY(0); } }
h1 {
    font-size: 2rem !important; font-weight: 700 !important;
    background: linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -0.5px; margin-bottom: 0.25rem !important;
}
/* ── Input ── */
div[data-testid="stTextInput"] input {
    background:rgba(255,255,255,0.05)!important; border:1.5px solid rgba(255,255,255,0.12)!important;
    border-radius:14px!important; padding:14px 18px!important; font-size:1rem!important;
    color:#e2e8f0!important; transition:all 0.3s ease!important; backdrop-filter:blur(8px);
}
div[data-testid="stTextInput"] input:focus {
    border-color:#60a5fa!important; box-shadow:0 0 0 3px rgba(96,165,250,0.15)!important;
    background:rgba(255,255,255,0.08)!important;
}
div[data-testid="stTextInput"] input::placeholder { color:rgba(255,255,255,0.3)!important; }
/* ── Primary button ── */
div[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#3b82f6,#8b5cf6)!important; border:none!important;
    border-radius:14px!important; font-weight:600!important; padding:0.7rem 1.5rem!important;
    transition:all 0.25s ease!important; box-shadow:0 4px 15px rgba(59,130,246,0.3)!important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    transform:translateY(-2px)!important; box-shadow:0 8px 25px rgba(59,130,246,0.5)!important;
}
/* ── Secondary buttons (welcome cards, suggestions) ── */
div[data-testid="stButton"] button[kind="secondary"] {
    background:rgba(255,255,255,0.04)!important;
    border:1px solid rgba(255,255,255,0.1)!important;
    border-radius:12px!important; font-size:0.85rem!important;
    color:#cbd5e1!important; transition:all 0.2s ease!important;
    white-space:normal!important; text-align:left!important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background:rgba(96,165,250,0.08)!important;
    border-color:rgba(96,165,250,0.3)!important;
    color:#e2e8f0!important;
}
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:rgba(255,255,255,0.04)!important; border-radius:12px!important;
    padding:4px!important; border:1px solid rgba(255,255,255,0.08)!important; gap:4px!important;
}
.stTabs [data-baseweb="tab"] { border-radius:9px!important; font-weight:500!important; }
.stTabs [aria-selected="true"] { background:rgba(96,165,250,0.15)!important; color:#60a5fa!important; }
/* ── Misc elements ── */
.stCode,[data-testid="stCode"] { border-radius:12px!important; border:1px solid rgba(255,255,255,0.08)!important; }
div[data-testid="stAlert"] { border-radius:12px!important; border:1px solid rgba(255,255,255,0.08)!important; }
hr { border-color:rgba(255,255,255,0.07)!important; margin:0.75rem 0!important; }
[data-testid="stDataFrame"] { border-radius:12px!important; overflow:hidden!important; }
/* ── Response card ── */
.response-card {
    background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
    border-radius:16px; padding:1.5rem 1.75rem; backdrop-filter:blur(12px);
    box-shadow:0 8px 32px rgba(0,0,0,0.2); animation:fadeUp 0.4s ease-out; line-height:1.7;
    font-size:0.97rem;
}
/* ── Metrics row — rendered as a single HTML block to avoid empty-div artefacts ── */
.metric-row {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:1rem 1.5rem; margin-top:0.25rem;
    display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem;
}
.metric-item { text-align:center; }
.metric-label {
    font-size:0.70rem; color:#94a3b8; text-transform:uppercase;
    letter-spacing:0.06em; margin-bottom:6px;
}
.metric-value { font-size:1.55rem; font-weight:700; color:#e2e8f0; line-height:1.1; }
.metric-value.good { color:#34d399; }
.metric-value.zero { color:#475569; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────

_SESSION_FILE = Path("data/session.json")
_FEEDBACK_LOG = Path("data/feedback.json")
_CONTEXT_THRESHOLD = 50   # messages before summarisation kicks in
_KEEP_RECENT = 20         # how many recent messages to preserve

_TEMPORAL_KW = {"data", "date", "mes", "ano", "year", "month", "periodo", "semana", "week", "dia", "trimestre"}
_CURRENCY_KW = {"valor", "receita", "preco", "total", "gasto", "ticket", "media", "avg", "sum"}

_EXAMPLES = [
    {"icon": "👑", "title": "Top gastadores",     "q": "Quais são os 5 clientes que mais gastaram?"},
    {"icon": "📦", "title": "Receita por categoria", "q": "Qual é a receita total por categoria de compra?"},
    {"icon": "📣", "title": "Campanhas efetivas",  "q": "Quais campanhas de marketing tiveram mais interações?"},
    {"icon": "🆘", "title": "Suporte pendente",   "q": "Quantos chamados de suporte estão sem resolução?"},
    {"icon": "🏙️", "title": "Clientes por cidade", "q": "Quais as 5 cidades com mais clientes?"},
    {"icon": "📅", "title": "Compras por mês",    "q": "Qual foi a receita total de compras por mês?"},
]


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_agent():
    return build_agent()


@st.cache_data
def _get_db_schema() -> dict:
    return get_schema()


@st.cache_data
def _get_table_preview(table_name: str) -> list[dict]:
    """Return the first 3 rows of a table for the sidebar schema preview."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")  # noqa: S608
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except (sqlite3.Error, OSError):
        return []


# ── Session persistence ───────────────────────────────────────────────────────

def _load_session() -> dict:
    """Load history and feedback from disk so they survive page refreshes."""
    if _SESSION_FILE.exists():
        try:
            return json.loads(_SESSION_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"history": [], "feedback": {}}


def _save_session() -> None:
    """Persist current history and feedback to disk (non-blocking on failure)."""
    try:
        data = {
            "history": st.session_state.history,
            "feedback": {str(k): v for k, v in st.session_state.feedback.items()},
            "saved_at": pd.Timestamp.now().isoformat(),
        }
        _SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _rebuild_memory(history: list[dict]) -> list:
    """
    Reconstruct minimal agent-compatible conversation memory from the saved
    history entries. Full tool-call messages are not restored (they're not
    stored), but Q&A pairs give the LLM enough context to understand
    follow-up questions after a page refresh.
    """
    messages = []
    for entry in history:
        messages.append(HumanMessage(content=entry["question"]))
        messages.append(AIMessage(content=entry["response"]))
    return messages


def _save_feedback(question: str, response: str, is_positive: bool) -> None:
    """Append a single feedback record to the JSON log."""
    try:
        entries: list = json.loads(_FEEDBACK_LOG.read_text(encoding="utf-8")) if _FEEDBACK_LOG.exists() else []
        entries.append({
            "timestamp": pd.Timestamp.now().isoformat(),
            "question": question,
            "response_excerpt": response[:300],
            "positive": is_positive,
        })
        _FEEDBACK_LOG.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


# ── AI helpers ────────────────────────────────────────────────────────────────

def _extract_content(content) -> str:
    """Normalise Gemini content — may be str or list of content blocks."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content) if content else ""


def _is_rate_limit_error(text: str) -> bool:
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _maybe_summarize_context(messages: list) -> list:
    """
    When the conversation grows beyond _CONTEXT_THRESHOLD messages, summarise
    the older portion into a single context block. This keeps token usage stable
    regardless of session length while preserving the most recent interactions.
    """
    if len(messages) <= _CONTEXT_THRESHOLD:
        return messages

    older, recent = messages[:-_KEEP_RECENT], messages[-_KEEP_RECENT:]

    # Build a readable digest of the older messages (only human/AI text, skip tool calls)
    digest_lines = []
    for msg in older:
        if isinstance(msg, HumanMessage):
            digest_lines.append(f"Usuário: {_extract_content(msg.content)}")
        elif isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", []):
            digest_lines.append(f"Assistente: {_extract_content(msg.content)[:200]}")

    if not digest_lines:
        return recent

    try:
        llm = get_llm(temperature=0)
        result = llm.invoke([
            SystemMessage(content="Resuma este histórico de conversa de forma concisa em português, mantendo os fatos e números-chave."),
            HumanMessage(content="\n".join(digest_lines)),
        ])
        summary = _extract_content(result.content)
        summary_msg = HumanMessage(content=f"[CONTEXTO RESUMIDO DE INTERAÇÕES ANTERIORES]\n{summary}")
        return [summary_msg] + recent
    except Exception:  # noqa: BLE001
        return recent  # on failure, silently drop the older messages


def _get_follow_ups(question: str, response: str) -> list[str]:
    """Generate 3 follow-up suggestions. Returns [] on any failure."""
    try:
        llm = get_llm(temperature=0.7)
        result = llm.invoke([
            SystemMessage(content="Based on the conversation, suggest exactly 3 short follow-up questions in Brazilian Portuguese. One per line, no numbering."),
            HumanMessage(content=f"Pergunta: {question}\n\nResposta: {response[:500]}"),
        ])
        content = _extract_content(result.content)
        return [q.strip() for q in content.strip().split("\n") if q.strip()][:3]
    except Exception:  # noqa: BLE001
        return []


def _extract_token_usage(messages: list) -> dict[str, int]:
    inp = out = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            meta = getattr(msg, "usage_metadata", None) or {}
            inp += meta.get("input_tokens", 0)
            out += meta.get("output_tokens", 0)
    return {"input": inp, "output": out, "total": inp + out}


# ── Data analysis helpers ─────────────────────────────────────────────────────

def _build_column_config(df: pd.DataFrame) -> dict:
    """
    Build column config for st.dataframe. Currency columns are expected to
    already be pre-formatted as strings by _build_display_df, so they get a
    TextColumn here. Non-currency numerics get a plain NumberColumn.
    """
    config = {}
    for col in df.columns:
        if col in df.select_dtypes(include="number").columns:
            config[col] = st.column_config.NumberColumn(col, format="%.2f")
        elif any(kw in col.lower() for kw in _CURRENCY_KW):
            # Pre-formatted string currency column
            config[col] = st.column_config.TextColumn(col)
    return config


def _detect_outliers(df: pd.DataFrame) -> list[str]:
    """IQR-based outlier detection — returns human-readable notes per column."""
    notes = []
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        count = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if count:
            notes.append(f"⚠️ `{col}`: {count} valor(es) fora do padrão detectado(s) (método IQR)")
    return notes


def _clean_response(text: str) -> str:
    """
    Sanitise the raw LLM output before displaying it:
    1. Strip lines that are predominantly non-Latin characters (Thai, Cyrillic,
       CJK, etc.). Small local models sometimes reason in another language before
       switching to Portuguese — we want only the Portuguese output.
    2. Convert LaTeX math notation to readable plain text.
    """
    # ── Step 1: remove non-Latin lines ────────────────────────────────────────
    # Characters in these Unicode blocks are never part of Portuguese text.
    _NON_LATIN = re.compile(
        r'[\u0400-\u04FF'   # Cyrillic (Russian, Bulgarian, etc.)
        r'\u0E00-\u0E7F'   # Thai
        r'\u3000-\u9FFF'   # CJK / Japanese / Korean
        r'\uAC00-\uD7AF'   # Hangul
        r'\u0600-\u06FF'   # Arabic
        r'\u0900-\u097F]'  # Devanagari (Hindi)
    )
    clean_lines = []
    for line in text.split('\n'):
        non_latin = len(_NON_LATIN.findall(line))
        # Skip if more than 20 % of the line's characters are non-Latin
        if len(line) > 0 and non_latin / len(line) > 0.20:
            continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines).strip()

    # ── Step 2: LaTeX cleanup ──────────────────────────────────────────────────
    text = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', text)
    text = text.replace(r'\times', '×').replace(r'\cdot', '·')
    text = text.replace(r'\approx', '≈').replace(r'\geq', '≥').replace(r'\leq', '≤')
    text = re.sub(r'\\\(|\\\)', '', text)
    text = re.sub(r'(?<!\$)\$(?!\$)', '', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)

    return text.strip()


def _format_brl(value) -> str:
    """
    Format a numeric value as Brazilian currency: R$ 1.234,56
    Swaps US-style separators (1,234.56) to Brazilian (1.234,56).
    """
    try:
        us = f"{float(value):,.2f}"          # "1,234.56"
        br = us.replace(",", "§").replace(".", ",").replace("§", ".")  # "1.234,56"
        return f"R$ {br}"
    except (TypeError, ValueError):
        return str(value)


def _build_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a display copy of the DataFrame with currency columns pre-formatted
    as Brazilian strings. Numeric-only columns keep their type for sorting.
    """
    out = df.copy()
    for col in df.select_dtypes(include="number").columns:
        if any(kw in col.lower() for kw in _CURRENCY_KW):
            out[col] = df[col].apply(lambda v: _format_brl(v) if pd.notna(v) else "")
    return out


def _export_conversation_md() -> str:
    lines = [
        "# Conversa — Assistente Virtual de Dados FRANQ\n",
        f"*Exportado em {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}*\n",
        "---\n",
    ]
    for i, entry in enumerate(st.session_state.history, 1):
        lines.append(f"## Pergunta {i}\n\n{entry['question']}\n")
        lines.append(f"### Resposta\n\n{entry['response']}\n\n---\n")
    return "\n".join(lines)


# ── Streaming execution ───────────────────────────────────────────────────────

def _run_with_streaming(messages: list) -> tuple[dict | None, list, bool]:
    """
    Stream the ReAct agent, displaying each reasoning step in real time.
    Returns (parsed_result, new_messages, was_rate_limited).
    """
    agent = get_agent()
    tool_steps: list[dict] = []
    pending: dict[str, dict] = {}
    last_sql_rows: list[dict] = []
    final_response = ""
    new_messages: list = []

    try:
        with st.status("🧠 Analisando sua pergunta...", expanded=True) as status:
            for event in agent.stream({"messages": messages}, stream_mode="updates"):
                for _node, update in event.items():
                    for msg in update.get("messages", []):
                        new_messages.append(msg)

                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    step: dict = {"tool": tc["name"], "args": tc["args"], "output": ""}
                                    tool_steps.append(step)
                                    pending[tc["id"]] = step
                                    if tc["name"] == "get_database_schema":
                                        st.write("📋 Consultando estrutura do banco de dados...")
                                    elif tc["name"] == "execute_sql_query":
                                        sql = tc["args"].get("sql", "")
                                        fmt = sqlparse.format(sql, reindent=True, keyword_case="upper", indent_width=4)
                                        st.write("⚙️ Executando consulta SQL:")
                                        st.code(fmt, language="sql")
                            else:
                                content = _extract_content(msg.content)
                                if content:
                                    final_response = _clean_response(content)

                        elif isinstance(msg, ToolMessage):
                            step = pending.get(msg.tool_call_id)
                            if step:
                                step["output"] = msg.content
                            if msg.content.startswith("ERROR"):
                                st.caption(f"⚠️ {msg.content[:180]}")
                            elif step and step["tool"] == "get_database_schema":
                                st.caption("✅ Schema obtido com sucesso")
                            else:
                                try:
                                    rows = json.loads(msg.content)
                                    if isinstance(rows, list) and rows:
                                        last_sql_rows = rows
                                        st.caption(f"✅ {len(rows)} registro(s) retornado(s)")
                                    else:
                                        st.caption(f"✅ {msg.content[:120]}")
                                except json.JSONDecodeError:
                                    st.caption(f"✅ {msg.content[:120]}")

            has_tools = bool(tool_steps)
            if final_response:
                label = (
                    f"✅ Análise concluída — {len(tool_steps)} etapa(s) executada(s) ▾"
                    if has_tools
                    else "✅ Resposta direta (sem consulta ao banco)"
                )
            else:
                label = "⚠️ Análise finalizada sem resposta"
            # Keep expanded when tools were used so the user sees the reasoning steps
            status.update(label=label, state="complete", expanded=has_tools)

    except Exception as exc:  # noqa: BLE001
        was_limited = _is_rate_limit_error(str(exc))
        if not was_limited:
            st.error(f"Ocorreu um erro inesperado.\n\n`{exc}`")
        return None, new_messages, was_limited

    return {"final_response": final_response, "tool_steps": tool_steps, "last_sql_rows": last_sql_rows}, new_messages, False


def _run_with_retry(messages: list) -> tuple[dict | None, list]:
    """Wraps _run_with_streaming with up to 2 automatic retries on 429 errors."""
    WAIT_SECONDS = [30, 60]
    for attempt in range(3):
        result, new_msgs, was_limited = _run_with_streaming(messages)
        if result is not None:
            return result, new_msgs
        if not was_limited or attempt == 2:
            if was_limited:
                st.warning("⚠️ **Limite de requisições esgotado.** Aguarde alguns minutos e tente novamente.")
            return None, new_msgs
        wait = WAIT_SECONDS[attempt]
        ph = st.empty()
        for t in range(wait, 0, -1):
            ph.info(f"⏳ Limite de requisições — tentativa {attempt + 2}/3 em **{t}s**...")
            time.sleep(1)
        ph.empty()
    return None, []


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_chart(df: pd.DataFrame, result_key: str = "chart") -> None:
    """
    Render an interactive chart with a type selector so the user can switch
    between bar, line, pie, and scatter without re-running the query.
    The chart type auto-detected from column names is pre-selected.
    """
    if len(df) < 2:
        return

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    # Determine which chart types make sense for this dataset
    options: list[str] = []
    if numeric_cols and text_cols:
        options += ["Barras", "Linha", "Pizza"]
    if len(numeric_cols) >= 2:
        options.append("Dispersão")

    if not options:
        return

    # Auto-detect a sensible default
    default = "Linha" if text_cols and any(kw in text_cols[0].lower() for kw in _TEMPORAL_KW) else "Barras"
    if default not in options:
        default = options[0]

    st.divider()
    st.caption("📊 Visualização dos dados")

    chart_type = st.radio(
        "Tipo de gráfico:",
        options,
        index=options.index(default),
        horizontal=True,
        key=f"chart_type_{result_key}",
    )

    common_layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Inter"),
    )
    x_col = text_cols[0] if text_cols else numeric_cols[0]
    y_col = numeric_cols[0]

    if chart_type == "Barras":
        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark", color=y_col, color_continuous_scale="Blues")
        common_layout["coloraxis_showscale"] = False

    elif chart_type == "Linha":
        fig = px.line(df, x=x_col, y=y_col, template="plotly_dark", markers=True)
        fig.update_traces(line=dict(color="#60a5fa", width=2.5))

    elif chart_type == "Pizza":
        if len(df) > 15:
            st.caption("ℹ️ Muitas categorias — considere usar barras para melhor legibilidade.")
        fig = px.pie(df, names=x_col, values=y_col, template="plotly_dark", hole=0.35)

    elif chart_type == "Dispersão":
        col1, col2 = st.columns(2)
        x_scatter = col1.selectbox("Eixo X:", numeric_cols, key=f"scatter_x_{result_key}")
        y_scatter = col2.selectbox("Eixo Y:", numeric_cols, index=min(1, len(numeric_cols) - 1), key=f"scatter_y_{result_key}")
        color_opt = ["Nenhum"] + text_cols
        color_sel = st.selectbox("Cor por:", color_opt, key=f"scatter_c_{result_key}")
        fig = px.scatter(
            df, x=x_scatter, y=y_scatter,
            color=None if color_sel == "Nenhum" else color_sel,
            template="plotly_dark", trendline="ols" if len(df) >= 5 else None,
        )
    else:
        return

    fig.update_layout(**common_layout)
    # Disable select/lasso tools — they trigger Streamlit reruns that clear the page.
    # Keep zoom/pan/hover which are the useful interactions.
    plotly_config = {
        "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        "displaylogo": False,
        "scrollZoom": False,
    }
    st.plotly_chart(fig, use_container_width=True, config=plotly_config)


def _render_stats(df: pd.DataFrame) -> None:
    """Show a compact statistical summary for all numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty or len(df) < 3:
        return

    with st.expander("📊 Resumo estatístico", expanded=False):
        stats = numeric_df.agg(["min", "max", "mean", "sum"]).round(2)
        stats.index = ["Mínimo", "Máximo", "Média", "Total"]
        st.dataframe(stats, use_container_width=True)


def _render_feedback(question: str, response: str, response_idx: int) -> None:
    current = st.session_state.feedback.get(response_idx)
    if current is None:
        col1, col2, col3 = st.columns([1, 1, 8])
        if col1.button("👍", key=f"up_{response_idx}", help="Resposta útil"):
            st.session_state.feedback[response_idx] = True
            _save_feedback(question, response, is_positive=True)
            _save_session()
            st.rerun()
        if col2.button("👎", key=f"down_{response_idx}", help="Resposta incorreta"):
            st.session_state.feedback[response_idx] = False
            _save_feedback(question, response, is_positive=False)
            _save_session()
            st.rerun()
        col3.caption("Esta resposta foi útil?")
    elif current:
        st.caption("👍 Obrigado! Feedback registrado.")
    else:
        st.caption("👎 Obrigado! Usaremos para melhorar.")


def _render_metrics(elapsed: float, tool_steps: list, new_messages: list) -> None:
    tokens = _extract_token_usage(new_messages)
    sql_count = sum(1 for s in tool_steps if s["tool"] == "execute_sql_query")
    tool_count = len(tool_steps)

    if is_local_provider():
        token_label = "🖥️ Tokens (local)"
        token_display = f"{tokens['total']:,}" if tokens["total"] else "—"
        token_title = "Processado localmente — sem custo de API"
    else:
        token_label = "🪙 Tokens"
        token_display = f"{tokens['total']:,}" if tokens["total"] else "—"
        token_title = "Tokens consumidos via API Gemini"

    # Colour-code values: green for non-zero counts, muted for zeros
    def _cls(val): return "good" if val and val != "—" and str(val) != "0" else "zero"

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="metric-label">⏱️ Tempo</div>
            <div class="metric-value good">{elapsed:.1f}s</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">🔧 Ferramentas</div>
            <div class="metric-value {_cls(tool_count)}">{tool_count}</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">🗄️ Queries SQL</div>
            <div class="metric-value {_cls(sql_count)}">{sql_count}</div>
        </div>
        <div class="metric-item" title="{token_title}">
            <div class="metric-label">{token_label}</div>
            <div class="metric-value {_cls(token_display)}">{token_display}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_result(agent_output: dict, question: str, response_idx: int) -> None:
    final_response = agent_output["final_response"]
    tool_steps = agent_output["tool_steps"]
    last_sql_rows = agent_output["last_sql_rows"]

    tab_response, tab_reasoning = st.tabs(["💬 Resposta", "🔍 Raciocínio do Agente"])

    with tab_response:
        if final_response:
            st.markdown(f'<div class="response-card">{final_response}</div>', unsafe_allow_html=True)
            st.divider()
            _render_feedback(question, final_response, response_idx)
        else:
            st.info("Nenhuma resposta foi gerada para esta pergunta.")

        if last_sql_rows:
            df = pd.DataFrame(last_sql_rows)
            df_display = _build_display_df(df)  # currency cols pre-formatted as Brazilian strings

            # Chart uses original numeric df (not the formatted display copy)
            _render_chart(df, result_key=str(response_idx))

            # Statistical summary (uses original numeric df)
            _render_stats(df)

            # Outlier notes
            for note in _detect_outliers(df):
                st.caption(note)

            st.divider()
            st.dataframe(df_display, use_container_width=True, hide_index=True, column_config=_build_column_config(df_display))
            st.caption(f"{len(df)} registro(s) — clique nas colunas para ordenar")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Exportar dados como CSV", csv_bytes, "resultado.csv", "text/csv", use_container_width=True)

    with tab_reasoning:
        if not tool_steps:
            st.info("Nenhuma ferramenta foi chamada — resposta direta sem consultar o banco.")
            return
        st.caption(f"O agente executou **{len(tool_steps)}** ação(ões).")
        st.divider()
        for idx, step in enumerate(tool_steps, 1):
            if step["tool"] == "get_database_schema":
                with st.expander(f"**Passo {idx} — Consultar estrutura do banco**", expanded=False):
                    st.code(step["output"], language="text")
            elif step["tool"] == "execute_sql_query":
                fmt_sql = sqlparse.format(step["args"].get("sql", ""), reindent=True, keyword_case="upper", indent_width=4)
                with st.expander(f"**Passo {idx} — Executar consulta SQL**", expanded=True):
                    st.code(fmt_sql, language="sql")
                    output = step["output"]
                    if output.startswith("ERROR"):
                        st.error(output)
                    else:
                        try:
                            rows = json.loads(output)
                            if isinstance(rows, list) and rows:
                                df_step = pd.DataFrame(rows)
                                st.dataframe(df_step, use_container_width=True, hide_index=True, column_config=_build_column_config(df_step))
                                st.caption(f"{len(df_step)} registro(s)")
                            else:
                                st.info(output)
                        except json.JSONDecodeError:
                            st.text(output)


# ── Session state initialisation ──────────────────────────────────────────────

if "session_loaded" not in st.session_state:
    saved = _load_session()
    st.session_state.history: list = saved.get("history", [])
    st.session_state.feedback: dict = {int(k): v for k, v in saved.get("feedback", {}).items()}
    st.session_state.conversation_messages: list = _rebuild_memory(st.session_state.history)
    st.session_state.follow_ups: list[str] = []
    st.session_state.auto_question: str = ""
    # Persists the last agent result across reruns (e.g. chart type toggle, zoom interactions)
    st.session_state.last_result: dict | None = None
    st.session_state.session_loaded = True

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── History & controls ──
    st.markdown("## 🗂️ Histórico")
    ctx = len([m for m in st.session_state.conversation_messages if isinstance(m, HumanMessage)])
    if ctx:
        st.caption(f"🧠 **{ctx}** pergunta(s) em memória")
        if st.button("🗑️ Nova Conversa", use_container_width=True):
            st.session_state.conversation_messages = []
            st.session_state.history = []
            st.session_state.follow_ups = []
            st.session_state.feedback = {}
            st.session_state.last_result = None
            _save_session()
            st.rerun()
    else:
        st.info("Nenhuma consulta realizada ainda.")

    if st.session_state.history:
        st.divider()
        for entry in reversed(st.session_state.history):
            label = entry["question"][:55] + "..." if len(entry["question"]) > 55 else entry["question"]
            with st.expander(f"**{label}**"):
                st.markdown(entry["response"])

        st.divider()
        md = _export_conversation_md()
        st.download_button(
            "📄 Exportar conversa (Markdown)",
            md.encode("utf-8"),
            "conversa_franq.md",
            "text/markdown",
            use_container_width=True,
        )

    # ── Example queries ──
    st.divider()
    st.markdown("#### 💡 Consultas de exemplo")
    for ex in _EXAMPLES:
        if st.button(f"{ex['icon']} {ex['title']}", key=f"ex_{ex['title']}", use_container_width=True):
            st.session_state.auto_question = ex["q"]
            st.rerun()

    # ── Database schema with data preview ──
    st.divider()
    st.markdown("#### 🗄️ Banco de Dados")
    try:
        db_schema = _get_db_schema()
        for table_name, columns in db_schema.items():
            with st.expander(f"**{table_name}** ({len(columns)} colunas)"):
                for col in columns:
                    pk = " 🔑" if col["pk"] else ""
                    st.caption(f"`{col['name']}` — {col['type']}{pk}")
                preview = _get_table_preview(table_name)
                if preview:
                    st.markdown("**Prévia (3 linhas):**")
                    st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
    except (sqlite3.Error, OSError, KeyError):
        st.caption("Schema indisponível.")

# ── Page header ───────────────────────────────────────────────────────────────

st.title("💹 Assistente Virtual de Dados")
st.markdown(
    f'<p style="color:#94a3b8;font-size:0.9rem;margin-top:-0.5rem;margin-bottom:0">'
    f'Faça perguntas em português sobre os dados da FRANQ. '
    f'<span style="color:#60a5fa;font-weight:500">{get_provider_label()}</span></p>',
    unsafe_allow_html=True,
)
st.divider()

# ── Welcome screen (only when no history) ────────────────────────────────────

if not st.session_state.history:
    st.markdown('<p style="font-weight:600;font-size:1rem;margin-bottom:0.75rem">🚀 Comece com um exemplo</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, ex in enumerate(_EXAMPLES[:3]):
        with cols[i]:
            if st.button(
                f"{ex['icon']} {ex['title']}\n\n_{ex['q']}_",
                key=f"welcome_{i}",
                use_container_width=True,
            ):
                st.session_state.auto_question = ex["q"]
                st.rerun()
    st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────

question = st.text_input(
    label="Sua pergunta",
    placeholder="Ex: Quais campanhas de marketing tiveram interação dos clientes?",
    label_visibility="collapsed",
    key="question_input",
)
submitted = st.button("Enviar", type="primary", use_container_width=True)

if st.session_state.auto_question:
    question = st.session_state.auto_question
    st.session_state.auto_question = ""
    submitted = True

# ── Agent execution ───────────────────────────────────────────────────────────
# Only runs when the user submits a new question. Results are stored in
# session_state.last_result so they survive reruns from chart interactions.

if submitted:
    if not question.strip():
        st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        st.session_state.follow_ups = []
        st.session_state.last_result = None   # clear previous result while processing
        messages_in = st.session_state.conversation_messages + [HumanMessage(content=question.strip())]
        messages_in = _maybe_summarize_context(messages_in)

        st.divider()
        start = time.monotonic()
        agent_output, new_messages = _run_with_retry(messages_in)
        elapsed = time.monotonic() - start

        if agent_output is not None:
            st.session_state.conversation_messages = messages_in + new_messages

            response_idx = len(st.session_state.history)
            # Persist so reruns (chart toggle, zoom, scroll) don't lose the results
            st.session_state.last_result = {
                "agent_output": agent_output,
                "question": question.strip(),
                "response_idx": response_idx,
                "elapsed": elapsed,
                "new_messages": new_messages,
            }

            response_text = agent_output["final_response"]
            if response_text and not _is_rate_limit_error(response_text):
                last_q = st.session_state.history[-1]["question"] if st.session_state.history else None
                if question.strip() != last_q:
                    st.session_state.history.append({"question": question.strip(), "response": response_text})
                    _save_session()
                follow_ups = _get_follow_ups(question.strip(), response_text)
                if follow_ups:
                    st.session_state.follow_ups = follow_ups

# ── Always render last result (persists across chart-toggle / zoom reruns) ────

if st.session_state.get("last_result"):
    lr = st.session_state.last_result
    # Only add divider if not already rendered inline during the submitted run above
    if not submitted:
        st.divider()
    _render_result(lr["agent_output"], lr["question"], lr["response_idx"])
    _render_metrics(lr["elapsed"], lr["agent_output"]["tool_steps"], lr["new_messages"])

# ── Follow-up suggestions ─────────────────────────────────────────────────────

if st.session_state.follow_ups:
    st.divider()
    st.caption("💡 **Perguntas sugeridas** — clique para enviar:")
    cols = st.columns(len(st.session_state.follow_ups))
    for i, suggestion in enumerate(st.session_state.follow_ups):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                st.session_state.auto_question = suggestion
                st.rerun()
