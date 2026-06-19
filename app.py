import json
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent import build_agent
from src.db_utils import get_schema

load_dotenv()

st.set_page_config(
    page_title="Assistente Virtual de Dados - FRANQ",
    page_icon="💹",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }
    .main .block-container {
        animation: fadeUp 0.6s ease-out;
        padding-top: 2.5rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }
    div[data-testid="stTextInput"] input {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(255,255,255,0.12) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96,165,250,0.15) !important;
        background: rgba(255,255,255,0.08) !important;
    }
    div[data-testid="stTextInput"] input::placeholder { color: rgba(255,255,255,0.3) !important; }
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59,130,246,0.5) !important;
        filter: brightness(1.1) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 9px !important; font-weight: 500 !important; }
    .stTabs [aria-selected="true"] { background: rgba(96,165,250,0.15) !important; color: #60a5fa !important; }
    .stCode, [data-testid="stCode"] { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
    div[data-testid="stAlert"] { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    [data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
    .response-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        animation: fadeUp 0.4s ease-out;
        line-height: 1.7;
    }
    .metric-row {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-top: 0.75rem;
    }
    .msg-user {
        background: rgba(96,165,250,0.1); border: 1px solid rgba(96,165,250,0.2);
        border-radius: 14px 14px 4px 14px; padding: 0.75rem 1.1rem;
        margin-bottom: 0.5rem; font-size: 0.95rem; color: #bfdbfe;
    }
    .msg-assistant {
        background: rgba(167,139,250,0.08); border: 1px solid rgba(167,139,250,0.15);
        border-radius: 14px 14px 14px 4px; padding: 0.75rem 1.1rem;
        margin-bottom: 1.25rem; font-size: 0.9rem; color: #ddd6fe; line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_agent():
    return build_agent()


@st.cache_data
def _get_db_schema() -> dict:
    """Cache the raw schema dict for the sidebar display."""
    return get_schema()


# ── Pure helper functions ─────────────────────────────────────────────────────

def _is_rate_limit_error(text: str) -> bool:
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _extract_content(content) -> str:
    """Normalise Gemini content — may be str or list of content blocks."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content) if content else ""


_TEMPORAL_KEYWORDS = {"data", "date", "mes", "ano", "year", "month", "periodo", "semana", "week", "dia", "trimestre"}
_CURRENCY_KEYWORDS = {"valor", "receita", "preco", "total", "gasto", "ticket", "media", "avg", "sum", "preco"}


def _detect_chart(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Return (x_col, y_col, chart_type) or None if no chart is appropriate."""
    if len(df) < 2:
        return None
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()
    if not numeric_cols or not text_cols:
        return None
    x_col, y_col = text_cols[0], numeric_cols[0]
    is_temporal = any(kw in x_col.lower() for kw in _TEMPORAL_KEYWORDS)
    return x_col, y_col, "line" if is_temporal else "bar"


def _build_column_config(df: pd.DataFrame) -> dict:
    """Build st.column_config entries so currency columns display with R$ prefix."""
    config = {}
    for col in df.select_dtypes(include="number").columns:
        col_key = col.lower().replace("(", " ").replace(")", " ")
        is_currency = any(kw in col_key for kw in _CURRENCY_KEYWORDS)
        config[col] = (
            st.column_config.NumberColumn(col, format="R$ %.2f")
            if is_currency
            else st.column_config.NumberColumn(col)
        )
    return config


def _get_follow_ups(question: str, response: str) -> list[str]:
    """Generate 3 follow-up suggestions. Returns [] on any failure."""
    try:
        from langchain_core.messages import SystemMessage
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        result = llm.invoke([
            SystemMessage(content=(
                "Based on the conversation, suggest exactly 3 short follow-up questions "
                "the user might ask next, in Brazilian Portuguese. "
                "Return only the questions, one per line, no numbering or bullets."
            )),
            HumanMessage(content=f"Pergunta: {question}\n\nResposta: {response[:500]}"),
        ])
        content = _extract_content(result.content)
        return [q.strip() for q in content.strip().split("\n") if q.strip()][:3]
    except Exception:  # noqa: BLE001
        return []


def _extract_token_usage(messages: list) -> dict[str, int]:
    """Sum token counts across all AIMessages that include usage_metadata."""
    inp = out = 0
    for msg in messages:
        if isinstance(msg, AIMessage):
            meta = getattr(msg, "usage_metadata", None) or {}
            inp += meta.get("input_tokens", 0)
            out += meta.get("output_tokens", 0)
    return {"input": inp, "output": out, "total": inp + out}


_FEEDBACK_LOG = Path(__file__).parent / "data" / "feedback.json"


def _save_feedback(question: str, response: str, is_positive: bool) -> None:
    """Persist feedback to a local JSON log for quality monitoring."""
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
        pass  # Non-critical: never block the UI on file write failures


# ── Streaming execution ───────────────────────────────────────────────────────

def _run_with_streaming(messages: list) -> tuple[dict | None, list, bool]:
    """
    Invoke the ReAct agent with streaming so each reasoning step appears in real
    time inside an st.status() container.

    Returns (parsed_result, new_messages, was_rate_limited).
    was_rate_limited=True signals the retry wrapper to wait and try again.
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
                                    final_response = content

                        elif isinstance(msg, ToolMessage):
                            step = pending.get(msg.tool_call_id)
                            if step:
                                step["output"] = msg.content
                            is_error = msg.content.startswith("ERROR")
                            is_schema = step and step["tool"] == "get_database_schema"
                            if is_error:
                                st.caption(f"⚠️ {msg.content[:180]}")
                            elif is_schema:
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

            label = "✅ Análise concluída!" if final_response else "⚠️ Análise finalizada"
            status.update(label=label, state="complete", expanded=False)

    except Exception as exc:  # noqa: BLE001
        was_limited = _is_rate_limit_error(str(exc))
        if not was_limited:
            st.error(f"Ocorreu um erro inesperado.\n\n`{exc}`")
        return None, new_messages, was_limited

    return {
        "final_response": final_response,
        "tool_steps": tool_steps,
        "last_sql_rows": last_sql_rows,
    }, new_messages, False


def _run_with_retry(messages: list) -> tuple[dict | None, list]:
    """
    Wraps _run_with_streaming with up to 2 automatic retries when a 429
    rate-limit error occurs. Shows a live countdown between attempts so
    the user knows the system is still working.
    """
    MAX_RETRIES = 2
    WAIT_SECONDS = [30, 60]

    for attempt in range(MAX_RETRIES + 1):
        result, new_msgs, was_limited = _run_with_streaming(messages)

        if result is not None:
            return result, new_msgs

        if not was_limited or attempt == MAX_RETRIES:
            if was_limited:
                st.warning(
                    "⚠️ **Limite de requisições esgotado após múltiplas tentativas.**\n\n"
                    "Aguarde alguns minutos e tente novamente."
                )
            return None, new_msgs

        # Show animated countdown before retrying
        wait = WAIT_SECONDS[attempt]
        placeholder = st.empty()
        for remaining in range(wait, 0, -1):
            placeholder.info(
                f"⏳ Limite de requisições atingido — "
                f"tentativa {attempt + 2}/{MAX_RETRIES + 1} em **{remaining}s**..."
            )
            time.sleep(1)
        placeholder.empty()

    return None, []


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_chart(df: pd.DataFrame) -> None:
    chart = _detect_chart(df)
    if not chart:
        return
    x_col, y_col, chart_type = chart
    st.divider()
    st.caption("📊 Visualização dos dados")
    layout = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Inter"),
    )
    if chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, template="plotly_dark", markers=True)
        fig.update_traces(line=dict(color="#60a5fa", width=2.5))
    else:
        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark", color=y_col, color_continuous_scale="Blues")
        layout["coloraxis_showscale"] = False
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(elapsed: float, tool_steps: list, new_messages: list) -> None:
    """Show a compact execution summary row below the result tabs."""
    tokens = _extract_token_usage(new_messages)
    tool_count = len(tool_steps)
    sql_count = sum(1 for s in tool_steps if s["tool"] == "execute_sql_query")

    st.markdown('<div class="metric-row">', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("⏱️ Tempo", f"{elapsed:.1f}s")
    cols[1].metric("🔧 Ferramentas", tool_count)
    cols[2].metric("🗄️ Queries SQL", sql_count)
    token_label = f"{tokens['total']:,}" if tokens["total"] else "—"
    cols[3].metric("🪙 Tokens", token_label)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_feedback(question: str, response: str, response_idx: int) -> None:
    """Thumbs-up / thumbs-down feedback row persisted in session state + JSON log."""
    current = st.session_state.feedback.get(response_idx)

    if current is None:
        col1, col2, col3 = st.columns([1, 1, 8])
        if col1.button("👍", key=f"up_{response_idx}", help="Resposta útil"):
            st.session_state.feedback[response_idx] = True
            _save_feedback(question, response, is_positive=True)
            st.rerun()
        if col2.button("👎", key=f"down_{response_idx}", help="Resposta incorreta"):
            st.session_state.feedback[response_idx] = False
            _save_feedback(question, response, is_positive=False)
            st.rerun()
        col3.caption("Esta resposta foi útil?")
    elif current:
        st.caption("👍 Obrigado! Feedback registrado.")
    else:
        st.caption("👎 Obrigado! Usaremos para melhorar.")


def _render_result(agent_output: dict, question: str, response_idx: int) -> None:
    """Render polished response + reasoning tabs after streaming completes."""
    final_response = agent_output["final_response"]
    tool_steps = agent_output["tool_steps"]
    last_sql_rows = agent_output["last_sql_rows"]

    tab_response, tab_reasoning = st.tabs(["💬 Resposta", "🔍 Raciocínio do Agente"])

    with tab_response:
        if final_response:
            st.markdown(
                f'<div class="response-card">{final_response}</div>',
                unsafe_allow_html=True,
            )
            st.divider()
            _render_feedback(question, final_response, response_idx)
        else:
            st.info("Nenhuma resposta foi gerada para esta pergunta.")

        if last_sql_rows:
            df = pd.DataFrame(last_sql_rows)
            _render_chart(df)
            st.divider()
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config=_build_column_config(df),
            )
            st.caption(f"{len(df)} registro(s) — ordenável por qualquer coluna")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Exportar dados como CSV",
                data=csv_bytes,
                file_name="resultado.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab_reasoning:
        if not tool_steps:
            st.info("Nenhuma ferramenta foi chamada — pergunta respondida diretamente.")
            return
        st.caption(f"O agente executou **{len(tool_steps)}** ação(ões).")
        st.divider()
        for idx, step in enumerate(tool_steps, 1):
            if step["tool"] == "get_database_schema":
                with st.expander(f"**Passo {idx} — Consultar estrutura do banco**", expanded=False):
                    st.code(step["output"], language="text")
            elif step["tool"] == "execute_sql_query":
                sql = step["args"].get("sql", "")
                fmt_sql = sqlparse.format(sql, reindent=True, keyword_case="upper", indent_width=4)
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
                                st.dataframe(
                                    df_step,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config=_build_column_config(df_step),
                                )
                                st.caption(f"{len(df_step)} registro(s)")
                            else:
                                st.info(output)
                        except json.JSONDecodeError:
                            st.text(output)


# ── Session state ─────────────────────────────────────────────────────────────

if "conversation_messages" not in st.session_state:
    st.session_state.conversation_messages: list = []

if "history" not in st.session_state:
    st.session_state.history: list = []

if "follow_ups" not in st.session_state:
    st.session_state.follow_ups: list[str] = []

if "auto_question" not in st.session_state:
    st.session_state.auto_question = ""

if "feedback" not in st.session_state:
    st.session_state.feedback: dict[int, bool] = {}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🗂️ Histórico")
    ctx_count = len([m for m in st.session_state.conversation_messages if isinstance(m, HumanMessage)])
    if ctx_count > 0:
        st.caption(f"🧠 **{ctx_count}** pergunta(s) em memória")
        if st.button("🗑️ Nova Conversa", use_container_width=True):
            st.session_state.conversation_messages = []
            st.session_state.history = []
            st.session_state.follow_ups = []
            st.session_state.feedback = {}
            st.rerun()
    else:
        st.info("Nenhuma consulta realizada ainda.")

    if st.session_state.history:
        st.divider()
        for entry in reversed(st.session_state.history):
            label = entry["question"][:55] + "..." if len(entry["question"]) > 55 else entry["question"]
            with st.expander(f"**{label}**"):
                st.markdown(entry["response"])

    # Database schema reference — shows tables and columns to guide the user
    st.divider()
    st.markdown("#### 🗄️ Banco de Dados")
    try:
        db_schema = _get_db_schema()
        for table_name, columns in db_schema.items():
            with st.expander(f"**{table_name}** ({len(columns)} colunas)"):
                for col in columns:
                    pk_badge = " 🔑" if col["pk"] else ""
                    st.caption(f"`{col['name']}` — {col['type']}{pk_badge}")
    except Exception:  # noqa: BLE001
        st.caption("Schema indisponível.")

# ── Page header ───────────────────────────────────────────────────────────────

st.title("💹 Assistente Virtual de Dados")
st.caption("Faça perguntas em português sobre os dados da FRANQ. O agente mantém contexto entre perguntas.")
st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────

question = st.text_input(
    label="Sua pergunta",
    placeholder="Ex: Quais campanhas de marketing tiveram interação dos clientes?",
    label_visibility="collapsed",
    key="question_input",
)
submitted = st.button("Enviar", type="primary", use_container_width=True)

# Follow-up suggestion buttons set auto_question and trigger a rerun;
# on this rerun the block below treats it like a normal submission.
if st.session_state.auto_question:
    question = st.session_state.auto_question
    st.session_state.auto_question = ""
    submitted = True

# ── Agent execution ───────────────────────────────────────────────────────────

if submitted:
    if not question.strip():
        st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        st.session_state.follow_ups = []
        messages_in = st.session_state.conversation_messages + [HumanMessage(content=question.strip())]

        st.divider()

        start_time = time.monotonic()
        agent_output, new_messages = _run_with_retry(messages_in)
        elapsed = time.monotonic() - start_time

        if agent_output is not None:
            st.session_state.conversation_messages = messages_in + new_messages

            response_idx = len(st.session_state.history)  # stable key before appending
            _render_result(agent_output, question.strip(), response_idx)
            _render_metrics(elapsed, agent_output["tool_steps"], new_messages)

            response_text = agent_output["final_response"]
            if response_text and not _is_rate_limit_error(response_text):
                st.session_state.history.append({"question": question.strip(), "response": response_text})
                follow_ups = _get_follow_ups(question.strip(), response_text)
                if follow_ups:
                    st.session_state.follow_ups = follow_ups

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
