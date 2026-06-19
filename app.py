import json

import pandas as pd
import plotly.express as px
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.agent import build_agent

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

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

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
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        color: #e2e8f0 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 3px rgba(96, 165, 250, 0.15) !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: rgba(255, 255, 255, 0.3) !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
        filter: brightness(1.1) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(96, 165, 250, 0.15) !important;
        color: #60a5fa !important;
    }

    .stCode, [data-testid="stCode"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(8px) !important;
    }

    hr { border-color: rgba(255, 255, 255, 0.08) !important; }

    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    .response-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        animation: fadeUp 0.4s ease-out;
        line-height: 1.7;
    }

    .suggestion-btn {
        background: rgba(96, 165, 250, 0.08) !important;
        border: 1px solid rgba(96, 165, 250, 0.2) !important;
        border-radius: 10px !important;
        color: #93c5fd !important;
        font-size: 0.88rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
    }
    .suggestion-btn:hover {
        background: rgba(96, 165, 250, 0.15) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
    }

    .msg-user {
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.2);
        border-radius: 14px 14px 4px 14px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
        color: #bfdbfe;
    }
    .msg-assistant {
        background: rgba(167, 139, 250, 0.08);
        border: 1px solid rgba(167, 139, 250, 0.15);
        border-radius: 14px 14px 14px 4px;
        padding: 0.75rem 1.1rem;
        margin-bottom: 1.25rem;
        font-size: 0.9rem;
        color: #ddd6fe;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def get_agent():
    return build_agent()


def _is_rate_limit_error(text: str) -> bool:
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _extract_content(content) -> str:
    """Normalise Gemini content — may be str or list of content blocks."""
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content) if content else ""


_TEMPORAL_KEYWORDS = {"data", "date", "mes", "ano", "year", "month", "periodo", "semana", "week", "dia", "trimestre"}


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


def _get_follow_ups(question: str, response: str) -> list[str]:
    """
    Generate 3 short follow-up questions using a lightweight LLM call.
    Returns an empty list on any failure so the feature degrades gracefully.
    """
    try:
        from langchain_core.messages import SystemMessage
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
        result = llm.invoke([
            SystemMessage(content=(
                "Based on the conversation below, suggest exactly 3 short follow-up questions "
                "the user might ask next, in Brazilian Portuguese. "
                "Return only the questions, one per line, no numbering or bullets."
            )),
            HumanMessage(content=f"Pergunta: {question}\n\nResposta: {response[:500]}"),
        ])
        content = _extract_content(result.content)
        return [q.strip() for q in content.strip().split("\n") if q.strip()][:3]
    except Exception:  # noqa: BLE001
        return []


# ── Streaming execution ───────────────────────────────────────────────────────

def _run_with_streaming(messages: list) -> tuple[dict | None, list]:
    """
    Invoke the ReAct agent with streaming so each reasoning step appears in real
    time inside an st.status() container. Returns (parsed_result, new_messages).

    stream_mode="updates" gives one event per node execution, each containing
    only the messages that node appended — perfect for incremental display.
    """
    agent = get_agent()
    tool_steps: list[dict] = []
    pending: dict[str, dict] = {}   # tool_call_id → step dict for matching ToolMessages
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

            label = "✅ Análise concluída!" if final_response else "⚠️ Análise finalizada sem resposta"
            status.update(label=label, state="complete", expanded=False)

    except Exception as exc:  # noqa: BLE001
        if _is_rate_limit_error(str(exc)):
            st.warning(
                "⚠️ **Limite de requisições atingido temporariamente.**\n\n"
                "Estamos na camada gratuita da API Gemini. "
                "Aguarde **30 a 60 segundos** e tente novamente."
            )
        else:
            st.error(f"Ocorreu um erro inesperado.\n\n`{exc}`")
        return None, new_messages

    return {
        "final_response": final_response,
        "tool_steps": tool_steps,
        "last_sql_rows": last_sql_rows,
    }, new_messages


# ── Result rendering ──────────────────────────────────────────────────────────

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


def _render_result(agent_output: dict) -> None:
    """Render the polished response + reasoning tabs after streaming completes."""
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
        else:
            st.info("Nenhuma resposta foi gerada para esta pergunta.")

        if last_sql_rows:
            df = pd.DataFrame(last_sql_rows)
            _render_chart(df)
            st.divider()

            # CSV export
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
            st.info("Nenhuma ferramenta foi utilizada (pergunta respondida sem consultar o banco).")
            return

        st.caption(f"O agente executou **{len(tool_steps)}** ação(ões).")
        st.divider()

        for idx, step in enumerate(tool_steps, 1):
            tool_name = step["tool"]
            args = step["args"]
            output = step["output"]

            if tool_name == "get_database_schema":
                with st.expander(f"**Passo {idx} — Consultar estrutura do banco**", expanded=False):
                    st.code(output, language="text")

            elif tool_name == "execute_sql_query":
                sql = args.get("sql", "")
                fmt_sql = sqlparse.format(sql, reindent=True, keyword_case="upper", indent_width=4)
                with st.expander(f"**Passo {idx} — Executar consulta SQL**", expanded=True):
                    st.code(fmt_sql, language="sql")
                    if output.startswith("ERROR"):
                        st.error(output)
                    else:
                        try:
                            rows = json.loads(output)
                            if isinstance(rows, list) and rows:
                                df_step = pd.DataFrame(rows)
                                st.dataframe(df_step, use_container_width=True, hide_index=True)
                                st.caption(f"{len(df_step)} registro(s)")
                            else:
                                st.info(output)
                        except json.JSONDecodeError:
                            st.text(output)


# ── Session state ─────────────────────────────────────────────────────────────

if "conversation_messages" not in st.session_state:
    # Preserves the full message history across questions for conversational memory
    st.session_state.conversation_messages: list = []

if "history" not in st.session_state:
    st.session_state.history: list = []  # sidebar display: [{question, response}]

if "follow_ups" not in st.session_state:
    st.session_state.follow_ups: list[str] = []

if "auto_question" not in st.session_state:
    st.session_state.auto_question = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🗂️ Histórico")

    ctx_count = len([m for m in st.session_state.conversation_messages if isinstance(m, HumanMessage)])
    if ctx_count > 0:
        st.caption(f"🧠 **{ctx_count}** pergunta(s) em memória — o agente tem contexto de toda a sessão.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Limpar", use_container_width=True, help="Apaga histórico e memória"):
                st.session_state.conversation_messages = []
                st.session_state.history = []
                st.session_state.follow_ups = []
                st.rerun()
    else:
        st.info("Nenhuma consulta realizada ainda.")

    if st.session_state.history:
        st.divider()
        for entry in reversed(st.session_state.history):
            label = entry["question"][:55] + "..." if len(entry["question"]) > 55 else entry["question"]
            with st.expander(f"**{label}**"):
                st.markdown(entry["response"])

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

# A follow-up suggestion button sets auto_question and triggers a rerun;
# on this rerun we treat it the same as a normal submission.
if st.session_state.auto_question:
    question = st.session_state.auto_question
    st.session_state.auto_question = ""
    submitted = True

# ── Agent execution ───────────────────────────────────────────────────────────

if submitted:
    if not question.strip():
        st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        st.session_state.follow_ups = []  # clear stale suggestions

        # Append new question to the full conversation history
        messages_in = st.session_state.conversation_messages + [HumanMessage(content=question.strip())]

        st.divider()
        agent_output, new_messages = _run_with_streaming(messages_in)

        if agent_output is not None:
            # Persist the complete updated history (enables follow-up context)
            st.session_state.conversation_messages = messages_in + new_messages

            _render_result(agent_output)

            response_text = agent_output["final_response"]

            if response_text and not _is_rate_limit_error(response_text):
                st.session_state.history.append(
                    {"question": question.strip(), "response": response_text}
                )

                # Generate follow-up suggestions (degrades gracefully on quota errors)
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
