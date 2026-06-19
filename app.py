import json

import pandas as pd
import plotly.express as px
import sqlparse
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage

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

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
    }

    /* ── Fade-in on load ── */
    .main .block-container {
        animation: fadeUp 0.6s ease-out;
        padding-top: 2.5rem;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Hero title ── */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
    }

    /* ── Input field ── */
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

    /* ── Primary send button ── */
    div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.3px !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
        filter: brightness(1.1) !important;
    }
    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(0px) !important;
    }

    /* ── Tabs ── */
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

    /* ── Code block (SQL) ── */
    .stCode, [data-testid="stCode"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    /* ── Alert boxes ── */
    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(8px) !important;
    }

    /* ── Divider ── */
    hr { border-color: rgba(255, 255, 255, 0.08) !important; }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ── Response card ── */
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

    /* ── Reasoning step card ── */
    .step-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 3px solid #60a5fa;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
    .step-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #60a5fa;
        margin-bottom: 0.4rem;
    }

    /* ── History message bubbles ── */
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


@st.cache_resource
def get_agent():
    return build_agent()


def run_agent(user_input: str) -> dict:
    return get_agent().invoke({"messages": [("human", user_input)]})


# Keywords that suggest a column represents a time axis
_TEMPORAL_KEYWORDS = {"data", "date", "mes", "ano", "year", "month", "periodo", "semana", "week", "dia", "trimestre"}


def _detect_chart(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """Return (x_col, y_col, chart_type) or None if no chart is appropriate."""
    if len(df) < 2:
        return None

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not numeric_cols or not text_cols:
        return None

    x_col = text_cols[0]
    y_col = numeric_cols[0]

    is_temporal = any(kw in x_col.lower() for kw in _TEMPORAL_KEYWORDS)
    chart_type = "line" if is_temporal else "bar"

    return x_col, y_col, chart_type


def _is_rate_limit_error(text: str) -> bool:
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _parse_agent_result(result: dict) -> dict:
    """
    Extract the final answer, tool call steps, and last SQL dataset from the
    ReAct agent's message list. Each AIMessage with tool_calls represents one
    reasoning step; the subsequent ToolMessages carry the observations.
    """
    messages = result.get("messages", [])

    final_response = ""
    tool_steps: list[dict] = []
    last_sql_rows: list[dict] = []

    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    # Match each tool call with its ToolMessage observation
                    output = next(
                        (m.content for m in messages[i + 1 :] if isinstance(m, ToolMessage) and m.tool_call_id == tc["id"]),
                        "",
                    )
                    tool_steps.append({
                        "tool": tc["name"],
                        "args": tc["args"],
                        "output": output,
                    })

                    # Keep the last successful SQL result for the chart / table
                    if tc["name"] == "execute_sql_query" and output and not output.startswith("ERROR"):
                        try:
                            rows = json.loads(output)
                            if isinstance(rows, list) and rows:
                                last_sql_rows = rows
                        except json.JSONDecodeError:
                            pass
            else:
                # Final AIMessage has no tool_calls — this is the synthesised answer
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                if content:
                    final_response = content

    return {
        "final_response": final_response,
        "tool_steps": tool_steps,
        "last_sql_rows": last_sql_rows,
    }


def _render_result(agent_output: dict) -> None:
    """Render result tabs for a single agent invocation."""
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

            if last_sql_rows:
                df = pd.DataFrame(last_sql_rows)
                chart = _detect_chart(df)
                if chart:
                    x_col, y_col, chart_type = chart
                    st.divider()
                    st.caption("📊 Visualização dos dados")

                    common_layout = dict(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=30, b=0),
                        font=dict(family="Inter"),
                    )

                    if chart_type == "line":
                        fig = px.line(df, x=x_col, y=y_col, template="plotly_dark", markers=True, hover_data=df.columns.tolist())
                        fig.update_traces(line=dict(color="#60a5fa", width=2.5))
                    else:
                        fig = px.bar(df, x=x_col, y=y_col, template="plotly_dark", color=y_col, color_continuous_scale="Blues", hover_data=df.columns.tolist())
                        common_layout["coloraxis_showscale"] = False

                    fig.update_layout(**common_layout)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Nenhuma resposta foi gerada para esta pergunta.")

    with tab_reasoning:
        if not tool_steps:
            st.info("Nenhuma ação foi registrada.")
            return

        st.caption(f"O agente executou **{len(tool_steps)}** ação(ões) para responder sua pergunta.")
        st.divider()

        for idx, step in enumerate(tool_steps, 1):
            tool_name = step["tool"]
            args = step["args"]
            output = step["output"]

            if tool_name == "get_database_schema":
                with st.expander(f"**Passo {idx} — Consultar estrutura do banco de dados**", expanded=False):
                    st.markdown('<div class="step-label">🗂️ Ferramenta: get_database_schema</div>', unsafe_allow_html=True)
                    st.code(output, language="text")

            elif tool_name == "execute_sql_query":
                sql = args.get("sql", "")
                formatted_sql = sqlparse.format(sql, reindent=True, keyword_case="upper", indent_width=4)

                with st.expander(f"**Passo {idx} — Executar consulta SQL**", expanded=True):
                    st.markdown('<div class="step-label">⚙️ Ferramenta: execute_sql_query</div>', unsafe_allow_html=True)
                    st.code(formatted_sql, language="sql")

                    if output.startswith("ERROR"):
                        st.error(output)
                    else:
                        try:
                            rows = json.loads(output)
                            if isinstance(rows, list) and rows:
                                df_step = pd.DataFrame(rows)
                                st.dataframe(df_step, use_container_width=True, hide_index=True)
                                st.caption(f"{len(df_step)} registro(s) retornado(s)")
                            else:
                                st.info(output)
                        except json.JSONDecodeError:
                            st.text(output)

            else:
                with st.expander(f"**Passo {idx} — {tool_name}**", expanded=False):
                    st.json(args)
                    st.text(output)


# ── Session state ─────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar — conversation history ────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🗂️ Histórico")
    st.caption("Perguntas desta sessão")

    if st.session_state.history:
        if st.button("🗑️ Limpar histórico", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.divider()

        for entry in reversed(st.session_state.history):
            label = entry["question"][:55] + "..." if len(entry["question"]) > 55 else entry["question"]
            with st.expander(f"**{label}**"):
                st.markdown(entry["response"])
    else:
        st.info("Nenhuma consulta realizada ainda.")

# ── Page header ───────────────────────────────────────────────────────────────

st.title("💹 Assistente Virtual de Dados")
st.caption("Faça perguntas em português sobre os dados da FRANQ e receba respostas instantâneas.")

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────

question = st.text_input(
    label="Sua pergunta",
    placeholder="Ex: Quais campanhas de marketing tiveram interação dos clientes?",
    label_visibility="collapsed",
)

submitted = st.button("Enviar", type="primary", use_container_width=True)

# ── Agent execution ───────────────────────────────────────────────────────────

if submitted:
    if not question.strip():
        st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        try:
            with st.spinner("Analisando e consultando os dados..."):
                raw_result = run_agent(question.strip())
        except Exception as exc:  # noqa: BLE001 — catch-all needed at UI boundary
            if _is_rate_limit_error(str(exc)):
                st.warning(
                    "⚠️ **Limite de requisições atingido temporariamente.**\n\n"
                    "Estamos na camada gratuita da API Gemini. "
                    "Aguarde **30 a 40 segundos** e tente novamente."
                )
            else:
                st.error(f"Ocorreu um erro inesperado. Tente novamente.\n\n`{exc}`")
            st.stop()

        parsed = _parse_agent_result(raw_result)

        st.divider()
        _render_result(parsed)

        response_text = parsed["final_response"]
        if response_text and not _is_rate_limit_error(response_text):
            st.session_state.history.append(
                {"question": question.strip(), "response": response_text}
            )
