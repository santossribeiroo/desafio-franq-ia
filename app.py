import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.agent import State, build_graph

load_dotenv()

st.set_page_config(
    page_title="Assistente Virtual de Dados - FRANQ",
    page_icon="💹",
    layout="centered",
)

# Compile once and cache so the graph isn't rebuilt on every interaction
@st.cache_resource
def get_graph():
    return build_graph()


def run_agent(question: str) -> State:
    initial_state: State = {
        "user_question": question,
        "generated_sql": "",
        "sql_result": [],
        "final_response": "",
        "error": "",
    }
    return get_graph().invoke(initial_state)


# ── Page header ──────────────────────────────────────────────────────────────

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

# ── Pipeline execution ────────────────────────────────────────────────────────

if submitted:
    if not question.strip():
        st.warning("Por favor, digite uma pergunta antes de enviar.")
    else:
        with st.spinner("Consultando os dados..."):
            result = run_agent(question)

        st.divider()

        # Surface a hard error that the LLM couldn't handle gracefully
        if result.get("error") and not result.get("final_response"):
            st.error(f"Não foi possível processar a solicitação.\n\n`{result['error']}`")
        else:
            # ── Response tab / agent reasoning expander ───────────────────────
            tab_response, tab_reasoning = st.tabs(["💬 Resposta", "🔍 Raciocínio do Agente"])

            with tab_response:
                if result.get("final_response"):
                    st.markdown(result["final_response"])
                else:
                    st.info("Nenhuma resposta foi gerada para esta pergunta.")

            with tab_reasoning:
                st.subheader("Query SQL gerada")
                if result.get("generated_sql"):
                    st.code(result["generated_sql"], language="sql")
                else:
                    st.info("Nenhuma query foi gerada.")

                st.subheader("Dados retornados pelo banco")
                if result.get("sql_result"):
                    st.dataframe(
                        pd.DataFrame(result["sql_result"]),
                        use_container_width=True,
                    )
                else:
                    st.info("A consulta não retornou registros.")

                # Show the error in context if the LLM still produced a response
                if result.get("error"):
                    st.subheader("Aviso do pipeline")
                    st.warning(result["error"])
