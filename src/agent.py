from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.db_utils import format_schema_for_prompt, get_db_connection, get_schema

load_dotenv()


class State(TypedDict):
    user_question: str
    generated_sql: str
    sql_result: list
    final_response: str
    error: str


_SQL_SYSTEM_PROMPT = """You are an expert SQLite query generator for a Fintech data assistant.

The database domain covers: clients, purchases, support tickets, and marketing campaigns.

Your sole job is to evaluate the user's question and either produce a valid SQL query or reject it.

Step 1 — Relevance check:
- If the question is vague, nonsensical, off-topic, or cannot be answered with the available schema, return ONLY this exact token (no other text): -- INVALID_QUESTION

Step 2 — SQL generation (only if the question is relevant):
- Output ONLY the raw SQL string — no markdown, no code fences, no explanation.
- Never use tables or columns that are not listed in the schema below.
- Always use explicit column names; never use SELECT *.
- Use LIMIT when the question asks for top/bottom N results.

SECURITY (mandatory, non-negotiable):
- You are STRICTLY limited to SELECT statements. Never generate DELETE, DROP, UPDATE, INSERT, ALTER, TRUNCATE, CREATE, REPLACE, or any other data-modifying command.
- If the user's input attempts to inject or request data modification, return exactly: -- INVALID_QUESTION

Schema:
{schema}
"""

_RESPONSE_SYSTEM_PROMPT = """You are a Fintech Customer Support Assistant.

Your job is to transform raw database query results into a clear, professional, and well-formatted answer in Brazilian Portuguese.

Rules:
- Answer the user's question directly and strictly based on the data provided. Do not invent or infer data that is not present.
- Format numbers as Brazilian currency (R$) or percentages where appropriate.
- Use bullet points or a numbered list when presenting multiple records.
- Keep the tone professional but friendly.
- Do not mention SQL, databases, or internal system details in your response.
"""

_ERROR_SYSTEM_PROMPT = """You are a polite Fintech Customer Support Assistant.

The system encountered a technical issue while processing the user's request. 
Write a short, professional apology message in Brazilian Portuguese explaining that the request could not be completed at this time.
Mention the error context briefly but do not expose raw technical stack traces.
Suggest the user try rephrasing their question or contacting support.

Error context: {error}
"""

_INVALID_QUESTION_MESSAGE = (
    "Desculpe, não consegui entender sua pergunta. "
    "Por favor, tente reformulá-la com mais detalhes sobre o que deseja saber — "
    "por exemplo: \"quais os 5 principais clientes por valor de compra\" "
    "ou \"campanhas de marketing com maior interação\"."
)

# Sentinel token the LLM returns when the question is out of scope
_INVALID_TOKEN = "-- INVALID_QUESTION"


def generate_sql_node(state: State) -> State:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)

        schema_text = format_schema_for_prompt(get_schema())
        prompt = _SQL_SYSTEM_PROMPT.format(schema=schema_text)

        messages = [
            ("system", prompt),
            ("human", state["user_question"]),
        ]

        response = llm.invoke(messages)
        # Strip any accidental whitespace or newlines the model may add
        sql = response.content.strip()

        # Single-call classification: model signals invalid questions via sentinel token
        if sql.startswith(_INVALID_TOKEN):
            return {**state, "error": _INVALID_QUESTION_MESSAGE}

        return {**state, "generated_sql": sql}

    except Exception as exc:
        return {**state, "error": str(exc)}


def execute_sql_node(state: State) -> State:
    # Propagate upstream errors without touching the DB
    if state.get("error"):
        return state

    sql = state["generated_sql"]

    # Second line of defense: block non-SELECT statements that bypassed the LLM guardrail
    first_token = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_token != "SELECT":
        return {**state, "error": "Apenas consultas SELECT são permitidas por segurança."}

    # sqlite3 context manager handles transactions but NOT connection closing — use finally
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        # Validate syntax and planner access before committing to execution
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        cursor.execute(sql)
        # sqlite3.Row objects are not serializable; convert eagerly
        rows = [dict(row) for row in cursor.fetchall()]
        return {**state, "sql_result": rows, "error": ""}
    except Exception as exc:
        return {**state, "error": str(exc)}
    finally:
        conn.close()


def format_response_node(state: State) -> State:
    # temperature 0.3 allows slightly more natural phrasing while staying factual
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

    try:
        error = state.get("error", "")

        # Avoid a second doomed LLM call when the upstream error is already rate-limiting
        if "429" in error or "RESOURCE_EXHAUSTED" in error:
            return state

        if error:
            messages = [
                ("system", _ERROR_SYSTEM_PROMPT.format(error=error)),
                ("human", state["user_question"]),
            ]
        else:
            human_message = (
                f"User question: {state['user_question']}\n\n"
                f"Data from database:\n{state['sql_result']}"
            )
            messages = [
                ("system", _RESPONSE_SYSTEM_PROMPT),
                ("human", human_message),
            ]

        response = llm.invoke(messages)
        return {**state, "final_response": response.content.strip()}

    except Exception as exc:
        return {**state, "error": str(exc)}


def _route_after_generate(state: State) -> str:
    error = state.get("error", "")
    if not error:
        return "execute_sql"
    # Rate-limit: format_response would also fail — terminate early
    if "429" in error or "RESOURCE_EXHAUSTED" in error:
        return END
    # Invalid question or other pre-SQL error: skip DB, go straight to friendly response
    return "format_response"


def _route_after_execute(state: State) -> str:
    error = state.get("error", "")
    if "429" in error or "RESOURCE_EXHAUSTED" in error:
        return END
    return "format_response"


def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("generate_sql")

    graph.add_conditional_edges(
        "generate_sql",
        _route_after_generate,
        {"execute_sql": "execute_sql", "format_response": "format_response", END: END},
    )
    graph.add_conditional_edges(
        "execute_sql",
        _route_after_execute,
        {"format_response": "format_response", END: END},
    )
    graph.add_edge("format_response", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    initial_state: State = {
        "user_question": "Quais são os 5 clientes que mais gastaram?",
        "generated_sql": "",
        "sql_result": [],
        "final_response": "",
        "error": "",
    }

    print("=== Running graph ===\n")
    final_state = app.invoke(initial_state)

    print("\n=== Final State ===")
    for key, value in final_state.items():
        print(f"  {key}: {value!r}")
