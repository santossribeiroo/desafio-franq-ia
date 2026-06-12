from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph

from src.db_utils import format_schema_for_prompt, get_db_connection, get_schema

load_dotenv()


class State(TypedDict):
    user_question: str
    generated_sql: str
    sql_result: list
    final_response: str
    error: str


_SQL_SYSTEM_PROMPT = """You are an expert SQLite query generator.

You will receive a user question and the full database schema.
Your sole job is to produce a single, executable SQLite SELECT query that answers the question.

Rules (violations will break the pipeline):
- Output ONLY the raw SQL string — no markdown, no code fences, no explanation.
- Never use tables or columns that are not listed in the schema below.
- Always use explicit column names; never use SELECT *.
- Use LIMIT when the question asks for top/bottom N results.

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


def generate_sql_node(state: State) -> State:
    try:
        llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0)

        schema_text = format_schema_for_prompt(get_schema())
        prompt = _SQL_SYSTEM_PROMPT.format(schema=schema_text)

        messages = [
            ("system", prompt),
            ("human", state["user_question"]),
        ]

        response = llm.invoke(messages)
        # Strip any accidental whitespace or newlines the model may add
        sql = response.content.strip()

        return {**state, "generated_sql": sql}

    except Exception as exc:
        return {**state, "error": str(exc)}


def execute_sql_node(state: State) -> State:
    # Propagate upstream errors without touching the DB
    if state.get("error"):
        return state

    sql = state["generated_sql"]

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            # sqlite3.Row objects are not serializable; convert eagerly
            rows = [dict(row) for row in cursor.fetchall()]

        return {**state, "sql_result": rows, "error": ""}

    except Exception as exc:
        return {**state, "error": str(exc)}


def format_response_node(state: State) -> State:
    # temperature 0.3 allows slightly more natural phrasing while staying factual
    llm = ChatVertexAI(model_name="gemini-1.5-flash", temperature=0.3)

    try:
        if state.get("error"):
            messages = [
                ("system", _ERROR_SYSTEM_PROMPT.format(error=state["error"])),
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


def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("generate_sql", generate_sql_node)
    graph.add_node("execute_sql", execute_sql_node)
    graph.add_node("format_response", format_response_node)

    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "format_response")
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
