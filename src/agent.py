from typing import TypedDict

from langgraph.graph import END, StateGraph


class State(TypedDict):
    user_question: str
    generated_sql: str
    sql_result: list
    final_response: str
    error: str


def generate_sql_node(state: State) -> State:
    print(f"[generate_sql_node] Received question: '{state.get('user_question')}'")
    return state


def execute_sql_node(state: State) -> State:
    print(f"[execute_sql_node] Would execute SQL: '{state.get('generated_sql')}'")
    return state


def format_response_node(state: State) -> State:
    print(f"[format_response_node] Would format result: {state.get('sql_result')}")
    return state


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

    print("=== Running graph with stub nodes ===\n")
    final_state = app.invoke(initial_state)

    print("\n=== Final State ===")
    for key, value in final_state.items():
        print(f"  {key}: {value!r}")
