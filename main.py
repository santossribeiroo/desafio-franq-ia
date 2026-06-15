from dotenv import load_dotenv

from src.agent import State, build_graph

load_dotenv()

if __name__ == "__main__":
    app = build_graph()

    initial_state: State = {
        "user_question": "Qual o número de reclamações não resolvidas por canal?",
        "generated_sql": "",
        "sql_result": [],
        "final_response": "",
        "error": "",
        "retry_count": 0,
    }

    print(f"Question: {initial_state['user_question']}\n")

    final_state = app.invoke(initial_state)

    print(f"Generated SQL:\n{final_state['generated_sql']}\n")
    print(f"Final Response:\n{final_state['final_response']}")

    if final_state.get("error"):
        print(f"\nError: {final_state['error']}")
