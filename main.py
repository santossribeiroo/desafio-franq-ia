"""
Quick end-to-end smoke test for the ReAct agent — run from the project root:
    python main.py
"""
import sys

# Force UTF-8 output so Portuguese characters don't break on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.agent import build_agent  # noqa: E402

if __name__ == "__main__":
    agent = build_agent()
    question = "Qual é o valor total de compras realizadas por clientes ativos?"

    print(f"Pergunta: {question}\n")
    print("=" * 60)

    result = agent.invoke({"messages": [("human", question)]})

    for msg in result["messages"]:
        role = type(msg).__name__
        content = msg.content

        # Gemini sometimes returns content as a list of content blocks
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"[{role}] -> tool '{tc['name']}' | args: {tc['args']}")
        elif content:
            print(f"[{role}]\n{content}\n")
