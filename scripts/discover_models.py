"""
Lists all Gemini models available for your GOOGLE_API_KEY that support generateContent.
Run this script to verify your API key and discover usable model identifiers.

Usage:
    python scripts/discover_models.py
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

print("Available models supporting generateContent:\n")
print(f"{'Model name':<40} | Supported methods")
print("-" * 75)

for model in client.models.list():
    methods = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", [])
    if "generateContent" in str(methods):
        print(f"{model.name:<40} | {methods}")
