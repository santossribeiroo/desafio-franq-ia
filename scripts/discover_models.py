"""
Lists all Gemini models available for your GOOGLE_API_KEY that support generateContent.
Run this script to verify your API key and discover usable model identifiers.
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found. Check your .env file.")

genai.configure(api_key=api_key)

print("Available models supporting generateContent:\n")
print(f"{'Model name':<40} | Supported methods")
print("-" * 75)

for model in genai.list_models():
    if "generateContent" in model.supported_generation_methods:
        print(f"{model.name:<40} | {model.supported_generation_methods}")
