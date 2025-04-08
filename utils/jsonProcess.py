import json
import os

def load_conversation_history(history_path: str, memory):
    """Load conversation history from a JSON file."""
    if os.path.exists(history_path):
        with open(memory, "r", encoding="utf-8") as file:
            try:
                return json.load(file)[-memory:]
            except json.JSONDecodeError:
                return []  # Return empty list if file is corrupted
    return []

def save_conversation_history(history):
    """Save conversation history to a JSON file."""
    with open(history, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)