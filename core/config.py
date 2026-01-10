from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, Optional, List
from dotenv import load_dotenv
import os

# 1. The Singleton Client
# This executes once when the module is imported.
# It holds the connection pool open for the entire lifetime of your app.
load_dotenv(dotenv_path='./google.env', verbose=True)
_SHARED_CLIENT = genai.Client(vertexai=True)


class VertexEngine:
    def __init__(self, model_id: str = 'gemini-1.5-pro'):
        # 2. Shared Connection
        # Every engine instance references the same open client.
        self.client = _SHARED_CLIENT

        # 3. Unique Configuration
        # Each engine gets its own independent config object.
        self.model_id = model_id
        self.config = types.GenerateContentConfig(
            temperature=0.0
        )

    def set_temperature(self, temp: float):
        self.config.temperature = temp
        return self

    def set_schema(self, schema: Type[BaseModel]):
        self.config.response_mime_type = "application/json"
        self.config.response_schema = schema
        return self

    def set_logprobs(self, count: int = 1):
        self.config.response_logprobs = True
        self.config.logprobs = count
        return self

    def set_system_instruction(self, instruction: str):
        self.config.system_instruction = instruction
        return self

    def set_model(self, model_id: str):
        self.model_id = model_id
        return self

    def generate(self, contents: List[types.Part]):
        """Executes the request using the unique config but the shared client."""
        return self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.config
        )

    def get_chat_session(self, history: Optional[List[types.Content]] = None):
        """Creates a chat session bound to this specific engine's config."""
        return self.client.chats.create(
            model=self.model_id,
            config=self.config,
            history=history or []
        )