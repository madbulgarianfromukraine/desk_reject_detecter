
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, Optional, Dict, Any, List

class VertexEngine:
    def __init__(self, model_id: str = 'gemini-2.5-flash'):
        load_dotenv(dotenv_path='./google.env', verbose=True)
        # Client handles Project/Location via env vars (GOOGLE_CLOUD_PROJECT, etc.)
        self.client = genai.Client()
        self.model_id = model_id
        self.config = types.GenerateContentConfig(temperature=0.0)

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
        """Executes the request with the current state."""
        return self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.config
        )

    def get_chat_session(self, history: Optional[List[types.Content]] = None):
        """Creates a stateful chat session, optionally resumed from history."""
        return self.client.chats.create(
            model=self.model_id,
            config=self.config,
            history=history or []
        )