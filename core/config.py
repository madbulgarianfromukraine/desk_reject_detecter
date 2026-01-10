from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, Optional, List
from dotenv import load_dotenv

# 1. The Singleton Client
# This executes once when the module is imported.
# It holds the connection pool open for the entire lifetime of your app.
# By sharing one genai.Client, we ensure efficient resource usage and connection pooling
# across all agent instances.
load_dotenv(dotenv_path='./google.env', verbose=True)
_SHARED_CLIENT = genai.Client(vertexai=True)


class VertexEngine:
    """
    A wrapper around the Google GenAI client that manages configuration state for LLM agents.

    Design Pattern: Singleton Client
    This class implements a pattern where multiple 'Engine' instances can exist (each with
    its own unique configuration like temperature, system instructions, or schema), but
    they all share a single underlying `genai.Client` (_SHARED_CLIENT).

    Rationale:
    - Efficient connection pooling: Maintaining one client prevents exhausting resources.
    - Simplified configuration: Allows chaining methods to build a complex GenerateContentConfig.
    """
    def __init__(self, model_id: str = 'gemini-1.5-pro'):
        """
        Initializes a new VertexEngine instance.

        :param model_id: The identifier of the model to use (default: 'gemini-1.5-pro').
        """
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
        """
        Sets the temperature for the generation config.

        :param temp: The temperature value (typically 0.0 to 1.0).
        :return: self (for method chaining).
        """
        self.config.temperature = temp
        return self

    def set_schema(self, schema: Type[BaseModel]):
        """
        Configures the engine to return structured JSON output matching the provided schema.

        :param schema: A Pydantic BaseModel class defining the output structure.
        :return: self (for method chaining).
        """
        self.config.response_mime_type = "application/json"
        self.config.response_schema = schema
        return self

    def set_logprobs(self, count: int = 1):
        """
        Enables log-probabilities in the response, which are used for confidence scoring.

        :param count: The number of log-probability candidates to return per token.
        :return: self (for method chaining).
        """
        self.config.response_logprobs = True
        self.config.logprobs = count
        return self

    def set_system_instruction(self, instruction: str):
        """
        Sets the system instructions (identity and task) for the model.

        :param instruction: The instruction string or Part object.
        :return: self (for method chaining).
        """
        self.config.system_instruction = instruction
        return self

    def set_model(self, model_id: str):
        """
        Updates the model identifier for this engine instance.

        :param model_id: The new model identifier.
        :return: self (for method chaining).
        """
        self.model_id = model_id
        return self

    def generate(self, contents: List[types.Part]):
        """
        Executes a generation request using the instance's unique config and the shared client.

        :param contents: A list of Parts (text, images, PDFs) to send to the model.
        :return: The GenerateContentResponse from the API.
        """
        return self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.config
        )

    def get_chat_session(self, history: Optional[List[types.Content]] = None):
        """
        Creates a new stateful chat session bound to this engine's specific configuration.

        :param history: Optional initial chat history.
        :return: A genai.chats.Chat object.
        """
        return self.client.chats.create(
            model=self.model_id,
            config=self.config,
            history=history or []
        )