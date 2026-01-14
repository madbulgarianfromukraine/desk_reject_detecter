from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import Type, Optional, List
from core.log import LOG

# 1. The Singleton Client
# This executes once when the module is imported.
# It holds the connection pool open for the entire lifetime of your app.
# By sharing one genai.Client, we ensure efficient resource usage and connection pooling
# across all agent instances.
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

    def create_cache(self, contents: List[types.Part], display_name: str = None, ttl_seconds : str ="180s"):
        """
        Creates a context cache for the current model.

        :param contents: The contents to cache (e.g., style guides).
        :param display_name: Optional display name for the cache.
        :return: The created CachedContent object.
        """
        if len(tools) <= 0:
            tools = None
        cache = self.client.caches.create(
            model=self.model_id,
            config=types.CreateCachedContentConfig(
                contents=contents,
                display_name=display_name,
                ttl=ttl_seconds
            )
        )

        return cache

    def set_cache(self, cache_name: str):
        """
        Configures the engine to use a specific context cache.

        :param cache_name: The resource name of the cache.
        :return: self (for method chaining).
        """
        self.config.cached_content = cache_name
        return self

    def get_model_limit(self) -> int:
        """
        Returns the input token limit for the current model dynamically via the API.
        """
        try:
            # Assuming self.client is your initialized genai.Client
            model_info = self.client.models.get(model=f'{self.model_id}')

            # input_token_limit is the attribute that stores the context window size
            return model_info.input_token_limit or self.__get_model_limit_local(model_id=self.model_id)
        except Exception as e:
            LOG.error(f"Error fetching model limits: {e}. Using conservative default.")
            return 32000

    def __get_model_limit_local(self, model_id: str = "gemini-2.5-flash"):
        model_id_parts = model_id.split("-", maxsplit=2)[1:]
        if model_id_parts[1].startswith("flash") or model_id_parts[1].startswith("pro"):
            if model_id_parts[1].endswith("image"):
                return 65_536
            elif len(model_id_parts[1]) >= 30:
                return 131_072
            elif len(model_id_parts) >= 20:
                return 8_192
            else:
                return 1_048_576

    def generate(self, contents: List[types.Part]):
        """
        Executes a generation request using the instance's unique config and the shared client.
        Supposed for the amount of tokens less than maximal.
        :param contents: A list of Parts (text, images, PDFs) to send to the model.
        :return: The GenerateContentResponse from the API.
        """
        return self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=self.config
        )

    def count_tokens(self, contents: List[types.Part]) -> int:
        """
        Counts the number of tokens in the provided contents.

        :param contents: A list of Parts to count tokens for.
        :return: Total token count.
        """
        return self.client.models.count_tokens(
            model=self.model_id,
            contents=contents
        ).total_tokens

    def split_contents(self, contents: List[types.Part], limit: int) -> List[List[types.Part]]:
        """
        Splits a list of Parts into multiple chunks, each within the token limit.

        :param contents: The list of Parts to split.
        :param limit: The maximum number of tokens allowed per chunk.
        :return: A list of chunks (each chunk is a list of Parts).
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for part in contents:
            part_tokens = self.count_tokens([part])
            if current_tokens + part_tokens > limit:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

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