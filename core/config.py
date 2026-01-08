
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv(dotenv_path="openai.env", verbose=True)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.0,
    max_retries=2,
    logprobs=True
)