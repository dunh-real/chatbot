from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")

def get_llm_service():
    """
    Khởi tạo và trả về dịch vụ LLM sử dụng LangChain và OpenAI.
    """
    if not api_key:
        raise ValueError("API_KEY is not set in environment variables.")

    llm = ChatOpenAI(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        openai_api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
    return llm