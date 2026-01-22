import asyncio
from agno.agent import Agent
from agno.knowledge.chunking.markdown import MarkdownChunking
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.markdown_reader import MarkdownReader
# from agno.vectordb.pgvector import PgVector
from agno.vectordb.qdrant import Qdrant

db_url = "http://localhost:6333"

knowledge = Knowledge(
    vector_db = Qdrant(collection = "test_collection2", url = db_url),
)

asyncio.run(knowledge.ainsert(
    url="https://github.com/agno-agi/agno/blob/main/README.md",
    reader=MarkdownReader(
        name="Markdown Chunking Reader",
        chunking_strategy=MarkdownChunking(),
    ),
))
agent = Agent(
    knowledge=knowledge,
    search_knowledge=True,
)

agent.print_response("What is Agno?", markdown=True)