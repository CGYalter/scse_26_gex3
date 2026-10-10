"""RAG tool.

Exposes the retriever built in ``rag.py`` as a single callable "tool" that the
agent can invoke with a user's question and receive the most relevant policy
excerpts back as a plain-text context block.
"""

from rag import retrieve

TOOL_NAME = "retrieve_policy"
TOOL_DESCRIPTION = (
    "Retrieve the university IT policy information relevant to a user's question. "
    "Input is the question text; output is the most relevant policy excerpts."
)


def rag_tool(query, k=3):
    """Return relevant policy excerpts for ``query`` as a formatted string."""
    matched = retrieve(query, k=k)

    if not matched:
        return "No relevant university IT policy information was found."

    blocks = []
    for doc in matched:
        topic = doc.metadata.get("topic", "general")
        blocks.append(f"[{topic}] {doc.page_content}")

    return "\n\n".join(blocks)
