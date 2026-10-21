"""Pre-built LangGraph nodes for Hindsight memory operations.

Provides node functions that can be added directly to a StateGraph to
inject memories at conversation start and store new memories after responses.
"""

import logging
from typing import Any, Optional

from hindsight_client import Hindsight
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from ._client import resolve_client
from .config import get_config
from .errors import HindsightError

logger = logging.getLogger(__name__)


def _extract_text_content(content: Any) -> str:
    """Extract text from a message content field.

    Handles both plain string content and multimodal content lists
    (where each item may be a dict with "type" and "text" keys).
    Returns the concatenated text parts, or an empty string if no
    text content is found.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return str(content) if content else ""


def create_recall_node(
    *,
    bank_id: Optional[str] = None,
    client: Optional[Hindsight] = None,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    budget: Optional[str] = None,
    max_tokens: Optional[int] = None,
    max_results: int = 10,
    tags: Optional[list[str]] = None,
    tags_match: Optional[str] = None,
    recall_types: Optional[list[str]] = None,
    recall_include_entities: bool = False,
    bank_id_from_config: str = "user_id",
    output_key: Optional[str] = None,
):
    """Create a node that injects relevant memories into the conversation.

    This node extracts the latest user message, recalls relevant memories
    from Hindsight, and returns them either as a SystemMessage in the
    ``messages`` list (default) or as a plain string under a custom state
    key via ``output_key``.

    **Message ordering:** When using the default ``messages`` output,
    ``MessagesState`` uses ``add_messages`` as its reducer, which appends.
    The memory SystemMessage will appear after existing messages, not at
    position 0. If your LLM provider requires system messages first, use
    ``output_key`` to write the memory text to a separate state field and
    inject it into your prompt in the agent node.

    Example with ``output_key`` (recommended for correct ordering)::

        from typing import Optional
        from langgraph.graph import MessagesState

        class AgentState(MessagesState):
            memory_context: Optional[str] = None

        recall = create_recall_node(
            client=client, bank_id="user-123", output_key="memory_context"
        )
        # In your agent node, read state["memory_context"] and prepend
        # it to the system prompt.

    The bank_id can be provided directly or resolved dynamically from
    the graph's RunnableConfig via the ``bank_id_from_config`` key.

    Args:
        bank_id: Static Hindsight memory bank ID.
        client: Pre-configured Hindsight client.
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        budget: Recall budget level (low/mid/high). Defaults to global config, then mid.
        max_tokens: Maximum tokens for recall results. Defaults to global config, then 4096.
        max_results: Maximum number of memories to inject.
        tags: Tags to filter recall results. Defaults to global recall_tags;
            pass [] to disable the inherited filter.
        tags_match: Tag matching mode. Defaults to global recall_tags_match, then any.
        recall_types: Fact types to filter (world, experience, observation).
        recall_include_entities: Include entity information in recall results.
        bank_id_from_config: Config key to read bank_id from at runtime.
            Looked up in ``config["configurable"][bank_id_from_config]``.
            Only used when ``bank_id`` is not provided.
        output_key: If set, write the memory text to this state key as a
            plain string instead of appending a SystemMessage to ``messages``.
            Use this with a custom state type to control where memory context
            appears in your prompt.

    Returns:
        An async node function compatible with LangGraph StateGraph.
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)

    # Resolve operation defaults at construction, like the tool factories. None
    # means omitted; explicit built-in defaults and [] must override configure().
    global_config = get_config()
    effective_budget = budget if budget is not None else (global_config.budget if global_config else "mid")
    effective_max_tokens = (
        max_tokens if max_tokens is not None else (global_config.max_tokens if global_config else 4096)
    )
    effective_tags = tags if tags is not None else (global_config.recall_tags if global_config else None)
    effective_tags_match = (
        tags_match if tags_match is not None else (global_config.recall_tags_match if global_config else "any")
    )

    async def recall_node(state: MessagesState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        # add_messages appends updates: an empty list leaves the previous recall
        # context intact. Remove only our own message when this turn has no memory.
        empty_messages: list[BaseMessage] = []
        if not output_key and any(msg.id == "hindsight_memory_context" for msg in state["messages"]):
            empty_messages = [RemoveMessage(id="hindsight_memory_context")]

        resolved_bank_id = bank_id
        if resolved_bank_id is None and config:
            configurable = config.get("configurable", {})
            resolved_bank_id = configurable.get(bank_id_from_config)

        if not resolved_bank_id:
            logger.warning("No bank_id available for recall node, skipping memory injection.")
            if output_key:
                return {output_key: None}
            return {"messages": empty_messages}

        # Extract query from the latest human message
        query = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = _extract_text_content(msg.content)
                break

        if not query:
            if output_key:
                return {output_key: None}
            return {"messages": empty_messages}

        try:
            recall_kwargs: dict[str, Any] = {
                "bank_id": resolved_bank_id,
                "query": query,
                "budget": effective_budget,
                "max_tokens": effective_max_tokens,
            }
            if effective_tags:
                recall_kwargs["tags"] = effective_tags
                recall_kwargs["tags_match"] = effective_tags_match
            if recall_types:
                recall_kwargs["types"] = recall_types
            if recall_include_entities:
                recall_kwargs["include_entities"] = True

            response = await resolved_client.arecall(**recall_kwargs)
            results = response.results[:max_results] if response.results else []

            if not results:
                if output_key:
                    return {output_key: None}
                return {"messages": empty_messages}

            lines = ["Relevant memories about this user:"]
            for i, result in enumerate(results, 1):
                lines.append(f"{i}. {result.text}")
            memory_text = "\n".join(lines)

            if output_key:
                return {output_key: memory_text}
            return {"messages": [SystemMessage(content=memory_text, id="hindsight_memory_context")]}
        except Exception as e:
            logger.error(f"Recall node failed: {e}")
            raise HindsightError(f"Recall node failed: {e}") from e

    return recall_node


def create_retain_node(
    *,
    bank_id: Optional[str] = None,
    client: Optional[Hindsight] = None,
    hindsight_api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    tags: Optional[list[str]] = None,
    metadata: Optional[dict[str, str]] = None,
    document_id: Optional[str] = None,
    bank_id_from_config: str = "user_id",
    retain_human: bool = True,
    retain_ai: bool = False,
):
    """Create a node that stores conversation messages as memories.

    This node extracts messages from the conversation and stores them
    via Hindsight retain. It should be placed after the LLM response
    node in your graph.

    Only ``HumanMessage`` and ``AIMessage`` text content is retained;
    ``ToolMessage`` / ``FunctionMessage`` content is intentionally
    skipped to avoid storing tool wire-protocol noise as memories.

    Args:
        bank_id: Static Hindsight memory bank ID.
        client: Pre-configured Hindsight client.
        hindsight_api_url: API URL (used if no client provided).
        api_key: API key (used if no client provided).
        tags: Tags to apply to stored memories. Defaults to global tags;
            pass [] to disable inherited tags.
        metadata: Metadata dict to attach to retained memories.
        document_id: Document ID for grouping/upserting memories.
        bank_id_from_config: Config key to read bank_id from at runtime.
        retain_human: Store human messages as memories.
        retain_ai: Store AI responses as memories.

    Returns:
        An async node function compatible with LangGraph StateGraph.
    """
    resolved_client = resolve_client(client, hindsight_api_url, api_key)
    global_config = get_config()
    effective_tags = tags if tags is not None else (global_config.tags if global_config else None)

    async def retain_node(state: MessagesState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
        resolved_bank_id = bank_id
        if resolved_bank_id is None and config:
            configurable = config.get("configurable", {})
            resolved_bank_id = configurable.get(bank_id_from_config)

        if not resolved_bank_id:
            logger.warning("No bank_id available for retain node, skipping memory storage.")
            return {"messages": []}

        # Only retain the latest human and/or AI message to avoid
        # duplicating memories that were already stored in prior calls.
        messages_to_retain = []
        if retain_human:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    text = _extract_text_content(msg.content)
                    if text:
                        messages_to_retain.append(text)
                    break
        if retain_ai:
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage):
                    text = _extract_text_content(msg.content)
                    if text:
                        messages_to_retain.append(text)
                    break

        if not messages_to_retain:
            return {"messages": []}

        content = "\n\n".join(messages_to_retain)

        try:
            retain_kwargs: dict[str, Any] = {
                "bank_id": resolved_bank_id,
                "content": content,
            }
            if effective_tags:
                retain_kwargs["tags"] = effective_tags
            if metadata:
                retain_kwargs["metadata"] = metadata
            if document_id:
                retain_kwargs["document_id"] = document_id
            await resolved_client.aretain(**retain_kwargs)
        except Exception as e:
            logger.error(f"Retain node failed: {e}")
            raise HindsightError(f"Retain node failed: {e}") from e

        return {"messages": []}

    return retain_node
