"""Global operation defaults must reach the SDK from compiled memory nodes."""

from collections.abc import Iterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from hindsight_client import Hindsight
from hindsight_client_api.models.recall_response import RecallResponse
from hindsight_client_api.models.recall_result import RecallResult
from hindsight_langgraph import configure, create_recall_node, create_retain_node, reset_config
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph


class AgentState(MessagesState):
    memory_context: str | None


@pytest.fixture(autouse=True)
def clean_config() -> Iterator[None]:
    reset_config()
    yield
    reset_config()


@pytest.mark.parametrize("scenario", ["unconfigured", "inherited", "overridden", "empty_tags", "snapshot"])
@pytest.mark.parametrize("output_key", [None, "memory_context"])
async def test_compiled_nodes_apply_operation_defaults(scenario: str, output_key: str | None) -> None:
    recall_requests = []
    retain_requests = []

    async def recall(request: web.Request) -> web.Response:
        recall_requests.append(await request.json())
        return web.json_response(RecallResponse(results=[RecallResult(id="m1", text="Favorite color: blue")]).to_dict())

    async def retain(request: web.Request) -> web.Response:
        retain_requests.append(await request.json())
        return web.json_response({"success": True, "bank_id": "test", "items_count": 1, "async": False})

    if scenario != "unconfigured":
        configure(
            budget="low",
            max_tokens=128,
            recall_tags=["project:alpha"],
            recall_tags_match="all_strict",
            tags=["source:langgraph"],
        )

    recall_options = {}
    retain_options = {}
    if scenario == "overridden":
        # Even an explicit built-in default must take precedence over configure().
        recall_options = {
            "budget": "mid",
            "max_tokens": 4096,
            "tags": ["explicit:recall"],
            "tags_match": "any",
        }
        retain_options = {"tags": ["explicit:retain"]}
    elif scenario == "empty_tags":
        recall_options = {"tags": []}
        retain_options = {"tags": []}

    app = web.Application()
    app.router.add_post("/v1/default/banks/test/memories/recall", recall)
    # The integration lock uses a client whose retain route is /memories;
    # current clients use /memories/retain. Exercise either published SDK.
    app.router.add_post("/v1/default/banks/test/memories", retain)
    app.router.add_post("/v1/default/banks/test/memories/retain", retain)
    async with TestServer(app) as server:
        client = Hindsight(base_url=str(server.make_url("")))
        try:
            recall_node = create_recall_node(client=client, output_key=output_key, **recall_options)
            retain_node = create_retain_node(client=client, **retain_options)
            if scenario == "snapshot":
                # Match the tools: defaults are resolved when the factory is called,
                # independently of RunnableConfig and later configure() calls.
                configure(
                    budget="high",
                    max_tokens=8192,
                    recall_tags=["project:beta"],
                    tags=["source:other"],
                )

            async def agent(state: AgentState) -> dict[str, list[AIMessage]]:
                context = (
                    state["memory_context"]
                    if output_key
                    else next(msg.content for msg in state["messages"] if isinstance(msg, SystemMessage))
                )
                assert context == "Relevant memories about this user:\n1. Favorite color: blue"
                return {"messages": [AIMessage(content="You like blue.")]}

            builder = StateGraph(AgentState)
            builder.add_node("recall", recall_node)
            builder.add_node("agent", agent)
            builder.add_node("retain", retain_node)
            builder.add_edge(START, "recall")
            builder.add_edge("recall", "agent")
            builder.add_edge("agent", "retain")
            builder.add_edge("retain", END)
            await builder.compile().ainvoke(
                {"messages": [HumanMessage(content="What color do I like?")]},
                config={"configurable": {"user_id": "test"}},
            )
        finally:
            await client.aclose()

    assert len(recall_requests) == len(retain_requests) == 1
    recalled = recall_requests[0]
    inherited = scenario in ("inherited", "empty_tags", "snapshot")
    assert recalled["query"] == "What color do I like?"
    assert recalled["budget"] == ("low" if inherited else "mid")
    assert recalled["max_tokens"] == (128 if inherited else 4096)
    if scenario in ("inherited", "snapshot"):
        assert recalled["tags"] == ["project:alpha"]
        assert recalled["tags_match"] == "all_strict"
    elif scenario == "overridden":
        assert recalled["tags"] == ["explicit:recall"]
        assert recalled["tags_match"] == "any"
    else:
        assert not recalled.get("tags")

    retained = retain_requests[0]["items"]
    assert len(retained) == 1
    assert retained[0]["content"] == "What color do I like?"
    expected_tags = (
        ["source:langgraph"]
        if scenario in ("inherited", "snapshot")
        else ["explicit:retain"]
        if scenario == "overridden"
        else None
    )
    assert (retained[0].get("tags") or None) == expected_tags
