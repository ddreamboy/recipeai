import operator
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from app.core.config import llm_settings
from app.tools.magnit import Product, search_products_tool

llm = ChatOpenAI(
    model=llm_settings.LLM_API_BASE_MODEL,
    base_url=llm_settings.LLM_API_BASE_URL,
    api_key=llm_settings.LLM_API_KEY,
)


class Cart(BaseModel):
    products: list[Product]
    total_price: float
    total_kkal: float
    total_protein: float
    total_fats: float
    total_carbs: float


class AgentState(MessagesState):
    cart: Optional[Cart]
    available_products: Annotated[list[Product], operator.add]


class SelectedProducts(BaseModel):
    selected_indices: list[int]


tools = [search_products_tool]
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(SelectedProducts)


SYSTEM_PROMPT = """\\
You are a grocery shopping assistant. You build a shopping cart in an online
grocery store based on the user's request. The store's search tool is
`search_products` (queries must be in Russian - the store catalog is Russian).

## Stage 1 - Build the ingredient list
The user may give you:
- a full recipe with quantities -> extract the ingredient list as-is;
- a dish name or a vague request ("ужин на двоих", "что-нибудь к завтраку") ->
    first propose a dish and compose a realistic ingredient list yourself, with
    quantities scaled to the stated number of servings (default: 2).

Skip pantry staples the user almost certainly has (water, salt, black pepper),
unless the recipe depends on a specific one (e.g., sea salt flakes for finishing).

## Stage 2 - Search and select
For EVERY ingredient call `search_products` with a short generic query:
the product name only, no quantities, fat content, or brands
("греческий йогурт", not "200г греческого йогурта 2%").
Issue ALL search calls in a single step (parallel tool calls), not one by one.

When all results are in, pick exactly one product per ingredient. Ranking
criteria, in order of priority:
1. Product type actually matches the ingredient (yogurt not equal yogurt drink).
2. Package size is close to the required quantity - prefer slightly more over not enough.
3. Macros (КБЖУ) close to what the recipe implies, when relevant.
4. Lower price per unit as a tiebreaker.

## Autonomy
You are running unattended - there is no human in the loop to answer questions.
NEVER ask clarifying questions; the conversation ends with your reply.
When a specific brand is unavailable, pick the closest
matching product yourself and note the substitution in the final answer.
When information seems missing, make a reasonable assumption and state it.

Respond in the same language as the user's request (Russian by default).

## Edge cases
- No good match -> retry once with a broader or synonymous query
  ("кинза" -> "кориандр зелень"). Still nothing -> list the ingredient under
  "не найдено" with a suggested substitute; never silently drop it.
- Several ingredients covered by one product (e.g., a spice mix) -> it's fine
  to map one product to multiple ingredients, say so explicitly.

## Output format
1. Short dish summary (only if you composed the menu yourself).
2. Cart list: ingredient -> chosen product -> package size -> price.
3. "Не найдено / замены" section, if any.

Do not invent products that were not in the search results.
"""


async def agent_node(state: AgentState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        state["messages"] = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


async def format_output_node(state: AgentState):
    selection = await structured_llm.ainvoke(
        [
            *state["messages"],
            HumanMessage(
                content=(
                    "Select the products you want to add to the cart based on the search results. "
                    "Return their indices [N] from the search results in selected_indices."
                )
            ),
        ]
    )

    available = state["available_products"]
    selected = [
        available[i] for i in selection.selected_indices if 0 <= i < len(available)
    ]

    return {
        "cart": Cart(
            products=selected,
            total_price=round(sum(p.price or 0 for p in selected), 2),
            total_kkal=sum(_to_float(p.kkal) for p in selected),
            total_protein=sum(_to_float(p.protein) for p in selected),
            total_fats=sum(_to_float(p.fats) for p in selected),
            total_carbs=sum(_to_float(p.carbs) for p in selected),
        )
    }


def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "format_output"


def collect_products(state: AgentState):
    messages = state["messages"]
    last_ai_idx = max(
        i for i, m in enumerate(messages) if isinstance(m, AIMessage) and m.tool_calls
    )
    tool_messages = messages[last_ai_idx + 1 :]

    available = state.get("available_products", [])
    new_products: list[Product] = []
    updated_messages: list[ToolMessage] = []

    for tm in tool_messages:
        if not isinstance(tm, ToolMessage) or tm.name != search_products_tool.name:
            continue
        products: list[Product] = tm.artifact or []
        start = len(available) + len(new_products)

        numbered = "\n".join(f"[{start + i}] {p}" for i, p in enumerate(products))
        new_products.extend(products)

        updated_messages.append(
            ToolMessage(
                content=numbered,
                tool_call_id=tm.tool_call_id,
                name=tm.name,
                id=tm.id,
            )
        )

    return {"messages": updated_messages, "available_products": new_products}


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("collect_products", collect_products)
builder.add_node("format_output", format_output_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "collect_products")
builder.add_edge("collect_products", "agent")
builder.add_edge("format_output", END)

graph = builder.compile()
