from typing import Optional

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from app.core.config import llm_settings
from app.tools.magnit import Product, search_products

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


tools = [search_products]
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(Cart)


class AgentState(MessagesState):
    cart: Optional[Cart]


SYSTEM_PROMPT = """\\
Ты подбираешь продукты для рецепта в Магните.
Для КАЖДОГО ингредиента вызови search_products с коротким запросом
(название продукта без количества, например "греческий йогурт", не "200г греческого йогурта 2%").
Вызывай все search_products за один шаг, не по одному.

Когда получишь результаты по всем ингредиентам — выбери по одному товару под каждый,
ориентируясь на соответствие виду продукта и близость КБЖУ к рецепту.
В конце верни список выбранных товаров с ценами и итоговое КБЖУ корзины.
"""


async def agent_node(state: AgentState):
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        state["messages"] = [SystemMessage(content=SYSTEM_PROMPT), *messages]
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def format_output_node(state: AgentState):
    messages = state["messages"]
    response = await structured_llm.ainvoke(messages)
    return {"cart": response}


def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "format_output"


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("format_output", format_output_node)

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")
builder.add_edge("format_output", END)

graph = builder.compile()
