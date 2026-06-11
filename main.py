import asyncio

from langchain_core.messages import HumanMessage

from app.agent.recipe import Cart, graph


async def main():
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Рецепт: греческий йогурт, низкокалорийные печеньки, сахарозаменитель, ягоды, какао"
                )
            ],
        }
    )

    cart = Cart.model_validate(result["cart"])


if __name__ == "__main__":
    asyncio.run(main())
