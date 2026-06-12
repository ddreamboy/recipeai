import asyncio

import markdown
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage

from app.agent.recipe import Cart, graph


async def main():
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""\\
Мука — 2000 г
Вода — 1150 г
Соль — 15 г
Сахар — 30 г
Дрожжи инстантные Angel — 15 г
Масло растительное — 20 г
"""
                )
            ],
        },
        verbose=True,
    )

    cart = Cart.model_validate(result["cart"])

    agent_response_raw = result["messages"][-1].content
    agent_response_html = markdown.markdown(agent_response_raw)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")
    rendered_html = template.render(cart=cart, agent_response=agent_response_html)

    with open("output.html", "w") as f:
        f.write(rendered_html)
    
    print("Output written to output.html")


if __name__ == "__main__":
    asyncio.run(main())
