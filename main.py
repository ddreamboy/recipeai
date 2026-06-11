import asyncio

from app.tools.magnit import search_products


async def main():
    query = input("Введите название продукта для поиска:\n") or "творог"
    products = await search_products(query)
    for product in products:
        print(product)


if __name__ == "__main__":
    asyncio.run(main())
