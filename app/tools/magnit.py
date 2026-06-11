import asyncio
import random
from pathlib import Path

from bs4 import BeautifulSoup
from loguru import logger
from lxml import html
from playwright.async_api import async_playwright
from pydantic import BaseModel

from app.core.cache import diskcache_cache

BASE_URL = "https://magnit.ru"
AUTH_DIR = Path("playwright", ".auth")
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_FILE = AUTH_DIR / "magnit.json"
CACHE_SALT = "magnit_search"

EXTRACT_RATIO = 0.35

CARDS_LAYOUT_FULL_XPATH = "/html/body/div[1]/div[1]/div/main/div/div/div/div[1]"
CARDS_RELATIVE_XPATH = ".//article"
CARD_TITLE_RELATIVE_XPATH = "./a/div[2]/div[1]/div[2]"
CARD_PRICE_RELATIVE_XPATH = "./a/div[2]/div[1]/div[1]/div/span"
CARD_IMAGE_RELATIVE_XPATH = "./a/div[1]/div/img"
CARD_HREF_RELATIVE_XPATH = "./a/@href"

KKAL_FULL_XPATH = "/html/body/div[2]/div[1]/div/div/main/div/div/div[1]/div/div/div/div[1]/div[2]/section[2]/div[2]/div[1]/div[2]"
PROTEIN_FULL_XPATH = "/html/body/div[2]/div[1]/div/div/main/div/div/div[1]/div/div/div/div[1]/div[2]/section[2]/div[2]/div[2]/div[2]"
FATS_FULL_XPATH = "/html/body/div[2]/div[1]/div/div/main/div/div/div[1]/div/div/div/div[1]/div[2]/section[2]/div[2]/div[3]/div[2]"
CARBS_FULL_XPATH = "/html/body/div[2]/div[1]/div/div/main/div/div/div[1]/div/div/div/div[1]/div[2]/section[2]/div[2]/div[4]/div[2]"


class Product(BaseModel):
    title: str
    price: str
    image_url: str
    product_url: str
    kkal: str = None
    protein: str = None
    fats: str = None
    carbs: str = None


async def search_products(query: str) -> list[Product]:
    """
    Returns a list of products from Magnit website based on the provided search query.
    Each product includes title, price, image URL, product URL, and nutritional information (kkal, protein, fats, carbs).
    Args:
        query (str): Text query for searching products on Magnit website.
    Returns:
        list[Product]: A list of Product objects containing the search results.
    """
    cache_tag = f"{CACHE_SALT}_{query}"
    if diskcache_cache.get(cache_tag):
        logger.info(f"Cache hit for query: {query}")
        return diskcache_cache.get(cache_tag)

    if not AUTH_FILE.exists():
        logger.error(
            f"Auth file not found at {AUTH_FILE}. Please run storage_session_cli.py to save the session."
        )
        return []

    products = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=AUTH_FILE)
        page = await context.new_page()
        search_url = f"{BASE_URL}/search?term={query}"
        await page.goto(search_url, wait_until="domcontentloaded", timeout=60000)
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")

        search_results_tree = html.fromstring(bytes(str(soup), encoding="utf-8"))

        cards_layout_el = search_results_tree.xpath(CARDS_LAYOUT_FULL_XPATH)[0]
        cards = cards_layout_el.xpath(CARDS_RELATIVE_XPATH)

        if len(cards) == 0:
            logger.warning("No products found")
            return

        need_to_extract = max(min(int(len(cards) * EXTRACT_RATIO), 20), 7)

        logger.info(f"Found products: {len(cards)}")
        logger.info(f"Will be extracted: {need_to_extract}")

        for i, card in enumerate(cards[:need_to_extract]):
            title_el = card.xpath(CARD_TITLE_RELATIVE_XPATH)
            price_el = card.xpath(CARD_PRICE_RELATIVE_XPATH)
            image_el = card.xpath(CARD_IMAGE_RELATIVE_XPATH)
            href_el = card.xpath(CARD_HREF_RELATIVE_XPATH)

            title = title_el[0].text_content().strip() if title_el else "no tittle"
            price_text = price_el[0].text_content().strip() if price_el else "no price"
            image_src = image_el[0].get("src") if image_el else "no image"
            href = href_el[0] if href_el else "no href"

            product = Product(
                title=title,
                price=price_text,
                image_url=image_src,
                product_url=f"{BASE_URL}{href}",
            )
            logger.debug(
                f"\t{i + 1}: {product.title} - {product.price} - {product.product_url}"
            )

            await asyncio.sleep(random.uniform(0.5, 2))
            await page.goto(
                product.product_url, wait_until="domcontentloaded", timeout=60000
            )
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            product_tree = html.fromstring(bytes(str(soup), encoding="utf-8"))

            kkal_el = product_tree.xpath(KKAL_FULL_XPATH)
            protein_el = product_tree.xpath(PROTEIN_FULL_XPATH)
            fats_el = product_tree.xpath(FATS_FULL_XPATH)
            carbs_el = product_tree.xpath(CARBS_FULL_XPATH)

            kkal = (
                kkal_el[0].text_content().strip()
                if len(kkal_el) > 0
                else "no kkal info"
            )
            protein = (
                protein_el[0].text_content().strip()
                if len(protein_el) > 0
                else "no protein info"
            )
            fats = (
                fats_el[0].text_content().strip()
                if len(fats_el) > 0
                else "no fats info"
            )
            carbs = (
                carbs_el[0].text_content().strip()
                if len(carbs_el) > 0
                else "no carbs info"
            )

            product.kkal = kkal
            product.protein = protein
            product.fats = fats
            product.carbs = carbs

            products.append(product)

        diskcache_cache.set(cache_tag, products)
        await browser.close()

    return products
