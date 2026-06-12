import asyncio
import random
import re
from pathlib import Path

from bs4 import BeautifulSoup
from langchain.tools import tool
from loguru import logger
from lxml import html
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright
from pydantic import BaseModel

from app.core.cache import diskcache_cache
from app.core.rate_limiter import rate_limiter

BASE_URL = "https://magnit.ru"
AUTH_DIR = Path("playwright", ".auth")
AUTH_DIR.mkdir(parents=True, exist_ok=True)
AUTH_FILE = AUTH_DIR / "magnit.json"
CACHE_SALT = "magnit_search"

EXTRACT_RATIO = 0.35
PRODUCT_PAGE_RETRIES = 2

_search_semaphore = asyncio.Semaphore(2)
_headless = True

CARDS_LAYOUT_FULL_XPATH = "/html/body/div[1]/div[1]/div/main/div/div/div/div[1]"
CARDS_RELATIVE_XPATH = ".//article"
CARD_TITLE_RELATIVE_XPATH = "./a/div[2]/div[1]/div[2]"
CARD_PRICE_RELATIVE_XPATH = "./a/div[2]/div[1]/div[1]/div/span"
PRICE_PATTERN = r"\d+(?:\.\d{2})?"
CARD_IMAGE_RELATIVE_XPATH = "./a/div[1]/div/img"
CARD_HREF_RELATIVE_XPATH = "./a/@href"

PRODUCT_DETAIL_NUTRITION_LAYOUT_FULL_XPATH = "/html/body/div[2]/div[1]/div/div/main/div/div/div[1]/div/div/div/div[1]/div[2]/section[2]"
KKAL_TEXT_PATTERN = "ккал"
PROTEIN_TEXT_PATTERN = "бел"
FATS_TEXT_PATTERN = "жир"
CARBS_TEXT_PATTERN = "углевод"


class Product(BaseModel):
    title: str
    price: float | None = None
    image_url: str
    product_url: str
    kkal: float | None = None
    protein: float | None = None
    fats: float | None = None
    carbs: float | None = None

    def __str__(self) -> str:
        parts: list[str] = [self.title]

        if self.price is not None:
            parts.append(f"{self.price:.2f} RUB")

        nutrition_parts: list[str] = []
        if self.kkal is not None:
            nutrition_parts.append(f"{self.kkal} kkal")
        if self.protein is not None:
            nutrition_parts.append(f"{self.protein}g protein")
        if self.fats is not None:
            nutrition_parts.append(f"{self.fats}g fats")
        if self.carbs is not None:
            nutrition_parts.append(f"{self.carbs}g carbs")

        if nutrition_parts:
            parts.append("in 100g: " + ", ".join(nutrition_parts))

        return " - ".join(parts)


def _format_products(products: list[Product]) -> str:
    if not products:
        return "Found no products"

    return "\n".join([f"- {p}" for p in products])


@tool(response_format="content_and_artifact")
async def search_products_tool(query: str) -> tuple[str, list[Product]]:
    """
    Returns a list of products from Magnit website based on the provided search query.
    Each product includes title, price, image URL, product URL, and nutritional information (kkal, protein, fats, carbs).
    Args:
        query (str): Text query for searching products on Magnit website.
    Returns:
        list[Product]: A list of Product objects containing the search results.
    """
    products = await search_products(query)
    return _format_products(products), products


@rate_limiter
async def search_products(query: str) -> list[Product]:
    """
    Returns a list of products from Magnit website based on the provided search query.
    Each product includes title, price, image URL, product URL, and nutritional information (kkal, protein, fats, carbs).
    Args:
        query (str): Text query for searching products on Magnit website.
    Returns:
        list[Product]: A list of Product objects containing the search results.
    """
    logger.info(f"Searching for products with query: {query}")
    cache_tag = f"{CACHE_SALT}_{query}"
    cached = diskcache_cache.get(cache_tag)
    if cached:
        logger.info(f"Cache hit for query: {query}")
        return cached

    if not AUTH_FILE.exists():
        logger.error(
            f"Auth file not found at {AUTH_FILE}. Please run storage_session_cli.py to save the session."
        )
        return []

    products = []
    async with _search_semaphore, async_playwright() as p:
        browser = await p.chromium.launch(headless=_headless)
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
            logger.warning(f"No products found for query: {query}")
            return []

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

            price_match = re.search(PRICE_PATTERN, price_text.replace(" ", ""))

            product = Product(
                title=title,
                price=float(price_match.group(0)) if price_match else None,
                image_url=image_src,
                product_url=f"{BASE_URL}{href}",
            )
            logger.debug(
                f"Product {i + 1}: {product.title} - {product.price} - {product.product_url}"
            )

            await asyncio.sleep(random.uniform(0.5, 2))

            opened = False
            for attempt in range(1, PRODUCT_PAGE_RETRIES + 1):
                try:
                    await page.goto(
                        product.product_url,
                        wait_until="domcontentloaded",
                        timeout=60000,
                    )
                    opened = True
                    break
                except PlaywrightTimeoutError:
                    logger.warning(
                        f"Timeout opening {product.product_url}, attempt {attempt}/{PRODUCT_PAGE_RETRIES}"
                    )

            if not opened:
                logger.warning(f"Skipping product {product.product_url}")
                continue

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            product_tree = html.fromstring(bytes(str(soup), encoding="utf-8"))

            product_detail_nutrition_layout = product_tree.xpath(
                PRODUCT_DETAIL_NUTRITION_LAYOUT_FULL_XPATH
            )

            kkal = protein = fats = carbs = None
            if product_detail_nutrition_layout:
                section = product_detail_nutrition_layout[0]

                rows = section.xpath(".//div/div")
                for row in rows:
                    name_el = row.xpath(".//div[1]")
                    value_el = row.xpath(".//div[2]")
                    if not name_el or not value_el:
                        continue
                    name = name_el[0].text_content().strip().lower()
                    value = value_el[0].text_content().strip().lower()
                    if KKAL_TEXT_PATTERN in name:
                        kkal = value
                    elif FATS_TEXT_PATTERN in name:
                        fats = value
                    elif CARBS_TEXT_PATTERN in name:
                        carbs = value
                    elif PROTEIN_TEXT_PATTERN in name:
                        protein = value

            product.kkal = kkal
            product.protein = protein
            product.fats = fats
            product.carbs = carbs

            products.append(product)

        diskcache_cache.set(cache_tag, products)
        await browser.close()

    return products
