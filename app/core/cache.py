from pathlib import Path
from typing import Any, Optional

import diskcache

from app.core.config import settings


class ICache:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError

    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError


class DiskCache(ICache):
    def __init__(self, cache_dir: str = "~/.cache/recipeai"):
        Path(cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(cache_dir)

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self.cache.set(key, value, expire=settings.CACHE_TTL_SECONDS)


diskcache_cache = DiskCache()
