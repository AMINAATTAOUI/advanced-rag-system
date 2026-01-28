"""
Caching utilities for the RAG system.
Implements in-memory caching with optional Redis support.
"""

import hashlib
import json
from typing import Any, Optional
from functools import wraps
from src.config import settings
from src.utils.logger import log


class SimpleCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self):
        self._cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        return self._cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in cache."""
        self._cache[key] = value
    
    def delete(self, key: str):
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self):
        """Clear all cache."""
        self._cache.clear()


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self):
        try:
            import redis
            self.client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True
            )
            self.client.ping()
            log.info("Redis cache initialized successfully")
        except Exception as e:
            log.warning(f"Redis not available, falling back to simple cache: {e}")
            self.client = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.client:
            return None
        try:
            value = self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            log.error(f"Error getting from cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set value in Redis cache."""
        if not self.client:
            return
        try:
            ttl = ttl or settings.cache_ttl
            self.client.setex(key, ttl, json.dumps(value))
        except Exception as e:
            log.error(f"Error setting cache: {e}")
    
    def delete(self, key: str):
        """Delete key from Redis cache."""
        if not self.client:
            return
        try:
            self.client.delete(key)
        except Exception as e:
            log.error(f"Error deleting from cache: {e}")
    
    def clear(self):
        """Clear all cache."""
        if not self.client:
            return
        try:
            self.client.flushdb()
        except Exception as e:
            log.error(f"Error clearing cache: {e}")


# Initialize cache based on settings
if settings.enable_cache:
    try:
        cache = RedisCache()
        if cache.client is None:
            cache = SimpleCache()
    except:
        cache = SimpleCache()
else:
    cache = SimpleCache()


def generate_cache_key(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.md5(key_data.encode()).hexdigest()


def cached(ttl: int = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (default: from settings)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                log.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            log.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator
