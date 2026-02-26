"""
Request / response logging middleware.
"""

import time
from fastapi import Request
from loguru import logger


async def logging_middleware(request: Request, call_next):
    start = time.perf_counter()
    logger.info(f"→ {request.method} {request.url.path}")

    response = await call_next(request)

    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info(f"← {request.method} {request.url.path} [{response.status_code}] {elapsed}ms")

    return response
