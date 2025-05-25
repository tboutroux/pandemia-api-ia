import pytest
from fastapi import HTTPException
from app.core.security import get_api_key
from app.config.settings import API_KEY

@pytest.mark.asyncio
async def test_valid_api_key_header():
    assert await get_api_key(api_key_header=API_KEY, api_key_query=None) == API_KEY

@pytest.mark.asyncio
async def test_valid_api_key_query():
    assert await get_api_key(api_key_header=None, api_key_query=API_KEY) == API_KEY

@pytest.mark.asyncio
async def test_invalid_api_key():
    with pytest.raises(HTTPException):
        await get_api_key(api_key_header="wrong", api_key_query=None)