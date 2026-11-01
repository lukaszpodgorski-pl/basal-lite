import fake_vllm
import pytest


@pytest.fixture(autouse=True)
def fresh_option_code_ids():
    """The fake tokenizer's option code ids from 100000 in every test, whatever ran before it."""
    fake_vllm.CODES.clear()
