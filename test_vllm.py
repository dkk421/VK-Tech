import asyncio
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "src"),
)
from medhack_ai_assistant.ml import AsyncVLLMClient, LLMMessage


async def main():
    client = AsyncVLLMClient()

    result = await client.chat_json([
        LLMMessage(
            role="system",
            content="Отвечай только JSON.",
        ),
        LLMMessage(
            role="user",
            content='Верни JSON: {"status":"ok"}',
        ),
    ])

    print(result)

    await client.aclose()


asyncio.run(main())