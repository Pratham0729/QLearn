
"""LLM abstraction layer using OpenRouter."""

import logging

import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPTS = {
    "beginner": """You are QubitAI, the AI quantum computing tutor for QuantumVerse.

Your job is to answer a broad range of questions about quantum computing,
quantum physics, quantum algorithms, quantum circuits, and related topics.

TEACHING STYLE:
- Explain concepts in simple, accessible language.
- Use analogies, practical examples, and step-by-step explanations.
- Avoid unnecessary mathematical notation.
- Introduce technical terms and explain what they mean.
- Keep answers concise but complete. Expand when the user asks for detail.

TOPICS YOU CAN EXPLAIN:
- Qubits, superposition, measurement, and entanglement.
- Quantum gates, circuits, and quantum states.
- Quantum algorithms, including Grover's and Shor's algorithms.
- Quantum hardware, error correction, and applications.
- Classical versus quantum computing.
- Quantum programming frameworks and learning exercises.

BEHAVIOR:
- Answer the user's actual question directly.
- Do not restrict answers to the lessons stored in the application.
- Handle follow-up questions using the conversation history.
- If a question is outside quantum computing, answer it if you can,
  while explaining any relevant connection to quantum computing.
- If you are uncertain, say so rather than inventing facts.
- Correct misconceptions politely.
- Never claim that quantum computers solve every problem faster.
- Use examples whenever they help understanding.
- Use proper quantum notation when useful, but explain it simply.
""",

    "intermediate": """You are QubitAI, an expert quantum computing tutor
for intermediate learners on QuantumVerse.

Answer a broad range of questions about quantum computing, quantum physics,
quantum information, algorithms, circuits, hardware, and applications.

Use linear algebra, complex amplitudes, Dirac notation, matrices, and
Bloch sphere representations when useful.

Explain mathematical steps and connect theoretical concepts to circuit
implementations and practical examples.

Cover topics including quantum gates, entanglement, measurement,
interference, Grover's algorithm, Shor's algorithm, quantum error correction,
quantum complexity, and quantum programming.

Answer the actual question, not just questions matching a predefined list.
Use conversation history to understand follow-up questions.
Do not limit explanations to the application's stored lessons.

Be accurate about quantum speedups and hardware limitations.
Distinguish established results from active research.
If uncertain, state the uncertainty instead of fabricating information.
Adapt the depth and length to the user's request.
""",

    "advanced": """You are QubitAI, an expert quantum computing mentor
for advanced learners on QuantumVerse.

Provide technically rigorous explanations across quantum information,
quantum mechanics, quantum algorithms, quantum complexity, quantum hardware,
and quantum error correction.

Use Dirac notation, Hilbert spaces, density matrices, tensor products,
unitary operators, Hamiltonians, and mathematical derivations where relevant.

Discuss topics such as BQP, quantum circuit complexity, Shor's algorithm,
Grover's algorithm, fault-tolerant computation, stabilizer codes,
variational algorithms, and NISQ-era limitations.

Answer open-ended questions and follow-ups using the conversation history.
Do not restrict answers to the application's lesson database or a fixed
set of predefined questions.

Distinguish proven results, assumptions, approximations, and active research.
Do not invent citations, experimental results, or mathematical claims.
If a question is ambiguous, state your interpretation or ask for clarification.

Adapt the level of mathematical detail to the user's request.
""",
}


def _fallback_response(message: str, difficulty: str) -> str:
    """Return a transparent response when the AI provider is unavailable."""

    logger.warning(
        "Returning fallback response. AI provider is unavailable. "
        "Difficulty=%s, message_length=%d",
        difficulty,
        len(message),
    )

    return (
        "I'm temporarily unable to reach the AI model, so I can't generate "
        "a proper answer to your question right now. Please try again shortly. "
        "If this continues, the AI service configuration or provider "
        "availability needs to be checked."
    )


async def get_ai_response(
    message: str,
    history: list[dict],
    difficulty: str = "beginner",
    context: dict | None = None,
) -> str:
    """
    Generate an AI response using OpenRouter.

    Logs provider errors without exposing the API key.
    Falls back gracefully if the provider is unavailable.
    """

    api_key = settings.OPENROUTER_API_KEY
    model = settings.AI_MODEL

    # Check whether the API key is configured.
    if (
        not api_key
        or api_key.strip() == ""
        or api_key == "your-openrouter-key-here"
    ):
        logger.error(
            "OPENROUTER_API_KEY is missing or contains the default "
            "placeholder. Check the Render environment variables."
        )
        return _fallback_response(message, difficulty)

    if not model or not model.strip():
        logger.error(
            "AI_MODEL is missing or empty. Check the Render environment."
        )
        return _fallback_response(message, difficulty)

    # Select the system prompt for the requested difficulty.
    system = SYSTEM_PROMPTS.get(
        difficulty.lower(),
        SYSTEM_PROMPTS["beginner"],
    )

    # Add relevant application context, if supplied.
    if context:
        system += (
            "\n\nAdditional learning context from the application:\n"
            f"{context}\n"
            "Use this context when relevant, but do not treat it as "
            "a restriction on which questions you can answer."
        )

    messages = [{"role": "system", "content": system}]

    # Include recent conversation history.
    # Only accept valid user/assistant messages.
    valid_history = []

    for item in history[-10:]:
        role = item.get("role")
        content = item.get("content")

        if role in ("user", "assistant") and isinstance(content, str):
            if content.strip():
                valid_history.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

    messages.extend(valid_history)

    # Append the current user message.
    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.APP_URL,
        "X-Title": settings.APP_NAME,
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1200,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
            )

            # Raise for HTTP errors so the status and response
            # body can be recorded in the logs.
            response.raise_for_status()

            data = response.json()

            choices = data.get("choices", [])

            if not choices:
                logger.error(
                    "OpenRouter returned no choices. Model=%s, response=%s",
                    model,
                    str(data)[:1000],
                )
                return _fallback_response(message, difficulty)

            assistant_message = choices[0].get("message", {})
            content = assistant_message.get("content")

            if isinstance(content, str) and content.strip():
                logger.info(
                    "OpenRouter response successful. Model=%s",
                    model,
                )
                return content.strip()

            logger.error(
                "OpenRouter returned an empty or invalid message. "
                "Model=%s, response=%s",
                model,
                str(data)[:1000],
            )

            return _fallback_response(message, difficulty)

    except httpx.HTTPStatusError as exc:
        # Log the HTTP status and provider error body.
        # Never log request headers or the API key.
        logger.error(
            "OpenRouter HTTP error. Status=%s, Model=%s, Body=%s",
            exc.response.status_code,
            model,
            exc.response.text[:1500],
        )

        return _fallback_response(message, difficulty)

    except httpx.TimeoutException:
        logger.exception(
            "OpenRouter request timed out. Model=%s",
            model,
        )

        return _fallback_response(message, difficulty)

    except httpx.RequestError:
        logger.exception(
            "Could not connect to OpenRouter. Model=%s",
            model,
        )

        return _fallback_response(message, difficulty)

    except (ValueError, KeyError, IndexError):
        logger.exception(
            "Could not parse OpenRouter response. Model=%s",
            model,
        )

        return _fallback_response(message, difficulty)

    except Exception:
        logger.exception(
            "Unexpected error in AI provider. Model=%s",
            model,
        )

        return _fallback_response(message, difficulty)
