#!/usr/bin/env python3
"""
RunPod Serverless Handler for Sophie AI Worker
Bridges RunPod Serverless jobs to llama-server OpenAI-compatible API

Session 751: Fix missing handler for job processing

Architecture:
  RunPod Job Queue → handler.py → llama-server (port 8080) → Response
"""

import os
import asyncio
import aiohttp
import runpod

# Configuration
LLAMA_SERVER_URL = f"http://localhost:{os.getenv('LLAMA_SERVER_PORT', '8080')}"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))  # 2 minutes default


async def async_handler(job: dict) -> dict:
    """
    Async handler for RunPod Serverless jobs

    Input formats supported:
    1. OpenAI-compatible (direct):
       {"input": {"messages": [...], "max_tokens": 200}}

    2. Wrapped OpenAI route:
       {"input": {"openai_route": "/v1/chat/completions", "openai_input": {...}}}

    3. Raw prompt:
       {"input": {"prompt": "Your question here"}}
    """
    job_input = job.get("input", {})

    # Determine the request type and build the request
    if "openai_route" in job_input:
        # Wrapped format: explicit route + input
        route = job_input["openai_route"]
        payload = job_input.get("openai_input", {})
    elif "messages" in job_input:
        # Direct OpenAI format
        route = "/v1/chat/completions"
        payload = job_input
    elif "prompt" in job_input:
        # Simple prompt format - convert to chat
        route = "/v1/chat/completions"
        payload = {
            "model": job_input.get("model", "buchhaltgenie"),
            "messages": [{"role": "user", "content": job_input["prompt"]}],
            "max_tokens": job_input.get("max_tokens", 500),
            "temperature": job_input.get("temperature", 0.7)
        }
    else:
        # Default: assume it's a chat completion request
        route = "/v1/chat/completions"
        payload = job_input

    # Ensure model is set
    if "model" not in payload:
        payload["model"] = "buchhaltgenie"

    # Make request to llama-server
    url = f"{LLAMA_SERVER_URL}{route}"

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    return {
                        "error": {
                            "message": f"llama-server returned HTTP {response.status}",
                            "details": error_text,
                            "type": "llama_server_error"
                        }
                    }
    except asyncio.TimeoutError:
        return {
            "error": {
                "message": f"Request timed out after {REQUEST_TIMEOUT}s",
                "type": "timeout_error"
            }
        }
    except aiohttp.ClientError as e:
        return {
            "error": {
                "message": f"Connection error: {str(e)}",
                "type": "connection_error"
            }
        }
    except Exception as e:
        return {
            "error": {
                "message": f"Unexpected error: {str(e)}",
                "type": "internal_error"
            }
        }


def handler(job: dict) -> dict:
    """
    Synchronous wrapper for the async handler
    Required for runpod.serverless.start()
    """
    return asyncio.get_event_loop().run_until_complete(async_handler(job))


if __name__ == "__main__":
    print("[Handler] Starting RunPod Serverless Handler")
    print(f"[Handler] llama-server URL: {LLAMA_SERVER_URL}")
    print(f"[Handler] Request timeout: {REQUEST_TIMEOUT}s")

    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True  # Support streaming responses
    })
