# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "requests",
#   "python-dotenv",
# ]
# ///

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
AGENT_ID = os.getenv("AGENT_ID", "agent_2501k520e4p6eqs87ga1ke8a67fs")

BASE_URL = "https://api.elevenlabs.io/v1/convai/agents"


def send_message(agent_id: str, message: str) -> dict:
    """Send a text message to the ElevenLabs agent and get response."""
    url = f"{BASE_URL}/{agent_id}/simulate-conversation"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "simulation_specification": {
            "user_turns": [
                {
                    "message": message
                }
            ],
            "simulated_user_config": {
                "user_name": "Test User"
            }
        }
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    return response.json()


def test_conversation(test_cases: list[str]):
    """Run a series of test messages through the agent."""
    print(f"Testing agent: {AGENT_ID}\n")

    for i, user_message in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"{'='*60}")
        print(f"👤 USER: {user_message}")

        try:
            result = send_message(AGENT_ID, user_message)

            # Get agent response
            agent_response = result.get("agent_response", "")
            print(f"🤖 ASSISTANT: {agent_response}")

            # Log any tools called
            if "tools_called" in result and result["tools_called"]:
                print(f"\n🛠️  TOOLS CALLED:")
                for tool in result["tools_called"]:
                    print(f"   - {tool.get('name')}: {tool.get('arguments')}")

            # Log token usage
            if "usage" in result:
                usage = result["usage"]
                print(f"\n📊 Tokens: {usage.get('total_tokens')} ({usage.get('input_tokens')} in, {usage.get('output_tokens')} out)")

            print()

        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            break


if __name__ == "__main__":
    # Test cases
    test_messages = [
        "I want to be connected to the Burnaby store",
        "I want to talk with someone from sales",
        "Goodbye"
    ]

    test_conversation(test_messages)
