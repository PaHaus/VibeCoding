import json
import os
from dotenv import load_dotenv
import litellm

# Načtení proměnných ze souboru .env
load_dotenv()

# 1. Funkce a jejich nástroje
def add(a: float, b: float) -> float:
    return a + b

def subtract(a: float, b: float) -> float:
    return a - b

def multiply(a: float, b: float) -> float:
    return a * b

def divide(a: float, b: float) -> float:
    return "Chyba: Dělení nulou" if b == 0 else a / b

AVAILABLE_TOOLS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Sečte dvě čísla.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subtract",
            "description": "Odečte druhé číslo od prvního.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Vynásobí dvě čísla.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Vydělí první číslo druhým.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]

# 2. Agent třída využitím LiteLLM
class LiteLLMMathAgent:
    def __init__(self, model: str = "openrouter/meta-llama/llama-3.3-70b-instruct:free"):
        # LiteLLM vyžaduje prefix 'openrouter/' pro OpenRouter modely
        self.model = model
        self.messages = [
            {
                "role": "system",
                "content": "Jsi přesný matematický asistent. Pro výpočty vždy používej dostupné nástroje.",
            }
        ]

    def chat(self, user_prompt: str, max_iterations: int = 10) -> str:
        self.messages.append({"role": "user", "content": user_prompt})

        for step in range(1, max_iterations + 1):
            # LiteLLM volání přes litellm.completion()
            response = litellm.completion(
                model=self.model,
                messages=self.messages,
                tools=tools_schema,
                tool_choice="auto",
            )

            response_message = response.choices[0].message
            # Převod odpovědi do dict formátu kompatibilního s konverzací
            self.messages.append(response_message.model_dump())

            # Pokud model nevyžaduje spuštění žádného nástroje, vrátíme odpověď
            if not response_message.tool_calls:
                return response_message.content

            # Volání funkcí požadovaných přes tool_calls
            for tool_call in response_message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                print(f"  [Tool]: {fn_name}({fn_args})")

                if fn_name in AVAILABLE_TOOLS:
                    result = AVAILABLE_TOOLS[fn_name](**fn_args)
                else:
                    result = f"Chyba: Nástroj {fn_name} neexistuje."

                self.messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": str(result),
                    }
                )

        return "Agent dosáhl maximálního počtu kroků."


# --- SPUŠTĚNÍ ---
if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        raise ValueError("Chybí OPENROUTER_API_KEY v .env souboru!")

    agent = LiteLLMMathAgent()

    print("--- Test 1: Složitější příkaz ---")
    print(agent.chat("Sečti 15 a 25, výsledek vynásob 4 a odečti 10."))

    print("\n--- Test 2: Navázání v paměti ---")
    print(agent.chat("Vyděl tento výsledek číslem 2."))