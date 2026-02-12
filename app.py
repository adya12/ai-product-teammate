import gradio as gr
import requests
import os

# ----------------------------------------
# Load API key from HuggingFace Secrets
# ----------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ----------------------------------------
# Main function
# ----------------------------------------

def generate_product_spec(user_idea):

    try:

        url = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        prompt = f"""
You are a senior product manager at Square designing AI-native fintech products.

Think deeply before answering.

IMPORTANT REQUIREMENTS:
- Avoid generic personas like “merchant” or “customer”; instead use behavioral or risk-based personas.
- Explain the mechanism of how AI changes decision-making or workflow.
- Explicitly discuss tradeoffs between fraud prevention, approval rate, and revenue.
- Avoid buzzwords; focus on operational reality.
- Think about how this would work at scale for thousands of merchants.

OUTPUT STRUCTURE:

## Problem Framing
## Behavioral User Segments
## Core Metrics (including risk metrics)
## Key Tradeoffs
## AI Mechanism (how the system works)
## Solution Strategy
## Recommended Experiments
## Product Recommendations (Must / Should / Could)

USER IDEA:
{IDEA}

"""

        payload = {
            "model":"llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are an expert product manager."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(url, headers=headers, json=payload)

        result = response.json()

        # DEBUG: print full API response in container logs
        print("FULL RESPONSE:", result)

        # Handle API errors safely
        if "choices" not in result:
            return f"API ERROR:\n{result}"

        return result["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR:\n{str(e)}"


# ----------------------------------------
# Gradio UI
# ----------------------------------------

demo = gr.Interface(
    fn=generate_product_spec,
    inputs=gr.Textbox(label="Enter your messy product idea"),
    outputs="text",
    title="AI Product Teammate"
)

demo.launch()



