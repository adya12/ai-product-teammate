import gradio as gr
from huggingface_hub import InferenceClient
import os
import requests

# -----------------------------
# LOAD HF TOKEN
# -----------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

# Create inference client
client = InferenceClient(
    model="google/flan-t5-large",   # stable free model
    token=HF_TOKEN
)

# -----------------------------
# MAIN FUNCTION
# -----------------------------


def generate_product_spec(user_idea):

    try:

        prompt = f"""
You are an expert Product Manager.

Create structured product thinking output:

## Problem framing
## User personas
## Metrics
## Hypotheses
## Experiments
## Product Spec

Idea:
{user_idea}
"""

        headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/google/flan-t5-large",
            headers=headers,
            json={"inputs": prompt}
        )

        # 🔥 THIS IS THE IMPORTANT PART
        return f"STATUS CODE: {response.status_code}\n\nRESPONSE:\n{response.text}"

    except Exception as e:
        return f"PYTHON ERROR:\n{str(e)}"
# -----------------------------
# GRADIO UI
# -----------------------------

demo = gr.Interface(
    fn=generate_product_spec,
    inputs=gr.Textbox(label="Enter your messy idea"),
    outputs="text",
    title="AI Product Teammate"
)

demo.launch()
