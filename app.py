import gradio as gr
from huggingface_hub import InferenceClient
import os

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

        result = client.text_generation(
            prompt,
            max_new_tokens=300
        )

        return result

    except Exception as e:
        # Show real error instead of generic "Error"
        return f"DEBUG ERROR:\n{str(e)}"

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
