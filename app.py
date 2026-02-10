import gradio as gr
from huggingface_hub import InferenceClient

# Use hosted inference (no local model loading)
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

def generate_product_spec(user_idea):

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

demo = gr.Interface(
    fn=generate_product_spec,
    inputs=gr.Textbox(label="Enter your messy idea"),
    outputs="text",
    title="AI Product Teammate"
)

demo.launch()
