import gradio as gr
from transformers import pipeline

# Load free hosted model
generator = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta"
)

def generate_product_spec(user_idea):

    prompt = f"""
You are an expert Product Manager.

Convert idea into structured product thinking.

OUTPUT:

## Problem framing
## User personas
## Metrics
## Hypotheses
## Experiments
## Product Spec

Idea:
{user_idea}
"""

    result = generator(prompt, max_new_tokens=400)[0]["generated_text"]

    return result

demo = gr.Interface(
    fn=generate_product_spec,
    inputs=gr.Textbox(label="Enter your messy idea"),
    outputs="text",
    title="AI Product Teammate"
)

demo.launch()
