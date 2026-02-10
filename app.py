import gradio as gr
from transformers import pipeline
import traceback

# ------------------------------------------------
# MODEL INITIALIZATION
# ------------------------------------------------

try:
    print("Loading model...")

    # Small fast model that works on free Spaces CPU
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    print("Model loaded successfully.")

except Exception as e:
    print("MODEL LOAD ERROR:")
    print(traceback.format_exc())
    generator = None

# ------------------------------------------------
# MAIN FUNCTION
# ------------------------------------------------

def generate_product_spec(user_idea):

    try:

        if generator is None:
            return "ERROR: Model failed to load. Check container logs."

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

        result = generator(
            prompt,
            max_new_tokens=200
        )

        return result[0]["generated_text"]

    except Exception as e:
        # Show detailed error inside UI
        error_message = f"""
DEBUG ERROR:

{str(e)}

TRACEBACK:
{traceback.format_exc()}
"""
        print(error_message)
        return error_message

# ------------------------------------------------
# UI
# ------------------------------------------------

demo = gr.Interface(
    fn=generate_product_spec,
    inputs=gr.Textbox(label="Enter your messy idea"),
    outputs="text",
    title="AI Product Teammate"
)

demo.launch()
