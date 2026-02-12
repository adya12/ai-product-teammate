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
You are a senior product manager designing AI-native fintech products.

Think deeply before answering.

IMPORTANT REQUIREMENTS:
- Avoid generic personas like “merchant” or “customer”; use behavioral or risk-based segments grounded in measurable signals.
- Focus on workflow design and decision systems (what the product does and how users experience it), not model architecture.
- Explain the mechanism: how AI changes decisions, reduces friction, or adds targeted friction.
- Explicitly discuss tradeoffs between fraud prevention, approval rate, revenue, and customer experience.
- Avoid buzzwords and unrealistic fintech changes (e.g., pricing controlled by banks/issuers). Keep it operational and shippable.
- Be concise, specific, and practical.

OUTPUT STRUCTURE (use these exact headings):

## Problem Framing
- What is the underlying problem and why now?
- Where in the journey does it occur and what causes it?

## Behavioral User Segments
- 3–5 segments based on observable behaviors (e.g., first-time buyer, high-AOV, repeat trusted, cross-border, high-velocity).
- For each: motivation + risk pattern + what they need.

## Core Metrics (including risk metrics)
- Business metrics (conversion, AOV, revenue lift)
- Risk metrics (chargeback rate, fraud loss rate, false positives/decline rate)
- Experience metrics (latency, step drop-off, CS contacts)

## Key Tradeoffs
- Name the key tensions and how you’ll balance them (e.g., approval vs losses, friction vs trust, speed vs accuracy).

## AI Mechanism (how the system works)
- Describe the decisioning loop at a product level: signals → risk/intent score → action → feedback loop.
- What actions change the user flow dynamically? (e.g., step-up verification, payment option ordering, retry guidance)

## Solution Strategy
- End-to-end approach: detection, decisioning, UX interventions, merchant controls, and transparency.
- Include how it scales operationally (automation, monitoring, safe rollout).

## Recommended Experiments
- 2–4 experiments with clear control/treatment, target metrics, and success criteria.
- Include guardrails (fraud loss ceiling, latency ceiling, CS contact rate).

## Product Recommendations
- Immediate priorities (ship in 4–8 weeks)
- Near-term enhancements (next 1–2 quarters)
- Future exploration (nice-to-have / longer-term)

USER IDEA:
{user_idea}

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



