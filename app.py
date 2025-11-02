import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re  # For formatting the output

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="🍳 Recipe Generator",
    page_icon="🍳",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- 2. Model Loading ---
MODEL_PATH = "./final_model"

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    print("--- Loading model and tokenizer ---")
    
    # Load the fine-tuned tokenizer
    # We set padding_side='left' for batch generation (as we learned)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = 'left'
    
    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    
    # Set the pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create the text-generation pipeline
    # We use device=-1 for CPU to ensure compatibility with services
    # like Streamlit Cloud. Change to device=0 if you have a GPU.
    generator_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1  # Use device=0 if you are running this on a GPU machine
    )
    print("--- Model and tokenizer loaded successfully ---")
    return generator_pipeline, tokenizer

# Load the model and show a spinner
with st.spinner("Loading the recipe model... This may take a moment."):
    generator, tokenizer = load_model()

# --- 3. App Interface ---
st.title("🍳 AI Recipe Generator")
st.markdown("This app uses a fine-tuned **GPT-2 model** to generate new recipes based on a title and ingredients. The model was trained on the `3A2M_EXTENDED.csv` dataset.")

# --- 4. User Input ---
with st.form(key="recipe_form"):
    # Input for Recipe Title
    title = st.text_input(
        "Enter a Recipe Title:",
        "Spicy Chicken Pasta"
    )

    # Input for Ingredients
    ingredients_raw = st.text_area(
        "Enter Ingredients (comma-separated):",
        "chicken breast, pasta, cayenne pepper, olive oil, garlic, tomatoes"
    )
    
    # Generation parameters in a collapsible section
    with st.expander("Advanced Settings"):
        temp = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=0.7, step=0.1)
        max_tokens = st.slider("Max Recipe Length (Tokens)", min_value=50, max_value=250, value=150, step=10)

    # Submit button for the form
    submit_button = st.form_submit_button(label="Generate Recipe")

# --- 5. Generation Logic ---
if submit_button:
    if not title or not ingredients_raw:
        st.error("Please provide both a title and ingredients.")
    else:
        with st.spinner("Brewing up your recipe... 🧑‍🍳"):
            # Clean and format the user's input
            title_clean = title.strip().lower()
            ingredients_clean = ", ".join([ing.strip().lower() for ing in ingredients_raw.split(',')])

            # This prompt format MUST match the one used during training
            prompt = (
                f"TITLE: {title_clean}\n"
                f"INGREDIENTS: {ingredients_clean}\n"
                f"RECIPE:"
            )

            try:
                # Call the pipeline
                generated_output = generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    no_repeat_ngram_size=2,
                    temperature=temp,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

                # --- 6. Process and Display Output ---
                full_text = generated_output[0]['generated_text']
                recipe_part = full_text[len(prompt):].strip()

                # Clean the output (remove extra EOS tokens)
                if tokenizer.eos_token in recipe_part:
                    recipe_part = recipe_part.split(tokenizer.eos_token)[0]

                # --- Improve readability by adding newlines ---
                # This turns "1. step one 2. step two" into:
                # 1. step one
                # 2. step two
                # We use regex to add a newline before any number followed by a period
                formatted_recipe = re.sub(r' (\d+\.)', r'\n\1', recipe_part).strip()

                st.subheader(f"Here's your recipe for: {title}")
                st.markdown(formatted_recipe)

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")