import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re  

st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🧑‍🍳",
    layout="wide",  
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f2f6; /* Light gray background */
        background-image: none;
    }

    /* Custom Title */
    .title-container {
        text-align: center;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .title-container h1 {
        font-size: 2.5em;
        font-weight: 700;
        color: #2a3a4b;
    }
    .title-container p {
        font-size: 1.1em;
        color: #556;
    }

    /* Custom Form Submit Button */
    [data-testid="stFormSubmitButton"] button {
        background-color: #28a745; /* Green */
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #218838;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Style for the generated recipe output */
    .recipe-box {
        background-color: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        min-height: 400px;
    }
    .recipe-box h3 {
        color: #2a3a4b;
        border-bottom: 2px solid #28a745;
        padding-bottom: 10px;
    }
    .recipe-box p {
        font-size: 1.05em;
        line-height: 1.6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


MODEL_PATH = "/kaggle/input/genai-project3/final_model/" 


@st.cache_resource
def load_model():
    print("--- Loading model and tokenizer ---")
    
    try:

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = 'left'

        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        
        tokenizer.pad_token = tokenizer.eos_token
        
        generator_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1  
        )
        print("--- Model and tokenizer loaded successfully ---")
        return generator_pipeline, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please make sure your model files are correctly located at {MODEL_PATH}")
        return None, None

with st.spinner("Warming up the AI chef... This may take a moment."):
    generator, tokenizer = load_model()

st.markdown("""
<div class="title-container">
    <h1>🧑‍🍳 AI Recipe Generator</h1>
    <p>Turn your ingredients into a delicious dish with the help of GPT-2.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.header("What's in your kitchen? 🛒")
    
    with st.form(key="recipe_form"):
        title = st.text_input(
            "What do you want to make?",
            "Spicy Chicken Pasta",
            help="Give your recipe a name."
        )

        ingredients_raw = st.text_area(
            "What ingredients do you have?",
            "chicken breast, pasta, cayenne pepper, olive oil, garlic, tomatoes",
            help="Enter ingredients, separated by commas.",
            height=150
        )
        
        with st.expander("✨ Tweak the AI's Creativity"):
            temp = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=0.7, step=0.1)
            max_tokens = st.slider("Recipe Length (Max Tokens)", min_value=50, max_value=250, value=150, step=10)

        submit_button = st.form_submit_button(label="🍽️ Generate Recipe!")

with col2:
    st.header("Your AI-Generated Recipe 📜")
    
    output_container = st.container()

if submit_button and generator:
    if not title or not ingredients_raw:
        st.error("Please provide both a title and ingredients.")
    else:
        with st.spinner("Brewing up your recipe... 🧑‍🍳"):
            title_clean = title.strip().lower()
            ingredients_clean = ", ".join([ing.strip().lower() for ing in ingredients_raw.split(',')])

            prompt = (
                f"TITLE: {title_clean}\n"
                f"INGREDIENTS: {ingredients_clean}\n"
                f"RECIPE:"
            )

            try:
                generated_output = generator(
                    prompt,
                    max_new_tokens=max_tokens,
                    no_repeat_ngram_size=2,
                    temperature=temp,
                    top_k=50,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

                # --- 7. Process and Display Output ---
                full_text = generated_output[0]['generated_text']
                recipe_part = full_text[len(prompt):].strip()

                if tokenizer.eos_token in recipe_part:
                    recipe_part = recipe_part.split(tokenizer.eos_token)[0]

                formatted_recipe_html = re.sub(r' (\d+\.)', r'<br><br>\1', recipe_part).strip()
                formatted_recipe_html = f"<p>{formatted_recipe_html}</p>"

                with output_container:
                    st.markdown(f"""
                    <div class="recipe-box">
                        <h3>{title.title()}</h3>
                        {formatted_recipe_html}
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during generation: {e}")

elif not generator:
    st.error("Model could not be loaded. The app cannot function.")

