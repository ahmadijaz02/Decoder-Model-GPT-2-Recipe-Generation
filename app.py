import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re  
import os
import subprocess
import json

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Recipe Generator",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- 2. Custom CSS for a New UI ---
st.markdown("""
<style>
    /* Remove Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Simple Title */
    .title-container {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 20px;
    }
    .title-container h1 {
        font-size: 3em;
        font-weight: 700;
        color: #111; /* Dark text */
    }
    .title-container p {
        font-size: 1.15em;
        color: #333; /* Darker subtext */
    }

    /* Card Headers (now simple headers) */
    [data-testid="stHorizontalBlock"] h2 {
        color: #111;
        font-weight: 600;
        font-size: 1.75em;
        border-bottom: 2px solid #eee;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }

    /* Custom Form Submit Button (Simple Blue) */
    [data-testid="stFormSubmitButton"] button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #0056b3;
    }

    /* Style for the generated recipe output */
    .recipe-box {
        background-color: #f8f8f8; /* Light gray box for output */
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 25px;
        min-height: 400px;
    }
    .recipe-box h3 {
        color: #111;
        border-bottom: 2px solid #007bff;
        padding-bottom: 10px;
        margin-top: 0;
    }
    .recipe-box p {
        font-size: 1.05em;
        line-height: 1.7;
        color: #333; /* Dark text */
    }
    
</style>
""", unsafe_allow_html=True)


# --- 3. Model Download Logic ---
MODEL_PATH = "final_model"

@st.cache_resource
def setup_and_download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model folder not found. Starting download from Kaggle...")
        
        if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
            st.error("Kaggle API secrets not found. Please add KAGGLE_USERNAME and KAGGLE_KEY to your Streamlit secrets.")
            return False

        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        api_creds = {
            "username": st.secrets["KAGGLE_USERNAME"],
            "key": st.secrets["KAGGLE_KEY"]
        }
        
        with open(kaggle_json_path, "w") as f:
            json.dump(api_creds, f)
            
        try:
            subprocess.run(["chmod", "600", kaggle_json_path], check=True)
        except Exception as e:
            print(f"Warning: Could not set file permissions. {e}")

        try:
            print("Downloading model from Kaggle...")
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "ahmadijaz92/genai-p3-t1",
                    "-p", ".",
                    "--unzip"
                ],
                check=True
            )
            print("Model downloaded and unzipped successfully.")
            return True
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download model from Kaggle: {e}")
            return False
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return False
    else:
        print("Model folder already exists.")
        return True

# --- 4. Model Loading ---

model_ready = setup_and_download_model()

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
        st.error(f"Please make sure your model files are in a folder named 'final_model' in the same directory as 'app.py'")
        return None, None

generator = None
tokenizer = None

if model_ready:
    with st.spinner("Warming up the AI chef... This may take a moment."):
        generator, tokenizer = load_model()

# --- 5. App Interface ---
st.markdown("""
<div class="title-container">
    <h1>AI Recipe Generator</h1>
    <p>Turn your ingredients into a delicious dish with the help of GPT-2.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h2>What's in your kitchen?</h2>", unsafe_allow_html=True)
    
    # --- 6. User Input Form ---
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
        
        with st.expander("Advanced Settings"):
            temp = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=0.7, step=0.1)
            max_tokens = st.slider("Recipe Length (Max Tokens)", min_value=50, max_value=250, value=150, step=10)

        submit_button = st.form_submit_button(label="Generate Recipe")

with col2:
    st.markdown("<h2>Your AI-Generated Recipe</h2>", unsafe_allow_html=True)
    
    output_container = st.container()

# --- 7. Generation Logic ---
if submit_button and generator:
    if not title or not ingredients_raw:
        st.error("Please provide both a title and ingredients.")
    else:
        with st.spinner("Generating your recipe..."):
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

                # --- 8. Process and Display Output ---
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

elif not generator and model_ready:
    st.error("Model is ready but failed to load. Please check the app logs.")
elif not model_ready:
    st.error("Model download failed. The app cannot function. Please check your Kaggle API secrets and dataset path.")

