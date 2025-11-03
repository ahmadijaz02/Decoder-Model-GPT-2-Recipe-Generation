import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re  
import os
import subprocess
import json

st.set_page_config(
    page_title="Recipe Generator",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #f0f4f7; /* A light, clean blue-gray */
        background-image: none;
    }

    /* Remove Streamlit Header/Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom Title */
    .title-container {
        text-align: center;
        padding: 25px 0;
        margin-bottom: 30px;
    }
    .title-container h1 {
        font-size: 3em;
        font-weight: 700;
        color: #1a2c4e; /* Dark navy text */
    }
    .title-container p {
        font-size: 1.15em;
        color: #5a6a8a; /* Muted subtext */
    }

    /* Main content columns as cards */
    [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.05);
        padding: 30px;
        border-top: 5px solid #007bff; /* Accent color */
    }
    
    /* Card Headers */
    [data-testid="stHorizontalBlock"] h2 {
        color: #1a2c4e;
        font-weight: 600;
        font-size: 1.75em;
        border-bottom: 2px solid #f0f4f7;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }

    /* Custom Form Submit Button */
    [data-testid="stFormSubmitButton"] button {
        background-image: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 15px rgba(0,123,255,0.2);
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-image: linear-gradient(45deg, #0056b3, #007bff);
        box-shadow: 0 6px 20px rgba(0,123,255,0.3);
        transform: translateY(-2px);
    }

    /* Style for the generated recipe output */
    .recipe-box {
        background-color: #f8f9fa; /* Slight off-white for contrast */
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 25px;
        min-height: 400px;
    }
    .recipe-box h3 {
        color: #1a2c4e;
        border-bottom: 2px solid #007bff;
        padding-bottom: 10px;
        margin-top: 0;
    }
    .recipe-box p {
        font-size: 1.05em;
        line-height: 1.7;
        color: #333;
    }
    
    /* Style for the text inputs */
    [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #d1d9e0;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
    }

</style>
""", unsafe_allow_html=True)


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

st.markdown("""
<div class="title-container">
    <h1>AI Recipe Generator</h1>
    <p>Turn your ingredients into a delicious dish with the help of GPT-2.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h2>What's in your kitchen?</h2>", unsafe_allow_html=True)

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

