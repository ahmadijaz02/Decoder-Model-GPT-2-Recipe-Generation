import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch
import re  # For formatting the output
import os
import subprocess
import json

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Recipe Generator",
    page_icon="🧑‍🍳",
    layout="wide",  # Use a wide layout for a more modern feel
    initial_sidebar_state="auto"
)

# --- 2. Custom CSS for a Unique UI ---
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
        color: #2a3a4b; /* Dark text */
    }
    .title-container p {
        font-size: 1.1em;
        color: #556;
    }

    /* Custom Form Submit Button (New Blue Theme) */
    [data-testid="stFormSubmitButton"] button {
        background-color: #0068c9; /* A clean, modern blue */
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
        background-color: #004a99; /* Darker blue on hover */
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
        border-bottom: 2px solid #0068c9; /* Matching blue border */
        padding-bottom: 10px;
    }
    .recipe-box p {
        font-size: 1.05em;
        line-height: 1.6;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)


# --- 3. Model Download Logic ---
MODEL_PATH = "final_model" # The model will be in this local folder

@st.cache_resource
def setup_and_download_model():
    # Only run this if the model folder doesn't already exist
    if not os.path.exists(MODEL_PATH):
        print("Model folder not found. Starting download from Kaggle...")
        
        # Check for secrets
        if "KAGGLE_USERNAME" not in st.secrets or "KAGGLE_KEY" not in st.secrets:
            # This error will be displayed on the Streamlit app
            st.error("Kaggle API secrets not found. Please add KAGGLE_USERNAME and KAGGLE_KEY to your Streamlit secrets.")
            return False

        # Set up the Kaggle API credentials
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        
        kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
        api_creds = {
            "username": st.secrets["KAGGLE_USERNAME"],
            "key": st.secrets["KAGGLE_KEY"]
        }
        
        with open(kaggle_json_path, "w") as f:
            json.dump(api_creds, f)
            
        # Set correct permissions for the API key
        try:
            subprocess.run(["chmod", "600", kaggle_json_path], check=True)
        except Exception as e:
            print(f"Warning: Could not set file permissions. {e}")

        # Download the dataset from Kaggle
        try:
            print("Downloading model from Kaggle...")
            # Command: kaggle datasets download -d ahmadijaz92/genai-project3 -p . --unzip
            subprocess.run(
                [
                    "kaggle", "datasets", "download",
                    "ahmadijaz92/genai-project3", # Your dataset path
                    "-p", ".",                   # Download to current directory
                    "--unzip"                    # Unzip the file
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

# Run the setup function first
model_ready = setup_and_download_model()

# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    print("--- Loading model and tokenizer ---")
    
    try:
        # Load the fine-tuned tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
        tokenizer.padding_side = 'left'
        
        # Load the fine-tuned model
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
        
        # Set the pad token
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create the text-generation pipeline
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

# Load the model and show a spinner
generator = None
tokenizer = None

if model_ready:
    with st.spinner("Warming up the AI chef... This may take a moment."):
        generator, tokenizer = load_model()

# --- 5. App Interface ---
st.markdown("""
<div class="title-container">
    <h1>🧑‍🍳 AI Recipe Generator</h1>
    <p>Turn your ingredients into a delicious dish with the help of GPT-2.</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Removed emoji from header
    st.header("What's in your kitchen?")
    
    # --- 6. User Input Form ---
    with st.form(key="recipe_form"):
        # Input for Recipe Title
        title = st.text_input(
            "What do you want to make?",
            "Spicy Chicken Pasta",
            help="Give your recipe a name."
        )

        # Input for Ingredients
        ingredients_raw = st.text_area(
            "What ingredients do you have?",
            "chicken breast, pasta, cayenne pepper, olive oil, garlic, tomatoes",
            help="Enter ingredients, separated by commas.",
            height=150
        )
        
        # Generation parameters in a collapsible section
        # Removed emoji from expander
        with st.expander("Advanced Settings"):
            temp = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.5, value=0.7, step=0.1)
            max_tokens = st.slider("Recipe Length (Max Tokens)", min_value=50, max_value=250, value=150, step=10)

        # Submit button for the form
        # Removed emoji from button label
        submit_button = st.form_submit_button(label="Generate Recipe!")

with col2:
    # Removed emoji from header
    st.header("Your AI-Generated Recipe")
    
    # This is where the output will be placed
    output_container = st.container()

# --- 7. Generation Logic ---
if submit_button and generator:
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

                # --- 8. Process and Display Output ---
                full_text = generated_output[0]['generated_text']
                recipe_part = full_text[len(prompt):].strip()

                # Clean the output (remove extra EOS tokens)
                if tokenizer.eos_token in recipe_part:
                    recipe_part = recipe_part.split(tokenizer.eos_token)[0]

                # --- Improve readability by adding HTML newlines ---
                # We use regex to add <br> tags for proper HTML rendering
                formatted_recipe_html = re.sub(r' (\d+\.)', r'<br><br>\1', recipe_part).strip()
                # Wrap in <p> tag for styling
                formatted_recipe_html = f"<p>{formatted_recipe_html}</p>"

                # Display the result in the right-hand column
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

