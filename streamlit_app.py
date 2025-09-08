import streamlit as st
from utils import extract_text_from_pdf, read_text_file, preprocess_text
from llm_model import QAGenerator
import pandas as pd
import base64
import io
import random 


#  Streamlit Configuration 
st.set_page_config(page_title="LLM-Powered Flashcard Generator", layout="wide")


#  Custom CSS for a more attractive and colorful UI 
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6; /* Light gray background */
    }
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4B0082; /* Indigo */
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    h2 {
        color: #6A5ACD; /* SlateBlue */
        text-align: center; /* Default for h2, overridden for "Provide Your Content" */
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #8A2BE2; /* BlueViolet */
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #9370DB; /* MediumPurple */
        transform: translateY(-2px);
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px;
    }
    .stRadio div[role="radiogroup"] label {
        background-color: black; /* Changed to black */
        color: white; /* Changed to white for readability */
        padding: 10px 20px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 2px solid #333; /* Darker border for contrast */
    }
    .stRadio div[role="radiogroup"] label:has(input:checked) {
        background-color: #333; /* Slightly lighter black when checked */
        border-color: #6A5ACD; /* SlateBlue */
        color: #FFFFFF; /* White */
        font-weight: bold;
    }

    /* Flashcard Styling */
    .flashcard-container {
        display: grid; /* Use CSS Grid for layout */
        /* Changed to ensure exactly 3 columns with equal width, regardless of screen size */
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
        justify-content: center;
        margin-top: 30px;
    }
    .flashcard {
        border-radius: 15px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
        padding: 25px;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.3s ease-in-out;
        border: 2px solid; /* Border color will be set dynamically */
        position: relative; /* Needed for absolute positioning of difficulty strip */
        overflow: hidden; /* Ensures strip stays within bounds to clip rotated elements */
    }
    .flashcard:hover {
        transform: translateY(-5px);
    }
    .flashcard-question {
        font-size: 1.25em;
        font-weight: bold;
        color: #8B4513; /* SaddleBrown */
        margin-bottom: 15px;
        z-index: 1; /* Ensure text is above the strip */
        padding-top: 30px; /* Add padding to push text below the diagonal strip */
    }
    .flashcard-answer {
        font-size: 1em;
        color: #2F4F4F; /* DarkSlateGray */
        line-height: 1.6;
        padding-top: 10px;
        border-top: 1px dashed; /* Border color will be set dynamically */
        z-index: 1; /* Ensure text is above the strip */
    }
    /* Difficulty Strip Styling (Diagonal) */
    .difficulty-strip {
        position: absolute;
        top: 0;
        right: 0; /* Position at top right */
        width: 120px; /* Increased width of the strip */
        height: 25px; /* Height of the strip */
        background-color: #FF0000; /* Default red, overridden by specific colors */
        transform-origin: top right; /* Set origin for rotation */
        transform: rotate(45deg) translate(25px, -8px); /* Rotate and then translate */
        border-bottom-left-radius: 10px; /* Rounded corner for a cleaner look */
        z-index: 2; /* Ensure strip is above content */
        display: flex;
        align-items: center;
        justify-content: center;
        color: white; /* Default text color for strip */
        font-weight: bold;
        font-size: 0.8em;
    }
    .stTextInput>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #D8BFD8;
        padding: 10px;
    }
    .stFileUploader label {
        background-color: #F0F8FF; /* AliceBlue */
        color: #4682B4; /* SteelBlue */
        border: 1px dashed #B0C4DE;
        border-radius: 8px;
        padding: 10px;
        cursor: pointer;
        text-align: center;
    }
    .stFileUploader label:hover {
        background-color: #E0FFFF; /* LightCyan */
    }
    /* Spinner customization */
    .stSpinner > div > div {
        color: #8A2BE2; /* BlueViolet */
    }
</style>
""", unsafe_allow_html=True)

# Model Loading 
@st.cache_resource
def get_qa_generator():
    """
    Initializes and returns the QAGenerator instance.
    This function is cached by Streamlit.
    """
    return QAGenerator()

# Get the cached QAGenerator instance
qa_generator = get_qa_generator()


#  Application Title and Description 
st.title("📚 LLM-Powered Flashcard Generator")
st.markdown("""
Welcome! This tool helps you convert educational content into interactive question-answer flashcards.
Simply upload a PDF or text file, or paste your study material directly, and let advanced
Large Language Models extract key information and generate relevant Q&A pairs for you.
""")


#  Input Section 
st.markdown("<h2 style='text-align: left; font-family: \"Inter\", sans-serif;'> Provide Your Content</h2>", unsafe_allow_html=True)

input_option = st.radio(
    "Select your preferred input method:",
    ("Upload File ⬆️", "Paste Text 📝"),
    index=0,
    help="Choose whether to upload a document (PDF/TXT) or paste text directly into the application."
)

uploaded_file = None
pasted_text = ""

if input_option == "Upload File ⬆️":
    uploaded_file = st.file_uploader(
        "Upload your educational content (.pdf or .txt):",
        type=["pdf", "txt"],
        help="Supported formats: PDF (.pdf) and Plain Text (.txt). Ensure text is clear and readable for best results."
    )
elif input_option == "Paste Text 📝":
    pasted_text = st.text_area(
        "Paste your educational content here:",
        height=300,
        placeholder="E.g., 'Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy into chemical energy...'",
        help="Paste any text content you want to convert into flashcards. Minimum ~50 characters for meaningful generation."
    )

num_qa_pairs = st.slider(
    "2. Number of Flashcards to Generate:",
    min_value=5,
    max_value=25,
    value=10,
    step=1,
    help="Adjust this slider to specify the approximate number of question-answer pairs you'd like to receive."
)

generate_button = st.button("3. Generate Flashcards ✨", type="primary", use_container_width=True)

flashcards_container = st.empty()

st.markdown("---")
st.header("Generated Flashcards")

if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

#  list of attractive color palettes for flashcards
flashcard_colors = [
    {"background_start": "#5D4037", "background_end": "#4E342E", "border": "#3E2723", "question_text": "#FFCCBC", "answer_text": "#D7CCC8", "line_color": "#8D6E63"}, # Deep Brown
    {"background_start": "#2E7D32", "background_end": "#1B5E20", "border": "#1B5E20", "question_text": "#C8E6C9", "answer_text": "#A5D6A7", "line_color": "#66BB6A"}, # Dark Green
    {"background_start": "#1565C0", "background_end": "#0D47A1", "border": "#0D47A1", "question_text": "#BBDEFB", "answer_text": "#90CAF9", "line_color": "#42A5F5"}, # Deep Blue
    {"background_start": "#EF6C00", "background_end": "#E65100", "border": "#E65100", "question_text": "#FFECB3", "answer_text": "#FFCC80", "line_color": "#FFA726"}, # Dark Orange
    {"background_start": "#C2185B", "background_end": "#AD1457", "border": "#AD1457", "question_text": "#F8BBD0", "answer_text": "#F48FB1", "line_color": "#EC407A"}, # Deep Pink
    {"background_start": "#6A1B9A", "background_end": "#4A148C", "border": "#4A148C", "question_text": "#E1BEE7", "answer_text": "#CE93D8", "line_color": "#AB47BC"}, # Deep Purple
    {"background_start": "#558B2F", "background_end": "#33691E", "border": "#33691E", "question_text": "#DCEDC8", "answer_text": "#C5E1A5", "line_color": "#9CCC65"}, # Dark Olive Green
    {"background_start": "#FBC02D", "background_end": "#F9A825", "border": "#F9A825", "question_text": "#FFFDE7", "answer_text": "#FFF59D", "line_color": "#FFEB3B"}, # Dark Yellow
]

# difficulty levels 
difficulty_colors = {
    "Easy": "white",
    "Medium": "yellow",
    "Hard": "black"
}


def get_simulated_difficulty(question):
    return random.choice(["Easy", "Medium", "Hard"])


if generate_button:
    input_content = ""
    if uploaded_file is not None:
        with st.spinner("🚀 Extracting text from your file... This may take a moment."):
            if uploaded_file.type == "application/pdf":
                input_content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "text/plain":
                input_content = read_text_file(uploaded_file)
            
            if input_content is None or not input_content.strip():
                st.error("❌ Could not extract text from the uploaded file. Please ensure it's a valid PDF or text file with readable content.")
                input_content = ""
            else:
                st.success("✅ Text extracted successfully!")

    elif pasted_text:
        input_content = pasted_text
    
    if input_content:
        with st.spinner(f"✨ Preprocessing text and generating {num_qa_pairs} flashcards... This can take a while, please be patient!"):
            preprocessed_text = preprocess_text(input_content)

            if not preprocessed_text.strip() or len(preprocessed_text.strip()) < 50:
                st.warning("⚠️ The input content is too short or empty after preprocessing. Please provide substantial text (at least 50 characters) for meaningful flashcard generation.")
                st.session_state.qa_pairs = []
            else:
                try:
                    raw_qa_pairs = qa_generator.generate_qa_pairs(preprocessed_text, num_qa=num_qa_pairs)
                    
                    st.session_state.qa_pairs = []
                    for qa in raw_qa_pairs:
                        difficulty = get_simulated_difficulty(qa['question'])
                        st.session_state.qa_pairs.append({
                            "question": qa['question'],
                            "answer": qa['answer'],
                            "difficulty": difficulty
                        })
                    
                    if st.session_state.qa_pairs:
                        st.info("""
                            **Difficulty Levels:**
                            - ⚪ **White Strip:** Easy Question
                            - 🟡 **Yellow Strip:** Medium Question
                            - ⚫ **Black Strip:** Hard Question
                            (Difficulty is currently simulated for demonstration purposes.)
                        """)

                        st.subheader(f"Generated {len(st.session_state.qa_pairs)} Flashcards:")

                        with flashcards_container.container():
                            st.markdown('<div class="flashcard-container">', unsafe_allow_html=True)

                            for i, qa in enumerate(st.session_state.qa_pairs):
                                color_palette = flashcard_colors[i % len(flashcard_colors)]
                                difficulty_strip_color = difficulty_colors[qa['difficulty']]
                                
                                text_color_on_strip = "black" if qa['difficulty'] == "Easy" or qa['difficulty'] == "Medium" else "white"

                                st.markdown(f"""
                                <div class="flashcard" style="background: linear-gradient(135deg, {color_palette['background_start']} 0%, {color_palette['background_end']} 100%); border-color: {color_palette['border']};">
                                    <div class="difficulty-strip" style="background-color: {difficulty_strip_color}; color: {text_color_on_strip};">{qa['difficulty'].upper()}</div>
                                    <div class="flashcard-question" style="color: {color_palette['question_text']};">Q{i+1}: {qa['question']}</div>
                                    <div class="flashcard-answer" style="color: {color_palette['answer_text']}; border-top-color: {color_palette['line_color']};">A{i+1}: {qa['answer']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        st.success("🎉 Flashcards generated successfully! Scroll down to view them.")
                    else:
                        st.warning("🧐 No flashcards could be generated from the provided content. This might happen with very short, ambiguous, or highly specialized text. Please try with different content or a larger input.")
                        st.session_state.qa_pairs = [] 
                except Exception as e:
                    st.error(f"Something went wrong during flashcard generation: {e}")
                    st.info("Please check the input text. Very short, irrelevant, or malformed text might cause issues with the models.")
                    st.session_state.qa_pairs = [] 
    else:
        st.info("⬆️ Please upload a file or paste some text above to begin generating flashcards.")
        st.session_state.qa_pairs = [] 


# --- Export Section ---
st.markdown("---")
st.header("Export Options")

if st.session_state.qa_pairs:
   
    df_export = pd.DataFrame([{'question': qa['question'], 'answer': qa['answer'], 'difficulty': qa['difficulty']} for qa in st.session_state.qa_pairs])
    
    csv_string = df_export.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Download Flashcards as CSV ⬇️",
        data=csv_string,
        file_name="flashcards.csv",
        mime="text/csv",
        help="Click to download your generated flashcards as a CSV file, including difficulty levels.",
        use_container_width=True
    )
else:
    st.info("Generate some flashcards first to enable the export option!")

st.markdown("---")


# Footer section
st.markdown("""
<div style="text-align: center; color: gray; margin-top: 50px; font-size: 0.9em;">
    <strong>Tip:</strong> The quality and relevance of the generated flashcards heavily depend on the clarity and content of your input text.
    For best results, provide well-structured and sufficiently detailed study material.

</div>
""", unsafe_allow_html=True)
