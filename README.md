# LLM-Powered Flashcard Generator

This project is a web-based application built with Streamlit that leverages Hugging Face's Large Language Models (LLMs) to automatically generate question-answer flashcards from provided text or PDF documents. It's designed to assist students and educators in creating study materials quickly and efficiently.

## Features

* **Flexible Input:** Upload PDF files or plain text documents, or paste content directly into the application.
* **AI-Powered Q&A Generation:** Utilizes a T5-based model for question generation and a DistilBERT model for extractive question answering to create relevant flashcards.
* **Customizable Output:** Specify the desired number of flashcards to generate.
* **Visually Appealing Interface:** Interactive flashcards displayed in a grid with dynamic color schemes.
* **Difficulty Indication:** Each flashcard features a diagonally tilted strip in the top-right corner indicating a simulated difficulty level (Easy, Medium, Hard) with distinct colors.
* **Export Functionality:** Download generated flashcards as a CSV file for offline use or integration with other tools.
* **Local Execution:** Runs entirely locally, leveraging your machine's resources (CPU or GPU).

##  Setup

Follow these steps to set up and run the Flashcard Generator on your local machine.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ShubhamKukreti07/Flashcard-Generator.git](https://github.com/ShubhamKukreti07/Flashcard-Generator.git)
    cd Flashcard-Generator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    This will install Streamlit, PyPDF2, Transformers, PyTorch, and SentencePiece.

##  Usage

Once the setup is complete, you can run the Streamlit application.

1.  **Start the Streamlit application:**
    Ensure your virtual environment is active.
    ```bash
    streamlit run app.py
    ```
    This command will open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Interact with the application:**
    * **Choose Input Method:** Select either "Upload File" to upload a PDF or `.txt` document, or "Paste Text" to directly input your content.
    * **Adjust Number of Flashcards:** Use the slider to specify how many flashcards you want to generate.
    * **Generate Flashcards:** Click the "Generate Flashcards " button. The application will process your content and display the generated Q&A pairs.
    * **Review Difficulty:** Observe the colored strips on each flashcard indicating its simulated difficulty.
        *  **White Strip:** Easy Question
        *  **Yellow Strip:** Medium Question
        *  **Black Strip:** Hard Question
        *(Note: Difficulty is currently simulated for demonstration purposes.)*
    * **Export:** Use the "Download Flashcards as CSV ⬇" button to save your generated flashcards.
      

## Project Structure

**Flashcard-Generator**
 * **app.py**  : Main Streamlit application file
 * **llm_model.py** :  Contains the QAGenerator class for LLM interactions
 * **requirements.txt**  : Lists all Python dependencies
 * **utils.py**  : Utility functions for text extraction and preprocessing
 * **README.md**  : This README file


## Sample Outputs

  ![Screenshot 2025-06-16 120054](https://github.com/user-attachments/assets/86254e13-7680-4d18-927b-741e91b64ef7)    
  
  ![Screenshot 2025-06-16 120218](https://github.com/user-attachments/assets/197ac0ab-6a5d-446e-8cfa-2c170d4d1bbc)

  ![Screenshot 2025-06-16 120404](https://github.com/user-attachments/assets/aa1f8e4a-05b2-42cd-a3d0-d491b7e9b636)  
  
  ![image](https://github.com/user-attachments/assets/dbb82287-ecbf-4b57-b54b-6f2a45cc825f)


##  Technologies Used

* **Streamlit:** For building the interactive web application.
* **Hugging Face Transformers:** For leveraging pre-trained LLMs for Question Generation (`valhalla/t5-base-qg-hl`) and Question Answering (`distilbert-base-uncased-distilled-squad`).
* **PyTorch:** The underlying deep learning framework used by the models.
* **PyPDF2:** For extracting text from PDF documents.
* **Pandas:** For easy handling and export of data to CSV.
