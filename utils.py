import PyPDF2
import io
import re


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.

    Args:
        pdf_file: An uploaded file object from Streamlit (st.file_uploader).

    Returns:
        str: The extracted text from the PDF, or None if an error occurs.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() if page.extract_text() else ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None
    return text


def read_text_file(text_file):
    """
    Reads text from an uploaded plain text file.

    Args:
        text_file: An uploaded file object from Streamlit (st.file_uploader).\

    Returns:
        str: The content of the text file, or None if an error occurs.
    """
    try:
        return text_file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading text file: {e}")
        return None


def preprocess_text(text):
    """
    Performs basic text preprocessing to clean the input for LLMs.
    - Removes extra whitespace, ensures consistent line breaks.
    - This helps in providing cleaner input to the models.

    Args:
        text (str): The raw input text.

    Returns:
        str: The preprocessed text.
    """
    if text is None:
        return ""
    

    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)

    # Remove leading/trailing whitespace from each line and then join them
    text = "\n".join([line.strip() for line in text.splitlines()])

    # Strip any leading/trailing whitespace from the entire text
    return text.strip()
