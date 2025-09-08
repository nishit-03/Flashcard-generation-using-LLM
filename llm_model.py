from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import pipeline
import torch
import re


class QAGenerator:
    def __init__(self, qg_model_name="valhalla/t5-base-qg-hl", qa_model_name="distilbert-base-uncased-distilled-squad"):
        """
        Initializes the Question Generation (QG) and Question Answering (QA) models.

        Args:
            qg_model_name (str): The Hugging Face model ID for question generation.
                                 'valhalla/t5-base-qg-hl' is a good choice for QG from text.
            qa_model_name (str): The Hugging Face model ID for extractive question answering.
                                 'distilbert-base-uncased-distilled-squad' is efficient and effective.
        """
      
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        
        self.qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
        self.qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name).to(self.device)
        print(f"Initialized QG model: {qg_model_name} on {self.device}")

    
        self.qa_pipeline = pipeline(
            "question-answering",
            model=qa_model_name,
            tokenizer=qa_model_name,
            device=0 if self.device == "cuda" else -1 # 0 for first GPU, -1 for CPU
        )
        print(f"Initialized QA model: {qa_model_name} on {self.device}")


    def generate_qa_pairs(self, text, num_qa=10, max_qg_length=512, max_qa_context_length=512, max_qa_answer_length=200):
        """
        Generates a list of question-answer pairs from the given text.

        The process involves:
        1. Using the QG model to generate candidate questions from the input text.
        2. Filtering for unique and valid questions.
        3. Using the QA model to find answers to each generated question within the original text.

        Args:
            text (str): The input educational content from which to generate Q&A.
            num_qa (int): The desired number of Q&A pairs to generate.
            max_qg_length (int): Maximum token length for the input to the question generation model.
                                 Longer texts will be truncated.
            max_qa_context_length (int): Maximum token length for the context provided to the
                                        question answering model.
            max_qa_answer_length (int): Maximum token length for the generated answer from the
                                        question answering model.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'question' (str) and 'answer' (str).
                  Returns an empty list if no Q&A pairs can be generated.
        """
        qa_pairs = []
        if not text or len(text.strip()) < 50: 
            print("Input text is too short for meaningful Q&A generation.")
            return qa_pairs

       
        input_text_for_qg = "generate question: " + text
        

        # Tokenize the input for QG, ensuring truncation if too long
        inputs = self.qg_tokenizer(
            input_text_for_qg,
            max_length=max_qg_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device) 

       
        dynamic_num_beams = max(10, num_qa * 2) 
        num_sequences_to_return = dynamic_num_beams


        # Generate questions using beam search for better quality and diversity
        generated_ids = self.qg_model.generate(
            inputs["input_ids"],
            num_beams=dynamic_num_beams, 
            max_length=64, 
            early_stopping=True, 
            num_return_sequences=num_sequences_to_return, 
            length_penalty=0.8, 
            no_repeat_ngram_size=2 
        )

        
        questions = [self.qg_tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]
        

        # Filter for unique and non-empty questions
        unique_questions = []
        seen_questions = set()
        for q in questions:
            cleaned_q = ' '.join(q.replace('\n', ' ').split())
            normalized_q_for_check = re.sub(r'[^\w\s]', '', cleaned_q).lower().strip()

            if cleaned_q and normalized_q_for_check not in seen_questions and "?" in cleaned_q: 
                unique_questions.append(cleaned_q)
                seen_questions.add(normalized_q_for_check) 
                if len(unique_questions) >= num_qa:
                    break


        # If not enough unique questions are generated, try generating more or adjust parameters
        if len(unique_questions) < num_qa and num_qa > 0:
            print(f"Warning: Only {len(unique_questions)} unique questions generated, targeting {num_qa}. "
                  f"Consider increasing input text length, adjusting parameters (e.g., max num_beams), or "
                  f"providing more diverse input content.")


        # Generate answers for each unique question using the QA pipeline
        for question in unique_questions:
            try:
                answer_result = self.qa_pipeline(
                    question=question,
                    context=text,
                    max_answer_len=max_qa_answer_length,
                    handle_impossible_answer=True, 
                )
                answer = answer_result['answer'].strip()
                cleaned_answer = ' '.join(answer.replace('\n', ' ').split())

                if cleaned_answer and cleaned_answer.lower() not in ["no answer", ""] and len(cleaned_answer) > 3 and cleaned_answer.lower() != question.lower():
                    qa_pairs.append({"question": question, "answer": cleaned_answer})
                    if len(qa_pairs) >= num_qa: 
                        break
            except Exception as e:
                print(f"Error generating answer for question '{question}': {e}")
                continue
        
        return qa_pairs