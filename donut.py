from transformers import DonutProcessor, VisionEncoderDecoderModel
from pdf2image import convert_from_path
from PIL import Image
import torch

# === Step 1: Convert PDF pages to images ===
def pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path, dpi=200)
    return images  # List of PIL Images

# === Step 2: Load Donut model and processor ===
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# === Step 3: Ask questions using Donut ===
def ask_question(image, question):
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    inputs = processor(image, return_tensors="pt").to(device)
    decoder_input_ids = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    
    outputs = model.generate(
        **inputs,
        decoder_input_ids=decoder_input_ids,
        max_length=512,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id
    )
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return result.replace(prompt, "").strip()

# === Step 4: Main Extraction Function ===
def extract_signatory_info(pdf_path):
    images = pdf_to_images(pdf_path)
    
    questions = [
        "What is the print full name?",
        "What is the print surname?",
        "What is the official position?",
    ]
    
    for page_num, image in enumerate(images):
        print(f"\n📄 Page {page_num + 1}")
        for q in questions:
            answer = ask_question(image, q)
            print(f"{q} → {answer}")

# === Run the extractor ===
extract_signatory_info("/content/regex_practise.docx")
