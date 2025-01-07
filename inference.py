from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Load the model and tokenizer
model_path = "./models"
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
print("Model loaded successfully.")

# Define request schema
class TextGenerationRequest(BaseModel):
    prompt: str
    max_length: int = 200
    num_return_sequences: int = 1

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text based on a given prompt.
    """
    result = pipe(
        request.prompt,
        max_length=request.max_length,
        num_return_sequences=request.num_return_sequences,
    )
    return {"generated_text": [r["generated_text"] for r in result]}
