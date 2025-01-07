from transformers import AutoModelForCausalLM, AutoTokenizer

# Specify the model ID
model_id = "tiiuae/falcon-7b-instruct"

# Download the model and tokenizer
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    trust_remote_code=True, 
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Save the model and tokenizer locally
model.save_pretrained("./models")
tokenizer.save_pretrained("./models")
print("Model downloaded and saved successfully.")
