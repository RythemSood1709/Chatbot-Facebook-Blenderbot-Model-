from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


conversation_history = []

while True:
    # Create conversation history string
    history_string = "\n".join(conversation_history)

    # Get the input data from the user
    input_text = input("> ")

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    # Generate the response from the model
    # Generate the response with "loop brakes"
    outputs = model.generate(
        **inputs, 
        max_length=100, 
        do_sample=True,        # Enables randomness so it doesn't just pick the #1 most likely word
        top_p=0.9,             # Nucleus sampling: only considers the most probable tokens
        temperature=0.7,       # Higher = more creative, Lower = more focused (0.7 is a sweet spot)
        repetition_penalty=1.2, # Directly penalizes words it has already used
        no_repeat_ngram_size=3  # Specifically stops it from repeating any 3-word phrase
    )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    print(f"Bot: {response}")

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)