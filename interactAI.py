import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Author: Sean Pepper


def load_trained_model_and_tokenizer(model_dir):
  """
    Loads a pre-trained GPT-2 model and its tokenizer from the specified directory.

    Args:
        model_dir (str): The directory containing the pre-trained model and tokenizer.

    Returns:
        tuple: A tuple containing the GPT2LMHeadModel and GPT2Tokenizer instances.
    """
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_length=100):
"""
    Generates a response from the GPT-2 model based on the given input text.

    Args:
        model (GPT2LMHeadModel): The GPT-2 model used for generating responses.
        tokenizer (GPT2Tokenizer): The tokenizer used for encoding/decoding text.
        input_text (str): The input text to generate a response from.
        max_length (int, optional): The maximum length of the generated response. Default is 100.

    Returns:
        str: The generated response from the GPT-2 model.
    """
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


def interact_with_gpt(model, tokenizer):
"""
Interactively chat with the GPT-2 model.

Args:
    model (GPT2LMHeadModel): The GPT-2 model used for generating responses.
    tokenizer (GPT2Tokenizer): The tokenizer used for encoding/decoding text.
"""
    print("GPT-2 AI is ready to chat! Type 'exit' to stop the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = generate_response(model, tokenizer, user_input)
        print("AI:", response)


if __name__ == "__main__":
    model_dir = "trained_gpt_model"
    model, tokenizer = load_trained_model_and_tokenizer(model_dir)
    interact_with_gpt(model, tokenizer)
    
""" 
This script, authored by Sean Pepper, provides a basic interface for interacting with a pre-trained GPT-2 model using the Hugging Face Transformers library. The main functions are:

1. `load_trained_model_and_tokenizer()`: Loads the pre-trained GPT-2 model and its tokenizer from a specified directory.
2. `generate_response()`: Generates a response from the GPT-2 model based on the given input text.
3. `interact_with_gpt()`: Interactively chats with the GPT-2 model.

When executed, the script loads a pre-trained GPT-2 model and its tokenizer using the load_trained_model_and_tokenizer() function. It then enters an interactive chat session with the GPT-2 model through the interact_with_gpt() function, where the user can type in text and receive responses generated by the GPT-2 model.

To exit the interactive session, the user can simply type 'exit', which will break the loop and end the conversation.
"""

