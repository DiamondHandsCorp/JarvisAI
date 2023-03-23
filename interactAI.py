import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_trained_model_and_tokenizer(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    return model, tokenizer


def generate_response(model, tokenizer, input_text, max_length=100):
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
