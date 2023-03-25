# Authored bySean Pepper
# Private license
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess data from a file


def load_and_preprocess_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    # Perform any additional preprocessing, if needed
    preprocessed_data = data
    return preprocessed_data

# Tokenize the data


def tokenize_data(data, max_length=512):
    tokenized_data = tokenizer(
        data, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    return tokenized_data

# Create a dataset for the tokenized data


class GPTDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data["input_ids"])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.tokenized_data.items()}

# Initialize a GPT-2 model


def initialize_gpt_model():
    config = GPT2Config.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)
    return model

# Load a GPT-2 model from a directory


def load_gpt_model(model_dir):
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    return model

# Train the GPT-2 model


def train_gpt_model(model, dataloader, epochs=2, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")

        for batch_num, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            if (batch_num + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_num + 1}, Loss: {loss.item():.4f}")

# Save the trained model and tokenizer to a directory


def save_trained_model(model, tokenizer, output_dir):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Main part of the script
if __name__ == "__main__":
    file_path = "preprocessed_conversations.txt"
    output_dir = "trained_gpt_model"

    preprocessed_data = load_and_preprocess_data(file_path)
    tokenized_data = tokenize_data(preprocessed_data)
    dataset = GPTDataset(tokenized_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = load_gpt_model(output_dir)  # Load the GPT-2 model
    train_gpt_model(model, dataloader)
    save_trained_model(model, tokenizer, output_dir)
    
"""

load_and_preprocess_data(file_path): This function loads and preprocesses data from a file. It takes the file path as an argument and returns the preprocessed data.

tokenize_data(data, max_length=512): This function tokenizes the data using the GPT2Tokenizer from the transformers library. It truncates the data to fit a maximum length (default is 512), pads the data to the maximum length, and returns a PyTorch tensor of the tokenized data.

GPTDataset(Dataset): This is a custom dataset class that takes in the tokenized data as an argument and returns a dictionary containing input_ids, attention_mask, and token_type_ids.

initialize_gpt_model(): This function initializes a new GPT-2 model using the GPT2LMHeadModel from the transformers library.

load_gpt_model(model_dir): This function loads a pre-trained GPT-2 model from a specified directory using the GPT2LMHeadModel from the transformers library.

train_gpt_model(model, dataloader, epochs=2, learning_rate=5e-5): This function trains the GPT-2 model on the given dataset using the specified number of epochs and learning rate. It takes in the model, dataloader, number of epochs (default is 2), and learning rate (default is 5e-5) as arguments.

save_trained_model(model, tokenizer, output_dir): This function saves the trained GPT-2 model and tokenizer to a specified output directory using the save_pretrained() method.

The main part of the script loads and preprocesses the data, tokenizes the data, creates a dataset and dataloader, loads a pre-trained GPT-2 model, trains the model on the dataset, and saves the trained model and tokenizer to a specified output directory.

"""
