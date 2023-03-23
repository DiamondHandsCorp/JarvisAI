# Authored bySean Pepper
# Private license or commercial license.By using this code you agree to
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