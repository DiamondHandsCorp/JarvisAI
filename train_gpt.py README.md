Authored by Sean Pepper

# train_gpt.py

## Description

This script provides functionality for training a GPT-2 model using PyTorch and the Hugging Face Transformers library. The script loads and preprocesses data from a file, tokenizes the data, creates a dataset and dataloader, initializes or loads a pre-trained GPT-2 model, trains the model on the dataset, and saves the trained model and tokenizer to a specified output directory.

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- tqdm

## Usage

The script can be run using the following command:

```python
python train_gpt.py
```

### Parameters

The script can be modified to take in the following parameters:

- `file_path`: A string specifying the file path to the input data file. Default is "preprocessed_conversations.txt".
- `output_dir`: A string specifying the directory to save the trained model and tokenizer. Default is "trained_gpt_model".
- `max_length`: An integer specifying the maximum length of the tokenized data. Default is 512.
- `batch_size`: An integer specifying the batch size for training the model. Default is 8.
- `epochs`: An integer specifying the number of epochs to train the model. Default is 2.
- `learning_rate`: A float specifying the learning rate for training the model. Default is 5e-5.

### Functions

The script contains the following functions:

- `load_and_preprocess_data(file_path)`: Loads and preprocesses data from a file. Returns preprocessed data.
- `tokenize_data(data, max_length=512)`: Tokenizes the data using the GPT2Tokenizer from the Transformers library. Returns a PyTorch tensor of the tokenized data.
- `GPTDataset(Dataset)`: A custom dataset class that takes in the tokenized data and returns a dictionary containing input_ids, attention_mask, and token_type_ids.
- `initialize_gpt_model()`: Initializes a new GPT-2 model using the GPT2LMHeadModel from the Transformers library.
- `load_gpt_model(model_dir)`: Loads a pre-trained GPT-2 model from a specified directory using the GPT2LMHeadModel from the Transformers library.
- `train_gpt_model(model, dataloader, epochs=2, learning_rate=5e-5)`: Trains the GPT-2 model on the given dataset using the specified number of epochs and learning rate.
- `save_trained_model(model, tokenizer, output_dir)`: Saves the trained GPT-2 model and tokenizer to a specified output directory.

## Example

```python
python train_gpt.py --file_path "data.txt" --output_dir "model_output" --max_length 1024 --batch_size 16 --epochs 4 --learning_rate 2e-5
```

This example will train a GPT-2 model on the data in "data.txt", with a maximum sequence length of 1024 tokens, a batch size of 16, over 4 epochs, and with a learning rate of 2e-5. The trained model and tokenizer will be saved to the "model_output" directory.
