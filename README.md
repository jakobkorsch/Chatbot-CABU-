# Chatbot(CABU) Arabic Chatbot

This project is an Arabic-language chatbot that uses natural language processing (NLP) and machine learning to understand and respond to user queries. The chatbot employs an RNN (Recurrent Neural Network) and GPT (Generative Pre-trained Transformer) as a fallback.

## Table of Contents
1. [Introduction](#Introduction)
2. [Installation](#Installation)
3. [Usage](#Usage)
4. [Project Structure](#Project-Structure)
5. [Training the Model](#Training-the-Model)
6. [API Keys](#API-Keys)
7. [Examples](#Examples)
8. [Troubleshooting](#Troubleshooting)
9. [License](#License)

## Introduction
This chatbot is designed to process Arabic text and respond to queries. It uses an RNN model for intent recognition and GPT as a fallback option for unknown inputs.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/your-repository-name.git
    https://github.com/jakobkorsch/Chatbot-CABU-.git
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Download the NLTK data:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

4. Ensure you have the Farasa library installed and have your `openai` API key.

## Usage
1. Start the main program:
    ```sh
    python main.py
    ```

2. Choose the input method (text or speech) and start interacting with the chatbot.

## Project Structure
- `main.py`: Main program to start the chatbot.
- `arabic_intents.json`: JSON file containing intents and their patterns.
- `chatbot_model.keras`: Saved RNN model.
- `words.pkl` & `classes.pkl`: Saved words and classes for processing.
- `requirements.txt`: List of dependencies.

## Training the Model
If you want to retrain the model, run the following code:
```sh
python train.py
