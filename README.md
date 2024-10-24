# Phishing Email Detector

This project aims to detect phishing emails using state-of-the-art NLP and ML algorithms.

> [!WARNING]
> This project is still under development. The final version will be released soon.

## Activity
![Alt](https://repobeats.axiom.co/api/embed/7c2df6ed93e5ada0c509e26d03271ba306db4557.svg "Repobeats analytics image")

## Project Structure

- **data/**: Contains raw and processed datasets.
- **notebooks/**: Jupyter notebooks for exploratory data analysis.
- **src/**: Source code for data preprocessing, model training, and evaluation.
- **models/**: Saved trained models and encoders.
- **tests/**: Unit tests for the project.
- **configs/**: Configuration files.
- **utils/**: Utility functions for data preprocessing, model training, and evaluation.

The project structure is as follows:
```
    project/
    ├── models/
    │   ├── local_llm.py
    │   ├── roberta_model.py
    │   ├── ollama_model.py
    │   ├── openai_model.py
    │   └── claude_model.py
    ├── tests/
    │   ├── test_local_llm.py
    │   ├── test_roberta_model.py
    │   ├── test_ollama_model.py
    │   ├── test_openai_model.py
    │   └── test_claude_model.py
    ├── utils/
    │   ├── preprocess.py
    │   ├── train.py
    │   └── data_loader.py
    ├── data/
    │   ├── processed/
    │   └── raw/
    ├── src/
    │   └── main.py
    ├── configs/
    ├── notebooks/
    ├── .env
    ├── setup.py
    ├── requirements.txt
    └── README.md
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/capstone-2024-T91/Image-Processing-and-NLP-for-Brand-Protection.git
   cd Image-Processing-and-NLP-for-Brand-Protection
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize Weights&Biases (Optional: Only if you're training a model)**

> [!TIP]
> Get your API key from your account before running this command

   ```bash
   wandb login
   ```

5. **Run the project**

> [!NOTE]
> Detailed instructions on how to run the project will be provided in the future.

   ## Usage

      ```bash
      python main.py [options] <email_text>
      ```

   <email_text>: The email content you want to classify.
   Options:
      -llm <model_name>: Specify a local LLM by name.
      -openai: Use OpenAI's GPT models.
      -claude: Use Anthropic's Claude models.
      -o <model_name>: Specify a model by name to use by Ollama.
      -r: Use the RoBERTa model.
      -local: Use the default local LLM.
      -train: Train the selected model.
      -v: Enable verbose output.
      -checkpoint: Resume training from the last checkpoint.

   ## Examples

   - Using OpenAI's GPT Model with Verbose Output

   ```bash
   python main.py -openai -v "Your email content here." # Use the finetuned OpenAI's gpt4o-mini model
   ```

   - Training the RoBERTa Model

   ```bash
   python main.py -r -train
   ```

   - Using a Specific Local LLM

   ```bash
   python main.py -llm "Your email content here." # Use the default local LLM (distilbert-base-uncased-finetuned)
   ```

   - Using an Ollama Model

   ```bash
   python main.py -o llama3 "Your email content here."
   ```

### **Citation**

**Conference Paper:**
> A. I. Champa, M. F. Rabbi, and M. F. Zibran, “Why phishing emails escape detection: A closer look at the failure points,” in *12th International Symposium on Digital Forensics and Security (ISDFS)*, 2024, pp. 1–6 (to appear).
