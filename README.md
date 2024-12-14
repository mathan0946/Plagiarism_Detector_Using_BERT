# Plagiarism Detection with BERT

This project is a Flask web application that uses a pre-trained BERT model to detect plagiarism by calculating the similarity between text inputs. It provides two modes:

1. **Check against a database**: Compare user-provided text with abstracts in a dataset.
2. **Check between two texts**: Compare two user-provided texts for similarity.

---

## Features

- **BERT-Based Text Vectorization**: Utilizes BERT embeddings for text representation.
- **Cosine Similarity**: Measures the similarity between text embeddings.
- **Interactive Web Interface**: A user-friendly interface built with Flask and HTML templates.
- **Dataset Preprocessing**: Preprocesses research paper abstracts and creates text vectors.

---

## Prerequisites

- Python 3.7+
- pip
- CUDA-enabled GPU (optional for faster vector generation)

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained BERT model:
   ```bash
   python -m transformers-cli download bert-base-uncased
   ```

4. Place your dataset in the `data/` directory as `database.csv`.
   - The dataset must include an `abstract` column and optionally a `paper_id` column.

---

## Running the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Application Flow

1. **Home Page**: Loads and preprocesses the dataset. Displays options to check plagiarism against the database or between two texts.

2. **Check Against Database**:
   - Input a text.
   - Compares the input with dataset abstracts.
   - Displays the most similar paper ID, similarity score, and plagiarism status.

3. **Check Between Two Texts**:
   - Input two texts.
   - Compares the texts and displays similarity score and plagiarism status.

---

## File Structure

```
.
├── app.py                  # Main Flask application
├── data/
│   └── database.csv        # Dataset file
├── templates/
│   ├── home.html           # Home page template
│   ├── check_database.html # Database comparison template
│   └── check_text.html     # Text-to-text comparison template
├── static/
│   └── styles.css          # CSS styles (optional)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Key Functions

### `preprocess_data(data_path, sample_size)`
Preprocesses the dataset by:
- Dropping rows without abstracts.
- Sampling a subset of the data.

### `create_vector_from_text(text, MAX_LEN=510)`
Generates a BERT embedding vector for a given text input.

### `create_vector_index(data)`
Generates BERT embeddings for all abstracts in the dataset.

### Routes:
- `/`: Home page.
- `/check_plagiarism_database`: Compare user text against dataset abstracts.
- `/check_plagiarism_text`: Compare two user-provided texts.

---

## Example Dataset Format

```csv
paper_id,abstract
1,"This paper introduces a novel approach to ..."
2,"We explore the effectiveness of deep learning in ..."
```

---

## Customization

- **Dataset**: Replace `data/database.csv` with your own dataset.
- **Threshold**: Adjust the plagiarism detection threshold (default: `0.9`) in the `/check_plagiarism_database` and `/check_plagiarism_text` routes.
- **Sample Size**: Change the `sample_size` variable in `app.py` to modify the number of abstracts processed.

---

## Dependencies

- Flask
- pandas
- numpy
- torch
- transformers
- keras
- tqdm

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- [Transformers Library](https://huggingface.co/transformers/)
- [BERT Pre-trained Models](https://github.com/google-research/bert)

---

Feel free to contribute or raise issues for further improvements!

