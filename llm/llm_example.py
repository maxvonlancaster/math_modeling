import transformers
import requests
import PyPDF2
from io import BytesIO
import requests
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import string 
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
# from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Replace with the URL of the PDF file you want to extract text from
pdf_urls = ["https://ia800903.us.archive.org/7/items/PDF4Kurd-English-SB/SS-5-CSSOSH.pdf"]

# Method to extract text from the PDF by link to it
def extract_text_from_pdf(url):
    try:
        # Fetch the PDF file from the URL
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Open the PDF using PyPDF2
            pdf_file = BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Initialize an empty string to store the extracted text
            extracted_text = ""

            # Loop through each page and extract text
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                extracted_text += page.extract_text()

            # Close the PDF file
            pdf_file.close()
            return extracted_text
        else:
            print(f"Failed to fetch PDF from URL. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def preprocess_text(text):
    # Remove non-printable characters and Unicode escape sequences
    text = text.encode('ascii', 'ignore').decode('utf-8')
    # TODO: Add preprocess steps as per data, Convert the text to lowercase
    text = text.lower()
    # # Tokenize the text into individual words
    tokens = word_tokenize(text)
    # # Remove stopwords and punctuation from the tokens
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tokens = [token for token in tokens if token not in stop_words and token not in punctuation]
    return " ".join(tokens)

# Collect and preprocess data from the PDFs
corpus = []
for pdf_url in pdf_urls:
    text = extract_text_from_pdf(pdf_url)
    processed_text = preprocess_text(text)
    corpus.append(processed_text)

print(corpus)

# Set hyperparameters
vocab_size = 10000
embedding_dim = 128
max_seq_length = 50
lstm_units = 256
output_units = vocab_size

data = corpus[0].split()
train, test = train_test_split(data, test_size=0.3, shuffle=False, random_state=None)
X_train, X_test, y_train, y_test = train_test_split(train, test,
    test_size=0.2, shuffle = True, random_state = 8)

# Pad sequences
X_train = pad_sequences(X_train, maxlen=max_seq_length)
X_val = pad_sequences(X_test, maxlen=max_seq_length)
y_val = y_test

# Build the model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_seq_length),
    LSTM(lstm_units),
    Dense(output_units, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the trained model
model.save('custom_llm_model.h5')
