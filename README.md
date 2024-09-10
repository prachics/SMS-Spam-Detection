# SMS Spam Detection

This project is an SMS Spam Detection system that classifies SMS messages as either **Spam** or **Not Spam**. The model is trained using machine learning techniques and can be deployed as a web app using Streamlit.

## Table of Contents

- [Installation](#installation)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Running the Web App](#running-the-web-app)
- [Contributing](#contributing)
- [License](#license)


## Installation

1. Prerequisites:
   - Ensure you have the following installed:
     - Python 3.x
     - pip (Python package installer)

2. Clone the Repository:
   - Open a terminal and clone the repository using SSH or HTTPS.

   - Using SSH:
     git clone git@github.com:ps8888/SMS-Spam-Detection.git

   - Using HTTPS:
     git clone https://github.com/ps8888/SMS-Spam-Detection.git

3. Navigate to the project directory:
   cd SMS-Spam-Detection

4. Install Dependencies:
   - You can install the required dependencies using pip by running:
     pip install -r requirements.txt


## How to Use

1. Train the model by running the Jupyter notebook:

    ```bash
    jupyter notebook sms_detection.ipynb
    ```

2. After training, pickle files (`vectorizer.pkl`, `model.pkl`) will be generated, which are used in the Streamlit web app for predictions.

## Project Structure

- `sms_detection.ipynb`: Jupyter notebook used for data preprocessing, model training, and evaluation.
- `host_website.py`: Streamlit application file that hosts the web app for SMS classification.
- `vectorizer.pkl`: Pickled TF-IDF vectorizer used for text feature extraction.
- `model.pkl`: Pickled machine learning model (e.g., Naive Bayes) for spam detection.
- `spam.csv`: Dataset used for training the model.

## Dataset

The project uses the **Spam/Ham dataset**, which consists of SMS messages labeled as either "spam" or "ham" (not spam).

## Model Training

The model is trained using a machine learning pipeline in `sms_detection.ipynb`:
- Preprocessing: Text cleaning, tokenization, stemming, stopword removal.
- Feature Extraction: TF-IDF vectorization.
- Classification: Naive Bayes classifier (or any other model of your choice).

## Running the Web App

To run the web app using Streamlit, follow these steps:

1. Ensure the pickled model and vectorizer files (`model.pkl`, `vectorizer.pkl`) are present in the project directory.
2. Run the Streamlit application:

    ```bash
    streamlit run host_website.py
    ```

3. Open the link provided by Streamlit in your browser. You should see the SMS Spam Classifier web interface.

4. Enter a message in the text box and click **Predict** to classify the message as **Spam** or **Not Spam**.

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


