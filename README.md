This **Python-based project** uses AI models to detect and **differentiate between human-written and AI-generated sentences** in a given text. 

**Clone the project from GitHub** : 
  https://github.com/arslan-ahmad-dev/ai_text_detector

**Install the required dependencies** :
  pip install -r requirements.txt

**Run the script with** :
  python ai_text_detector.py

**Open Postman and set up a POST request to** :
  http://127.0.0.1:5000/ai_text_detector

**In the request body, select form-data**
  
  **key** : input_string
  
  **value** : paste your text from any source to check the model's result (EITHER FROM CHATGPT OR SOME HUMAN CONTENT)

The response will indicate the total number of sentences, as well as how many are human-written versus AI-generated.
This tool leverages advanced NLP techniques for accurate classification, making it useful for distinguishing between human and AI text.
