from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("akshayvkt/detect-ai-text")
pipe = pipeline("text-classification", model="akshayvkt/detect-ai-text")
model = AutoModelForSequenceClassification.from_pretrained("akshayvkt/detect-ai-text")

@app.route('/ai_text_detector', methods=['POST'])
def ai_text_detector():
    input_string = request.form.get('input_string')

    paragraphs = input_string.split('\n')

    output_html = ""
    ai_sentence_count = 0
    human_sentence_count = 0
    total_sentences = 0

    for paragraph in paragraphs:
        sentences = [sentence.strip() + '.' for sentence in paragraph.split('.') if sentence.strip()]
        total_sentences += len(sentences)

        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors="pt")

            sentence_length = len(inputs['input_ids'][0])
            repeats = max(1, 110 // sentence_length)
            repeated_sentence = (sentence + " ") * repeats
            truncated_sentence = " ".join(repeated_sentence.split()[:110])

            inputs = tokenizer(truncated_sentence, return_tensors="pt")
            outputs = model(**inputs)
            probabilities = outputs.logits.softmax(dim=1).detach().numpy()[0]

            if probabilities[0] > probabilities[1]:
                class_name = "human"
                human_sentence_count += 1
            else:
                class_name = "ai"
                ai_sentence_count += 1

            formatted_sentence = f'<span class="{class_name}">{sentence}</span>'
            output_html += formatted_sentence

        if paragraph.strip():
            output_html += '\n'

    if output_html.endswith('\n'):
        output_html = output_html[:-1]

    human_percentage = round((human_sentence_count / total_sentences) * 100, 2)
    ai_percentage = round((ai_sentence_count / total_sentences) * 100, 2)

    message = ""
    if ai_percentage < 20:
        message = "The Input Text Demonstrates Strong Affinity Towards Human-Like Expression."
    elif 20 <= ai_percentage <= 40:
        message = "The Input Text Shows Some Affinity Towards Human-Like Expression."
    elif 40 < ai_percentage < 50:
        message = "The Input Text Shows Proximity to Human-Like Expression."
    elif 50 <= ai_percentage <= 60:
        message = "The Input Text Shows Some Affinity Towards AI-Like Expression."
    elif 60 < ai_percentage < 70:
        message = "The Input Text Shows Proximity to AI-Like Expression."
    elif ai_percentage >= 70:
        message = "The Input Text Demonstrates Strong Affinity Towards AI-Like Expression."

    results_dict = {
        "output_html": output_html,
        "human_sentence_count": human_sentence_count,
        "ai_sentence_count": ai_sentence_count,
        "total_sentences": total_sentences,
        "human_percentage": human_percentage,
        "ai_percentage": ai_percentage,
        "text_behavior": message
    }

    return jsonify(results_dict)

if __name__ == '__main__':
    app.run(debug=True)
