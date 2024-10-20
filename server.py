from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize the pipeline
pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    
    # Use the pipeline to get the emotion
    result = pipe(text)
    
    # Extract the emotion with the highest score
    if result:
        highest_emotion = max(result, key=lambda x: x['score'])
        emotion = highest_emotion['label']
    else:
        emotion = "Unknown"
    
    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
