from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# Load optimized model
model = torch.load('checkpoints/best_model.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict_retrosynthesis():
    """
    API endpoint for retrosynthesis prediction.
    """
    data = request.json
    input_data = data['input']
    with torch.no_grad():
        prediction = model(input_data)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    API endpoint for prediction explanation.
    """
    data = request.json
    input_data = data['input']
    with torch.no_grad():
        explanation = model.explain(input_data)
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)