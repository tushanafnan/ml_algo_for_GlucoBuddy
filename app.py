import pickle as pkl
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load scaler and model
script_dir = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(script_dir, 'scaler.pkl')
scaler = pkl.load(open(scaler_path, 'rb'))

model_path = os.path.join(script_dir, 'nb.pkl')
with open(model_path, 'rb') as f:
    model = pkl.load(f)

# Define validation for abnormal values
def check_abnormalities(data):
    """Check for abnormal input values."""
    funny_responses = []

    if data['Age'] > 120:
        funny_responses.append({
            'message': "Are you a vampire? 🧛 That's an incredible age!",
            'gif_url': "https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif"
        })

    if data['BMI'] > 60:
        funny_responses.append({
            'message': "Are you sure about that BMI? That seems quite high! 🤔",
            'gif_url': "https://media.giphy.com/media/1gXcPPRjzXtQxHsx3c/giphy.gif"
        })

    if data['Glucose'] > 300:
        funny_responses.append({
            'message': "Yikes! That glucose level is off the charts! 🚀",
            'gif_url': "https://media.giphy.com/media/3o7qE6LVRcOya0RJLa/giphy.gif"
        })

    if data['Insulin'] > 600:
        funny_responses.append({
            'message': "That insulin level seems unusually high! 🤯",
            'gif_url': "https://media.giphy.com/media/l0HU20BZ6LbSEITza/giphy.gif"
        })

    if data['BloodPressure'] > 180:
        funny_responses.append({
            'message': "Is that even possible? That blood pressure is wild! 😵",
            'gif_url': "https://media.giphy.com/media/dyQWeSIHWuQdyMjq9q/giphy.gif"
        })

    return funny_responses

# Prediction function
def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age):
    input_data = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction == 1:
        result = {
            'prediction': "You have high chances of Diabetes! Please consult a Doctor.",
            'gif_url': "https://media1.tenor.com/m/dHVat9e2S38AAAAC/rat-cry-mouse-cutie.gif"
        }
    else:
        result = {
            'prediction': "You have low chances of Diabetes. Please maintain a healthy lifestyle.",
            'gif_url': "https://media.giphy.com/media/W1GG6RYUcWxoHl3jV9/giphy.gif"
        }

    return result

# Flask route to handle predictions
@app.route('/predict', methods=['POST'])
def predictions():
    if request.method == 'POST':
        data = request.get_json()

        # Extract input data
        Age = data.get('Age', 0)
        Pregnancies = data.get('Pregnancies', 0)
        Glucose = data.get('Glucose', 0)
        BloodPressure = data.get('BloodPressure', 0)
        Insulin = data.get('Insulin', 0)
        BMI = data.get('BMI', 0)
        SkinThickness = data.get('SkinThickness', 0)
        DPF = data.get('DPF', 0)

        # Check for abnormalities
        funny_responses = check_abnormalities(data)
        if funny_responses:
            return jsonify({'abnormalities': funny_responses}), 400

        # Perform prediction
        result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age)
        return jsonify(result), 200

    return "Invalid request method", 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
