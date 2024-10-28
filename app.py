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
    """Check for abnormal input values and return funny messages if found."""
    funny_responses = []

    # Convert values to numbers to avoid type comparison errors
    age = int(data.get('Age', 0))
    bmi = float(data.get('BMI', 0.0))
    glucose = int(data.get('Glucose', 0))
    insulin = int(data.get('Insulin', 0))
    bp = int(data.get('BloodPressure', 0))

    if age > 120:
        funny_responses.append({
            'message': "Are you a vampire? 🧛 That's an incredible age!",
            'gif_url': "https://media.giphy.com/media/3o6ZtpxSZbQRRnwCKQ/giphy.gif"
        })

    if bmi > 60:
        funny_responses.append({
            'message': "Are you sure about that BMI? That seems quite high! 🤔",
            'gif_url': "https://media.giphy.com/media/1gXcPPRjzXtQxHsx3c/giphy.gif"
        })

    if glucose > 300:
        funny_responses.append({
            'message': "Yikes! That glucose level is off the charts! 🚀",
            'gif_url': "https://media.giphy.com/media/3o7qE6LVRcOya0RJLa/giphy.gif"
        })

    if insulin > 600:
        funny_responses.append({
            'message': "That insulin level seems unusually high! 🤯",
            'gif_url': "https://media.giphy.com/media/l0HU20BZ6LbSEITza/giphy.gif"
        })

    if bp > 180:
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
        Age = int(data.get('Age', 0))
        Pregnancies = int(data.get('Pregnancies', 0))
        Glucose = int(data.get('Glucose', 0))
        BloodPressure = int(data.get('BloodPressure', 0))
        Insulin = int(data.get('Insulin', 0))
        BMI = float(data.get('BMI', 0.0))
        SkinThickness = int(data.get('SkinThickness', 0))
        DPF = float(data.get('DPF', 0.0))

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
