import pickle

# Save the trained model to a file
with open('stress_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('stress_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.json
    input_features = np.array([data['Age'], data['Gender'], data['HeartRate'], data['SleepHours']]).reshape(1, -1)

    # Predict stress level
    prediction = model.predict(input_features)

    # Return prediction as JSON
    return jsonify({'StressLevel': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)


#Example 


{
  "Age": 25,
  "Gender": 0,
  "HeartRate": 70,
  "SleepHours": 7
}


