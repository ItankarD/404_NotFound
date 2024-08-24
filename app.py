from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle

# Load the models and scalers
model = pickle.load(open('404_NotFound/model.pkl', 'rb'))
sc = pickle.load(open('404_NotFound/standscaler.pkl', 'rb'))
mx = pickle.load(open('404_NotFound/minmaxscaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        N = float(data['Nitrogen'])
        P = float(data['Phosporus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])

        # Prepare the features for the model
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply transformations
        mx_features = mx.transform(single_pred)
        sc_mx_features = sc.transform(mx_features)

        # Predict the crop
        prediction = model.predict(sc_mx_features)

        # Define the crop dictionary
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
            8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
            19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        # Get the crop name or return 'Unknown crop'
        crop = crop_dict.get(int(prediction[0]), "Unknown crop")
        result = f"{crop} is the best crop to be cultivated right there."

        # Return a JSON response
        return jsonify({"result": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
