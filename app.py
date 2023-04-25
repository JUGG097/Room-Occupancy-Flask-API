from flask import Flask, request
from flask_cors import CORS
import json
import joblib
import pandas as pd
from modules import LIME_explain

app = Flask(__name__)
CORS(app)

url = "https://drive.google.com/file/d/1qXVdhUH1Q8HYjWdH7Ob0LMZ3OQRWacCM/view?usp=share_link"
url ='https://drive.google.com/uc?id=' + url.split('/')[-2]
raw_data = pd.read_csv(url)
model = joblib.load("models/rf_clf.joblib")


@app.route("/check", methods=["GET"])
def health_check():
    return {"success": True}, 200


@app.route("/prediction", methods=["POST"])
def occupancy_prediction():
    # try:
    req_body = json.loads(request.data)
    # allow for dynamic arrangement of request body
    data = {
        "Temp": req_body["Temp"],
        "Light": req_body["Light"],
        "Sound": req_body["Sound"],
        "PIR": req_body["PIR"],
        "Day_Period": req_body["Day_Period"],
        "S5_CO2": req_body["S5_CO2"],
        "S5_CO2_Slope": 0.5,
    }

    df_data = pd.DataFrame(data, index=[0])
    pred = model.predict(df_data)
  
    exp_output = LIME_explain.prediction_explainer(raw_data, df_data, model, pred[0])
    
    return {
        "success": True,
        "prediction": str(pred[0]),
        "explanation": exp_output,
    }, 200
    # except:
    #     return {"success": False, "error": "Internal Server Error"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
