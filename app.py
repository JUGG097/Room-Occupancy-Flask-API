from flask import Flask, request
from flask_cors import CORS
import json
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

model = joblib.load("models/rf_clf.joblib")

@app.route("/check", methods=["GET"])
def health_check():
    return {"success": True}, 200


@app.route("/prediction", methods=["POST"])
def occupancy_prediction():
    try:
        req_body = json.loads(request.data)
        req_body["S5_CO2_Slope"] = 0.5
        # print(req_body)
        df_data = pd.DataFrame(req_body, index=[0])
        pred = model.predict(df_data)
        # print(pred)
        return {"success": True, "prediction": str(pred[0])}, 200
    except:
        return {"success": False, "error": "Internal Server Error"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
