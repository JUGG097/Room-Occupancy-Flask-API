import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lime import lime_tabular


def prediction_explainer(input_data, model, pred):
    data = pd.read_csv("data/ALL_SET.csv")

    features = data.drop("Room_Occupancy_Count", axis=1)
    target = data["Room_Occupancy_Count"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.30, random_state=42
    )

    cat_features = [3, 4]

    cat_names = {3: ["No Motion", "Motion"], 4: ["Morning", "Afternoon", "Evening"]}

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        class_names=["0", "1", "2", "3"],
        feature_names=X_train.columns,
        categorical_features=cat_features,
        categorical_names=cat_names,
    )

    exp_output = explainer.explain_instance(
        data_row=input_data.to_numpy().flatten(),
        predict_fn=model.predict_proba,
        labels=(pred,),
    )

    # return exp_output.as_html(predict_proba=False, show_predicted_value=False)
    return exp_output.as_list(label=pred)
