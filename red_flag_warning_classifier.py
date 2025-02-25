import os
import requests
import pandas as pd
import argparse
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle
from datetime import datetime, timezone



# Basic logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File path for archived alerts
ALERT_ARCHIVE_FILE = "red_flag_alert_archive.csv"

def fetch_alerts():
    """
    Fetch active alerts from the NWS API for California.
    Returns:
        DataFrame: A pandas df with alert data.
    """
    url = "https://api.weather.gov/alerts/active?area=CA&limit=200"
    headers = {"User-Agent": "MyApp/1.0 (your_email@example.com)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    alerts = data.get("features", [])
    logging.info(f"Total alerts retrieved: {len(alerts)}")


    alert_data = []
    for alert in alerts:
        properties = alert.get("properties", {})
        alert_data.append({
            "id": alert.get("id"),
            "event": properties.get("event"),
            "headline": properties.get("headline"),
            "description": properties.get("description"),
            "instruction": properties.get("instruction"),
            "area": properties.get("areaDesc"),
            "effective": properties.get("effective"),
            "expires": properties.get("expires"),
            "sent": properties.get("sent"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    return pd.DataFrame(alert_data)

def archive_alerts(df, archive_file=ALERT_ARCHIVE_FILE):
    """
    Append the df.
        df: The alerts DataFrame to archive.
        archive_file: Path to the CSV file.
    """
    if os.path.exists(archive_file):
        df.to_csv(archive_file, mode='a', index=False, header=False)
        logging.info(f"Appended {len(df)} alerts to {archive_file}")
    else:
        df.to_csv(archive_file, mode='w', index=False, header=True)
        logging.info(f"Created {archive_file} with {len(df)} alerts")

def train_model(archive_file=ALERT_ARCHIVE_FILE):
    """
    The target is label: 1 if the alert is a red flag warning, 0 otherwise.
        args: archive_file (str)
    """
    if not os.path.exists(archive_file):
        logging.error(f"Archive file {archive_file} does not exist.")
        return
    
    df = pd.read_csv(archive_file)
    if df.empty:
        logging.error("Archive file is empty.")
        return

 
    df["target"] = df["event"].apply(lambda x: 1 if isinstance(x, str) and "red flag warning" in x.lower() else 0)


    df["text"] = df["headline"].fillna('') + " " + df["description"].fillna('')
    
    # Log class distribution
    logging.info("Class distribution:\n" + str(df["target"].value_counts()))

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df["text"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluating model 
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info("Classification Report:\n" + report)

    # Saving trained model and TF-IDF vectorizer 
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    logging.info("Model and vectorizer saved as xgb_model.pkl and tfidf_vectorizer.pkl.")

def main():
    parser = argparse.ArgumentParser(description="Alert Archiving and Modeling")
    parser.add_argument("action", nargs="?", default="fetch", choices=["fetch", "train"],
                        help="Action to perform: fetch alerts or train model (default: fetch)")
    args = parser.parse_args()

    if args.action == "fetch":
        logging.info("Fetching alerts from API...")
        df = fetch_alerts()
        archive_alerts(df)
    elif args.action == "train":
        logging.info("Training model on archived alerts...")
        train_model()

if __name__ == "__main__":
    main()

# sample_alert = {
#     "id": "sample_id",
#     "properties": {
#         "event": "Red Flag Warning",
#         "headline": "Sample Headline",
#         "description": "This is a sample description of the alert.",
#         "instruction": "Take necessary precautions.",
#         "areaDesc": "Some County, CA",
#         "effective": "2025-01-01T00:00:00Z",
#         "expires": "2025-01-01T06:00:00Z",
#         "sent": "2025-01-01T00:00:00Z"
#     }
# }
# alerts = [sample_alert]

