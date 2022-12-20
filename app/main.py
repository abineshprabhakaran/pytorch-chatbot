from flask import Flask, abort,jsonify, request
from infer import IntentClassifier
import csv
import datetime
from os.path import exists

app = Flask(__name__)

# Transformers
# Panda 

@app.route("/", methods=["GET"])
def home():
    return "API is running on port 5000..."


@app.route("/torch", methods=["POST"])
def torch_live():
    dataDict = request.get_json()
    if(not isinstance(dataDict["message"], str)):
        return abort(403)   
    classifierObj = IntentClassifier("model_chatbot.pth","tokenizer_chatbot","id2label.json")
    input_message = dataDict["message"]
    response_json = classifierObj.getIntent(input_message)
    intent = response_json["result"]["intent"]
    if(exists("names.csv")):     
        with open('names.csv', newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            last_row = []
            for row in csv_reader:
                last_row = row
            last_serial_number = last_row[0]
            csvfile.close()
        with open('names.csv', 'a') as csvfile:
            append_list = [last_serial_number, datetime.datetime.now().timestamp(), input_message, intent]
            writer_object = csv.writer(csvfile)
            writer_object.writerow(append_list)
            csvfile.close()
    else:
        with open("names.csv", "w", newline="") as csvfile:
            fieldnames = ["s.no", "timestamp", "input", "output"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"s.no": 1, "timestamp": datetime.datetime.now().timestamp(), "input": input_message, "output": intent})
    return jsonify(response_json)

if __name__ == "__main__":
    app.run(debug=True)