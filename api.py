from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from training import get_evaluate, training_process
import time

app = Flask(__name__)
CORS(app)

@app.route('/recommender/<string:user_id>', methods=['GET'])
def get_products(user_id):
    with open('data/model_to_apply.json', 'r') as file:
        data = json.load(file)
    for item in data['data']:
        if item['user_id'] == user_id:
            return jsonify(item)
    return jsonify({"error": "User not found"}), 404

@app.route('/model', methods=['GET'])
def get_model():
    with open('data/model.json', 'r') as file:
        result = json.load(file)
    return jsonify(result)

@app.route('/model', methods=['POST'])
def apply_model():
    data = request.get_json()
    model = data['model']
    if (model == "Neutral Collaborative Filtering"):
        model_key = "ncf"
    elif (model == "Deep Matrix Factorization"):
        model_key = "dmf"
    else:
        model_key = "cdl"
    PREDICTED_FILE_PATH = f'model_state/{model_key}_predicted.json'
    EVALUATE_FILE_PATH = f'evaluate/{model_key}.json'
    
    with open(EVALUATE_FILE_PATH, 'r') as r_file:
        evaluate = json.load(r_file)

    formatted_json = json.dumps(evaluate, indent=4, sort_keys=False, ensure_ascii=False)
    with open('data/model.json', 'w', encoding='utf-8') as file:
        file.write(formatted_json)    

    RESULT_FILE_PATH = 'data/result.json'
    with open(RESULT_FILE_PATH, 'r') as file:
        result = json.load(file)

    result['models'] = {}

    formatted_json = json.dumps(result, indent=4, sort_keys=False, ensure_ascii=False)
    with open(RESULT_FILE_PATH, 'w', encoding='utf-8') as file:
        file.write(formatted_json)

    with open(PREDICTED_FILE_PATH, 'r') as r_file:
        predicted = json.load(r_file)

    APPLY_FILE_PATH = 'data/model_to_apply.json'
    with open(APPLY_FILE_PATH, 'w') as w_file:
        json.dump(predicted, w_file)

    return jsonify(f"Model {model_key} applied successfully")

@app.route('/result', methods=['GET'])
def get_result():
    with open('data/result.json', 'r') as file:
        result = json.load(file)
    return jsonify(result)

@app.route('/train', methods=['GET'])
def start_training():
    with open('data/result.json', 'r') as file:
        result = json.load(file)

    result['isTraining'] = True
    result['models'] = {}

    formatted_json = json.dumps(result, indent=4, sort_keys=False, ensure_ascii=False)
    with open('data/result.json', 'w', encoding='utf-8') as file:
        file.write(formatted_json)

    total_run_time_start = time.time()
    # print("start training dmf")
    # dmf_start_time = time.time()
    # training_process("dmf")
    # dmf_end_time = time.time()
    # print("DMF done")
    # print("Execution time for DMF: {} seconds".format(dmf_end_time - dmf_start_time))

    # print("start training cdl")
    # cdl_start_time = time.time()
    # training_process("cdl")
    # cdl_end_time = time.time()
    # print("CDL done")
    # print("Execution time for CDL: {} seconds".format(cdl_end_time - cdl_start_time))

    print("start training ncf")
    ncf_start_time = time.time()
    training_process("ncf")
    ncf_end_time = time.time()
    print("NCF done")
    print("Execution time for NCF: {} seconds".format(ncf_end_time - ncf_start_time))

    CDL_FILE_PATH = 'evaluate/cdl.json'
    cdl = get_evaluate(CDL_FILE_PATH)

    NCF_FILE_PATH = 'evaluate/ncf.json'
    ncf = get_evaluate(NCF_FILE_PATH)

    DMF_FILE_PATH = 'evaluate/dmf.json'
    dmf = get_evaluate(DMF_FILE_PATH)

    result = {
        "isTraining": False,
        "models": {}
    }

    result["models"]["Neutral Collaborative Filtering"] = ncf["Neutral Collaborative Filtering"]
    result["models"]["Deep Matrix Factorization"] = dmf["Deep Matrix Factorization"]
    result["models"]["Custom Deep Learning"] = cdl["Custom Deep Learning"]

    formatted_json = json.dumps(result, indent=4, sort_keys=False, ensure_ascii=False)
    with open('data/result.json', 'w', encoding='utf-8') as file:
        file.write(formatted_json)
    
    total_run_time_end = time.time()
    print("Execution time: {} seconds".format(total_run_time_end - total_run_time_start))
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8888)