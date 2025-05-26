import pandas as pd
import requests
import random
import time
random.seed(42)

def threshold_serverity(sample):
    # higher is less stressed
    serverity = sample['avg_severity_normalised']
    
    # non stress as low stress
    if not sample['is_stressor']:
        return "low_stress"

    if(serverity <= -0.5):
        return "high_stress"
    elif (serverity < 0.5):
        return "moderate_stress"
    else:
        return "low_stress"


df = pd.read_excel("dataset/SAD_semantic_dataset_test.xlsx")
url = "http://localhost:5000/scripted/chat"
reset_url = "http://localhost:5000/scripted/reset"
results = []
for index, row in df.iterrows():
    print(f"INDEX NUMBER {index}/580")
    sentence = row['sentence']
    true_severity = threshold_serverity(row)

    response = requests.post(reset_url)

    payload = {
        "message": sentence,
        "session_id": "test_session"
    }
    # send dummy response to get sentiment value
    response = requests.post(url, json=payload)

    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    time_taken = end_time - start_time
    try:
        predicted = response.json()['stressLevel']
    except Exception:
        predicted = "ERROR"

    print(f"Input sentence: {len(sentence)}")
    print(f"Response Time: {time_taken}")
    print(f"Response: {predicted}")
    print(f"True: {true_severity}")
    results.append([sentence, time_taken, predicted, true_severity])

results_df = pd.DataFrame(results, columns=['sentence', 'response_time', 'predicted_stress', 'true_stress'])
results_df.to_csv("scriptedTimingReport.csv", index=False)