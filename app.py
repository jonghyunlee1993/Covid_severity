import gradio as gr
import pickle
import numpy as np

# 모델 로드
with open("weights/rf_with_3_roc-7231.pkl", "rb") as f:
    model_3 = pickle.load(f)

with open("weights/rf_with_5_roc-6964.pkl", "rb") as f:
    model_5 = pickle.load(f)

with open("weights/rf_with_10_roc-7648.pkl", "rb") as f:
    model_10 = pickle.load(f)

def predict(model_choice, *features):
    if model_choice == "3 Features Model":
        model = model_3
        features = np.array(features[:3]).reshape(1, -1)
    elif model_choice == "5 Features Model":
        model = model_5
        features = np.array(features[:5]).reshape(1, -1)
    else:
        model = model_10
        features = np.array(features[:10]).reshape(1, -1)
    
    probs = model.predict_proba(features)[0]
    pred_class = np.argmax(probs)
    return {"Healthy Probability": probs[0], "Moderate Probability": probs[1], "Severe Probability": probs[2], "Predicted Class": ["Healthy", "Moderate", "Severe"][pred_class]}

def update_inputs(model_choice):
    if model_choice == "3 Features Model":
        return [gr.update(visible=True)] * 3 + [gr.update(visible=False)] * 7
    elif model_choice == "5 Features Model":
        return [gr.update(visible=True)] * 5 + [gr.update(visible=False)] * 5
    else:
        return [gr.update(visible=True)] * 10

# 3 환자 데이터
patient_data = [
    [0.00950199627,	0.002873627955,	0.05917470807,	0.0006082679383,	0.005489430406,	0.01875617968,	0.002332945343,	0.001732186886,	0.07225121716,	0.00271592886], # Healthy
    [0.0009448129617,	0.0002792178881,	0.009025529571,	0.003314391796,	0.01380996581,	0.0008029400889,	0.003318919653,	0,	0.001177242987,	0], # Moderate
    [0,	0,	0.008366306725,	0,	0.0004390779363,	0.0003573890179,	0.001832895106,	0,	0.0008934725449,	0] # Severe
]

def fill_patient_data(model_choice, patient_index):
    data = patient_data[patient_index]
    if model_choice == "3 Features Model":
        return data[:3] + [None] * 7
    elif model_choice == "5 Features Model":
        return data[:5] + [None] * 5
    else:
        return data[:10]

def clear_inputs():
    return [None] * 10

features = [
   " CD8+ T Cell (EM CD27hi)", 
    "Conventional DC",	
    "CD4+ T Cell (naive)",	
    "CD8+ T Cell (EMRA CD57hi)",
    "NK Cell (CD56low CD16hi CD57hi)",	
    "CD8+ T Cell (CM)",
    "CD8+ NKT Cell",
    "Plasmacytoid DC",
    "CD4+ T Cell (EM CD27hi)",
    "CD8+ T Cell (EM CD27low)"
]


# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## COVID-19 Severity Classification App")
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(["3 Features Model", "5 Features Model", "10 Features Model"], label="Choose Model", value="5 Features Model")
            inputs = [gr.Number(label=features[i], visible=(i < 5)) for i in range(10)]
            with gr.Group():
                gr.Markdown("### Example Data")
                with gr.Row():
                    gr.Button("Healthy").click(fill_patient_data, inputs=[model_choice, gr.State(0)], outputs=inputs)
                    gr.Button("Moderate").click(fill_patient_data, inputs=[model_choice, gr.State(1)], outputs=inputs)
                    gr.Button("Severe").click(fill_patient_data, inputs=[model_choice, gr.State(2)], outputs=inputs)
            run_button = gr.Button("Run Prediction")
        with gr.Column(scale=1):
            output = gr.JSON(label="Prediction Output")
            gr.Button("Clear").click(clear_inputs, outputs=inputs)
            
            gr.Markdown("""
            ### Model Performance Metrics
            <div style='width: 100%; overflow-x: auto;'>
            <table style='width: 100%; text-align: center; margin: auto;'>
            <tr>
                <th>Metric</th><th>Full model</th><th>10 features</th><th>5 features</th><th>3 features</th>
            </tr>
            <tr>
                <td>Accuracy</td><td>0.6552</td><td>0.6899</td><td>0.6207</td><td>0.5517</td>
            </tr>
            <tr>
                <td>AUROC</td><td>0.8096</td><td>0.7648</td><td>0.6964</td><td>0.7321</td>
            </tr>
            </table>
            </div>
            """)
    
    model_choice.change(update_inputs, inputs=[model_choice], outputs=inputs)
    run_button.click(predict, inputs=[model_choice] + inputs, outputs=output)

# 앱 실행
demo.launch()
