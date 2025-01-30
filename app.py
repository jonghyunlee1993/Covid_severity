import gradio as gr
import pickle
import numpy as np

# 모델 로드
with open("weights/rf_with_3_roc-7596.pkl", "rb") as f:
    model_3 = pickle.load(f)

with open("weights/rf_with_5_roc-8557.pkl", "rb") as f:
    model_5 = pickle.load(f)

with open("weights/rf_with_10_roc-8502.pkl", "rb") as f:
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
    [3796, 1935, 1148, 2193, 650, 1091, 243, 10135, 260, 23640],  # 환자 1
    [626, 515, 185, 9150, 1863, 587, 2196, 7021, 1149, 5980],  # 환자 2
    [0, 328, 0, 258, 123, 116, 0, 423, 1145, 4916]   # 환자 3
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
    "CD8+ T Cell (EM CD27hi)",
    "CD8+ T Cell (EMRA CD57hi)",
    "gd T Cell", 
    "Conventional DC", 
    "NK Cell (CD56Low CD16hi CD571ow)",
    "NK Cell (CD56low CD1Ghi CDS7hi)", 
    "CD8+ T Cell (EMRA CDS7low)", 
    "Monocytes (CD14+ CD16+)",
    "B Cell (Plasmablast)", 
    "CD4+ T Cell (Naive)"
]


# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## COVID-19 Severity Classification App")
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(["3 Features Model", "5 Features Model", "10 Features Model"], label="Choose Model", value="5 Features Model")
            inputs = [gr.Number(label=features[i], visible=(i < 5)) for i in range(10)]
            with gr.Row():
                gr.Button("Healthy").click(fill_patient_data, inputs=[model_choice, gr.State(0)], outputs=inputs)
                gr.Button("Moderate").click(fill_patient_data, inputs=[model_choice, gr.State(1)], outputs=inputs)
                gr.Button("Severe").click(fill_patient_data, inputs=[model_choice, gr.State(2)], outputs=inputs)
            run_button = gr.Button("Run Prediction")
        with gr.Column(scale=1):
            output = gr.JSON(label="Prediction Output")
            gr.Button("Clear").click(clear_inputs, outputs=inputs)
    
    model_choice.change(update_inputs, inputs=[model_choice], outputs=inputs)
    run_button.click(predict, inputs=[model_choice] + inputs, outputs=output)

# 앱 실행
demo.launch()
