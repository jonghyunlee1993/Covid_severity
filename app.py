import gradio as gr
import pickle
import numpy as np

# 모델 로드
with open("weights/final_model_with_3_features.pkl", "rb") as f:
    model_3 = pickle.load(f)

with open("weights/final_model_with_9_features.pkl", "rb") as f:
    model_9 = pickle.load(f)

with open("weights/final_model_with_17_features.pkl", "rb") as f:
    model_17 = pickle.load(f)

# 예측 함수
def predict(model_choice, *features):
    features = np.array(features, dtype=np.float32)  # 입력 데이터 변환
    if model_choice == "3 Features Model":
        model = model_3
        features = features[:3].reshape(1, -1)
    elif model_choice == "9 Features Model":
        model = model_9
        features = features[:9].reshape(1, -1)
    else:  # "17 Features Model"
        model = model_17
        features = features[:17].reshape(1, -1)
    
    probs = model.predict_proba(features)[0]
    pred_class = np.argmax(probs)
    return {
        "Healthy Probability": probs[0], 
        "Moderate Probability": probs[1], 
        "Severe Probability": probs[2], 
        "Predicted Class": ["Healthy", "Moderate", "Severe"][pred_class]
    }

# 입력 UI 업데이트 함수
def update_inputs(model_choice):
    if model_choice == "3 Features Model":
        return [gr.update(visible=True)] * 3 + [gr.update(visible=False)] * 14
    elif model_choice == "9 Features Model":
        return [gr.update(visible=True)] * 9 + [gr.update(visible=False)] * 8
    else:
        return [gr.update(visible=True)] * 17

# 환자 데이터 예제
patient_data = [
    [0.001890597, 0.010370961, 0.008972138, 0.003136423, 0.064586282, 0.00177585, 0.027249728, 
     0.005286567, 0.027689592, 0.000232226, 0.000663895, 0.007474961, 0.004833043, 0.000535487, 
     0.020471447, 0.010196108, 0.021006934, 0.003415095],  # Healthy

    [0.0, 0.000970547, 0.0, 0.000286823, 0.009271361, 0.002888386, 0.001596907, 
     0.000798453, 0.010885322, 0.001894582, 0.003404667, 0.001119385, 0.000644964, 
     0.0, 0.00082481, 0.007626392, 0.000728686, 0.002767455],  # Moderate

    [0.0, 0.0, 0.001030931, 0.0, 0.008546474, 0.000213836, 0.002607752, 0.000570229, 
     0.000735386, 0.001128288, 0.0, 0.0, 0.0, 0.0, 0.000365085, 0.000671062, 0.001703732, 0.0]  # Severe
]

# 예제 데이터 채우기
def fill_patient_data(model_choice, patient_index):
    data = patient_data[patient_index]
    if model_choice == "3 Features Model":
        return data[:3] + [None] * 14
    elif model_choice == "9 Features Model":
        return data[:9] + [None] * 8
    else:
        return data[:17]

# 입력 필드 초기화
def clear_inputs():
    return [None] * 17

features = [
    "Plasmacytoid.DC",
    "CD8..T.Cell..EM.CD27hi.",
    "CD8..T.Cell..CD161..MAIT.",
    "Conventional.DC",
    "CD4..T.Cell..naive.",
    "CD8..T.Cell..EMRA.CD57low.",
    "CD8..T.Cell..naive.",
    "gd.T.Cell",
    "NK.Cell..CD56low.CD16hi.CD57low.",
    "CD4..T.Cell..EM.CD57hi.",
    "CD8..T.Cell..EMRA.CD57hi.",
    "CD8..T.Cell..EM.CD57hi.",
    "NK.Cell..CD56hi.CD16low.",
    "ILC",
    "CD8..T.Cell..CM.",
    "B.Cell..naive.",
    "CD4..T.Cell..EM.CD27low.",
    "B.Cell..transitional."
]

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("## Minimmune: COVID-19 Severity Classification")
    
    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(
                ["3 Features Model", "9 Features Model", "17 Features Model"], 
                label="Choose Model", value="17 Features Model"
            )
            
            # Feature 입력 필드: 17개를 2개 Column으로 배치
            with gr.Row():
                with gr.Column():
                    inputs = [gr.Number(label=features[i], visible=(i < 9)) for i in range(9)]
                with gr.Column():
                    inputs += [gr.Number(label=features[i], visible=(i >= 9)) for i in range(9, 17)]

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
            
            gr.Markdown("### Model Performance Metrics")
            gr.Image("figs/image.png")
    
    model_choice.change(update_inputs, inputs=[model_choice], outputs=inputs)
    run_button.click(predict, inputs=[model_choice] + inputs, outputs=output)

# 앱 실행
demo.launch()
