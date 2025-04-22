import streamlit as st
import numpy as np
import cv2
import os
from PIL import Image
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ensemble Model Class
class DeepfakeEnsemble:
    def __init__(self, model_paths):
        self.models = {}
        self.weights = {}
        
        for name, path in model_paths.items():
            self.models[name] = load_model(path)
            self.weights[name] = 1.0 / len(model_paths)
    
    def preprocess_input(self, img_array):
        img_array = cv2.resize(img_array, (224, 224))
        img_array = image.img_to_array(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    
    def predict_all(self, img_array):
        preprocessed = self.preprocess_input(img_array)
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(preprocessed)[0][0]
            predictions[name] = pred
        
        return predictions
    
    def weighted_predict(self, predictions):
        weighted_sum = 0
        total_weight = 0
        
        for name, pred in predictions.items():
            # Higher confidence = higher weight (if prediction is close to 0 or 1)
            confidence = max(pred, 1 - pred)  
            self.weights[name] = confidence  # Update weight dynamically
            
            weighted_sum += pred * self.weights[name]
            total_weight += self.weights[name]
        
        return weighted_sum / total_weight

# Initialize models
model_paths = {
    'VGG16': 'VGG16_deepfake_model.h5',
    'InceptionV3': 'InceptionV3_deepfake_model.h5',
    'DenseNet121': 'DenseNet121_deepfake_model.h5',
    'Custom': 'custom_deepfake_model.h5'
}
ensemble = DeepfakeEnsemble(model_paths)

# Streamlit UI
st.title("Deepfake Detection Analyzer")

input_type = st.radio("Select input type:", ["Image", "Video"], index=0)

if input_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])
    num_frames = st.slider("Number of frames to analyze:", 1, 50, 10)

def extract_frames(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return frames

if st.button("Analyze"):
    if uploaded_file:
        if input_type == "Image":
            # Process image
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            # Display original image
            st.subheader("Uploaded Image")
            st.image(img, use_container_width=True)
            
            # Get predictions
            predictions = ensemble.predict_all(img_array)
            ensemble_pred = ensemble.weighted_predict(predictions)
            
            # Display results in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Predictions")
                for model_name, pred in predictions.items():
                    confidence = pred if pred > 0.5 else 1-pred
                    st.write(f"**{model_name}**")
                    st.progress(int(confidence*100))
                    st.caption(f"{'REAL' if pred > 0.5 else 'FAKE'} ({pred:.4f})")
            
            with col2:
                st.subheader("Ensemble Result")
                confidence = ensemble_pred if ensemble_pred > 0.5 else 1-ensemble_pred
                if ensemble_pred > 0.5:
                    st.success(f"**REAL** ({ensemble_pred:.4f})")
                else:
                    st.error(f"**FAKE** ({ensemble_pred:.4f})")
                
                st.metric("Confidence", f"{confidence:.2%}")
                
                # Show weights
                st.write("**Model Weights:**")
                for name, weight in ensemble.weights.items():
                    st.write(f"- {name}: {weight:.2f}")

        else:  # Video processing - UPDATED TO SHOW WEIGHTS
            video_path = f"temp_{uploaded_file.name}"
            with open(video_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            frames = extract_frames(video_path, num_frames)
            os.remove(video_path)
            
            st.subheader("Video Analysis Results")
            
            # SECTION 1: Show frames in a scrollable row
            st.write("### Extracted Frames")
            cols = st.columns(4)  # 4 frames per row
            for i, frame in enumerate(frames):
                with cols[i % 4]:
                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
            
            # SECTION 2: Frame-by-frame predictions
            st.write("---")
            st.write("### Frame Predictions")
            
            # Process all frames
            all_predictions = {name: [] for name in ensemble.models.keys()}
            frame_results = []
            weight_history = {name: [] for name in ensemble.models.keys()}
            
            for i, frame in enumerate(frames):
                predictions = ensemble.predict_all(frame)
                ensemble_pred = ensemble.weighted_predict(predictions)
                
                for name, pred in predictions.items():
                    all_predictions[name].append(pred)
                    weight_history[name].append(ensemble.weights[name])
                
                frame_results.append({
                    'Frame': i+1,
                    'Ensemble': ensemble_pred,
                    **predictions
                })
            
            # Show predictions table
            st.dataframe(
                pd.DataFrame(frame_results).set_index('Frame').style.format("{:.4f}"),
                height=300
            )
            
            # SECTION 3: Final analysis
            st.write("---")
            st.write("### Final Video Analysis")
            
            # Calculate averages
            avg_predictions = {name: np.mean(preds) for name, preds in all_predictions.items()}
            ensemble_avg = np.mean([x['Ensemble'] for x in frame_results])
            avg_weights = {name: np.mean(weights) for name, weights in weight_history.items()}
            
            # Display in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Average Model Predictions:**")
                for name, pred in avg_predictions.items():
                    st.write(f"- {name}: {'REAL' if pred > 0.5 else 'FAKE'} ({pred:.4f})")
                
                st.write("\n**Average Model Weights:**")
                for name, weight in avg_weights.items():
                    st.write(f"- {name}: {weight:.4f}")
            
            with col2:
                st.write("**Final Ensemble Result:**")
                if ensemble_avg > 0.5:
                    st.success(f"REAL (average confidence: {ensemble_avg:.4f})")
                else:
                    st.error(f"FAKE (average confidence: {1-ensemble_avg:.4f})")
                
                st.metric("Overall Confidence", 
                         f"{ensemble_avg:.2%}" if ensemble_avg > 0.5 else f"{1-ensemble_avg:.2%}")
            
            # SECTION 4: Visualization
            st.write("---")
            st.write("### Prediction Trends")
            
            tab1, tab2, tab3 = st.tabs(["Line Chart", "Histogram", "Weights"])
            
            with tab1:
                st.line_chart(
                    pd.DataFrame(frame_results).set_index('Frame')[['Ensemble'] + list(ensemble.models.keys())]
                )
            
            with tab2:
                st.bar_chart(
                    pd.DataFrame(avg_predictions, index=[0]).T.rename(columns={0: "Score"})
                )
                
            with tab3:
                st.line_chart(
                    pd.DataFrame(weight_history)
                )
                st.write("Model weights evolution across frames")
    else:
        st.warning("Please upload a file first")