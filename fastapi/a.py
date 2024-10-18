from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
import torch
from PIL import Image
import time
from io import BytesIO
import logging
import tempfile
from categories import initialized_categories, device  # Importing from categories.py
from categories import model, processor, class_names
from typing import Dict

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

app = FastAPI()

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe Holistic model initialization
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return results

def extract_keypoints(results):
    # Check if face and hand landmarks are present
    has_face_landmarks = results.face_landmarks is not None
    has_hand_landmarks = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)

    # Extract keypoints as before
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Return the extracted keypoints and information about presence of face/hand landmarks
    return np.concatenate([pose, face, lh, rh]), has_face_landmarks, has_hand_landmarks

def process_video_file(file_obj, sequence_length=30, fps=10):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        file_path = temp_video.name
        with open(file_path, "wb") as buffer:
            buffer.write(file_obj.file.read())
        print(f"Saved video to temporary file: {file_path}")
    
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    keypoints_sequence = []
    frame_interval = max(int(video_fps / fps), 1)  # Ensure interval is at least 1

    # Flag to check if landmarks were detected
    landmarks_detected = False

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read the frame.")
                break

            if frame_num % frame_interval == 0:
                print(f"Processing frame {frame_num + 1}")
                results = mediapipe_detection(frame, holistic)
                keypoints, has_face_landmarks, has_hand_landmarks = extract_keypoints(results)

                # Check for face and hand landmarks
                if has_face_landmarks or has_hand_landmarks:
                    landmarks_detected = True
                keypoints_sequence.append(keypoints)

            frame_num += 1

            if len(keypoints_sequence) == sequence_length:
                print("Collected sufficient frames for prediction.")
                break

        cap.release()

        if not landmarks_detected:
            print(f"No face or hand landmarks detected in any of the frames.")
            return "no_landmarks"

        if len(keypoints_sequence) < sequence_length:
            print(f"Insufficient frames captured. Expected {sequence_length}, but got {len(keypoints_sequence)}.")
            # Pad the sequence by repeating last frame
            while len(keypoints_sequence) < sequence_length:
                keypoints_sequence.append(keypoints_sequence[-1])
            print(f"Padded sequence to {sequence_length} frames.")

        print(f"Successfully processed {len(keypoints_sequence)} frames.")
        
        # Remove the temporary file after processing
        print(f"Temporary file {file_path} deleted.")
        
        return np.array(keypoints_sequence)
    
def predict(model_video, scaler, video_sequence, actions, device, confidence_threshold=0.75):
    try:
        # Ensure model is on the correct device
        model_video.to(device)

        # Shape manipulation before scaling
        video_sequence = np.expand_dims(video_sequence, axis=0)  # Shape is (1, sequence_length, num_features)
        num_samples, seq_len, num_features = video_sequence.shape
        
        # Scaling logic
        video_sequence = video_sequence.reshape(-1, num_features)  # Reshape for scaling
        video_sequence = scaler.transform(video_sequence)  # Scale the features
        video_sequence = video_sequence.reshape(num_samples, seq_len, num_features)  # Reshape back
        
        # Convert to tensor and move to device
        video_sequence = torch.tensor(video_sequence).float().to(device)
        
        with torch.no_grad():
            outputs = model_video(video_sequence)

            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)

            # Get the maximum probability (confidence) and the predicted class
            confidence, predicted = torch.max(probabilities, dim=1)

            # Ensure we get the predicted label
            predicted_label = predicted.item() if predicted.shape[0] == 1 else predicted.cpu().numpy()[0]
            predicted_confidence = confidence.item() if predicted.shape[0] == 1 else confidence.cpu().numpy()[0]

            # Check if the confidence is below the threshold
            if predicted_confidence < confidence_threshold:
                logging.info("Invalid Action Detected")
                return "Invalid Action Detected"
            else:
                predicted_action = actions[predicted_label]
                logging.info(f"Prediction completed: {predicted_action} (Confidence: {predicted_confidence:.2f})")
                return predicted_action

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
        return None



@app.post('/upload-image')
async def upload_image(image: UploadFile = File(...)):
    img_data = await image.read()
    image = Image.open(BytesIO(img_data)).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = class_names[predicted_class_idx]

    logging.info(f"Predicted label: {label}")

    return {"label": label}

@app.post('/upload-video')
async def upload_video(video: UploadFile = File(...), category: str = Form(...)) -> Dict[str, str]:
    request_id = time.time()
    logging.info(f"Request ID {request_id}: Received request to upload video.")

    if not category or category not in initialized_categories:
        logging.info(f"Request ID {request_id}: Invalid category.")
        return {"message": "Invalid category"}

    if not video:
        logging.info(f"Request ID {request_id}: No video part in the request.")
        return {"message": "No video part in the request"}

    logging.info(f"Request ID {request_id}: Video file received.")

    # Process the video and make a prediction
    keypoints_sequence = process_video_file(video)

    # Check for landmarks
    if isinstance(keypoints_sequence, str) and keypoints_sequence == "no_landmarks":
        logging.info("No face or hand landmarks detected")
        return {"message": "No landmarks detected"}  # Return simple message

    elif isinstance(keypoints_sequence, list) and len(keypoints_sequence) == 0:
        logging.info("No face or hand landmarks detected")
        return {"message": "No landmarks detected"}  # Return simple message

    if keypoints_sequence is not None:
        category_data = initialized_categories[category]
        predicted_action = predict(
            model_video=category_data['model'],
            scaler=category_data['scaler'],
            video_sequence=keypoints_sequence,
            actions=category_data['actions'],
            device=device
        )
        logging.info(f"Request ID {request_id}: Sending prediction to client: {predicted_action}")
        return {
            "message": "Video uploaded successfully!",
            "predicted_action": predicted_action
        }
    else:
        logging.info(f"Request ID {request_id}: Prediction could not be made due to insufficient data.")
        return {"message": "Prediction could not be made due to insufficient data"}


# @app.post('/upload-video')
# async def upload_video(video: UploadFile = File(...), category: str = Form(...)) -> Dict[str, str]:
#     request_id = time.time()
#     print(f"Request ID {request_id}: Received request to upload video.")

#     if not category or category not in initialized_categories:
#         print(f"Request ID {request_id}: Invalid category.")
#         raise HTTPException(status_code=400, detail="Invalid category")

#     if not video:
#         print(f"Request ID {request_id}: No video part in the request.")
#         raise HTTPException(status_code=400, detail="No video part in the request")

#     print(f"Request ID {request_id}: Video file received.")

#     # Process the video and make a prediction
#     keypoints_sequence = process_video_file(video)

#     # Check for landmarks
#     # if isinstance(keypoints_sequence, str) and keypoints_sequence == "no_landmarks":
#     # Handle the case when there are no landmarks detected
#     if isinstance(keypoints_sequence, str) and keypoints_sequence == "no_landmarks":
#     # Handle the case when there are no landmarks detected
#         # return JSONResponse(content={"detail": "No face or hand landmarks detected"}, status_code=400)
#         logging.info("Invalid Action")
#     elif isinstance(keypoints_sequence, list) and len(keypoints_sequence) == 0:
#     # If keypoints_sequence is a list and it's empty, handle as no landmarks
#         # return JSONResponse(content={"detail": "No face or hand landmarks detected"}, status_code=400)
#         logging.info("Invalid Action")

#     if keypoints_sequence is not None:
#         category_data = initialized_categories[category]
#         predicted_action = predict(
#             model_video=category_data['model'],
#             scaler=category_data['scaler'],
#             video_sequence=keypoints_sequence,
#             actions=category_data['actions']
#         )
#         print(f"Request ID {request_id}: Sending prediction to client: {predicted_action}")
#         return {
#             "message": "Video uploaded successfully!",
#             "predicted_action": predicted_action
#         }
#     else:
#         print(f"Request ID {request_id}: Prediction could not be made due to insufficient data.")
#         raise HTTPException(status_code=400, detail="Prediction could not be made due to insufficient data.")

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)