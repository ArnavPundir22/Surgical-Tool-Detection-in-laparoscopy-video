import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from collections import deque  
import statistics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ['clipper', 'grasper', 'hook', 'scissor']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # FIX: Added ImageNet normalization below
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# ================= LOAD MODEL =================
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load("resnet50_tools.pth", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ================= PROCESS VIDEO =================
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25  # fallback fix

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/output.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise Exception("Error opening VideoWriter")

    frame_count = 0
    
    # FIX: Added a temporal smoothing buffer.
    # Surgical videos have high motion blur, causing the classifier to "flicker" 
    # its predictions rapidly from frame to frame. This deque stores the last 
    # 5 predictions, acting as a simple low-pass filter to stabilize the UI.
    prediction_buffer = deque(maxlen=5) 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)

        # Append the current frame's prediction to the buffer
        prediction_buffer.append(pred.item())
        
        # Select the most frequent prediction from the last 5 frames
        smoothed_pred = statistics.mode(prediction_buffer)

        # Use the smoothed prediction for the UI label
        label = classes[smoothed_pred]
        
        # PRO-FIX: Get the specific probability of the smoothed prediction class, 
        # not just the maximum probability of the current frame.
        confidence = probs[0][smoothed_pred].item()
        # <------------------------------>

        text = f"{label} ({confidence:.2f})"

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    # Ensure file exists
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise Exception("Output video not created properly!")

    print(f"Processed {frame_count} frames")
    print("Saved at:", output_path)

    return output_path
