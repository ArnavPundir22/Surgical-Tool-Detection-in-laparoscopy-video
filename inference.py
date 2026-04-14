import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

classes = ['clipper', 'grasper', 'hook', 'scissor']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
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

        label = classes[pred.item()]
        confidence = conf.item()

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
