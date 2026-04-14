# 🏥 Surgical Tool Detection in Laparoscopy Video

> AI-powered system for detecting surgical instruments in laparoscopic videos using Deep Learning.

---

## 🚀 Overview

This project leverages **Computer Vision + Deep Learning** to automatically detect and classify surgical tools in laparoscopic videos.

It processes videos frame-by-frame and overlays:

- Tool name
- Confidence score

The system is powered by a fine-tuned **ResNet50 model** and provides a **modern Streamlit interface** for easy interaction.

---

## ✨ Features

- 🎥 Upload surgical videos or use YouTube links
- 🤖 Deep Learning-based tool classification
- 📊 Real-time prediction with confidence scores
- 💻 Interactive Streamlit UI
- 📦 Pre-trained model support (`resnet50_tools.pth`)
- 📥 Download processed output video
- ⚡ GPU acceleration support (CUDA)

---

## 🧠 Model Details

- Architecture: **ResNet50**
- Framework: **PyTorch**
- Classes:
    - Clipper
    - Grasper
    - Hook
    - Scissor

The model is trained on a structured dataset using transfer learning.

---

## 🏗️ Project Structure

.  
├── app.py                  # Streamlit UI  
├── inference.py            # Video processing & prediction  
├── train.py                # Model training pipeline  
├── resnet50_tools.pth      # Trained model weights  
├── requirements.txt        # Dependencies  
├── outputs/                # Generated results

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

git clone https://github.com/ArnavPundir22/Surgical-Tool-Detection-in-laparoscopy-video.git  
cd Surgical-Tool-Detection-in-laparoscopy-video

### 2️⃣ Create Virtual Environment

python -m venv .venv  
source .venv/bin/activate   # Linux/Mac  
.venv\Scripts\activate      # Windows

### 3️⃣ Install Dependencies

pip install -r requirements.txt

Dependencies include PyTorch, OpenCV, Streamlit, and more.

---

## ▶️ Usage

### Run the Application

streamlit run app.py

---

## 🖥️ Application Workflow

1. Upload a video OR paste a YouTube link
2. Click **Run Detection**
3. Model processes each frame
4. Output video is generated with predictions
5. Download the processed video

---

## 🔍 How It Works

### 🧩 Inference Pipeline

- Video is read frame-by-frame using OpenCV
- Each frame is:
    - Resized to 224×224
    - Converted to tensor
- Passed through the trained ResNet model
- Softmax applied to get probabilities
- Highest confidence class is selected
- Label is drawn on frame

Implemented in:

---

## 🎨 Streamlit UI Highlights

- Dual input support (Upload / YouTube)
- Live video preview
- Downloadable results
- Clean dark-themed UI

Implemented in:

---

## 📊 Training Details

- Data Augmentation:
    - Random flip
    - Rotation
- Optimizer: Adam
- Loss: CrossEntropy
- Metrics:
    - Training Accuracy
    - Validation Accuracy

Outputs generated:

- Loss curve
- Accuracy graph
- Confusion matrix

---

## ⚡ Performance

- Real-time capable (depending on hardware)
- GPU acceleration supported
- Efficient frame-wise inference

---

## 🔮 Future Improvements

- Multi-tool detection per frame
- Bounding box localization (Object Detection)
- Integration with YOLO / Detectron2
- Real-time webcam inference
- Deployment on cloud (AWS / GCP)

---

# 🤝 Contributing

Contributions are welcome!

### Fork the repo  
### Create a new branch  
`git checkout -b feature/your-feature`
  
### Commit changes  
`git commit -m "Add your feature"  `
  
### Push  
`git push origin feature/your-feature`

Then create a Pull Request 🚀

---

## 📜 License

This project is open-source 

---

## 💡 Acknowledgements

- PyTorch
- OpenCV
- Streamlit
- Medical imaging research community

---

## 👨‍💻 Author

**Arnav Pundir**
