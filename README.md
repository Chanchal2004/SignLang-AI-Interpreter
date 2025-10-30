# 🖐️ Sign Language Recognition & Text-to-Speech System

A real-time **American Sign Language (ASL)** recognition system that converts hand gestures into **text and speech** using **Computer Vision** and **Deep Learning**. Built with **Python, OpenCV, TensorFlow**, and a custom-trained **CNN model**, this project empowers accessibility by enabling **hands-free communication** through sign language.

---

## 📸 Demo Video  
🔗 https://drive.google.com/file/d/16yT3klPjl3PsZSrgbyzHKy-9Tp2QJMB_/view?usp=sharing

---

## 🚀 Features

### ✅ Core Functionalities
- **Hand Gesture Recognition:** Real-time detection using `cvzone`'s HandDetector (21-point landmarks).  
- **Alphabet Prediction (A–Z):** CNN trained on 26 classes using skeletonized hand images (accuracy **>95%**).  
- **Text-to-Speech Conversion:** Converts constructed sentences into speech using `pyttsx3`.  

### ✅ Word & Sentence Suggestions
- **Next Word Prediction:** Trigram Language Model trained using the Brown Corpus (`NLTK`).  
- **Auto Spell Correction:** Implemented using **SymSpell** based on dictionary frequency & edit distance.

### ✅ Gesture Controls
- ✋ **Open Hand → Insert Space**  
- 👍 **Thumb Out → Backspace**  
- ✅ Completely **hands-free operation** (no keyboard needed)

---

## 🧠 Model Architecture

### 🔹 Input
- **55×55 grayscale skeletonized gesture images**

### 🔹 Layers
- **3× Conv2D + BatchNorm + MaxPooling + Dropout**  
- Dense Layers: **512 → 256** neurons (+ Dropout)  
- Output: **Softmax (26 classes A–Z)**

### 🔹 Training Techniques
- Data Augmentation (Rotation, Zoom, Shear, Flip)  
- Early Stopping  
- ReduceLROnPlateau  
- ModelCheckpoint (best validation accuracy)  

---

## 📸 GUI Overview (Tkinter)
- Live camera feed panel  
- Skeleton drawing panel  
- Real-time prediction + confidence score  
- Sentence builder with:
  - Word suggestions  
  - Next-word prediction  
- Control buttons: **Speak**, **Auto-Correct**, **Backspace**, **Clear**

---

## 📁 File Structure
```
├── alphabetPred.py         # Skeleton-to-letter prediction
├── final1pred.py           # Full ASL-to-speech GUI system
├── handAcquisition.py      # Data collection tool
├── trainmodel.py           # CNN training script
├── AtoZ_3.1/               # Dataset (26 folders A–Z)
├── model-bw.weights.h5     # Trained model weights
├── best_model.h5           # Best accuracy model
├── model-bw.json           # CNN architecture
```

---

## 🆕 Unique Aspects
- **Skeleton-Based Images:** More noise-resistant and adaptable than raw RGB.  
- **Gesture-Based Control:** Space & backspace using hand gestures only.  
- **Context-Aware Suggestions:** Powered by NLP models.  

---

## 🧪 Future Enhancements
- Dynamic gesture recognition (motion-based signs)  
- Number and phrase-level ASL support  
- Web / mobile deployment (Flask, Kivy)  
- Multi-hand detection & multi-user support  

---

## 🛠️ Technologies Used
**Python, OpenCV, TensorFlow/Keras, cvzone, NLTK, SymSpell, Tkinter, Pyttsx3**
