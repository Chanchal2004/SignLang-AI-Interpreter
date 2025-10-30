🖐️ Sign Language Recognition & Text-to-Speech System
A real-time American Sign Language (ASL) recognition system that converts hand gestures into text and speech using Computer Vision and Deep Learning. Built with Python, OpenCV, TensorFlow, and a custom-trained CNN model, this project empowers accessibility by enabling hands-free communication through sign language.

📸 Demo Video
https://drive.google.com/file/d/16yT3klPjl3PsZSrgbyzHKy-9Tp2QJMB_/view?usp=sharing

🚀 Features
✅ Core Functionalities
Hand Gesture Recognition: Real-time detection using cvzone's HandDetector (21-point landmarks).

Deep Learning Alphabet Prediction: CNN trained on 26 classes (A–Z) using skeletonized hand images (accuracy >95%).

Text-to-Speech Integration: Converts constructed sentences into audible speech with pyttsx3.

Word & Sentence Suggestions:

Next Word Prediction: NLTK-trained Trigram Language Model on the Brown Corpus.
Auto Spell Correction: SymSpell based on dictionary frequency & edit distance.
Gesture Controls:

✋ Open Hand → Insert Space
👍 Thumb Out → Backspace
Hands-Free Operation: No keyboard required — control everything with gestures.

🧠 Model Architecture
Input: 55×55 grayscale skeletonized gesture images

Layers:

3× Conv2D + BatchNormalization + MaxPooling + Dropout
Fully connected: 512 & 256 neurons + Dropout
Output: Softmax (26 classes: A–Z)
Training Techniques:

Data Augmentation (Rotation, Zoom, Shear, Flip)
Early Stopping, ReduceLROnPlateau
ModelCheckpoint for best accuracy
📸 GUI Overview
Built with Tkinter featuring:

Live camera feed

Skeleton drawing panel

Real-time prediction + confidence

Sentence builder with:

Word suggestions
Next word prediction
Control buttons: Speak, Auto-Correct, Backspace, Clear

📁 File Structure
├── alphabetPred.py         # Skeleton-to-letter prediction
├── final1pred.py           # Full ASL-to-speech GUI system
├── handAcquisition.py      # Data collection tool
├── trainmodel.py           # CNN training script
├── AtoZ_3.1/               # Dataset (26 folders, A–Z)
├── model-bw.weights.h5     # Trained model weights
├── best_model.h5           # Best validation model
├── model-bw.json           # Model architecture
🆕 Unique Aspects
Skeleton-Based Images: More generalizable & noise-resistant than raw RGB.
Gesture-Based Control: Space & backspace without touching the keyboard.
Context-Aware Suggestions: NLP-powered next word prediction.
🧪 Future Enhancements
Support for dynamic gestures (motion-based signs)
Expand to numbers & ASL-specific phrases
Web/mobile deployment (Flask / Kivy)
Multi-hand tracking & recognition
🛠️ Technologies Used
Python, OpenCV, TensorFlow/Keras, cvzone, NLTK, SymSpell, Tkinter, Pyttsx3
