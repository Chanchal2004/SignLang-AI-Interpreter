import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
from collections import defaultdict, Counter
import pyttsx3
import nltk
from nltk.corpus import words, brown
from nltk import trigrams
from difflib import get_close_matches
import pickle
import time
import pkg_resources
from symspellpy import SymSpell, Verbosity

class ASLRecognitionApp:
    def _init_(self, root):
        self.root = root
        self.root.title("ASL Recognition System")
        
        # Set initial window size
        self.root.geometry("1000x800")
        self.root.minsize(900, 700)
        
        # Configure grid weights
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        
        # Initialize components
        self.initialize_tts()
        self.initialize_nltk()
        self.initialize_spell_checker()
        self.load_models()
        self.initialize_variables()
        
        # Setup GUI
        self.setup_gui()
        
        # Start video processing
        self.start_video_processing()
        
        # White background for skeleton
        self.white = np.ones((350, 350, 3), np.uint8) * 255
        
        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
    
    def initialize_nltk(self):
        """Download required NLTK data"""
        nltk.download('words', quiet=True)
        nltk.download('brown', quiet=True)
        self.english_words = set(words.words())
    
    def initialize_spell_checker(self):
        """Initialize the spell checker"""
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt")
        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    
    def load_models(self):
        """Load ASL and language models"""
        # ASL recognition model
        self.model = load_model('best_model.h5')
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
        # Language model for next-word prediction
        self.language_model_file = 'language_model.pkl'
        if oss.path.exists(self.language_model_file):
            with open(self.language_model_file, 'rb') as f:
                self.language_model = pickle.load(f)
        else:
            self.train_language_model()
    
    def train_language_model(self):
        """Train a trigram language model on Brown corpus"""
        print("Training language model...")
        self.language_model = defaultdict(Counter)
        
        for sentence in brown.sents():
            # Filter and normalize words
            words = [word.lower() for word in sentence if word.isalpha()]
            # Create trigrams and count frequencies
            for w1, w2, w3 in trigrams(words, pad_right=True, pad_left=True):
                self.language_model[(w1, w2)][w3] += 1
        
        # Save the model
        with open(self.language_model_file, 'wb') as f:
            pickle.dump(self.language_model, f)
    
    def initialize_variables(self):
        """Initialize all application variables"""
        # Video capture
        self.capture = cv2.VideoCapture(0)
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.offset = 20
        
        # Recognition parameters
        self.threshold = 0.80
        self.letter_cooldown = 1.0  # seconds
        self.last_letter_time = 0
        self.letter_hold_frames = 5  # frames to hold for letter input
        
        # Gesture recognition
        self.space_gesture_threshold = 0.8
        self.backspace_gesture_threshold = 0.8
        self.gesture_hold_frames = 10
        self.space_gesture_counter = 0
        self.backspace_gesture_counter = 0
        
        # Sentence management
        self.sentence = ""
        self.last_letter = None
        self.letter_count = 0
        self.current_suggestions = []
        self.current_sentence_suggestions = []
        
        # Caches
        self.suggestions_cache = {}
    
    def setup_gui(self):
        """Setup the complete GUI interface"""
        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', padding=3)
        style.configure('TLabelFrame', padding=5)
        style.configure('TButton', padding=3, font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10))
        
        # Main frames
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed", padding=5)
        self.video_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        self.skeleton_frame = ttk.LabelFrame(self.root, text="Hand Skeleton", padding=5)
        self.skeleton_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        self.info_frame = ttk.LabelFrame(self.root, text="Recognition Info", padding=5)
        self.info_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        
        # Configure grid weights for video frames
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        self.skeleton_frame.grid_rowconfigure(0, weight=1)
        self.skeleton_frame.grid_columnconfigure(0, weight=1)
        
        # Video labels
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill='both')
        
        self.skeleton_label = ttk.Label(self.skeleton_frame)
        self.skeleton_label.pack(expand=True, fill='both')
        
        # Info panel with prediction, confidence and gesture info
        info_panel = ttk.Frame(self.info_frame)
        info_panel.pack(fill='x', padx=5, pady=5)
        
        self.prediction_label = ttk.Label(info_panel, text="Predicted: ", font=('Helvetica', 11))
        self.prediction_label.pack(side='left', padx=5)
        
        self.confidence_label = ttk.Label(info_panel, text="Confidence: ", font=('Helvetica', 11))
        self.confidence_label.pack(side='left', padx=5)
        
        self.gesture_label = ttk.Label(info_panel, text="Gesture: ", font=('Helvetica', 11))
        self.gesture_label.pack(side='left', padx=5)
        
        # Sentence display with wrapping
        self.sentence_frame = ttk.Frame(self.info_frame)
        self.sentence_frame.pack(fill='x', padx=10, pady=5)
        
        self.sentence_label = ttk.Label(self.sentence_frame, text="Sentence:", font=('Helvetica', 12))
        self.sentence_label.pack(side='left')
        
        self.sentence_text = ttk.Label(self.sentence_frame, text="", font=('Helvetica', 12), 
                                     wraplength=800, foreground="blue")
        self.sentence_text.pack(side='left', fill='x', expand=True)
        
        # Word suggestions (completion for current word)
        self.suggestions_frame = ttk.LabelFrame(self.info_frame, text="Word Completion", padding=5)
        self.suggestions_frame.pack(fill='x', padx=10, pady=5)
        
        self.suggestion_buttons = []
        for i in range(3):
            btn = ttk.Button(self.suggestions_frame, text="", width=12,
                           command=lambda i=i: self.add_suggestion_to_sentence(i))
            btn.pack(side='left', padx=5, pady=2)
            self.suggestion_buttons.append(btn)
        
        # Sentence suggestions (next word prediction)
        self.sentence_suggestions_frame = ttk.LabelFrame(self.info_frame, text="Next Word Prediction", padding=5)
        self.sentence_suggestions_frame.pack(fill='x', padx=10, pady=5)
        
        self.sentence_suggestion_buttons = []
        for i in range(3):
            btn = ttk.Button(self.sentence_suggestions_frame, text="", width=12,
                           command=lambda i=i: self.add_sentence_suggestion(i))
            btn.pack(side='left', padx=5, pady=2)
            self.sentence_suggestion_buttons.append(btn)
        
        # Control buttons
        self.control_frame = ttk.Frame(self.info_frame)
        self.control_frame.pack(fill='x', padx=10, pady=5)
        
        buttons = [
            ("Speak", self.speak_sentence),
            ("Clear", self.clear_sentence),
            ("Space", self.add_space),
            ("Backspace", self.backspace),
            ("Auto-Correct", self.auto_correct_last_word)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(self.control_frame, text=text, width=10, command=command)
            btn.pack(side='left', padx=5, pady=2)
        
        # Threshold control
        self.threshold_frame = ttk.Frame(self.info_frame)
        self.threshold_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(self.threshold_frame, text="Confidence Threshold:").pack(side='left', padx=5)
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.5, to=1.0, value=0.8, 
                                        command=self.update_threshold)
        self.threshold_slider.pack(side='left', padx=5, expand=True, fill='x')
        self.threshold_value = ttk.Label(self.threshold_frame, text="0.80", width=5)
        self.threshold_value.pack(side='left', padx=5)
    
    def start_video_processing(self):
        """Start the video processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()
    
    def predict_next_words(self, context):
        """Predict the next likely words given the context using trigram model"""
        context = context.lower().split()
        predictions = []
        
        # Use trigram model if enough context
        if len(context) >= 2:
            last_two = tuple(context[-2:])
            if last_two in self.language_model:
                predictions = [word for word, count in self.language_model[last_two].most_common(3)]
        
        # Fall back to bigram if not enough predictions
        if len(predictions) < 3 and len(context) >= 1:
            last_one = (context[-1],)
            if last_one in self.language_model:
                bigram_preds = [word for word, count in self.language_model[last_one].most_common(3)]
                predictions.extend(bigram_preds[:3-len(predictions)])
        
        # Fall back to unigram (most common words) if still not enough
        if len(predictions) < 3:
            predictions.extend(["you", "the", "and"][:3-len(predictions)])
        
        return list(dict.fromkeys(predictions))[:3]  # Remove duplicates and limit to 3
    
    def update_sentence_suggestions(self):
        """Update the sentence completion suggestions based on current sentence"""
        if not self.sentence.strip():
            self.current_sentence_suggestions = []
            for btn in self.sentence_suggestion_buttons:
                btn.config(text="")
            return
        
        # Get predictions
        predictions = self.predict_next_words(self.sentence)
        self.current_sentence_suggestions = predictions
        
        # Update buttons
        for i in range(3):
            if i < len(self.current_sentence_suggestions):
                self.sentence_suggestion_buttons[i].config(text=self.current_sentence_suggestions[i])
            else:
                self.sentence_suggestion_buttons[i].config(text="")
    
    def add_sentence_suggestion(self, index):
        """Add the selected sentence suggestion to the sentence"""
        if index < len(self.current_sentence_suggestions):
            # Add space if needed
            if self.sentence and not self.sentence[-1].isspace():
                self.sentence += " "
            self.sentence += self.current_sentence_suggestions[index]
            self.update_sentence_display()
            self.engine.say(self.current_sentence_suggestions[index])
            self.engine.runAndWait()
    
    def update_suggestions(self, letter):
        """Update word completion suggestions based on current letter"""
        if not letter:
            self.current_suggestions = []
            for btn in self.suggestion_buttons:
                btn.config(text="")
            return

        if letter in self.suggestions_cache:
            self.current_suggestions = self.suggestions_cache[letter]
        else:
            # First try to find words that start with the letter
            prefix_matches = [word.capitalize() for word in self.english_words 
                            if word.lower().startswith(letter.lower())]
            
            # Sort by length to prefer shorter, more common words
            prefix_matches = sorted(prefix_matches, key=len)
            
            # If we don't have enough prefix matches, get close matches
            if len(prefix_matches) < 3:
                close_matches = [match.capitalize() for match in 
                               get_close_matches(letter.lower(), 
                                                [word.lower() for word in self.english_words],
                                                n=3, cutoff=0.6)]
                # Combine and remove duplicates
                all_matches = list(dict.fromkeys(prefix_matches + close_matches))
                self.current_suggestions = all_matches[:3]
            else:
                self.current_suggestions = prefix_matches[:3]
            
            # Cache the results
            self.suggestions_cache[letter] = self.current_suggestions
        
        # Update buttons
        for i in range(3):
            if i < len(self.current_suggestions):
                self.suggestion_buttons[i].config(text=self.current_suggestions[i])
            else:
                self.suggestion_buttons[i].config(text="")
    
    def update_sentence_display(self):
        """Update the sentence display and refresh suggestions"""
        self.sentence_text.config(text=self.sentence)
        self.update_sentence_suggestions()
    
    def clear_sentence(self):
        """Clear the current sentence"""
        self.sentence = ""
        self.sentence_text.config(text="")
        self.last_letter = None
        self.letter_count = 0
        self.update_sentence_suggestions()
        self.engine.say("Sentence cleared")
        self.engine.runAndWait()
    
    def add_space(self):
        """Add a space to the sentence with auto-correction"""
        if not self.sentence.strip():
            self.sentence += " "
            self.update_sentence_display()
            return
            
        words = self.sentence.split()
        if words:  # Only correct if there are words
            last_word = words[-1]
            suggestions = self.sym_spell.lookup(last_word.lower(), 
                                              Verbosity.CLOSEST,
                                              max_edit_distance=2)
            if suggestions and suggestions[0].term != last_word.lower():
                corrected_word = suggestions[0].term.capitalize()
                words[-1] = corrected_word
                self.sentence = " ".join(words) + " "
                self.update_sentence_display()
                self.engine.say(f"Corrected to {corrected_word}")
                self.engine.runAndWait()
                return
        
        self.sentence += " "
        self.update_sentence_display()
        self.engine.say("Space added")
        self.engine.runAndWait()
    
    def auto_correct_last_word(self):
        """Manually trigger auto-correction for the last word"""
        if not self.sentence.strip():
            return
            
        words = self.sentence.split()
        if words:
            last_word = words[-1]
            suggestions = self.sym_spell.lookup(last_word.lower(), 
                                            Verbosity.CLOSEST,
                                            max_edit_distance=2)
            if suggestions and suggestions[0].term != last_word.lower():
                corrected_word = suggestions[0].term.capitalize()
                words[-1] = corrected_word
                self.sentence = " ".join(words)
                self.update_sentence_display()
                self.engine.say(f"Corrected to {corrected_word}")
                self.engine.runAndWait()
            else:
                self.engine.say("No correction needed")
                self.engine.runAndWait()
    
    def backspace(self):
        """Remove the last character from the sentence"""
        if len(self.sentence) > 0:
            self.sentence = self.sentence[:-1]
            self.update_sentence_display()
            self.engine.say("Backspace")
            self.engine.runAndWait()
    
    def speak_sentence(self):
        """Speak the current sentence"""
        if self.sentence.strip():
            self.engine.say(self.sentence)
            self.engine.runAndWait()
        else:
            self.engine.say("No sentence to speak")
            self.engine.runAndWait()
    
    def add_suggestion_to_sentence(self, index):
        """Add the selected word suggestion to the sentence"""
        if index < len(self.current_suggestions):
            # Add space if needed
            if self.sentence and not self.sentence[-1].isspace():
                self.sentence += " "
            self.sentence += self.current_suggestions[index]
            self.update_sentence_display()
            self.last_letter = None
            self.letter_count = 0
            self.engine.say(f"Added {self.current_suggestions[index]}")
            self.engine.runAndWait()
    
    def update_threshold(self, value):
        """Update the confidence threshold for letter recognition"""
        self.threshold = float(value)
        self.threshold_value.config(text=f"{self.threshold:.2f}")
    
    def detect_gestures(self, hand):
        """Detect special gestures for space and backspace"""
        fingers = self.hd2.fingersUp(hand)
        
        # Space gesture: All fingers extended (open hand)
        if sum(fingers) == 5:
            self.space_gesture_counter += 1
            self.backspace_gesture_counter = 0
            if self.space_gesture_counter >= self.gesture_hold_frames:
                self.space_gesture_counter = 0
                return "space"
        # Backspace gesture: Only thumb extended (thumb out)
        elif fingers == [1, 0, 0, 0, 0]:
            self.backspace_gesture_counter += 1
            self.space_gesture_counter = 0
            if self.backspace_gesture_counter >= self.gesture_hold_frames:
                self.backspace_gesture_counter = 0
                return "backspace"
        else:
            self.space_gesture_counter = 0
            self.backspace_gesture_counter = 0
            
        return None
    
    def process_video(self):
        """Main video processing loop"""
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    continue
                    
                frame = cv2.flip(frame, 1)
                hands, _ = self.hd.findHands(frame, draw=False, flipType=True)
                white = self.white.copy()
                predicted_letter = None
                confidence = 0
                current_gesture = None
                
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    
                    # Ensure bounding box is within frame with offset
                    x1, y1 = max(0, x - self.offset), max(0, y - self.offset)
                    x2, y2 = min(frame.shape[1], x + w + self.offset), min(frame.shape[0], y + h + self.offset)
                    
                    image = frame[y1:y2, x1:x2]
                    
                    if image.size == 0:
                        continue
                        
                    handz, imz = self.hd2.findHands(image, draw=True, flipType=True)
                    
                    if handz:
                        hand = handz[0]
                        pts = hand['lmList']
                        os = ((350 - w) // 2) - 15
                        os1 = ((350 - h) // 2) - 15
                        
                        # Check for special gestures first
                        gesture = self.detect_gestures(hand)
                        
                        if gesture == "space":
                            self.add_space()
                            current_gesture = "SPACE GESTURE (OPEN HAND)"
                        elif gesture == "backspace":
                            self.backspace()
                            current_gesture = "BACKSPACE GESTURE (THUMB OUT)"
                        else:
                            # Draw skeleton lines
                            connections = [
                                (0, 1, 2, 3, 4),
                                (0, 5, 6, 7, 8),
                                (0, 9, 10, 11, 12),
                                (0, 13, 14, 15, 16),
                                (0, 17, 18, 19, 20),
                                (5, 9, 13, 17)
                            ]
                            
                            for connection in connections:
                                for i in range(len(connection)-1):
                                    cv2.line(white, 
                                            (pts[connection[i]][0]+os, pts[connection[i]][1]+os1),
                                            (pts[connection[i+1]][0]+os, pts[connection[i+1]][1]+os1),
                                            (0, 255, 0), 2)
                            
                            # Draw landmarks
                            for i in range(21):
                                cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)
                            
                            # Preprocess for model prediction
                            gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                            resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
                            normalized = resized / 255.0
                            input_img = np.expand_dims(normalized, axis=(0, -1))
                            
                            predictions = self.model.predict(input_img)
                            predicted_class = np.argmax(predictions[0])
                            predicted_letter = self.class_names[predicted_class]
                            confidence = np.max(predictions[0])
                            
                            # Update word suggestions
                            self.update_suggestions(predicted_letter)
                            
                            # Add to sentence if confidence is high enough
                            if confidence >= self.threshold:
                                current_time = time.time()
                                if current_time - self.last_letter_time > self.letter_cooldown:
                                    if predicted_letter != self.last_letter:
                                        self.last_letter = predicted_letter
                                        self.letter_count = 1
                                    else:
                                        self.letter_count += 1
                                        
                                    if self.letter_count >= self.letter_hold_frames:
                                        self.sentence += predicted_letter
                                        self.update_sentence_display()
                                        self.engine.say(predicted_letter)
                                        self.engine.runAndWait()
                                        self.last_letter = None
                                        self.letter_count = 0
                                        self.last_letter_time = current_time
                else:
                    self.update_suggestions(None)
                    self.space_gesture_counter = 0
                    self.backspace_gesture_counter = 0
                
                # Update GUI
                self.update_frames(frame, white, predicted_letter, confidence, current_gesture)
                
            except Exception as e:
                print("Error:", traceback.format_exc())
                continue
    
    def update_frames(self, frame, skeleton, letter=None, confidence=0, gesture=None):
        """Update the video and skeleton frames in the GUI"""
        # Resize frames to fit in the window
        frame = cv2.resize(frame, (450, 350))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(frame)
        frame_img = ImageTk.PhotoImage(image=frame_img)
        
        skeleton = cv2.resize(skeleton, (350, 350))
        skeleton = cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB)
        skeleton_img = Image.fromarray(skeleton)
        skeleton_img = ImageTk.PhotoImage(image=skeleton_img)
        
        self.video_label.config(image=frame_img)
        self.video_label.image = frame_img
        
        self.skeleton_label.config(image=skeleton_img)
        self.skeleton_label.image = skeleton_img
        
        # Update prediction info
        if letter:
            self.prediction_label.config(text=f"Pred: {letter}")
            self.confidence_label.config(text=f"Conf: {confidence:.2f}")
        else:
            self.prediction_label.config(text="Pred: ")
            self.confidence_label.config(text="Conf: ")
        
        if gesture:
            self.gesture_label.config(text=f"Gest: {gesture}")
        else:
            self.gesture_label.config(text="Gest: ")
    
    def on_close(self):
        """Clean up when closing the application"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.capture.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ASLRecognitionApp(root)
    root.mainloop()