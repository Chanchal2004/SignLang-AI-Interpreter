import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
from collections import defaultdict
import pyttsx3
import time

# Import for word suggestions and autocorrect using NLTK
import nltk
from nltk.corpus import words
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk import trigrams
from collections import Counter

try:
    nltk.data.find('corpora/words')
except nltk.downloader.DownloadError:
    nltk.download('words')

try:
    nltk.data.find('corpora/brown')
except nltk.downloader.DownloadError:
    nltk.download('brown')
from nltk.corpus import brown

class ASLRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Recognition System")

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 170)  # Increased speech rate

        # Load model and initialize variables
        try:
            self.model = load_model('best_model.h5')
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {str(e)}")
            self.root.destroy()
            return

        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        # Cooldown parameters - Reduced for responsiveness
        self.letter_cooldown = 0.3  # Slightly reduced cooldown
        self.last_letter_time = 0
        self.cooldown_active = False

        # Gesture recognition parameters - Tuned for faster detection
        self.space_gesture_threshold = 0.75  # Slightly lower threshold
        self.backspace_gesture_threshold = 0.75  # Slightly lower threshold
        self.gesture_hold_frames = 4  # Further reduced hold frames
        self.space_gesture_counter = 0
        self.backspace_gesture_counter = 0

        # Word suggestion and autocorrect setup using NLTK
        self.english_words = set(words.words())
        self.current_suggestions = []
        self.partial_word = "" # To store the currently forming word
        self.suggestion_history = defaultdict(int) # Track frequency of suggestions
        self.suggestion_cooldown = 0.1  # Further reduced cooldown for suggestions

        # Sentence autocorrect setup using NLTK trigrams
        self.trigram_model = self.train_trigram_model()
        self.autocorrect_threshold = 0.6 # Threshold for applying autocorrect
        self.sentence_autocorrect_cooldown = 1.0 # Cooldown for full sentence autocorrect
        self.last_sentence_autocorrect_time = 0

        # Video capture setup with multiple camera attempts
        self.capture = None
        self.camera_index = 0
        while self.camera_index < 3:
            self.capture = cv2.VideoCapture(self.camera_index)
            if self.capture.isOpened():
                break
            self.camera_index += 1

        if not self.capture or not self.capture.isOpened():
            messagebox.showerror("Error", "Could not open any webcam")
            self.root.destroy()
            return

        # Reduced detection confidence for potentially faster tracking
        self.hd = HandDetector(maxHands=1, detectionCon=0.4)
        self.hd2 = HandDetector(maxHands=1, detectionCon=0.4)
        self.offset = 20

        # Sentence formation
        self.sentence = ""
        self.last_letter = None
        self.letter_count = 0
        self.threshold = 0.55 # Further reduced threshold for quicker recognition
        self.hand_detected = False # Flag to track if a hand is currently detected
        self.last_suggestion_update = 0
        self.frame_delay = 0.01 # Small delay to allow GUI updates - can be adjusted

        # Create GUI elements
        self.setup_gui()

        # Start video processing thread
        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.daemon = True
        self.thread.start()

        # White background for skeleton (3-channel color image)
        self.skeleton_width = 400
        self.skeleton_height = 400
        self.white = np.ones((self.skeleton_height, self.skeleton_width, 3), np.uint8) * 255

        # Bind window closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def train_trigram_model(self):
        all_words = brown.words(categories='news')[:50000] # Use a subset for faster training
        n_grams = trigrams(all_words)
        return Counter(n_grams)

    def suggest_next_word(self, sentence):
        if not sentence:
            return []
        words = sentence.lower().split()
        if len(words) < 2:
            return self.get_frequent_suggestions()

        last_two_words = tuple(words[-2:])
        possible_next_words = defaultdict(float)
        for trigram, count in self.trigram_model.items():
            if trigram[:2] == last_two_words:
                possible_next_words[trigram[2]] += count

        sorted_suggestions = sorted(possible_next_words.items(), key=lambda item: item[1], reverse=True)
        return [word.capitalize() for word, _ in sorted_suggestions[:3]]

    def autocorrect_sentence(self):
        current_time = time.time()
        if (current_time - self.last_sentence_autocorrect_time) < self.sentence_autocorrect_cooldown:
            return

        words = self.sentence.lower().split()
        if len(words) < 2:
            return

        corrected_words = list(words)
        modified = False
        for i in range(len(words) - 1):
            context = tuple(corrected_words[max(0, i-1):i+1])
            if len(context) == 2:
                possible_next = self.suggest_next_word(" ".join(context))
                if possible_next and possible_next[0].lower() != corrected_words[i+1] and self.calculate_similarity(corrected_words[i+1], possible_next[0].lower()) < self.autocorrect_threshold:
                    corrected_words[i+1] = possible_next[0].lower()
                    print(f"Autocorrected '{words[i+1]}' to '{corrected_words[i+1]}'")
                    modified = True

        if modified:
            self.sentence = " ".join(corrected_words)
            self.update_sentence_display()
            self.last_sentence_autocorrect_time = current_time

    def calculate_similarity(self, word1, word2):
        if not word1 or not word2:
            return 1.0
        n = 2
        ngrams1 = set(ngrams(word1, n))
        ngrams2 = set(ngrams(word2, n))
        return jaccard_distance(ngrams1, ngrams2)

    def setup_gui(self):
        # Main frames
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        self.skeleton_frame = ttk.LabelFrame(self.root, text="Hand Skeleton")
        self.skeleton_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.info_frame = ttk.LabelFrame(self.root, text="Recognition Info")
        self.info_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Video labels
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")

        self.skeleton_label = ttk.Label(self.skeleton_frame)
        self.skeleton_label.pack(expand=True, fill="both")

        # Info labels
        self.prediction_label = ttk.Label(self.info_frame, text="Predicted Letter: ", font=('Helvetica', 14))
        self.prediction_label.pack(anchor='w', padx=10, pady=5)

        self.confidence_label = ttk.Label(self.info_frame, text="Confidence: ", font=('Helvetica', 14))
        self.confidence_label.pack(anchor='w', padx=10, pady=5)

        # Gesture status label
        self.gesture_label = ttk.Label(self.info_frame, text="Gesture: ", font=('Helvetica', 14))
        self.gesture_label.pack(anchor='w', padx=10, pady=5)

        # Word suggestions frame
        self.suggestions_frame = ttk.LabelFrame(self.info_frame, text="Word Suggestions")
        self.suggestions_frame.pack(fill='x', padx=10, pady=5)

        self.suggestion_buttons = []
        for i in range(5):
            btn = ttk.Button(self.suggestions_frame, text="",
                             command=lambda i=i: self.add_suggestion_to_sentence(i))
            btn.pack(side='left', expand=True, fill='x', padx=5, pady=5)
            self.suggestion_buttons.append(btn)

        # Sentence display
        self.sentence_frame = ttk.Frame(self.info_frame)
        self.sentence_frame.pack(fill='x', padx=10, pady=10)

        self.sentence_label = ttk.Label(self.sentence_frame, text="Sentence: ", font=('Helvetica', 16))
        self.sentence_label.pack(side='left', anchor='w')

        # Control buttons frame
        self.control_frame = ttk.Frame(self.info_frame)
        self.control_frame.pack(side='right', padx=10, pady=10)

        # Text-to-speech button
        self.speak_button = ttk.Button(self.control_frame, text="Speak", command=self.speak_sentence)
        self.speak_button.pack(side='left', padx=5)

        # Clear button
        self.clear_button = ttk.Button(self.control_frame, text="Clear", command=self.clear_sentence)
        self.clear_button.pack(side='left', padx=5)

        # Space button
        self.space_button = ttk.Button(self.control_frame, text="Space", command=self.add_space)
        self.space_button.pack(side='left', padx=5)

        # Backspace button
        self.backspace_button = ttk.Button(self.control_frame, text="Backspace", command=self.backspace)
        self.backspace_button.pack(side='left', padx=5)

        # Threshold control
        self.threshold_frame = ttk.Frame(self.info_frame)
        self.threshold_frame.pack(anchor='w', padx=10, pady=5)

        ttk.Label(self.threshold_frame, text="Confidence Threshold:").pack(side='left')
        self.threshold_slider = ttk.Scale(self.threshold_frame, from_=0.5, to=1.0, value=self.threshold,
                                            command=self.update_threshold)
        self.threshold_slider.pack(side='left', padx=5)
        self.threshold_value = ttk.Label(self.threshold_frame, text=f"{self.threshold:.2f}")
        self.threshold_value.pack(side='left')

        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_value.config(text=f"{self.threshold:.2f}")

    def clear_sentence(self):
        self.sentence = ""
        self.sentence_label.config(text="Sentence: ")
        self.last_letter = None
        self.letter_count = 0
        self.partial_word = ""
        self.update_suggestions("") # Clear suggestions
        self.engine.say("Sentence cleared")
        self.engine.runAndWait()

    def add_space(self):
        if self.partial_word:
            self.sentence += self.partial_word + " "
            self.partial_word = ""
            self.autocorrect_sentence() # Attempt autocorrect after adding a word
        elif self.sentence and not self.sentence.endswith(" "):
            self.sentence += " "
            self.autocorrect_sentence()
        self.update_sentence_display()
        self.update_suggestions("") # Clear suggestions after space
        self.last_letter = None
        self.letter_count = 0
        self.engine.say("Space added")
        self.engine.runAndWait()

    def backspace(self):
        if self.partial_word:
            self.partial_word = self.partial_word[:-1]
            self.update_suggestions(self.partial_word)
        elif self.sentence:
            self.sentence = self.sentence[:-1]
        self.update_sentence_display()
        self.last_letter = None
        self.letter_count = 0
        self.engine.say("Backspace")
        self.engine.runAndWait()

    def speak_sentence(self):
        final_sentence = (self.sentence + " " + self.partial_word).strip()
        if final_sentence:
            self.engine.say(final_sentence)
            self.engine.runAndWait()
        else:
            self.engine.say("Nothing to speak")
            self.engine.runAndWait()

    def update_sentence_display(self):
        display_text = f"Sentence: {self.sentence} {self.partial_word}"
        self.sentence_label.config(text=display_text.strip())

    def add_suggestion_to_sentence(self, index):
        if index < len(self.current_suggestions):
            word = self.current_suggestions[index]
            self.sentence += self.partial_word + " " + word + " " if self.partial_word else word + " "
            self.partial_word = ""
            self.update_sentence_display()
            self.update_suggestions("") # Clear suggestions after adding word
            self.last_letter = None
            self.letter_count = 0
            self.suggestion_history[word.lower()] += 1 # Increment frequency
            self.autocorrect_sentence() # Attempt autocorrect after adding a suggested word
            self.engine.say(f"Added {word}")
            self.engine.runAndWait()

    def generate_suggestions(self, current_word):
        suggestions = []
        if not current_word:
            # Suggest based on the last two words of the sentence
            suggestions.extend(self.suggest_next_word(self.sentence))
            suggestions.extend(self.get_frequent_suggestions()[:2])
            return list(dict.fromkeys(suggestions[:5]))

        n = 2  # Consider n-grams of size 2 (bigrams)
        current_ngrams = set(ngrams(current_word.lower(), n))

        scored_suggestions = []
        for word in self.english_words:
            if len(word) >= max(0, len(current_word) - 1) and len(word) <= len(current_word) + 2: # More aggressive length filter
                word_ngrams = set(ngrams(word.lower(), n))
                distance = jaccard_distance(current_ngrams, word_ngrams)
                score = distance - (self.suggestion_history.get(word.lower(), 0) * 0.005) # Slightly increased frequency weight
                scored_suggestions.append((score, word.capitalize()))

        scored_suggestions.sort(key=lambda item: item[0])
        top_suggestions = [suggestion[1] for suggestion in scored_suggestions[:2]]

        # Add next word suggestions based on sentence context
        top_suggestions.extend(self.suggest_next_word)
        (self.sentence)

        frequent_suggestions = self.get_frequent_suggestions(count=5)
        for f_suggestion in frequent_suggestions:
            if f_suggestion not in top_suggestions:
                top_suggestions.append(f_suggestion)

        return list(dict.fromkeys(top_suggestions[:5])) # Remove duplicates and limit to 5

    def get_frequent_suggestions(self, count=5):
        sorted_history = sorted(self.suggestion_history.items(), key=lambda item: item[1], reverse=True)
        return [item[0].capitalize() for item in sorted_history[:count]]

    def update_suggestions(self, current_word):
        current_time = time.time()
        if (current_time - self.last_suggestion_update) > self.suggestion_cooldown:
            self.current_suggestions = self.generate_suggestions(current_word)
            for i, btn in enumerate(self.suggestion_buttons):
                if i < len(self.current_suggestions):
                    btn.config(text=self.current_suggestions[i])
                else:
                    btn.config(text="")
            self.last_suggestion_update = current_time

    def detect_gestures(self, hand):
        """Detect special gestures for space and backspace"""
        fingers = self.hd2.fingersUp(hand)

        # Space gesture: All fingers extended (open hand)
        if sum(fingers) == 5:  # All fingers up
            self.space_gesture_counter += 1
            self.backspace_gesture_counter = 0
            if self.space_gesture_counter >= self.gesture_hold_frames:
                self.space_gesture_counter = 0
                return "space"
        # Backspace gesture: Only thumb extended (thumb out)
        elif fingers == [1, 0, 0, 0, 0]:  # Only thumb is up
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
        while self.running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    continue

                # Further optimize frame resizing
                scale_factor = 0.4  # Slightly smaller for camera feed
                frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
                frame_small = cv2.flip(frame_small, 1)
                hand_results = self.hd.findHands(frame_small, draw=False, flipType=True)

                if hand_results:
                    if isinstance(hand_results, tuple):
                        hands, _ = hand_results
                    else:
                        hands = hand_results
                else:
                    hands = []
                    self.hand_detected = False
                    self.update_suggestions("")
                    self.partial_word = ""

                white = self.white.copy()
                predicted_letter = None
                confidence = 0
                current_gesture = None

                if hands:
                    self.hand_detected = True
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    # Scale back bounding box
                    x, y, w, h = int(x / scale_factor), int(y / scale_factor), int(w / scale_factor), int(h / scale_factor)
                    # Adjust cropping slightly to ensure hand is fully captured
                    crop_offset = 25
                    image = np.array(frame[max(0, y - crop_offset):min(frame.shape[0], y + h + crop_offset),
                                            max(0, x - crop_offset):min(frame.shape[1], x + w + crop_offset)])

                    if image.size == 0:
                        continue

                    hand_results2 = self.hd2.findHands(image, draw=True, flipType=True)
                    if hand_results2:
                        if isinstance(hand_results2, tuple):
                            handz, _ = hand_results2
                        else:
                            handz = hand_results2

                        if handz:
                            hand = handz[0]
                            pts = hand['lmList']
                            # Adjust offset based on cropped image size for better skeleton positioning
                            ch, cw, _ = white.shape
                            bh, bw, _ = image.shape
                            os_x = (cw - bw) // 2
                            os_y = (ch - bh) // 2

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
                                    (0, 1, 2, 3, 4),     # Thumb
                                    (0, 5, 6, 7, 8),     # Index
                                    (0, 9, 10, 11, 12),    # Middle
                                    (0, 13, 14, 15, 16),   # Ring
                                    (0, 17, 18, 19, 20),   # Pinky
                                    (5, 9, 13, 17)      # Palm connections
                                ]

                                for connection in connections:
                                    for i in range(len(connection)-1):
                                        try:
                                            cv2.line(white,
                                                     (pts[connection[i]][0] + os_x, pts[connection[i]][1] + os_y),
                                                     (pts[connection[i+1]][0] + os_x, pts[connection[i+1]][1] + os_y),
                                                     (0, 255, 0), 3)
                                        except IndexError:
                                            continue # Handle cases where not all landmarks are detected

                                # Draw landmarks
                                for i in range(21):
                                    try:
                                        cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)
                                    except IndexError:
                                        continue # Handle cases where not all landmarks are detected

                                # Convert to grayscale if needed
                                if len(white.shape) == 2:
                                    white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)

                                # Preprocess for model prediction
                                gray = cv2.cvtColor(white, cv2.COLOR_BGR2GRAY)
                                resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
                                normalized = resized / 255.0
                                input_img = np.expand_dims(normalized, axis=(0, -1))

                                # Get prediction
                                predictions = self.model.predict(input_img)
                                predicted_class = np.argmax(predictions[0])
                                predicted_letter = self.class_names[predicted_class]
                                confidence = np.max(predictions[0])

                                # Update word suggestions based on the predicted letter and current partial word
                                if predicted_letter in self.class_names and confidence > self.threshold:
                                    current_time = time.time()
                                    if predicted_letter != self.last_letter or (current_time - self.last_letter_time) > self.letter_cooldown:
                                        self.partial_word += predicted_letter
                                        self.update_suggestions(self.partial_word)
                                        self.last_letter = predicted_letter
                                        self.last_letter_time = current_time
                    else:
                        # If no hand detected in the cropped image, still try to suggest
                        self.update_suggestions(self.partial_word)

                # If no hand detected in the main frame, still try to suggest based on partial word
                if not hands:
                    self.update_suggestions(self.partial_word)
                    if self.partial_word:
                        self.autocorrect_partial_word()
                        if self.partial_word.endswith(" "):
                            self.autocorrect_sentence()
                            self.partial_word = ""

                # Update GUI
                self.update_frames(frame_small, white, predicted_letter, confidence, current_gesture)

                time.sleep(self.frame_delay) # Introduce a small delay

            except Exception as e:
                print(f"Error in video processing: {str(e)}")
                traceback.print_exc() # Print the full traceback for debugging
                continue

    def autocorrect_partial_word(self):
        if self.partial_word and not self.partial_word[-1].isspace():
            n = 2
            current_ngrams = set(ngrams(self.partial_word.lower(), n))
            scored_suggestions = []
            for word in self.english_words:
                if word.lower().startswith(self.partial_word.lower()):
                    word_ngrams = set(ngrams(word.lower(), n))
                    distance = jaccard_distance(current_ngrams, word_ngrams)
                    scored_suggestions.append((distance, word))
            if scored_suggestions:
                scored_suggestions.sort(key=lambda item: item[0])
                best_match = scored_suggestions[0][1]
                if scored_suggestions[0][0] < 0.4 and len(best_match) > len(self.partial_word) and time.time() - self.last_letter_time > 0.8: # More relaxed autocorrect
                    print(f"Autocorrected partial word '{self.partial_word}' to '{best_match}'")
                    self.partial_word = best_match
                    self.update_suggestions(self.partial_word)
                    self.update_sentence_display()

    def update_frames(self, frame, skeleton, letter=None, confidence=0, gesture=None):
        try:
            # Convert frames to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_img = Image.fromarray(frame)
            frame_img = ImageTk.PhotoImage(image=frame_img)

            # Ensure skeleton is 3-channel for display
            if len(skeleton.shape) == 2:
                skeleton = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
            skeleton_img = Image.fromarray(skeleton)
            skeleton_img = ImageTk.PhotoImage(image=skeleton_img)

            # Update labels
            self.video_label.config(image=frame_img)
            self.video_label.image = frame_img

            self.skeleton_label.config(image=skeleton_img)
            self.skeleton_label.image = skeleton_img

            # Update prediction info
            pred_text = f"Predicted Letter: {letter}" if letter else "Predicted Letter: "
            self.prediction_label.config(text=pred_text)

            conf_text = f"Confidence: {confidence:.2f}" if letter else "Confidence: "
            self.confidence_label.config(text=conf_text)

            # Update gesture info
            gesture_text = f"Gesture: {gesture}" if gesture else "Gesture: "
            self.gesture_label.config(text=gesture_text)

            # Update sentence display
            self.update_sentence_display()
        except Exception as e:
            print(f"Error updating frames: {e}")

    def on_close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
        self.root.destroy()

if __name__== "__main__":
    root = tk.Tk()
    try:
        app = ASLRecognitionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        traceback.print_exc()
        root.destroy()