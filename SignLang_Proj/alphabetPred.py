import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os as oss
import traceback
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('best_model.h5')  # Replace with your actual model path
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 15
step = 1
flag = False
suv = 0
count = len(oss.listdir("./AtoZ_3.1/A"))
c_dir = 'A'
white = np.ones((400, 400), np.uint8) * 255
cv2.imwrite("./white.jpg", white)

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)
        hands, _ = hd.findHands(frame, draw=False, flipType=True)
        white = cv2.imread("./white.jpg")
        predicted_letter = None  # To store the current prediction

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            image = np.array(frame[y - offset:y + h + offset, x - offset:x + w + offset])
            cv2.imshow("image", image)
            handz, imz = hd2.findHands(image, draw=True, flipType=True)
            if handz:
                hand = handz[0]
                pts = hand['lmList']
                os = ((400 - w) // 2) - 15
                os1 = ((400 - h) // 2) - 15
                
                # Draw skeleton lines (keep your original skeleton drawing code)
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0]+os, pts[t][1]+os1), (pts[t+1][0]+os, pts[t+1][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[5][0]+os, pts[5][1]+os1), (pts[9][0]+os, pts[9][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0]+os, pts[9][1]+os1), (pts[13][0]+os, pts[13][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0]+os, pts[13][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[5][0]+os, pts[5][1]+os1), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0]+os, pts[0][1]+os1), (pts[17][0]+os, pts[17][1]+os1), (0, 255, 0), 3)

                skeleton0 = np.array(white)
                zz = np.array(white)
                for i in range(21):
                    cv2.circle(white, (pts[i][0]+os, pts[i][1]+os1), 2, (0, 0, 255), 1)

                skeleton1 = np.array(white)
                cv2.imshow("1", skeleton1)
                
                # Preprocess the skeleton image for prediction
                gray = cv2.cvtColor(skeleton1, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (55, 55), interpolation=cv2.INTER_AREA)
                normalized = resized / 255.0
                input_img = np.expand_dims(normalized, axis=0)  # Add batch dimension
                input_img = np.expand_dims(input_img, axis=-1)  # Add channel dimension
                
                # Make prediction
                predictions = model.predict(input_img)
                predicted_class = np.argmax(predictions[0])
                predicted_letter = class_names[predicted_class]
                confidence = np.max(predictions[0])
                # print(predictions)

        # Display prediction on frame (only when we have a prediction)
        if predicted_letter:
            frame = cv2.putText(frame, f"Predicted: {predicted_letter} ({confidence:.2f})", 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow("frame", frame)
        
        interrupt = cv2.waitKey(1)
        if interrupt & 0xFF == 27:  # esc key
            break

        

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()