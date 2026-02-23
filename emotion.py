import threading
import time
from flask import Flask, render_template, Response, redirect, url_for
import cv2
import speech_recognition as sr
from deepface import DeepFace
from textblob import TextBlob
import queue
import tkinter as tk
from tkinter import Label

# Flask app
app = Flask(__name__)

# Queue for streaming logs
log_queue = queue.Queue()

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the speech recognizer
recognizer = sr.Recognizer()

def recognize_speech():
    """Recognizes speech from the microphone and analyzes sentiment."""
    log_queue.put("Listening for voice input...")
    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            log_queue.put(f"You said: {text}")
            sentiment = TextBlob(text).sentiment
            if sentiment.polarity > 0.1:
                return "happy"
            elif sentiment.polarity < -0.1:
                return "sad"
            else:
                return "neutral"
        except sr.UnknownValueError:
            log_queue.put("Could not understand the audio.")
            return "speech unclear"
        except sr.RequestError as e:
            log_queue.put(f"Speech Recognition error: {e}")
            return "error"
        except Exception as e:
            log_queue.put(f"Unexpected error: {e}")
            return "error"

def combine_emotions(face_emotion, speech_emotion):
    """Combines face and speech emotions to determine the final result."""
    if face_emotion == speech_emotion:
        return face_emotion
    elif face_emotion in ["Error", "Unknown"]:
        return speech_emotion
    elif speech_emotion in ["error", "speech unclear"]:
        return face_emotion
    else:
        return "confuse"

@app.route('/')
def index():
    """Render the home page with two buttons."""
    return render_template('index.html')

@app.route('/emotion_page')
def emotion_page():
    """Render the emotion detection page."""
    return render_template('emotion_page.html')

@app.route('/logs')
def logs():
    """Render the logs page."""
    return render_template('logs.html')

@app.route('/stream')
def stream():
    """Stream real-time log messages."""
    def generate():
        while True:
            log_message = log_queue.get()
            yield f"data: {log_message}\n\n"
    return Response(generate(), content_type='text/event-stream')

@app.route('/run_emotion_detection')
def run_emotion_detection():
    """Run emotion detection process and display the result on the webcam screen."""
    cap = cv2.VideoCapture(0)
    
    # Define font for displaying text
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        face_emotion = "Unknown"
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = rgb_frame[y:y + h, x:x + w]
            
            try:
                # Analyze emotion of the face
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list) and len(result) > 0:
                    face_emotion = result[0]['dominant_emotion']
                else:
                    face_emotion = "Unknown"
            except Exception as e:
                log_queue.put(f"DeepFace analysis error: {str(e)}")
                face_emotion = "Error"

            # Draw a rectangle around the face and display the emotion text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {face_emotion}", (x, y - 10), font, 0.9, (0, 255, 0), 2)

        # Perform speech emotion detection every 50th frame
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 50 == 0:
            speech_emotion = recognize_speech()
            log_queue.put(f"Speech Emotion: {speech_emotion}")
        else:
            speech_emotion = "Listening..."

        # Combine face and speech emotion
        final_emotion = combine_emotions(face_emotion, speech_emotion)
        log_queue.put(f"Final Emotion Prediction: {final_emotion}")
        
        # Display final emotion on the screen
        cv2.putText(frame, f"Final Emotion: {final_emotion}", (10, 30), font, 1, (0, 255, 255), 2)
        
        # Show the frame with face detection and emotion text
        cv2.imshow('Emotion Detection', frame)
        
        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('index'))

# Tkinter GUI
def tkinter_gui():
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Emotion Detection")

    # Set the window size
    root.geometry("1000x600")

    # Create frames for layout
    left_frame = tk.Frame(root)
    left_frame.pack(side="left", padx=20)

    right_frame = tk.Frame(root)
    right_frame.pack(side="right", padx=20)

    # Label to display the webcam feed
    video_label = Label(right_frame)
    video_label.pack()

    # Labels to display audio emotion, video emotion, and final emotion
    audio_emotion_label = Label(left_frame, text="Audio Emotion: Unknown", font=("Arial", 15), bg="black", fg="white")
    audio_emotion_label.pack(pady=20)

    video_emotion_label = Label(left_frame, text="Video Emotion: Unknown", font=("Arial", 15), bg="black", fg="white")
    video_emotion_label.pack(pady=20)

    final_emotion_label = Label(left_frame, text="Final Emotion: Unknown", font=("Arial", 20), bg="black", fg="white")
    final_emotion_label.pack(pady=20)

    # Load pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def show_webcam():
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        face_emotion = "Unknown"
        audio_emotion = "Listening..."

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Process each detected face
            for (x, y, w, h) in faces:
                face_roi = rgb_frame[y:y + h, x:x + w]
                try:
                    # Analyze emotion of the face
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(result, list) and len(result) > 0:
                        face_emotion = result[0]['dominant_emotion']
                    else:
                        face_emotion = "Unknown"
                except Exception as e:
                    face_emotion = "Error"

                # Draw a rectangle around the face and display the emotion text
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Face: {face_emotion}", (x, y - 10), font, 0.9, (0, 255, 0), 2)

            # Update the labels in Tkinter
            audio_emotion_label.config(text=f"Audio Emotion: {audio_emotion}")
            video_emotion_label.config(text=f"Video Emotion: {face_emotion}")
            final_emotion = combine_emotions(face_emotion, audio_emotion)
            final_emotion_label.config(text=f"Final Emotion: {final_emotion}")

            # Convert the frame to an image format that Tkinter can display
            img = cv2.imencode('.png', frame)[1].tobytes()  # Use the original BGR frame here
            img_tk = tk.PhotoImage(data=img)
            video_label.config(image=img_tk)
            video_label.image = img_tk  # Keep a reference to avoid garbage collection

            # Update Tkinter window
            root.update_idletasks()
            root.update()

        cap.release()

    # Run the webcam display in a separate thread
    threading.Thread(target=show_webcam, daemon=True).start()

    root.mainloop()

if __name__ == '__main__':
    # Run Flask server in a separate thread
    threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}, daemon=True).start()

    # Run Tkinter GUI
    tkinter_gui()
