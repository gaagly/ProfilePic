import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default camera (usually the built-in webcam)

# Function to capture photo and save cropped face
def capture_photo():
    ret, frame = cap.read()
    process_image(frame)

# Function to process image and save cropped face
def process_image(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        for i, (x, y, w, h) in enumerate(faces):
            aspect_ratio = w / h

            new_width = 320
            new_height = int(new_width / aspect_ratio)

            y_adjusted = max(0, y - 50)
            h_adjusted = min(frame.shape[0], y + h + 100)
            x_adjusted = max(0, x - int((h_adjusted - y_adjusted) * aspect_ratio - w) // 2)
            w_adjusted = min(frame.shape[1], x + w + int((h_adjusted - y_adjusted) * aspect_ratio - w) // 2)

            cropped_face = cv2.resize(frame[y_adjusted:h_adjusted, x_adjusted:w_adjusted], (new_width, new_height))
            cv2.imwrite(f'cropped_face_{i}.jpg', cropped_face)
            print(f"Face {i+1} cropped and saved successfully.")
    else:
        print("No face detected.")

# Function to handle the choice of source
def select_source():
    selected_source = source_var.get()

    if selected_source == 0:  # Camera
        ret, frame = cap.read()
        if ret:
            process_image(frame)
        else:
            print("Error capturing photo from camera.")
    elif selected_source == 1:  # Image/PDF
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            process_image(image)

# GUI setup
root = tk.Tk()
root.title("Face Crop App")

# Create a label to display the webcam feed
label = tk.Label(root)
label.pack(padx=10, pady=10)

# Create radio buttons for selecting the source
source_var = tk.IntVar()
camera_radio = tk.Radiobutton(root, text="Use Camera", variable=source_var, value=0)
camera_radio.pack(anchor=tk.W)
file_radio = tk.Radiobutton(root, text="Use Image/PDF", variable=source_var, value=1)
file_radio.pack(anchor=tk.W)

# Create a button to process the selected source
process_button = tk.Button(root, text="Process", command=select_source)
process_button.pack(pady=10)

# Function to update the webcam feed
def update_feed():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        label.img = img
        label.config(image=img)
    root.after(10, update_feed)

# Start updating the webcam feed
update_feed()

# Run the GUI main loop
root.mainloop()

# Release the webcam
cap.release()
cv2.destroyAllWindows()
