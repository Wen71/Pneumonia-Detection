
# Importing libraries
import tkinter as tk
from tkinter import Label
from tkinter import filedialog
from PIL import Image, ImageTk
import heatmap
import torch
from transformers import pipeline

model_name = "chest-xray-classification"
pipe = pipeline('image-classification', model=model_name, device=0)

label_mappping = {"LABEL_0": 'NORMAL', "LABEL_1": "PNEUMONIA"}

# image uploader function
def imageUploader():
    global img_path  # Make img_path global to access it in predictButton

    #fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    img_path = tk.filedialog.askopenfilename()
 
    # if file is selected
    if len(img_path):
        img = Image.open(img_path)
        img = img.resize((200, 200))
        pic = ImageTk.PhotoImage(img)
 
        # re-sizing the app window in order to fit picture
        # and buttom
        app.geometry("560x400")
        label.config(image=pic)
        label.image = pic
 
    # if no file is selected, then we are displaying below message
    else:
        print("No file is Choosen !! Please choose a file:(")
 
def predictButton():
    global img_path
    if img_path:
        img = Image.open(img_path)
        prediction = pipe(img)[0]
        prediction_label = label_mappping[prediction['label']]
        result_label.config(text=f"Prediction: {prediction_label} with score: {prediction['score']:.3f}")
    else:
        result_label.config(text="No image uploaded")

def diagnoseButton():
    global img_path
    if img_path:
        # heatmap.display_heatmap(img_path)
        # heatmap.enhanced_heatmap(img_path)
        heatmap.generate_gradcam(img_path)
    else:
        result_label.config(text="No image uploaded")

# Main method
if __name__ == "__main__":
    global img_path
    img_path = None

    # defining tkinter object
    app = tk.Tk()
 
    # setting title and basic size to our App
    app.title("Chest x-ray pneumonia detection")
    app.geometry("560x400")
 
    label = tk.Label(app)
    label.grid(row=0, column=0, columnspan=2, pady=10)

    button_frame = tk.Frame(app)
    button_frame.grid(row=1, column=0, columnspan=2, pady=20)

    # Defining our upload button
    uploadButton = tk.Button(button_frame, text="Locate Image", command=imageUploader)
    uploadButton.grid(row=0, column=0, padx=10)

    # Defining our predict button
    predictButton = tk.Button(button_frame, text="Predict", command=predictButton)
    predictButton.grid(row=0, column=1, padx=10)

    diagnoseButton = tk.Button(button_frame, text="Diagnose", command=diagnoseButton)
    diagnoseButton.grid(row=0, column=2, padx=10)

    result_label = tk.Label(app, text="", font=("Helvetica", 16))
    result_label.grid(row=2, column=0, columnspan=2, pady=20)

    app.mainloop()