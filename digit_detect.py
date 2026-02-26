import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import joblib


# Load saved model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# Create Tkinter Window
window = tk.Tk()
window.title("🖌 Digit Recognizer App")
window.configure(bg="#f0f0f0") 
window.geometry("600x600")    


# Canvas
canvas_width = 450
canvas_height = 450
canvas_frame = tk.Frame(window, bg="#ddd", bd=2, relief="ridge")
canvas_frame.pack(pady=20)

canvas = tk.Canvas(canvas_frame, width=canvas_width, height=canvas_height, bg="white", cursor="cross")
canvas.pack()


# PIL image to draw on
image = Image.new("L", (canvas_width, canvas_height), 255)
draw = ImageDraw.Draw(image)


# Drawing function
def draw_lines(event):
    x, y = event.x, event.y
    r = 12 
    canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

canvas.bind("<B1-Motion>", draw_lines)



# Predict function
def predict_digit():
    img_array = np.array(image)

    img_array = 255 - img_array

    coords = np.column_stack(np.where(img_array > 20))
    if coords.size == 0:
        result_label.config(text="Draw something first!")
        return

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    img_array = img_array[y_min:y_max, x_min:x_max]

    img_pil = Image.fromarray(img_array)
    img_pil = img_pil.resize((20, 20))

    new_img = Image.new("L", (28, 28), 0)
    new_img.paste(img_pil, (4, 4))

    img_array = np.array(new_img)

    img_array = img_array.reshape(1, -1)

    img_array = scaler.transform(img_array)

    prediction = model.predict(img_array)

    result_label.config(text=f"Prediction: {prediction[0]}", font=("Arial", 22, "bold"))


# Clear canvas function
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,canvas_width,canvas_height], fill=255)
    result_label.config(text="")


# Buttons & Label Frame
button_frame = tk.Frame(window, bg="#f0f0f0")
button_frame.pack(pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_digit,
                           bg="#4CAF50", fg="white", font=("Arial", 14, "bold"), width=10)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(button_frame, text="Clear", command=clear_canvas,
                         bg="#f44336", fg="white", font=("Arial", 14, "bold"), width=10)
clear_button.grid(row=0, column=1, padx=10)

result_label = tk.Label(window, text="", bg="#f0f0f0", font=("Arial", 20, "bold"))
result_label.pack(pady=10)


# Start the App
window.mainloop()