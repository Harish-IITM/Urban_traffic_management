import tkinter as tk
import time
import random
import os
from threading import Thread
import torch
from ultralytics import YOLO 

# Load the YOLOv5 model (pretrained)
model = YOLO(r"C:\Users\rudra\Downloads\best.pt")

# List of vehicle classes in the YOLOv5 COCO dataset
vehicle_classes = ['car', 'motorbike', 'bus', 'truck']

def count_vehicles(img_path):
    results = model.predict(source=img_path, imgsz=640, conf=0.7)
    
    vehicle_count = 0
    
    for result in results:
        # result.boxes contains the detected boxes
        for detection in result.boxes:
            label = detection.cls  # This is the tensor containing the class index
            
            # The label is a tensor; check if it's class 0 (Vehicle)
            if int(label.item()) == 0:  # Convert the tensor to a Python integer and check if it's 0
                vehicle_count += 1

    return vehicle_count


def process_multiple_images(image_paths):
    vehicle_counts = []
    
    for img_path in image_paths:
        count = count_vehicles(img_path)
        vehicle_counts.append(count)
    
    return vehicle_counts

def select_random_images(directory, num_images=4):
    # List all image files in the directory (assuming common image extensions)
    all_images = [file for file in os.listdir(directory) if file.endswith(('jpg', 'jpeg', 'png'))]
    
    # Select `num_images` random images from the directory
    if len(all_images) < num_images:
        raise ValueError(f"Not enough images in directory. Found {len(all_images)}, but need {num_images}.")
    
    selected_images = random.sample(all_images, num_images)
    
    # Prepend the directory path to the file names
    selected_images = [os.path.join(directory, img) for img in selected_images]
    
    return selected_images

# Directory where your images are stored
image_directory = r"C:\Users\rudra\OneDrive\Desktop\UTM\images"  # Set your actual image directory here
imgs=select_random_images(image_directory)
vehicles=process_multiple_images(imgs)
print(vehicles)
# Define the countdown timer function
# def countdown_timer(label, initial_count):
#     count = initial_count
#     while True:
#         while count > 0:
#             minutes, seconds = divmod(count, 60)
#             time_formatted = f"{minutes:02}:{seconds:02}"
#             label.config(text=time_formatted)
#             label.update()
#             time.sleep(1)
#             count -= 1
#         # Regenerate vehicle counts and reset timer with average count
#         random_images = select_random_images(image_directory, num_images=4)
#         vehicle_counts = process_multiple_images(random_images)
#         if vehicle_counts:
#             average_count = int(sum(vehicle_counts) / len(vehicle_counts))  # Calculate the average
#         else:
#             average_count = 20  # Fallback in case no vehicles are detected

#         count = max(average_count/(sum(vehicle_counts)) , 20)  # Use the average count for countdown (convert to seconds, min 20 sec)

# # Create the GUI window
# def create_gui():
#     window = tk.Tk()
#     window.title("Vehicle Countdown Timer")

#     # Set the window size and background color
#     window.geometry("300x150")
#     window.configure(bg='lightblue')

#     # Create a label to display the countdown
#     label = tk.Label(window, text="00:00", font=("Helvetica", 48), bg='lightblue', fg='black')
#     label.pack(expand=True)

#     # Start the countdown timer in a separate thread
#     random_images = select_random_images(image_directory, num_images=4)
#     initial_counts = process_multiple_images(random_images)
#     if initial_counts:
#         initial_count = int(sum(initial_counts) / len(initial_counts)) * 60  # Use average count as initial countdown (in seconds)
#     else:
#         initial_count = 60  # Fallback to 60 seconds if no vehicles detected
    
#     timer_thread = Thread(target=countdown_timer, args=(label, initial_count))
#     timer_thread.daemon = True
#     timer_thread.start()

#     # Run the GUI loop
#     window.mainloop()

# if __name__ == "__main__":
#     create_gui()
