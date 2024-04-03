import serial
import time

# Bring your packages onto the path
from model import Model
from camera_helper import CameraHelper
from dataset import train_dataset, val_dataset
model = Model(train_dataset=train_dataset, val_dataset=val_dataset)

# Load the model
model.load( version=6, checkpoint_name="epoch=52-step=2000")

# Open the Webcam
camera = CameraHelper(model)
# Open the serial port
ser = serial.Serial('COM3', 9600)  # Change '/dev/ttyACM0' to the correct port

# Wait for Arduino to initialize
time.sleep(2)



# Simulate user input (replace with actual user input logic)
while True:
    # Wait for response from Arduino
    response = ser.readline().decode()

    # If Arduino sends a check message
    if(response.strip() == "Check"):
        # Then we wait for the user's selection
        answer = camera.waitForSelection(redraw=False)

        # And write the answer
        ser.write((str(answer)).encode())
     
    time.sleep(.5)

# Close the serial port
ser.close()