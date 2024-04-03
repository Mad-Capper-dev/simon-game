        
import cv2

from image_helper import COLORS, box_cxcywh_to_xyxy, rescale_bboxes 
from dataset import feature_extractor
from model import Model
import time

from threading import Thread
import cv2
import queue
import threading

class CameraHelper():
    vid: cv2.VideoCapture
    size: [int, int]
    balls: [int]

    def __init__(self, model: Model, show: bool = True):
        self.model = model
        self.balls = []
        self.input_buffer = queue.Queue()
        self.detection_buffer = queue.Queue()
        
        t = threading.Thread(target=self.show)
        t.start()

        return
    
    def redraw_frame(self, size, frame, outputs, threshold=0.9):
            # keep only predictions with confidence >= threshold
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        
        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), size)

        colors = COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(probas[keep], bboxes_scaled.tolist(), colors):
            
                frame = cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax + 10), int(ymax + 10)), (0,0,255), 2)
        
        # Display the resulting frame 
        cv2.imshow('frame', frame) 

    def show(self):
        self.capture()

        while True:
            ret, frame = self.vid.read()
            if ret:
                self.input_buffer.put(frame)
                cv2.imshow("Video", frame)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        return

    def capture(self):
        self.vid = cv2.VideoCapture(0)
        self.size = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        print("Sleeping")
        time.sleep(1)
        print("Reading frame")
        ret, frame = self.vid.read() 
        print("Predicting")
        encoding = feature_extractor(images=frame, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        pixel_values = pixel_values.unsqueeze(0)
                
        outputs = self.model.output(pixel_values=pixel_values)
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        tensor = box_cxcywh_to_xyxy(outputs.pred_boxes[0, keep].cpu())
        balls_coord = tensor.detach().numpy()

        for ball in balls_coord:
            self.balls.append(ball[0])

        print(self.balls)
        self.balls.sort()
         
        print(self.balls)

     

    # Capture the video and returns the first masked ball
    def waitForSelection(self, redraw=True):
        while True:
            # Capture the video frame 
            # by frame 
            ret, frame = self.vid.read() 

            encoding = feature_extractor(images=frame, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            pixel_values = pixel_values.unsqueeze(0)
                
            outputs = self.model.output(pixel_values=pixel_values)
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.9

            tensor = box_cxcywh_to_xyxy(outputs.pred_boxes[0, keep].cpu())
            balls_coord = tensor.detach().numpy()

            if(len(balls_coord) == 3):
                balls = list(map(lambda ball: ball[0], balls_coord))
                balls.sort()
                
                if(self.balls[0] - 0.05 <= balls[0] <= self.balls[0] + 0.05 ):
                    if(self.balls[1] - 0.05 <= balls[1] <= self.balls[1] + 0.05):
                        if(self.balls[2] - 0.05 <= balls[2] <= self.balls[2] + 0.05):
                            return 3
                        else:
                            return 2
                    else:
                        return 1
                else:
                    return 0

            if(redraw):
                self.redraw_frame(self.size, frame, outputs)

            # the 'q' button is set as the 
            # quitting button you may use any 
            # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                # After the loop release the cap object 
                self.vid.release() 
                # Destroy all the windows 
                cv2.destroyAllWindows() 
                break

    def captureAndPredict(self, redraw=True):
        # define a video capture object 
        self.vid = cv2.VideoCapture(0) 
        self.size = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)



        while(True): 
            
            # Capture the video frame 
            # by frame 
            ret, frame = self.vid.read() 

            encoding = feature_extractor(images=frame, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            pixel_values = pixel_values.unsqueeze(0)
            
            outputs = self.model.output(pixel_values=pixel_values)
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.9

            print(f"keep : {probas[keep]}")
            if(redraw):
                self.redraw_frame(self.size, frame, outputs)


            
            # the 'q' button is set as the 
            # quitting button you may use any 
            # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        # After the loop release the cap object 
        self.vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 