        
import cv2

from image_helper import COLORS, rescale_bboxes 
from dataset import feature_extractor
from model import Model

class CameraHelper():
    def __init__(self, model: Model):
        self.model = model
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

    def captureAndPredict(self):
        # define a video capture object 
        vid = cv2.VideoCapture(0) 
        size = vid.get(cv2.CAP_PROP_FRAME_WIDTH), vid.get(cv2.CAP_PROP_FRAME_HEIGHT)



        while(True): 
            
            # Capture the video frame 
            # by frame 
            ret, frame = vid.read() 

            # height, width, channels = frame.shape
            # scale = 512 / max(height, width)
            # frame = cv2.resize(frame, (round(scale * width), round(scale * height)))


            encoding = feature_extractor(images=frame, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()
            pixel_values = pixel_values.unsqueeze(0)
            
            outputs = self.model.output(pixel_values=pixel_values)
            self.redraw_frame(size, frame, outputs)
            
            # the 'q' button is set as the 
            # quitting button you may use any 
            # desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 