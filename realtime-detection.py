import cv2
import numpy as np
import time

# Load Yolo
net = cv2.dnn.readNet("weights/yolov3.weights", "cfg/yolov3.cfg")
classes = []
with open(r"C:/Users/ritish.dhiman/Documents/python codes/coco.names.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


cam_feed = cv2.VideoCapture(0)
cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)



##################################   loading image   #######################################

font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0 #sensor data arrives in the frame_id of the sensor collecting it

while True:
    #This code initiates an infinite loop (to be broken later by a break statement), where we have ret and frame being defined as the cap.read().
    #Basically, ret is a boolean regarding whether or not there was a return at all, at the frame is each frame that is returned.
    #If there is no frame, you wont get an error, you will get None
    _,frame= cam_feed.read()
    frame_id+=1
    
    height,width,channels = frame.shape

##################################    Detecting objects    ###################################
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    #Sets the input value blob for the network net
    outs = net.forward(output_layers)
    #Above line is where the exact feed forward through the network happens.
    #If we don’t specify the output layer names, by default, it will return the predictions only from final output layer.
    #Any intermediate output layer will be ignored
    #We need go through each detection from each output layer to get the class id, confidence and bounding box corners 
    #and more importantly ignore the weak detections (detections with low confidence value).


    #######################   Showing informations on the screen    ##############################
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            #note that the output consists of 85 values instead of 80 from our name list.
            #The first 5 values are as followed: center x, center y, width, height, confidence object present.
            #The rest of the 80 values are the probability of that object displayed
            
            scores = detection[5:]
            #We will then get the ‘classID’ of the max probability of that object. 
            #We will also get that probability value by finding the index of the scores and put it in ‘confidence’.
            
            class_id = np.argmax(scores)
            #argmax() function returns indices of the max element of the array in a particular axis
            
            confidence = scores[class_id]
            if confidence > 0.5:
                ######################### Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                ######################### Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #When we perform the detection, it happens that we have more boxes for the same object, so we should use another function to remove this “noise”.
    #It’s called Non maximum suppresion.


    #Box: contain the coordinates of the rectangle sorrounding the object detected.
    #Label: it’s the name of the object detected
    #Confidence: the confidence about the detection from 0 to 1.
    
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            #From boxes, we extract the x,y,w,h coordinates of the object and label them with their class_ids. 
            #Then random colors generated earlier is assigned to color variable
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
            cv2.putText(frame, label, (x, y + 30), font, 1.5, color, 1)
 
    
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
        
    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;

    
cam_feed.release()    
cv2.destroyAllWindows()        