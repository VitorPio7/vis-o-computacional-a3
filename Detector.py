import cv2
import numpy as np
import time

np.random.seed(20)
class Detector:
    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            
        self.classesList.insert(0, '__Background__')
        
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList),3))
        
        print(self.classesList)
    
    def onVideo(self): 
        cap = cv2.VideoCapture(self.videoPath)
        
        if(cap.isOpened()==False):
            print("Error opening file...")
            return
        (success, image) = cap.read()
        
        startTime = 0
        while success:
            currentTime = time.time()
            fps = 1/(currentTime - startTime)
            startTime = currentTime
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold = 0.5)
            
            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float, confidences))
            
            bboxIdx = cv2.dnn.NMSBoxes(bboxs,confidences, score_threshold=0.5, nms_threshold=0.2)
            object_count = {}
            if len(bboxIdx) != 0:
                for i in range(0, len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelID = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelID]
                    classColor = [int(c) for c in self.colorList[classLabelID]]
                    
                    ##Exibe a contagem dos objetos
                    object_count[classLabel] = object_count.get(classLabel, 0)+1
                    
                    displayText = "{}:{:.2f}".format(classLabel, classConfidence)
                    
                    x,y,w,h = bbox
                    
                    cv2.rectangle(image, (x,y), (x+w, y+h), color=classColor, thickness=1)
                    cv2.putText(image, displayText, (x+10, y), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2,1,True)
                    
                    lineWidth = min(int(w * 0.3), int(h*0.3))
                    
                    cv2.line(image, (x,y), (x+lineWidth,y), classColor, thickness=5)
                    cv2.line(image, (x,y), (x,y+lineWidth), classColor, thickness=5)
                    
                    cv2.line(image, (x + w, y), (x + w - lineWidth,y), classColor, thickness=5)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, thickness=5)
                    
                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, thickness=5)
                    cv2.line(image, (x, y), (x, y+lineWidth), classColor, thickness=5)
                    
                    cv2.line(image, (x + w, y + h), (x + w - lineWidth,y + h), classColor, thickness=5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, thickness=5)
                    
            ## código para demonstrar a contagem no canto superior esquerdo
            y_offset = 20
            for obj, count in object_count.items():
                text = f"{obj}: {count}"
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                y_offset +=30
            fps_text = f"FPS: {int(fps)}"
            
            (text_width, text_height), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_PLAIN, 2, 2)
            
            x_pos = image.shape[1] - text_width - 10
            y_pos = 40
            
            cv2.putText(image, fps_text, (x_pos, y_pos),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            cv2.imshow("Result", image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            
            (success, image) = cap.read()
        cv2.destroyAllWindows()