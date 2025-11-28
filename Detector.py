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

            # -----------------------------------------------------
            # 1) Segmentação por limiar (THRESHOLD)
            # -----------------------------------------------------
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
            cv2.imshow("Threshold", thresh)

            # -----------------------------------------------------
            # 2) Detecção de bordas (CANNY)
            # -----------------------------------------------------
            edges = cv2.Canny(gray, 50, 150)
            cv2.imshow("Canny", edges)

            # -----------------------------------------------------
            # 3) Segmentação K-MEANS (AGRUPAMENTO DE CORES)
            # -----------------------------------------------------
            Z = image.reshape((-1, 3))
            Z = np.float32(Z)

            K = 3  # número de clusters (pode trocar p/ 4 ou 5 se quiser mais bonito)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)
            segmented_img = center[label.flatten()]
            segmented_img = segmented_img.reshape(image.shape)

            cv2.imshow("KMeans", segmented_img)

            # -----------------------------------------------------
            # DETECÇÃO DE OBJETOS (SEU YOLO)
            # -----------------------------------------------------
            classLabelIDs, confidences, bboxs = self.net.detect(image, confThreshold=0.5)

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

                    object_count[classLabel] = object_count.get(classLabel, 0) + 1

                    x, y, w, h = bbox
                    area = w * h

                    texts = [
                        f"{classLabel}: {classConfidence:.2f}",
                        f"Tamanho: {w}x{h}px",
                        f"Area: {area}px"
                    ]

                    padding = 5
                    line_height = 18
                    box_height = line_height * len(texts) + padding * 2
                    box_width = 220

                    overlay = image.copy()
                    box_x1 = x
                    box_y1 = max(0, y - box_height - 5)
                    box_x2 = box_x1 + box_width
                    box_y2 = box_y1 + box_height

                    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0)

                    text_y = box_y1 + padding + 12
                    for txt in texts:
                        cv2.putText(image, txt, (box_x1 + padding, text_y),
                                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
                        text_y += line_height

                    cv2.rectangle(image, (x, y), (x + w, y + h), classColor, 1)

                    lineWidth = min(int(w * 0.3), int(h * 0.3))
                    cv2.line(image, (x, y), (x + lineWidth, y), classColor, 5)
                    cv2.line(image, (x, y), (x, y + lineWidth), classColor, 5)

                    cv2.line(image, (x + w, y), (x + w - lineWidth, y), classColor, 5)
                    cv2.line(image, (x + w, y), (x + w, y + lineWidth), classColor, 5)

                    cv2.line(image, (x, y + h), (x + lineWidth, y + h), classColor, 5)
                    cv2.line(image, (x, y + h), (x, y + h - lineWidth), classColor, 5)

                    cv2.line(image, (x + w, y + h), (x + w - lineWidth, y + h), classColor, 5)
                    cv2.line(image, (x + w, y + h), (x + w, y + h - lineWidth), classColor, 5)

            y_offset = 20
            for obj, count in object_count.items():
                text = f"{obj}: {count}"
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                y_offset += 30

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
