from Detector import *
import os

##Estamos usando o SSD MobileNet v3 do tensorFlow, carregado via OpenCV
##Ele foi pré-treinado no dataset COCO

def main():
    videoPath = 0 #"test_videos/cozinhando.mp4" ## para usar webcam podemos colocar 0 e para videos já gravados é necessário colocar o path dele
    
    configPath = os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data","frozen_inference_graph.pb")
    classesPath = os.path.join("model_data","coco.names")

    detector = Detector(videoPath, configPath, modelPath, classesPath)
    detector.onVideo()
    
if __name__ == '__main__':   
    main()