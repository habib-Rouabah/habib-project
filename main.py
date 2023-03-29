import matplotlib.pyplot as plt

import numpy as np

import os
import cv2
import torch
from PIL import Image



def matrix_to_vector(matrix):
    matrix = np.array(matrix)
    vector = matrix.flatten()
    return vector


def Tesnor_to_vector_array(tensor):
    mask = tensor[:, -1] == 0
    rows_with_zero = tensor[mask]
    resu = rows_with_zero[:, :4]
    b = resu.numpy()

    return b.flatten()


def show_image_with_bounding(img, resu):
    color = (0, 255, 0)
    thickness = 2
    for i in resu:
        start_point = (int(i[0]), int(i[1]))
        end_point = (int(i[2]), int(i[3]))
        cv2.rectangle(img, start_point, end_point, color, thickness)

    cv2.imshow("Bounding Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()







def get_bounding(folder_path):
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Set device (CPU/GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    my_dict = {}
    i = 1
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):

            for file in os.listdir(subfolder_path):
                matrix = []
                video = None  # initialize video variable to None
                if file.endswith('.mp4') or file.endswith('.avi'):
                    video_path = os.path.join(subfolder_path, file)
                    video = cv2.VideoCapture(video_path)  # assign VideoCapture object to video variable
                    fps = video.get(cv2.CAP_PROP_FPS)
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    #print(total_frames)
                    for frame_num in range(total_frames):
                        # Read the frame
                        ret, frame = video.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = frame
                        results = model(frame)
                        tensor = results.xyxy[0]

                        vector = Tesnor_to_vector_array(tensor)
                        if len(vector)==0:
                            print("here")
                            print(tensor)
                            results.show()
                        if len(vector)==8:
                            print("here with 8")
                            print(tensor)
                            results.show()

                        #print(len(vector))
                            # print(vector)
                            # print("sep")

                            # show_image_with_bounding(img, resu)
                        matrix.append(vector)
                            #print(len(matrix))


                        # print("sep")
                if video is not None:  # check if video variable has been assigned a value before releasing it
                    video.release()
                print(len(matrix))

                m = np.concatenate(matrix)
                #print(len(m))
                #print(m)

                my_dict[f'{i}'] = m
                i = i + 1
            # print(my_dict)
    return my_dict


folder_path_fall = "yolov5-master/dataset/train/fall"
fall = get_bounding(folder_path_fall)