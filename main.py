import matplotlib.pyplot as plt

import numpy as np

import os
import cv2
import torch
from PIL import Image
import csv
import pandas as pd
import glob
from ultralytics import YOLO


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

def check_the_frame(frame,v):

    img = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    results = model(img)
    #results.show()
    tensor = results.xyxy[0]
    vecteur1 = Tesnor_to_vector_array(tensor)
    if len(vecteur1) == 4:
        v[1]=1
        vecteur_ccw1 = [vecteur1[2], vecteur1[1], vecteur1[0], vecteur1[3]]


    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resultss = model(img)

    tensor = resultss.xyxy[0]
    vecteur2 = Tesnor_to_vector_array(tensor)
    if len(vecteur2) == 4:
        v[2] = 1
        vecteur_ccw2 = [vecteur2[2], vecteur2[3], vecteur2[0], vecteur2[1]]
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    resultss = model(img)

    tensor = resultss.xyxy[0]
    vecteur3 = Tesnor_to_vector_array(tensor)
    if len(vecteur3) == 4 :
        v[3] = 1
        vecteur_ccw3 = [vecteur3[0], vecteur3[3], vecteur3[2], vecteur3[1]]


    if len(vecteur1) == 4:
        return vecteur_ccw1,v
    elif len(vecteur2) == 4:
        return vecteur_ccw2,v
    elif len(vecteur3) == 4:
        return vecteur_ccw3,v
    else:
        vecteur=[]
        return vecteur,v


def dict_to_csv(dict):
    with open('my_file.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, value in dict.items():
            writer.writerow(value)
def dict_to_csv_n_f(dict):
    with open('my_file_no_fall.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, value in dict.items():
            writer.writerow(value)
def all_lines_equal(path):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        first_row = next(reader)
        length = len(first_row)
        for row in reader:
            if len(row) != length:
                return False
    return True
def left_padding(path):



    with open(path, 'r') as input_file, open('output.csv', 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)


        max_length = max(len(row) for row in reader)

        input_file.seek(0)


        for row in reader:

            first_12 = row[:12]

            print(first_12)
            if len(row) == max_length:

                writer.writerow(row)
            else:

                num_pads = int((max_length - len(row)) / len(first_12))

                v=first_12*(num_pads+1)

                padded_row = v + row[12:]

                writer.writerow(padded_row)


def calcul_surf_cont_cen(vector):
    longueur =abs( vector[0] - vector[2])
    largeur = abs(vector[1] - vector[3])
    surface = longueur * largeur
    contour = (longueur + largeur)*2
    centre_x = (vector[0] + vector[2]) / 2
    centre_y = (vector[1] + vector[3]) / 2
    return surface,contour,centre_x,centre_y
def get_the_eight_values(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    coins = [x1, y1, x2, y1, x1, y2, x2, y2]
    return coins



def compter_lignes(folder_path):

    all_files = glob.glob(folder_path + "/*.csv")
    resultats = []

    for file in all_files:
        df = pd.read_csv(file)

        count = (df[(df.iloc[:, 1:9] == -1).all(axis=1) & (df.iloc[:, 9:] == 0).all(axis=1)].count().sum())/13

        resultats.append({'Fichier': file, 'Nombre de lignes': count})

    resultats_df = pd.DataFrame(resultats)
    return resultats_df

def get_bounding(folder_path):


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
                matrix_with_frame = []

                video = None  # initialize video variable to None
                if file.endswith('.mp4') or file.endswith('.avi'):
                    video_path = os.path.join(subfolder_path, file)
                    video = cv2.VideoCapture(video_path)  # assign VideoCapture object to video variable
                    fps = video.get(cv2.CAP_PROP_FPS)
                    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    #print(total_frames)
                    j=1
                    v=[0,0,0,0]
                    for frame_num in range(total_frames):
                        # Read the frame
                        ret, frame = video.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        #img = frame
                        results = model(frame)
                        tensor = results.xyxy[0]

                        vecteur0 = Tesnor_to_vector_array(tensor)
                        if len(vecteur0)==4:
                            v[0]=1



                        vector_with_check,vec=check_the_frame(frame,v)
                        if len(vecteur0)==4:
                            vector = vecteur0
                            surface, contour, centre_x, centre_y = calcul_surf_cont_cen(vector)
                            vector = get_the_eight_values(vector[:2], vector[2:])
                            valeurs_supplementaires = [centre_x, centre_y, surface, contour]
                            vector += valeurs_supplementaires
                            vector +=v
                            vector_with_frame=[j]+vector[:]
                            matrix.append(vector)
                            matrix_with_frame.append(vector_with_frame)
                        elif len(vector_with_check)==4:
                            surface, contour, centre_x, centre_y = calcul_surf_cont_cen(vector_with_check)
                            vector_with_check = get_the_eight_values(vector_with_check[:2], vector_with_check[2:])
                            valeurs_supplementaires = [centre_x, centre_y, surface, contour]
                            vector_with_check += valeurs_supplementaires
                            vector_with_check +=v
                            vector_with_frame = [j] + vector_with_check[:]
                            matrix.append(vector_with_check)
                            matrix_with_frame.append(vector_with_frame)
                        elif len(vecteur0)==0 and len(vector_with_check)==0:
                            vector=[-1,-1,-1,-1,-1,-1,-1,-1]
                            surface=contour=centre_x=centre_y=0
                            valeurs_supplementaires = [centre_x, centre_y, surface, contour]
                            vector += valeurs_supplementaires

                            vector_with_frame = [j] + vector[:]
                            matrix.append(vector)
                            matrix_with_frame.append(vector_with_frame)


                        j=j+1

                if video is not None:  # check if video variable has been assigned a value before releasing it
                    video.release()

                file_name = os.path.basename(file)
                video_name =os.path.splitext(file_name)[0]
                df = pd.DataFrame(matrix_with_frame)
                df.to_csv(f"{video_name}.csv", index=False)

                m = np.concatenate(matrix)


                my_dict[f'{i}'] = m
                i = i + 1

    return my_dict

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
folder_path_all = "yolov5-master/dataset/train_all"
All_the_dataset=get_bounding(folder_path_all)
dict_to_csv(All_the_dataset)
#path='my_file.csv'
#left_padding(path)


