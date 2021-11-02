from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

import os
import pandas
import argparse

img_type = ['jpg', 'jpeg', 'png', 'heic', 'JPG', 'JPEG', 'PNG', 'HEIC']

def ocr(input_path):
    # ocr
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False)
    img_path = input_path
    result = ocr.ocr(img_path, cls=True)
    res_list = []
    for idx in range(len(result)):
        if not(result[idx][0][0][1] < 220 or result[idx][0][0][0] > 800):
            res_list.append(result[idx])

    result = res_list

    # filter
    exp_list = []
    tmp_set = []
    for line in result:
        if line[1][0].find('以内') != -1:
            if(len(tmp_set) == 0):
                tmp_set.append('')
            tmp_set.append(line[1][0])
            exp_list.append(tmp_set)
            tmp_set = []
        else:
            if (len(tmp_set) == 0):
                tmp_set.append(line[1][0])
            else:
                tmp_set.append('')
                exp_list.append(tmp_set)
                tmp_set = [line[1][0]]

    output_name = img_path.split('/')[-1].split('.')[0]

    folder = os.path.exists('output/imgs')
    if not folder:
        os.makedirs('output/imgs')  

    # export as *.jpg
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('output/imgs/' + output_name + '.jpg')
    return exp_list

if __name__ == '__main__':
    # handle argument
    parser = argparse.ArgumentParser()
    parser.description='please enter two parameters a and b ...'
    parser.add_argument('-i', '--input_dir', help='input dir', type=str, default='input')
    args = parser.parse_args()
    
    files= os.listdir(args.input_dir)

    exp_list = []

    for file in files:
        if not os.path.isdir(file) and (file.split('.')[-1] in img_type):
            print(args.input_dir + '/' + file)
            exp_list += ocr(args.input_dir + '/' + file)

    # export as *.csv
    column=['Username', 'Distance']
    pandas.DataFrame(columns=column,data=exp_list).to_csv('output/res.csv')