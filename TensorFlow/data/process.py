# TODO 换 os.API
from skimage import io
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

image_path = 'test'
json_path = 'test\\label'
label_path = 'test\\label.txt'

def show_image(image_path="train\\000.png"):
    """ 
    展示图片信息
    """
    im = io.imread(image_path)
    print(im.shape)
    plt.imshow(im)
    plt.show()



def imageRename(image_path='train'):
    """ 
    对训练集图片重命名
    """
    for i, old_name in enumerate(os.listdir(image_path)):
        # print(f'{i}, {old_name}\n')
        old_name = image_path + os.sep + old_name   # os.sep 系统分隔符
        new_name = image_path + os.sep + f'{i:0>3}'+'.png'
        os.rename(old_name, new_name)
        print(f'{old_name}-->{new_name}')


def label_txt(json_path='json', label_path='train\\label.txt'):
    # TODO 不用单独再分一个文件夹存 json
    """ 
    生成 label.txt 文件 
    """
    with open(label_path, 'w') as file:
        # content = 'path,point0_x,point0_y,point1_x,point1_y,point2_x,point2_y,point3_x,point3_y,point4_x,point4_y'
        # file.write(content + '\n')
        for i, json_name in enumerate(os.listdir(json_path)):
            content = ''
            json_name = json_path + os.sep + json_name 
            j = open(json_name)
            info = json.load(j)
            image_path = info['imagePath']  # "000.png"
            content += image_path + ','
            shapes = info['shapes']
            for j in range(5):
                # print(f"labels: {shapes[i]['label']}, points: {shapes[i]['points'][0]}")
                x = shapes[j]['points'][0][0]
                y = shapes[j]['points'][0][1]
                x = float(format(x, '.2f'))
                y = float(format(y, '.2f'))
                content += str(x) + ',' + str(y)
                if j < 4:
                    content += ','
            file.write(content + '\n')
            # break
    file.close()  # 关闭文件


def main():
    # imageRename(image_path)
    # show_image("test/000.png")
    label_txt(json_path=json_path, label_path=label_path)

if __name__ == '__main__':
    main()
