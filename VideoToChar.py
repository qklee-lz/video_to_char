import os
import argparse

import numpy as np

from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import cv2

sample_rate = 0.4

def mkdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)    

def ascii_art(file):
    im = Image.open(file)

    # Compute letter aspect ratio
    font = ImageFont.load_default()
    # font = ImageFont.truetype("SourceCodePro-Bold.ttf", size=12)
    aspect_ratio = font.getsize("x")[0] / font.getsize("x")[1]
    new_im_size = np.array(
        [im.size[0] * sample_rate, im.size[1] * sample_rate * aspect_ratio]
    ).astype(int)

    # Downsample the image
    im = im.resize(new_im_size)

    # Keep a copy of image for color sampling
    im_color = np.array(im)

    # Convert to gray scale image
    im = im.convert("L")

    # Convert to numpy array for image manipulation
    im = np.array(im)

    # Defines all the symbols in ascending order that will form the final ascii
    symbols = np.array(list(" .-vM"))

    # Normalize minimum and maximum to [0, max_symbol_index)
    im = (im - im.min()) / (im.max() - im.min()) * (symbols.size - 1)

    # Generate the ascii art
    ascii = symbols[im.astype(int)]
    # lines = "\n".join(("".join(r) for r in ascii))

    # Create an output image for drawing ascii text
    letter_size = font.getsize("x")
    im_out_size = new_im_size * letter_size
    bg_color = "black"
    im_out = Image.new("RGB", tuple(im_out_size), bg_color)
    draw = ImageDraw.Draw(im_out)

    # Draw text
    y = 0
    for i, line in enumerate(ascii):
        for j, ch in enumerate(line):
            color = tuple(im_color[i, j])  # sample color from original image
            draw.text((letter_size[0] * j, y), ch[0], fill=color, font=font)
        y += letter_size[1]  # increase y by letter height

    # Save image file
    # im_out.save(file + ".ascii.png")
    
    # resize
    im_out = im_out.resize(args.save_size)
    return im_out

def unlock_movie(src_path):
# """ 将视频转换成图片path: 视频路径 """
    cap = cv2.VideoCapture(src_path)
    print("视频宽度: {}".format(cap.get(3)))
    print("视频高度: {}".format(cap.get(4)))
    print("视频fps: {}".format(cap.get(5)))
    print("视频帧数: {}".format(cap.get(7)))
    input("?")
    CV_CAP_PROP_FRAME_COUNT = cap.get(7)

    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    frame_save = 0
    save_folder_path = args.save_folder
    
    while suc:
        suc, frame = cap.read()
        if frame_count % args.gap_frame == 0:
            path = save_folder_path + "temp_{}.jpg".format(frame_save)
            cv2.imwrite(path, frame)
        
            out = ascii_art(path)
            out.save(path)

            frame_save += 1

            # plt.cla()
            # plt.imshow(out)
            # print(1)
            # plt.pause(0.01)  
        frame_count += 1

        print("进度: {}/{}".format(frame_count, CV_CAP_PROP_FRAME_COUNT))

    cap.release()
    print('unlock movie: ', frame_count)

def show():
    img_name_list = os.listdir(args.save_folder)
    img_name_list.sort(key= lambda x:int(x[5:-4]))
    img_list = [args.save_folder + i for i in img_name_list]
    for i in range(0, len(img_list)):
        a = cv2.imread(img_list[i])
        cv2.imshow("dragon ball", a)
        cv2.waitKey(300)

    cv2.waitKey (0) 
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert any image to ascii art.")
    parser.add_argument("--file", default='/Users/lqk/Desktop/temp_project/【1080P-60帧】龙珠史上最震撼的变身合集.mp4', type=str, help="input image file",)
    parser.add_argument("--gap_frame", type=int, default=60, help="sample gap of frame(EXTRACT_FREQUENCY)",)
    parser.add_argument("--save_size", type=tuple, default=(1280, 720), help="output image size (h, w)",)
    parser.add_argument("--save_folder", type=str, default='temp_debug/', help="output iamge folder",)
    parser.add_argument("--mode",  type=str, default="make", help="if you have finished make, choose 'show' to show")
    args = parser.parse_args()
    
    mkdir(args.save_folder)
    
    if args.mode == 'show':
        show()
    else:        
        if args.file.endswith(".mp4"):
            unlock_movie(args.file)
            show()
        else:
            img_origin = cv2.imread(args.file)
            args.save_size = (img_origin.shape[1], img_origin.shape[0])
            out = ascii_art(args.file)
            path = "{}_char.jpg".format(args.file.split(".jpg")[0])
            out.save(path)

        
