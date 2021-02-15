import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image_folder',
    type=str,
    help='',
)
parser.add_argument(
    '--video_name',
    type=str,
    help='',
)
parser.add_argument(
    '--rate',
    type=int,
    default=16,
    help='',
)
args = parser.parse_args()

image_folder = args.image_folder
video_name = f'{args.video_name}.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = [int(x[:-4]) for x in images]
images.sort()
images = [f"{x}.png" for x in images]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, args.rate, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
