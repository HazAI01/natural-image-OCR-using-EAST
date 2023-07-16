import pytesseract
import cv2
import numpy as np
from utils import forward_passer, box_extractor
from imutils.object_detection import non_max_suppression

def resize_image(image, new_width, new_height):
    #Re-sizes image to given width & height
    #return: modified image, ratio of new & old height and width
    h, w = image.shape[:2]

    ratio_w = w / new_width
    ratio_h = h / new_height

    image = cv2.resize(image, (new_width, new_height))

    return image, ratio_w, ratio_h



def main():
    image_path = "image.jpg"
    resized_width = 320
    resized_height = 320
    padding = 0.0
    min_confidence = 0.5
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    #reading image
    image = cv2.imread(image_path)
    orig_image = image.copy()
    orig_h, orig_w = orig_image.shape[:2]

    #resizing image
    image, ratio_w, ratio_h = resize_image(image, resized_width, resized_height)

    #layers used for ROI recognition
    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']
    
    #getting results from the model
    scores, geometry = forward_passer(net, image, layers=layer_names)

    #decoding results from the model
    rectangles, confidences = box_extractor(scores, geometry, min_confidence)

    #applying non-max suppression to get boxes depicting text regions
    boxes = non_max_suppression(np.array(rectangles), probs=confidences)

    results = []

    #text recognition main loop
    for (start_x, start_y, end_x, end_y) in boxes:
        start_x = int(start_x * ratio_w)
        start_y = int(start_y * ratio_h)
        end_x = int(end_x * ratio_w)
        end_y = int(end_y * ratio_h)

        dx = int((end_x - start_x) * padding)
        dy = int((end_y - start_y) * padding)

        start_x = max(0, start_x - dx)
        start_y = max(0, start_y - dy)
        end_x = min(orig_w, end_x + (dx*2))
        end_y = min(orig_h, end_y + (dy*2))

        #ROI to be recognized
        roi = orig_image[start_y:end_y, start_x:end_x]

        #recognizing text
        config = '-l eng --oem 1 --psm 7'
        text = pytesseract.image_to_string(roi, config=config)

        #collating results
        results.append(((start_x, start_y, end_x, end_y), text))
    
    #sorting results top to bottom
    results.sort(key=lambda r: r[0][1])

    #creating and flushing a text file
    file = open("recognized_text.txt", "w+")
    file.write("")
    file.close()

    #writing extracted text into text file
    for (start_x, start_y, end_x, end_y), text in results:
        text = ''.join([c if ord(c) < 128 else "" for c in text]).strip()
        file = open("recognized_text.txt", "a")
        file.write(text)
        file.write("\n")
        file.close()



if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    main()
