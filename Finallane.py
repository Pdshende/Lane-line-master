import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
scaling_factorx=1
scaling_factory=1
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with
    #depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)

    draw_lines(line_img, lines)
    return line_img

def draw_lines(img, lines, color=[0, 255, 0], thickness=15):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            print(line)
def draw_lines(img, lines, color=[0, 255, 0], thickness=15):
    """
    This function draws `lines` with `color` and `thickness`.
    """
    imshape = img.shape

    # these variables represent the y-axis coordinates to which
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]

    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []

    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)

            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]

    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)

    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)

    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global),
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global),
                      (lower_right_x, ymax_global), color, thickness)


# grayscale the image

def process_image(image):
  grayscaled = grayscale(image)
  #plt.imshow(grayscaled, cmap='gray')
  # apply gaussian blur
  kernelSize = 5
  gaussianBlur = gaussian_blur(grayscaled, kernelSize)
  # cannyscaling_factorx=0.4

  minThreshold = 10
  maxThreshold = 90
  edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)
  #apply mask
  lowerLeftPoint = [50, 800]
  upperLeftPoint = [470, 430]
  upperRightPoint = [900, 430]
  lowerRightPoint = [1300, 750]

  pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint,
  lowerRightPoint]], dtype=np.int32)
  masked_image = region_of_interest(edgeDetectedImage, pts)
  #hough lines
  rho = 1
  theta = np.pi/180
  threshold = 30
  min_line_len = 20
  max_line_gap = 20

  houged = hough_lines(masked_image, rho, theta,
                  threshold, min_line_len, max_line_gap)
  combo_image = cv2.addWeighted(image, 1 , houged, 1,1)

  combo = cv2.add(image,combo_image)
  return combo
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _ , frame = cap.read()
    #frame1=cv2.resize(frame,None,fx=scaling_factorx,fy=scaling_factory,interpolation=cv2.INTER_AREA)
    #image = cv2.imread('test.jpg')
    p = process_image(frame)
    frame75 = rescale_frame(p, percent=75)

    cv2.imshow("re",frame75)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
     # The following frees up resources and closes all windows

cap.release()
cv2.destroyAllWindows()
