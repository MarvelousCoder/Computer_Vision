import cv2
import argparse
import numpy as np

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
    global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed

    if event == cv2.EVENT_LBUTTONDOWN:
        if cropping: # Segundo clique
            cropping = False
            refPt.append((x,y))
            cv2.line(param, refPt[0], refPt[1], (255,0,0), 5)
            print(np.sqrt((refPt[1][0] - refPt[0][0])**2 + (refPt[1][1] - refPt[0][1])**2))
        else:
            refPt = [(x, y)]
            cropping = True
        


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())
    
    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args["image"])
    clone = image.copy()
    cv2.namedWindow("image")

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        cv2.setMouseCallback("image", click_and_crop, image)
        key = cv2.waitKey(1) & 0xFF
    
        # if the 'r' key is pressed, reset the cropping region
        if  key == ord("r"):
            image = clone.copy()
    
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    
    # if there are two reference points, then crop the region of interest
    # from the image and display it
    # if len(refPt) == 2:
    #     roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    #     cv2.imshow("ROI", roi)
    #     cv2.waitKey(0)
    
    # close all open windows

    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.setMouseCallback("image", click_and_crop, frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()