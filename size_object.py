# Filename: init.py
# Usage: This script will measure different objects in the frame using a reference object of known dimension. 
# The object with known dimension must be the leftmost object.

from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

# Function to show array of images (intermediate results)
def show_images(images):
    for i, img in enumerate(images):
        cv2.imshow("image_" + str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image
img_path = "images/1234.jpeg"

# Read image and preprocess
image = cv2.imread(img_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)

edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as the leftmost contour is the reference object
(cnts, _) = contours.sort_contours(cnts)

# Remove contours which are not large enough
cnts = [x for x in cnts if cv2.contourArea(x) > 100]

# Reference object dimensions
# Here, for reference, I have used a 2cm x 2cm square
ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2.5
pixel_per_cm = dist_in_pixel/dist_in_cm

# Counter variable to keep track of the number of objects
object_count = 0
total_length = 0
total_width = 0

# Draw remaining contours and count objects
for cnt in cnts:
    box = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    box = perspective.order_points(box)
    (tl, tr, br, bl) = box

    # Check if contour area is large enough to be considered as an object
    if cv2.contourArea(cnt) > 100:
        # Increment the object count
        object_count += 1

        cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
        mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
        mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
        wid = euclidean(tl, tr) / pixel_per_cm
        ht = euclidean(tr, br) / pixel_per_cm
        total_length += wid
        total_width += ht
        cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

# Calculate average length and width
avg_length = total_length / object_count
avg_width = total_width / object_count

# Display the number of objects, average length, and average width on the image
cv2.putText(image, "Objects Count: {}".format(object_count), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(image, "Average Length: {:.2f}cm".format(avg_length), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(image, "Average Width: {:.2f}cm".format(avg_width), (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the final result
show_images([image])

# Log the results
print("Number of objects:", object_count)
print("Average length:", avg_length, "cm")
print("Average width:", avg_width, "cm")
