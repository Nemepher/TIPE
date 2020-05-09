import numpy as np
import cv2

img = cv2.imread("assets/testtree.png", cv2.IMREAD_GRAYSCALE)

#cv2 medianBlur
img_mblur = cv2.medianBlur(img, 5)

#cv2 threeshold

#ret,img_th = cv2.threshold(img_mblur,70,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)  #THRESH_BINARY_INV) #+ cv2.THRESH_OTSU)

ret,img_th = cv2.threshold(img_mblur,20,100,cv2.THRESH_TOZERO) #+ cv2.THRESH_OTSU)
cv2.imshow("e",img_th)
cv2.waitKey(0)
ret,img_th = cv2.threshold(img_mblur,70,255,cv2.THRESH_BINARY_INV) #+ cv2.THRESH_OTSU)

#cv2 errosion/dilatation

kernel = np.ones((8,8), np.uint8) 
img_erosion = cv2.erode(img_th, kernel, iterations=3)

kernel = np.ones((7,7),np.uint8)
img_dilate = cv2.dilate(img_erosion,kernel,iterations=1)



#kernel = np.ones((2,2), np.uint8) 
#img_dilate = cv2.erode(img_dilate, kernel, iterations=2)

#cv2 contours
contours, hierarchie= cv2.findContours(img_dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    M = cv2.moments(c)
    if M["m00"]!=0:
        cX = int(M["m10"]/M["m00"] )
        cY = int(M["m01"]/M["m00"] )	
        cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
print(len(contours))
#cv2 edges detection
img_edges = cv2.Canny(img_th,100,200)

#cv2 floodFill
img_floodfill = img.copy()
h,w = img.shape[:2]
mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(img_floodfill, mask, (50,50), 70)



for image in [img,img_mblur, img_th, img_erosion, img_dilate, img]:
    cv2.imshow("image",image)
    cv2.waitKey(0)
cv2.destroyAllWindows()


