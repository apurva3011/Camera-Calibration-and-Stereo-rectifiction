import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt 
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float


f = cv.FileStorage('.../output/task_2/1stereoRectification_parameters.xml',cv.FileStorage_READ)
R1 = f.getNode("R1").mat()
P1 = f.getNode("P1").mat()
R2 = f.getNode("R2").mat()
P2 = f.getNode("P2").mat()
Q = f.getNode("Q").mat()
f.release()

f = cv.FileStorage('.../output/task_1/Left_intrinsics.xml',cv.FileStorage_READ)
mtxL = f.getNode("LeftIntrinsics").mat()
distL = f.getNode("DistortionCoefficients").mat()
f.release()

f = cv.FileStorage('.../output/task_1/Right_intrinsics.xml',cv.FileStorage_READ)
mtxR = f.getNode("RightIntrinsics").mat()
distR = f.getNode("DistortionCoefficients").mat()
f.release()


imgL = cv.imread('.../images/task_3_and_4/left_0.png')
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
h,  w = imgL.shape[:2]
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 1, (w,h))
mapxL, mapyL = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, (w,h), 5)
Ldst = cv.remap(imgL, mapxL, mapyL, cv.INTER_LINEAR)
#x, y, w, h = roi
#Ldst = Ldst[y:y+h, x:x+w]
cv.imwrite('undistL.png', Ldst)
print(Ldst.shape[:2])
#cv.imshow('undistL',Ldst)
#cv.waitKey(0)
#cv.destroyAllWindows()

imgR = cv.imread('.../images/task_3_and_4/right_0.png')
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
h,  w = imgR.shape[:2]
#print(h,w)
#newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtxR, distR, (w,h), 1, (w,h))
mapxR, mapyR = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, (w,h), 5)
Rdst = cv.remap(imgR, mapxR, mapyR, cv.INTER_LINEAR)
#x, y, w, h = roi
#Rdst = Rdst[y:y+h, x:x+w]
cv.imwrite('undistR.png', Rdst)
print(Rdst.shape[:2])
#cv.imshow('undistR',Rdst)
#cv.waitKey(0)
#cv.destroyAllWindows()

orbL = cv.ORB_create()
# find the keypoints with ORB
kpL = orbL.detect(Ldst,None)
# compute the descriptors with ORB
kpL, desL = orbL.compute(Ldst, kpL)
# draw only keypoints location,not size and orientation
orbfpL = cv.drawKeypoints(Ldst, kpL, None, color=(0,255,0), flags=0)
#plt.imshow(orbfpL), plt.show()
cv.imwrite('featurepointsL.png',orbfpL)
h, w = orbfpL.shape[:2]
print(orbfpL.shape[:2])

for p in kpL:
    px=np.array(p.pt)
    for q in kpL:
        qx=np.array(q.pt)
        r = np.linalg.norm(px-qx)
        if r<6 and p.response>=q.response:
            kpL.remove(q)
kpL, desL = orbL.compute(Ldst, kpL)            
orbfpL1 = cv.drawKeypoints(Ldst, kpL, None, color=(0,255,0), flags=0)
cv.imwrite('featurepointsL1.png',orbfpL1)

print(orbfpL1.shape[:2])


orbR = cv.ORB_create()
# find the keypoints with ORB
kpR = orbR.detect(Rdst,None)
# compute the descriptors with ORB
kpR, desR = orbR.compute(Rdst, kpR)
# draw only keypoints location,not size and orientation
orbfpR = cv.drawKeypoints(Rdst, kpR, None, color=(0,255,0), flags=0)
#plt.imshow(orbfpR), plt.show()
cv.imwrite('featurepointsR.png',orbfpR)
print(orbfpR.shape[:2])



for p in kpR:
    px=np.array(p.pt)
    qx=np.array(q.pt)
    r = np.linalg.norm(px-qx)
    if r<6 and p.response>=q.response:
        kpR.remove(q)
            
orbfpR1 = cv.drawKeypoints(Rdst, kpR, None, color=(0,255,0), flags=0)
cv.imwrite('featurepointsR1.png',orbfpR1)
print(orbfpR1.shape[:2])
kpR, desR = orbR.compute(Rdst, kpR)
          
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(desL,desR)
matches = sorted(matches, key = lambda x:x.distance)
img = cv.drawMatches(Ldst,kpL,Rdst,kpR,matches[:10],None, flags=2)

plt.imshow(img),plt.show()
cv.imwrite('img.png',img)
