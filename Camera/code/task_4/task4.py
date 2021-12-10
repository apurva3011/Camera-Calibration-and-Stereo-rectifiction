import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt 

f = cv.FileStorage('.../output/task_1/Left_intrinsics.xml',cv.FileStorage_READ)
mtxL = f.getNode("LeftIntrinsics").mat()
distL = f.getNode("DistortionCoefficients").mat()
f.release()

f = cv.FileStorage('.../output/task_1/Right_intrinsics.xml',cv.FileStorage_READ)
mtxR = f.getNode("RightIntrinsics").mat()
distR = f.getNode("RightDistortionCoefficients").mat()
f.release()

f = cv.FileStorage('.../output/task_2/1stereoCalibration_parameters.xml',cv.FileStorage_READ)
mL = f.getNode("LeftIntrinsics").mat()
dL = f.getNode("LeftDistortionCoefficients").mat()
f.release()

f = cv.FileStorage('.../output/task_2/1stereoCalibration_parameters.xml',cv.FileStorage_READ)
mR = f.getNode("RightIntrinsics").mat()
dR = f.getNode("RightDistortionCoefficients").mat()
f.release()

f = cv.FileStorage('.../output/task_2/1stereoRectification_parameters.xml',cv.FileStorage_READ)
R1 = f.getNode("R1").mat()
P1 = f.getNode("P1").mat()
R2 = f.getNode("R2").mat()
P2 = f.getNode("P2").mat()
Q = f.getNode("Q").mat()
f.release()

imgL = cv.imread('.../images/task_3_and_4/left_8.png')
imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
#print(imgL.shape[::-1])
mapLx, mapLy = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, imgL.shape[::-1], cv.CV_32FC1)
#print(mapLx)
#plt.imshow(mapLx)
imgL = cv.imread('.../images/task_3_and_4/left_8.png')
hL,  wL = imgL.shape[:2]
#print(h, w)

dstL = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)

#xL, yL, wL, hL = roiL
#dstL = dstL[yL:yL+hL, xL:xL+wL]
#cv.imwrite('leftrectify_depth.png', dstL)

cv.imshow('dstL',dstL)
cv.waitKey(0)
cv.destroyAllWindows()
print(dstL.shape[:2])

imgR = cv.imread('.../images/task_3_and_4/right_8.png')
imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
print(imgR.shape[::-1])
newcameramtxR, roiR = cv.getOptimalNewCameraMatrix(mR, dR, imgR.shape[::-1], 1, imgR.shape[::-1])
mapRx, mapRy = cv.initUndistortRectifyMap(mtxR, distR, R2, P2, imgR.shape[::-1], cv.CV_32FC1)
#print(mapLx)
#plt.imshow(mapLx)
imgR = cv.imread('.../images/task_3_and_4/right_8.png')
hR,  wR = imgR.shape[:2]
#print(h, w)

dstR = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)

#xR, yR, wR, hR = roiR
#dstR = dstR[yR:yR+hR, xR:xR+wR]
#cv.imwrite('rightrectify_depth.png', dstR)

cv.imshow('dstR',dstR)
cv.waitKey(0)
cv.destroyAllWindows()
print(dstR.shape[:2])

win_size = 5
min_disp = -1
max_disp = 63 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object. 
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
 numDisparities = num_disp,
 blockSize = 3,
 uniquenessRatio = 5,
 speckleWindowSize = 100,
 speckleRange = 2,
 disp12MaxDiff = 12,
 P1 = 8*3*win_size**2,
 P2 =32*3*win_size**2,
 mode=cv.STEREO_SGBM_MODE_HH) 
#Compute disparity map
print ("\nComputing the disparity  map...")
disparity_map = stereo.compute(dstL, dstR)
disparity_map = cv.normalize(src=disparity_map, dst=disparity_map, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
disparity_map=np.uint8(disparity_map)
#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()
cv.imwrite('disparity1.png',disparity_map)
#cv.imshow('disparity',disparity_map)
#cv.waitkey(0)
#cv.destroyAllWindows()

image=cv.reprojectImageTo3D(disparity_map,Q)
cv.imwrite('3dimage1.png',image)
