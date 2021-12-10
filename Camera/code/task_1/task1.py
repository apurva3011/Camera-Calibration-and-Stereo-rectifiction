import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt 

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objpL = np.zeros((6*9,3), np.float32)
objpL[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpointsL = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.

imagesL = glob.glob('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_1/left_*.png')
for fname in imagesL:
    imgL = cv.imread(fname)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    retL, cornersL = cv.findChessboardCorners(grayL,(9,6), None)
    
    if retL == True:
        objpointsL.append(objpL)
        corners2L = cv.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        cv.drawChessboardCorners(imgL, (9,6), corners2L, retL)
        #cv.imshow('img', img)
        #cv.waitKey(0)
#cv.destroyAllWindows()
retL, mtxL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpointsL, imgpointsL, grayL.shape[::-1], None, None)

objpR = np.zeros((6*9,3), np.float32)
objpR[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpointsR = [] # 3d point in real world space
imgpointsR = [] # 2d points in image plane.

imagesR = glob.glob('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_1/right_*.png')
for fname in imagesR:
    imgR = cv.imread(fname)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    retR, cornersR = cv.findChessboardCorners(grayR,(9,6), None)
    
    if retR == True:
        objpointsR.append(objpR)
        corners2R = cv.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)
        cv.drawChessboardCorners(imgR, (9,6), corners2R, retR)
        #cv.imshow('img', img)
        #cv.waitKey(0)
#cv.destroyAllWindows()
retR, mtxR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpointsR, imgpointsR, grayR.shape[::-1], None, None)

imgL = cv.imread('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_1/left_2.png')
h,  w = imgL.shape[:2]
newcameramtxL, roiL = cv.getOptimalNewCameraMatrix(mtxL, distL, (w,h), 1, (w,h))
mapxL, mapyL = cv.initUndistortRectifyMap(mtxL, distL, None, newcameramtxL, (w,h), 5)
dstL = cv.remap(imgL, mapxL, mapyL, cv.INTER_LINEAR)
x, y, w, h = roiL
dstL = dstL[y:y+h, x:x+w]
cv.imwrite('calib1.png', dstL)
print(dstL.shape[:2])

imgR = cv.imread('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_1/right_2.png')
h,  w = imgR.shape[:2]
newcameramtxR, roiR = cv.getOptimalNewCameraMatrix(mtxR, distR, (w,h), 1, (w,h))
mapxR, mapyR = cv.initUndistortRectifyMap(mtxR, distR, None, newcameramtxR, (w,h), 5)
dstR = cv.remap(imgR, mapxR, mapyR, cv.INTER_LINEAR)
x, y, w, h = roiR
dstR = dstR[y:y+h, x:x+w]
cv.imwrite('calib2.png', dstR)
print(dstR.shape[:2])

f = cv.FileStorage("C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_1/Left_intrinsics.xml", cv.FileStorage_WRITE)
f.write("LeftIntrinsics",mtxL)
f.write('LeftDistortionCoefficients',distL)
f.release()

f = cv.FileStorage("C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_1/Right_intrinsics.xml", cv.FileStorage_WRITE)
f.write("RightIntrinsics",mtxL)
f.write('RightDistortionCoefficients',distL)
f.release()