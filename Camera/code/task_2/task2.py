import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.

imageL = glob.glob('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_2/left_0.png')
imageR = glob.glob('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_2/right_0.png')

for fname in imageL:
    imgL = cv.imread(fname)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    retL, cornersL = cv.findChessboardCorners(grayL,(9,6), None)

for fname in imageR:
    imgR = cv.imread(fname)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    retR, cornersR = cv.findChessboardCorners(grayR,(9,6), None)
    
objpoints.append(objp)
        
if retL == True:
            corners2L = cv.cornerSubPix(grayL,cornersL, (11,11), (-1,-1), criteria)
            imgpointsL.append(cornersL)
            cv.drawChessboardCorners(imgL, (9,6), corners2L, retL)
            #cv.imshow('Limg', imgL)
            #cv.waitKey(0)
#cv.destroyAllWindows()
            
if retR == True:
            corners2R = cv.cornerSubPix(grayR,cornersR, (11,11), (-1,-1), criteria)
            imgpointsR.append(cornersR)
            cv.drawChessboardCorners(imgR, (9,6), corners2R, retR)
            #cv.imshow('Rimg', imgR)
            #cv.waitKey(0)
#cv.destroyAllWindows()

f = cv.FileStorage('C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_1/Left_intrinsics.xml',cv.FileStorage_READ)
mtxL = f.getNode("LeftIntrinsics").mat()
distL = f.getNode("DistortionCoefficients").mat()
f.release()

f = cv.FileStorage('C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_1/Right_intrinsics.xml',cv.FileStorage_READ)
mtxR = f.getNode("RightIntrinsics").mat()
distR = f.getNode("RightDistortionCoefficients").mat()
f.release()


flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
stereo_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
ret,mL,dL,mR,dR,R,T,E,F = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, mtxL, distL, mtxR, distR, grayL.shape[::-1], criteria=stereo_criteria, flags=flags)

f = cv.FileStorage("C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_2/1stereoCalibration_parameters.xml", cv.FileStorage_WRITE)
f.write("LeftIntrinsics",mL)
f.write('LeftDistortionCoefficients',dL)
f.write("RightIntrinsics",mR)
f.write('RightDistortionCoefficients',dR)
f.write("R",R)
f.write("T",T)
f.write("E",E)
f.write("F",F)
f.release()

PL = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]])

PR = np.array([[9.9545, -5.7585, 9.5049, -3.5426],
               [6.0435, 9.9997, -2.7108, -7.2782],
               [-9.5031, 3.2729, 9.9546, -8.1241]])

RL = np.array([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(mL, dL, mR, dR, grayL.shape[::-1], R, T)

f = cv.FileStorage("C:/Users/Apurva/Documents/ras/cse598/project_2a/output/task_2/stereoRectification_parameters.xml", cv.FileStorage_WRITE)
f.write("R1",R1)
f.write('R2',R2)
f.write("P1",P1)
f.write('P2',P2)
f.write("Q",Q)
f.release()

mapLx, mapLy = cv.initUndistortRectifyMap(mtxL, distL, R1, P1, grayL.shape[::-1], cv.CV_32FC1)
#print(mapLx)
#plt.imshow(mapLx)
img = cv.imread('C:/Users/Apurva/Documents/ras/cse598/project_2a/images/task_2/left_1.png')
h,  w = img.shape[:2]
#print(h, w)

dstL = cv.remap(img, mapLx, mapLy, cv.INTER_LINEAR)
#x, y, w, h = roi
#dstL = dstL[y:y+h, x:x+w]
cv.imwrite('1rectify.png', dstL)