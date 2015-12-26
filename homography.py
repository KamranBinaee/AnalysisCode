import numpy as np
import cv2
import os
import scipy.io as sio
import matplotlib
import matplotlib.pylab as plt
from sklearn.neighbors import NearestNeighbors

def pixelsToMetric(x_pixel, y_pixel):
    x_pixel = (0.126/1920)*np.subtract(x_pixel, 1920.0/2.0)
    y_pixel = (0.071/1080.0)*np.subtract(y_pixel, 1080/2.0)
    return x_pixel, y_pixel


def metricToPixels(x, y):
    x_pixel = (1920.0/0.126)*np.add(x, 0.126/2.0)
    y_pixel = (1080.0/0.071)*np.subtract(0.071/2.0, y)
    return x_pixel, y_pixel
def icp(a, b, init_pose=(0,0,0), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)

    #Initialise with the initial pose estimation
    Tr = np.array([[np.cos(init_pose[2]),-np.sin(init_pose[2]),init_pose[0]],
                   [np.sin(init_pose[2]), np.cos(init_pose[2]),init_pose[1]],
                   [0,                    0,                   1          ]])

    src = cv2.transform(src, Tr[0:2])

    for i in range(no_iterations):
        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst[0]) # ,warn_on_equidistant=False
        distances, indices = nbrs.kneighbors(src[0])

        #Compute the transformation between the current source
        #and destination cloudpoint
        T = cv2.estimateRigidTransform(src, dst[0, indices.T], False)
        #Transform the previous source and update the
        #current source cloudpoint
        src = cv2.transform(src, T)
        #Save the transformation from the actual source cloudpoint
        #to the destination
        Tr = np.dot(Tr, np.vstack((T,[0,0,1])))
    return Tr[0:2]


def findResidualError(projectedPoints, referrencePoints):
    e2 = np.zeros((projectedPoints.shape[0],2))
    for i in range(projectedPoints.shape[0]):
        temp = np.subtract(projectedPoints[i], referrencePoints[i])
        #print 'temp', temp
        e2[i,:] = np.power(temp[0:2], 2)
    return [np.sqrt(sum(sum(e2[:])))]

def calibrateData(cyclopeanPOR_XY, truePOR_XY, method = cv2.LMEDS, threshold = 10):

    global startFrame, endFrame
    result = cv2.findHomography(cyclopeanPOR_XY, truePOR_XY, method , ransacReprojThreshold = threshold)
    #print result[0]
    #print 'size', len(result[1]),'H=', result[1]
    totalFrameNumber = truePOR_XY.shape[0]
    arrayOfOnes = np.ones((totalFrameNumber,1), dtype = float)

    homogrophy = result[0]
    print 'H=', homogrophy, '\n'
    #print 'Res', result[1]
    cyclopeanPOR_XY = np.hstack((cyclopeanPOR_XY, arrayOfOnes))
    truePOR_XY = np.hstack((truePOR_XY, arrayOfOnes))
    projectedPOR_XY = np.zeros((totalFrameNumber,3))
    
    for i in range(totalFrameNumber):
        projectedPOR_XY[i,:] = np.dot(homogrophy, cyclopeanPOR_XY[i,:])
        #print projectedPOR_XY[i,:]
    
    #projectedPOR_XY
    projectedPOR_XY[:, 0], projectedPOR_XY[:, 1] = metricToPixels(projectedPOR_XY[:, 0], projectedPOR_XY[:, 1])
    cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1] = metricToPixels(cyclopeanPOR_XY[:, 0], cyclopeanPOR_XY[:, 1])
    truePOR_XY[:, 0], truePOR_XY[:, 1] = metricToPixels(truePOR_XY[:, 0], truePOR_XY[:, 1])
    data = projectedPOR_XY
    frameCount = range(myRange)
    xmin = 550#min(cyclopeanPOR_XY[frameCount,0])
    xmax = 1350#max(cyclopeanPOR_XY[frameCount,0])
    ymin = 250#min(cyclopeanPOR_XY[frameCount,1])
    ymax = 800#max(cyclopeanPOR_XY[frameCount,1])
    #print xmin, xmax, ymin, ymax
    fig1 = plt.figure()
    plt.plot(data[frameCount,0], data[frameCount,1], 'bx', label='Calibrated POR')
    plt.plot(cyclopeanPOR_XY[frameCount,0], cyclopeanPOR_XY[frameCount,1], 'g2', label='Uncalibrated POR')
    plt.plot(truePOR_XY[frameCount,0], truePOR_XY[frameCount,1], 'r8', label='Ground Truth POR')
    #l1, = plt.plot([],[])
    
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('X')
    plt.ylabel('Y')
    if ( method == cv2.RANSAC):
        methodTitle = ' RANSAC '
    elif( method == cv2.LMEDS ):
        methodTitle = ' Least Median '
    elif( method == 0 ):
        methodTitle = ' Homography '
    plt.title('Calibration Result using'+ methodTitle+'\nWith System Calibration for ['+str(startFrame) +' '+ str(endFrame)+'] Frames ')
    plt.grid(True)
    #plt.axis('equal')
    #line_ani = animation.FuncAnimation(fig1, update_line1, frames = 11448, fargs=(sessionData, l1), interval=14, blit=True)
    legend = plt.legend(loc=[0.6,0.9], shadow=True, fontsize='small')# 'upper center'
    plt.show()
    print 'MSE_after = ', findResidualError(projectedPOR_XY, truePOR_XY)
    print 'MSE_before = ', findResidualError(cyclopeanPOR_XY, truePOR_XY)


matFilename = 'preProcessedDataexp_data-2015-10-6-20-8'#'preProcessedDataexp_data-2015-9-30-14-26'#'preProcessedDataexp_data-2015-9-25-20-34'#'preProcessedDataexp_data-2015-8-18-18-34'
undistortedFilename = 'preProcessedDataUDexp_data-2015-10-6-20-8'
datapath = os.getcwd()
#os.chdir(dataPath  + '\\MatFile')
#print 'Changed cwd to ==> ', os.getcwd()
rawMatFile = sio.loadmat(matFilename);
undistortedMatFile = sio.loadmat(undistortedFilename);
#os.chdir(dataPath)
#print 'Returned cwd to ==> ', os.getcwd()

framesPerPoint = range(100)

startFrame = 0
endFrame = 2700


#startFrame = 2700
#endFrame = 5400

#startFrame = 8507
#endFrame = 10707

#startFrame = 13550
#endFrame = 16250

#startFrame = 20533
#endFrame = 23233

#startFrame = 26610
#endFrame = 29310


frameIndexRange = range(startFrame, endFrame)
numberOfPointsPerPlane = range(9)
#frontPlaneIndex = 3*numberOfPointsPerPlane
#middlePlaneIndex = 3*numberOfPointsPerPlane +1 
#farPlaneIndex = 3*numberOfPointsPerPlane +2
global startFrame, endFrame
myRange = endFrame - startFrame
#eventFlag
#calibrationStatus = np.array(map(float,rawMatFile['calibrationStatus'][:]), dtype= float)
gx = map(float, rawMatFile['truePOR_XY'][0, frameIndexRange])
gy = map(float, rawMatFile['truePOR_XY'][1, frameIndexRange])
#gx, gy = metricToPixels(gx,gy)
truePOR_XY = np.array([gx, gy], dtype = float)
truePOR_XY = truePOR_XY.T
#print arrayOfOnes.shape, truePOR_XY.shape
#truePOR_XY = np.hstack((truePOR_XY, arrayOfOnes))
#print truePOR_XY.shape

rgx = map(float, rawMatFile['rightEyePOR_XY'][0, frameIndexRange])
rgy = map(float, rawMatFile['rightEyePOR_XY'][1, frameIndexRange])
#rgx, rgy = metricToPixels(rgx,rgy)
rightPOR_XY = np.array([rgx, rgy], dtype = float)
rightPOR_XY = rightPOR_XY.T
#rightPOR_XY = np.hstack((rightPOR_XY,arrayOfOnes))

lgx = map(float, rawMatFile['leftEyePOR_XY'][0, frameIndexRange])
lgy = map(float, rawMatFile['leftEyePOR_XY'][1, frameIndexRange])
#lgx, lgy = metricToPixels(lgx,lgy)
leftPOR_XY = np.array([lgx, lgy], dtype = float)
leftPOR_XY = leftPOR_XY.T
#leftPOR_XY = np.hstack((leftPOR_XY,arrayOfOnes))

cgx = map(float, rawMatFile['cyclopeanPOR_XY'][0, frameIndexRange])
cgy = map(float, rawMatFile['cyclopeanPOR_XY'][1, frameIndexRange])
#cgx, cgy = metricToPixels(cgx,cgy)
cyclopeanPOR_XY = np.array([cgx, cgy], dtype = float)
cyclopeanPOR_XY = cyclopeanPOR_XY.T
#cyclopeanPOR_XY = np.hstack((cyclopeanPOR_XY, arrayOfOnes))
#print cyclopeanPOR_XY.shape

#cgx = map(float, undistortedMatFile['unDistortedEyePOR_XY'].T[0, frameIndexRange])
#cgy = map(float, undistortedMatFile['unDistortedEyePOR_XY'].T[1, frameIndexRange])
##cgx, cgy = metricToPixels(cgx,cgy)
#unDistortedCyclopeanPOR_XY = np.array([cgx, cgy], dtype = float)
#unDistortedCyclopeanPOR_XY = unDistortedCyclopeanPOR_XY.T
#unDistortedCyclopeanPOR_XY[:, 0], unDistortedCyclopeanPOR_XY[:, 1] = pixelsToMetric(unDistortedCyclopeanPOR_XY[:, 0], unDistortedCyclopeanPOR_XY[:, 1])

calibrateData(cyclopeanPOR_XY, truePOR_XY, cv2.RANSAC, 50)
calibrateData(cyclopeanPOR_XY, truePOR_XY, cv2.LMEDS, 10)

#calibrateData(unDistortedCyclopeanPOR_XY, truePOR_XY, cv2.RANSAC, 50)
#calibrateData(unDistortedCyclopeanPOR_XY, truePOR_XY, cv2.LMEDS, 10)

# ICP Method as another matching algorithm
#Create the datasets
#ang = np.linspace(-np.pi/2, np.pi/2, 320)
#a = np.array([ang, np.sin(ang)])
#th = np.pi/2
#rot = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
#b = np.dot(rot, a) + np.array([[0.2], [0.3]])
#
##Run the icp
##M2 = icp(a, b, [0.1,  0.33, np.pi/2.2], 30)
#M2 = icp( cyclopeanPOR_XY.T, truePOR_XY.T, [0.1,  0.33, np.pi/2.2], 30)
#print 'M = ', M2
#
##Plot the result
#src = np.array([cyclopeanPOR_XY.T]).astype(np.float32)
#res = cv2.transform(src, M2)
#plt.figure()
#plt.plot(b[0],b[1])
#plt.plot(res[0].T[0], res[0].T[1], 'r.')
#plt.plot(a[0], a[1])
#plt.grid(True)
#plt.show()