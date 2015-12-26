import os
import numpy as np
import scipy
import scipy.signal as scisignal
import scipy.io as sio
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
import time
import pandas
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

def plotPORVelocity(velocity, POR, time, xtitle, ytitle, title):
    plt.figure()
    plt.plot(time, velocity)
    plt.hold(True)    
    plt.plot(time, POR, 'r.')
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def plotGazePoints(idx, gazePoints, calibrationPoints, viewPoint, title, myMarker):
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(calibrationPoints[idx,0], calibrationPoints[idx,2], calibrationPoints[idx,1], 'r.')
    ax.hold(True)
    ax.plot(gazePoints[idx,0], gazePoints[idx,2], gazePoints[idx,1], myMarker,label=title)
    ax.hold(True)
    ax.plot(viewPoint[idx,0], viewPoint[idx,2], viewPoint[idx,1], 'D',label=title)
    ax.legend()
    plt.grid(True)
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    plt.show()

def plot3D(x, y, z, title, myMarker):
    mpl.rcParams['legend.fontsize'] = 10
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, y, z, myMarker,label=title)
    ax.legend()
    plt.grid(True)
    ax.set_xlabel('X(m)')
    ax.set_ylabel('Y(m)')
    ax.set_zlabel('Z(m)')
    plt.show()

def plot2D(x, y, xtitle, ytitle, title):

    plt.figure()
    plt.plot(x, y, 'b.')
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.grid(True)
    plt.show()

def isInList(needle, haystack):
  try:
    return haystack.index(needle)
  except :
    return None
    
def findTrialIndex(dataFlag):
    indexList = []
    for x in range(28):
        indexList.append(x*100)
    # TODO: This should be fixed: Due to a data type problem data.index didn't work for ndarray
    
    #print type(dataFlag)
    #for counter in range(27):
    #    indexList.append(isInList(counter, dataFlag)) #dataFlag.index(counter))
    return indexList

def plotPOR(rawMatFile, eyeString):
    tempVar = rawMatFile[eyeString]
#    plot2D( tempVar[0:2700,0], tempVar[0:2700,1], 'EYE POR X', 'EYE POR Y', eyeString)
    x = tempVar[0:2700,0]
    y = tempVar[0:2700,1]
    xtitle = 'EYE POR X'
    ytitle = 'EYE POR Y'
    
    plt.figure()
    plt.plot(x, y,'b.')
    
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.grid(True)
    plt.show()
    
def createLine(point1, point2):
    vector = np.subtract(point2, point1)
    vector = np.divide(vector, np.linalg.norm(vector))
    return [vector, point1]

    
numberOfCalibrationPoints = 27
rawMatFileName = 'rawMat_exp_data-2015-8-18-18-34.mat'

dataPath = os.getcwd()
print 'Current Work Directory ==>', dataPath
os.chdir(dataPath  + '/MatFile')
print 'Changed cwd to ==> ', os.getcwd()
rawMatFile =  sio.loadmat(rawMatFileName)
os.chdir(dataPath)
print 'Returned cwd to ==> ', os.getcwd()

#trialStartFrameIndex = findTrialIndex(rawMatFile['calibrationCounter'][:])
#print 'Trial Starting Frame = ', trialStartFrameIndex
tempVar = rawMatFile['eyePOR_XY']

indexRange = range(0,2700)

calibrationPosition = np.array(rawMatFile['calibrationPosition_XYZ'][indexRange, :], dtype = float)
cyclopEyePosition = np.array(rawMatFile['view_XYZ_Pos'][indexRange, :], dtype = float)
leftPixelatedPOR = np.array(rawMatFile['leftPOR_XY'][indexRange, :], dtype = float)
rightPixelatedPOR = np.array(rawMatFile['rightPOR_XY'][indexRange, :], dtype = float)

rightPupilPos_XYZ = np.array(rawMatFile['rightPupilPos_XYZ'], dtype = float)
leftPupilPos_XYZ = np.array(rawMatFile['leftPupilPos_XYZ'], dtype = float)
por_Z = np.zeros(len(indexRange))
por_Z = por_Z + 0.049 + cyclopEyePosition[:,2]
por_Z = np.array([por_Z])
#print pixelatedPOR, por_Z
leftPixelatedPOR  = np.concatenate((leftPixelatedPOR, por_Z.T), axis=1)
rightPixelatedPOR  = np.concatenate((rightPixelatedPOR, por_Z.T), axis=1)
print rightPixelatedPOR

#pixelatedPOR = np.hstack((pixelatedPOR, [por_Z]))

lines = np.array(rawMatFile['eyeGazeDir_XYZ'][indexRange, :], dtype = float)

# Here I want to plot the gaze error for each calibration sphere.
# We should calculate the Intersection point of gaze vector and target plane
# Remember that I draw a plane containing the target point and prependicular to the eye-sphere vector
# So if I find the intersection point of this plane and the gaze vector that will give me a point to plot
# For detail calculation refer to the following link:
# https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection


spherePositions = calibrationPosition + cyclopEyePosition


plot2D( range(100), calibrationPosition[0:100,2], 'X', 'Y', 'CalibPos')
