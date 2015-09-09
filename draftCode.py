import os
import numpy as np
import scipy
import scipy.signal as scisignal
import scipy.io as sio
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
import time
#import pandas
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
    plt.plot(x, y)
    
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

trialStartFrameIndex = findTrialIndex(rawMatFile['calibrationCounter'][:])
#print 'Trial Starting Frame = ', trialStartFrameIndex
#plot2D( range(len(tempVar)), tempVar, 'Frame Number', 'Calibration Counter', 'Sample Figure')

#plotPOR(rawMatFile, 'eyePOR_XY')
tempVar = rawMatFile['eyePOR_XY']
#plot2D( tempVar[2700:-1,0], tempVar[2700:-1,1], 'EYE POR X', 'EYE POR Y', 'eyePOR')
#plot2D(rawMatFile['calibrationPosition_XYZ'][0:2700, 0], rawMatFile['calibrationPosition_XYZ'][0:2700, 1], 'GazePoint X', 'GazePoint Y', 'Gaze Points')
#print createLine([1,1,1], [2,2,2])

calibrationPosition = np.array(rawMatFile['calibrationPosition_XYZ'][0:2700, :], dtype = float)
cyclopEyePosition = np.array(rawMatFile['view_XYZ_Pos'][0:2700, :], dtype = float)
lines = np.array(rawMatFile['eyeGazeDir_XYZ'][0:2700, :], dtype = float)

#plot3D(range(100), range(100), range(100), 'Sample Fig')

# Here I want to plot the gaze error for each calibration sphere.
# We should calculate the Intersection point of gaze vector and target plane
# Remember that I draw a plane containing the target point and prependicular to the eye-sphere vector
# So if I find the intersection point of this plane and the gaze vector that will give me a point to plot
# For detail calculation refer to the following link:
# https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

normal = -calibrationPosition
trialNumber = 0
frameIndex = range(trialStartFrameIndex[trialNumber], trialStartFrameIndex[trialNumber+27])
#print frameIndex
point_0 = cyclopEyePosition + calibrationPosition
s = point_0 - cyclopEyePosition
numerator = np.inner(s, (normal))
denumerator = np.inner(lines, (normal))
print numerator.shape, denumerator.shape
d = np.divide(numerator, denumerator)
print d.shape 
answer = []
for i in range(2700):
    answer.append(d[i,i])
answer=np.array(answer)
print answer.shape

finalPoints = np.empty([2700,3], dtype=float)
for i in range(2700):
    finalPoints[i,:] = answer[i]*(lines[i,:]) + cyclopEyePosition[i,:]
data = finalPoints
#plot3D(data[frameIndex,0], data[frameIndex,1], data[frameIndex,2], 'Calibration Points', 'r.')

#plot3D(calibrationPosition[frameIndex,0], calibrationPosition[frameIndex,2], calibrationPosition[frameIndex,1], 'Calibration Points', 'b.')

plotGazePoints(frameIndex, data, point_0, cyclopEyePosition, 'Gaze Points', 'b.')

eyePOR = np.array(rawMatFile['eyePOR_XY'][2700:-1,0], dtype = float)
timeStamp = np.array(rawMatFile['eyeTimeStamp'][2700:-1], dtype = float)
tempVarCopy = np.roll(eyePOR, -1, axis=0)
#tempVarCopy[-1,:] = [0.0,0.0] 
#print 'Copy', tempVarCopy[0:5,:]
#print 'original', tempVar[0:5,:]
diff = tempVarCopy - eyePOR
#diff = np.linalg.norm(diff, axis=0)
velocity = scisignal.medfilt(diff, kernel_size=5)
print timeStamp[0:10]
print len(tempVar), len(timeStamp)
#tempVar.vstack
#plot2D(timeStamp[0:-2], tempVar[0:-2], 'EYE POR X', 'EYE POR Y', 'eyePOR')
#plot2D(timeStamp[0:-2], tempVar[0:-2], 'EYE POR X', 'EYE POR Y', 'eyePOR')
print len(velocity[0:-2])
plotPORVelocity(velocity[0:-2], eyePOR[0:-2]/120, timeStamp[0:-2], 'EYE POR X', 'EYE POR Y', 'Lateral Velocity')

eyePOR = np.array(rawMatFile['eyePOR_XY'][2700:-1,1], dtype = float)
timeStamp = np.array(rawMatFile['eyeTimeStamp'][2700:-1], dtype = float)
tempVarCopy = np.roll(eyePOR, -1, axis=0)
#tempVarCopy[-1,:] = [0.0,0.0] 
#print 'Copy', tempVarCopy[0:5,:]
#print 'original', tempVar[0:5,:]
diff = tempVarCopy - eyePOR
#diff = np.linalg.norm(diff, axis=0)
velocity = scisignal.medfilt(diff, kernel_size=5)
print timeStamp[0:10]
print len(tempVar), len(timeStamp)
#tempVar.vstack
#plot2D(timeStamp[0:-2], tempVar[0:-2], 'EYE POR X', 'EYE POR Y', 'eyePOR')
#plot2D(timeStamp[0:-2], tempVar[0:-2], 'EYE POR X', 'EYE POR Y', 'eyePOR')
print len(velocity[0:-2])
plotPORVelocity(velocity[0:-2], eyePOR[0:-2]/100, timeStamp[0:-2], 'EYE POR X', 'EYE POR Y', 'Vertical Velocity')