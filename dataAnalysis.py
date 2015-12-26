# -*- coding: utf-8-sig -*-
import os
import numpy as np
import scipy
import scipy.io as sio
import matplotlib
import matplotlib.pylab as plt
import matplotlib.animation as animation
import time

class sessionData():

	def __init__(self):
		
		self.textFileList = []
		self.medianFilterLength = 5
		self.dataStructure = {}

	def parseTextFile(self, textFileName):

		self.rawMatFileName = 'rawMat_' + textFileName  + '.mat'
		dataPath = os.getcwd()
		print 'Current Work Directory ==>', dataPath
		os.chdir(dataPath  + '/MatFile')
		print 'Changed cwd to ==> ', os.getcwd()
#		if ( os.path.isfile(dataPath + self.rawMatFileName)):#dataPath + '\\MatFile'  + '\\rawMat_' + textFileName +'.mat') ):
#			print '.mat File found ==> Skip Text Parsing'
#			return
		os.chdir(dataPath)
		print 'Returned cwd to ==> ', os.getcwd()

		os.chdir(dataPath  + '/RawTextFile')
		print 'Changed cwd to ==> ', os.getcwd()
		self.txtFile = open(textFileName  +'.txt',"r")
		os.chdir(dataPath)
		print 'Returned cwd to ==> ', os.getcwd()
		values = np.array([],dtype = float);
		i = 0
		
		X_Matrix = np.array([], dtype = float);
		Y_Matrix = np.array([], dtype = float);
		Z_Matrix = np.array([], dtype = float);
		C_Matrix = np.array([], dtype = float);
		F_Matrix = np.array([], dtype = float);
		inCalibrate = np.array([], dtype = float);
		EventFlag = np.array([], dtype = float);
		TrialType = np.array([], dtype = str);
		view_XYZ_Matrix = np.array([],  dtype = float);
		view_Quat_Matrix = np.array([], dtype = float);
		Paddle_XYZ_Matrix = np.array([],  dtype = float);
		Paddle_Quat_Matrix = np.array([], dtype = float);
		Ball_XYZ_Matrix = np.array([],  dtype = float);
		Ball_Vel_XYZ_Matrix = np.array([],  dtype = float);
		Inv_view_Matrix = np.array([],  dtype = float);
		Inv_Pro_Matrix = np.array([],  dtype = float);
		BallPix_XYDist = np.array([],  dtype = float);
		
		eyeTimeStamp = np.array([],  dtype = float);
		IOD = np.array([],  dtype = float);
		IPD = np.array([],  dtype = float);

		eyePOR_XY = np.array([],  dtype = float);
		eyeGazeDir_XYZ = np.array([],  dtype = float);
		eyeGazePoint_XYZ = np.array([],  dtype = float);

		rightPOR_XY = np.array([],  dtype = float);
		rightPupilRadius = np.array([],  dtype = float);
		rightEyeLensDistance = np.array([],  dtype = float);
		rightEyeScreenDistance = np.array([],  dtype = float);
		rightGazePoint_XYZ = np.array([],  dtype = float);
		rightGazeDir_XYZ = np.array([],  dtype = float);
		rightPupilPos_XYZ = np.array([],  dtype = float);
		
		leftPOR_XY = np.array([],  dtype = float);
		leftPupilRadius = np.array([],  dtype = float);
		leftEyeLensDistance = np.array([],  dtype = float);
		leftEyeScreenDistance = np.array([],  dtype = float);
		leftGazePoint_XYZ = np.array([],  dtype = float);
		leftGazeDir_XYZ = np.array([],  dtype = float);
		leftPupilPos_XYZ = np.array([],  dtype = float);        

                calibrationPosition_XYZ = np.array([], dtype = float)
                calibrationCounter = np.array([], dtype = float)
                calibrationInProgress = np.array([], dtype = float)
				
		ballInitialPos_XYZ = np.array([], dtype = float)
		ballFinalPos_XYZ = np.array([], dtype = float)
		initialVelocity_XYZ = np.array([], dtype = float)
		presentationDuration = np.array([], dtype = float)
		blankDuration = np.array([], dtype = float)
		postBlankDuration = np.array([], dtype = float)
		timeToContact = np.array([], dtype = float)
		beta = np.array([], dtype = float)
		theta = np.array([], dtype = float)
		
		eyePOR_XY.resize((1,2))
		eyeGazeDir_XYZ.resize((1,3))
		eyeGazePoint_XYZ.resize((1,3))
		
		rightPOR_XY.resize((1,2))
		rightGazeDir_XYZ.resize((1,3))
		rightGazePoint_XYZ.resize((1,3))
		rightPupilPos_XYZ.resize((1,3))

		leftPOR_XY.resize((1,2))
		leftGazeDir_XYZ.resize((1,3))
		leftGazePoint_XYZ.resize((1,3))
		leftPupilPos_XYZ.resize((1,3))                

                calibrationPosition_XYZ.resize((1,3))
                
		view_XYZ_Matrix.resize((1,3))
		view_Quat_Matrix.resize((1,4))
		Paddle_XYZ_Matrix.resize((1,3))
		Paddle_Quat_Matrix.resize((1,4))
		Ball_XYZ_Matrix.resize((1,3))
		Ball_Vel_XYZ_Matrix.resize((1,3))
		Inv_view_Matrix.resize((1,16))
		Inv_Pro_Matrix.resize((1,16))
		BallPix_XYDist.resize((1,3))

		ballInitialPos_XYZ.resize((1,3))
		ballFinalPos_XYZ.resize((1,3))
		initialVelocity_XYZ.resize((1,3))
		
		print 'Parsing TextData in progress for',textFileName,'.txt ....\n' 
		
		for aline in self.txtFile:
			Line = aline.split()
			for i in range(len(Line)):
				#print 'TextFile Parsing'
				if (Line[i] == 'frameTime'):
					#F_Matrix.append(float(Line[i+1]));
					F_Matrix = np.hstack((F_Matrix, Line[i+1]))
					#print 'F=\n', F_Matrix
				elif (Line[i] == 'inCalibrateBool'):
					inCalibrate = np.hstack((inCalibrate, Line[i+1]))
				elif (Line[i] == 'eventFlag'):
					EventFlag = np.hstack((EventFlag,Line[i+1]));
				elif (Line[i] == 'trialType'):
					TrialType = np.hstack((TrialType, Line[i+1]));
				elif (Line[i] == 'eyeTimeStamp'):
					eyeTimeStamp = np.hstack((eyeTimeStamp, Line[i+1]));
				elif (Line[i] == 'IOD'):
					IOD = np.hstack((IOD, Line[i+1]));
				elif (Line[i] == 'IPD'):
					IPD = np.hstack((IPD, Line[i+1]));
					
				elif (Line[i] == 'eyePOR_XY'):
					eyePOR_XY = np.vstack((eyePOR_XY, np.array([float(Line[i+1]), float(Line[i+2])])) );
				elif (Line[i] == 'eyeGazeDir_XYZ'):
					eyeGazeDir_XYZ = np.vstack((eyeGazeDir_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
				elif (Line[i] == 'eyeGazePoint_XYZ'):
					eyeGazePoint_XYZ = np.vstack((eyeGazePoint_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
					
				elif (Line[i] == 'rightPupilRadius'):
					rightPupilRadius = np.hstack((rightPupilRadius, Line[i+1]));
				elif (Line[i] == 'rightEyeScreenDistance'):
					rightEyeScreenDistance = np.hstack((rightEyeScreenDistance, Line[i+1]));
				elif (Line[i] == 'rightEyeLensDistance'):
					rightEyeLensDistance = np.hstack((rightEyeLensDistance, Line[i+1]));
				elif (Line[i] == 'rightPOR_XY'):
					rightPOR_XY = np.vstack((rightPOR_XY, np.array([float(Line[i+1]), float(Line[i+2])])) );				
				elif (Line[i] == 'rightGazeDir_XYZ'):
					rightGazeDir_XYZ = np.vstack((rightGazeDir_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
				elif (Line[i] == 'rightGazePoint_XYZ'):
					rightGazePoint_XYZ = np.vstack((rightGazePoint_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
				elif (Line[i] == 'rightPupilPos_XYZ'):
					rightPupilPos_XYZ = np.vstack((rightPupilPos_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));


				elif (Line[i] == 'leftPupilRadius'):
					leftPupilRadius = np.hstack((leftPupilRadius, Line[i+1]));
				elif (Line[i] == 'leftEyeScreenDistance'):
					leftEyeScreenDistance = np.hstack((leftEyeScreenDistance, Line[i+1]));
				elif (Line[i] == 'leftEyeLensDistance'):
					leftEyeLensDistance = np.hstack((leftEyeLensDistance, Line[i+1]));
				elif (Line[i] == 'leftPOR_XY'):
					leftPOR_XY = np.vstack((leftPOR_XY, np.array([float(Line[i+1]), float(Line[i+2])])) );				
				elif (Line[i] == 'leftGazeDir_XYZ'):
					leftGazeDir_XYZ = np.vstack((leftGazeDir_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
				elif (Line[i] == 'leftGazePoint_XYZ'):
					leftGazePoint_XYZ = np.vstack((leftGazePoint_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
				elif (Line[i] == 'leftPupilPos_XYZ'):
					leftPupilPos_XYZ = np.vstack((leftPupilPos_XYZ, np.array([float(Line[i+1]), float(Line[i+2]), float(Line[i+3])]) ));
                                
				elif (Line[i] == 'calibrationInProgress'):
				       calibrationInProgress = 	np.hstack((calibrationInProgress, Line[i+1]));
				elif (Line[i] == 'calibrationPosition'):
					calibrationPosition_XYZ = np.vstack((calibrationPosition_XYZ, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));
				elif (Line[i] == 'calibrationCounter'):
					calibrationCounter = np.hstack((calibrationCounter, Line[i+1]));
					
				elif (Line[i] == 'viewPos_XYZ'):
					view_XYZ_Matrix = np.vstack((view_XYZ_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));
				elif (Line[i] == 'viewQUAT_WXYZ'):
					view_Quat_Matrix = np.vstack((view_Quat_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3], Line[i+4]])))
				elif (Line[i] == 'paddlePos_XYZ'):
					Paddle_XYZ_Matrix = np.vstack((Paddle_XYZ_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));    
				elif (Line[i] == 'paddleQUAT_WXYZ'):
					Paddle_Quat_Matrix = np.vstack((Paddle_Quat_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3], Line[i+4]])))
				elif (Line[i] == 'ballPos_XYZ'):
					Ball_XYZ_Matrix = np.vstack((Ball_XYZ_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));
				elif (Line[i] == 'ballVel_XYZ'):
					Ball_Vel_XYZ_Matrix = np.vstack((Ball_Vel_XYZ_Matrix, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));
				elif (Line[i] == 'ballPix_XYDist'):
					BallPix_XYDist = np.vstack((BallPix_XYDist, np.array([Line[i+1], Line[i+2], Line[i+3]]) ));

				elif (Line[i] == 'ballInitialPos_XYZ'):
					ballInitialPos_XYZ = np.vstack((ballInitialPos_XYZ, np.array([Line[i+1], Line[i+2], Line[i+3]])));
				elif (Line[i] == 'ballFinalPos_XYZ'):
					ballFinalPos_XYZ = np.vstack((ballFinalPos_XYZ, np.array([Line[i+1], Line[i+2], Line[i+3]])));
				elif (Line[i] == 'initialVelocity_XYZ'):
					initialVelocity_XYZ = np.vstack((initialVelocity_XYZ, np.array([Line[i+1], Line[i+2], Line[i+3]])));
					
				elif (Line[i] == 'PD'):
					presentationDuration = np.hstack((presentationDuration, Line[i+1]));
				elif (Line[i] == 'BD'):
					blankDuration = np.hstack((blankDuration, Line[i+1]));
				elif (Line[i] == 'PBD'):
					postBlankDuration = np.hstack((postBlankDuration, Line[i+1]));
				elif (Line[i] == 'TTC'):
					timeToContact = np.hstack((timeToContact, Line[i+1]));
				elif (Line[i] == 'Beta'):
					beta = np.hstack((beta, Line[i+1]));
				elif (Line[i] == 'Theta'):
					theta = np.hstack((theta, Line[i+1]));
					

		self.txtFile.close()
		
		view_XYZ_Matrix = np.delete(view_XYZ_Matrix, 0, 0)
		view_Quat_Matrix = np.delete(view_Quat_Matrix, 0, 0)
		Paddle_XYZ_Matrix = np.delete(Paddle_XYZ_Matrix, 0, 0)
		Paddle_Quat_Matrix = np.delete(Paddle_Quat_Matrix, 0, 0)
		Ball_XYZ_Matrix = np.delete(Ball_XYZ_Matrix, 0, 0)
		Ball_Vel_XYZ_Matrix = np.delete(Ball_Vel_XYZ_Matrix, 0, 0)
		BallPix_XYDist = np.delete(BallPix_XYDist, 0, 0)
		ballInitialPos_XYZ = np.delete(ballInitialPos_XYZ, 0, 0)
		ballFinalPos_XYZ = np.delete(ballFinalPos_XYZ, 0, 0)
		
		eyePOR_XY = np.delete(eyePOR_XY, 0, 0)
		eyeGazeDir_XYZ = np.delete(eyeGazeDir_XYZ, 0, 0)
		eyeGazePoint_XYZ = np.delete(eyeGazePoint_XYZ, 0, 0)
		
		
		rightPOR_XY = np.delete(rightPOR_XY,0,0)
		rightGazePoint_XYZ = np.delete(rightGazePoint_XYZ,0,0)
		rightGazeDir_XYZ = np.delete(rightGazeDir_XYZ,0,0)
		rightPupilPos_XYZ= np.delete(rightPupilPos_XYZ,0,0)
		
		leftPOR_XY = np.delete(leftPOR_XY,0,0)
		leftGazePoint_XYZ = np.delete(leftGazePoint_XYZ,0,0)
		leftGazeDir_XYZ = np.delete(leftGazeDir_XYZ,0,0)
		leftPupilPos_XYZ= np.delete(leftPupilPos_XYZ,0,0)
		
		calibrationPosition_XYZ = np.delete(calibrationPosition_XYZ, 0, 0)
		#print 'Pos size=\n', XYZ_Matrix.shape
		#print 'F size=\n', F_Matrix.shape
	#	print 'E size=\n', eyePOR_XY.shape
	#	print 'T size=\n', eyeGazeDir_XYZ.shape
	#	print 'Q size=\n', eyeGazePoint_XYZ.shape

		print '...Experiment Text File Parsing Done!!'
		print 'Type List1 : ', type(F_Matrix), type(inCalibrate), type(TrialType), type(view_XYZ_Matrix), type(view_Quat_Matrix),
		type(Paddle_XYZ_Matrix), type(Paddle_Quat_Matrix), type(Ball_XYZ_Matrix), type(Ball_Vel_XYZ_Matrix)
		
#		print '\nType List2 : ', type(BallPix_XYDist), type(Inv_view_Matrix), type(Inv_Pro_Matrix)
#		print '\nType List3 : ', type(eyeTimeStamp), type(IOD), type(IPD), type(eyePOR_XY), type(eyeGazeDir_XYZ), type(eyeGazePoint_XYZ)
#		print '\nType List4 : ', type(rightEyeLensDistance), type(rightPupilRadius), type(rightPOR_XY), type(rightGazePoint_XYZ), type(rightGazeDir_XYZ), type(rightPupilPos_XYZ)
#		print '\nType List5 : ', type(leftEyeLensDistance), type(leftPupilRadius), type(leftPOR_XY), type(leftGazePoint_XYZ), type(leftGazeDir_XYZ), type(leftPupilPos_XYZ)
#		print '\nType List6 : ', type(ballFinalPos_XYZ), type(ballFinalPos_XYZ), type(initialVelocity_XYZ), type(presentationDuration), type(blankDuration), type(postBlankDuration),
#		type(timeToContact), type(beta), type(theta)
		
		
				
		self.rawDataStructure = {
		
		'FrameTime':F_Matrix,'inCalibrateBool':inCalibrate, 'calibrationInProgress':calibrationInProgress, 'EventFlag':EventFlag,'TrialType':TrialType, 
		'view_XYZ_Pos':view_XYZ_Matrix,'Quat_Matrix':view_Quat_Matrix, 'paddlePos_XYZ':Paddle_XYZ_Matrix,
		'paddleQUAT_WXYZ':Paddle_Quat_Matrix, 'ballPos_XYZ':Ball_XYZ_Matrix, 'ballVel_XYZ':Ball_Vel_XYZ_Matrix,
		'ballPix_XYDist': BallPix_XYDist, 'invViewMat': Inv_view_Matrix, 'invProMat':Inv_Pro_Matrix,

                'calibrationCounter' : calibrationCounter, 'calibrationPosition_XYZ' : calibrationPosition_XYZ,
                
		'eyeTimeStamp':eyeTimeStamp, 'IOD':IOD, 'IPD':IPD,
		
		'eyePOR_XY':(eyePOR_XY), 'eyeGazeDir_XYZ':eyeGazeDir_XYZ, 'eyeGazePoint_XYZ':(eyeGazePoint_XYZ),

		'rightEyeLensDistance':(rightEyeLensDistance), 'rightEyeScreenDistance' : (rightEyeScreenDistance), 'rightPupilRadius':(rightPupilRadius), 'rightPOR_XY':(rightPOR_XY),
		'rightGazePoint_XYZ':(rightGazePoint_XYZ), 'rightGazeDir_XYZ':(rightGazeDir_XYZ), 'rightPupilPos_XYZ':(rightPupilPos_XYZ),
		
		'leftEyeLensDistance':(leftEyeLensDistance), 'leftEyeScreenDistance' : (leftEyeScreenDistance), 'leftPupilRadius':(leftPupilRadius), 'leftPOR_XY':(leftPOR_XY),
		'leftGazePoint_XYZ':(leftGazePoint_XYZ), 'leftGazeDir_XYZ':(leftGazeDir_XYZ), 'leftPupilPos_XYZ':(leftPupilPos_XYZ),
		
		'ballInitialPos_XYZ': (ballInitialPos_XYZ), 'ballFinalPos_XYZ': (ballFinalPos_XYZ), 'initialVelocity_XYZ': (initialVelocity_XYZ),
		
		'PD': (presentationDuration), 'BD':(blankDuration), 'PD' : (postBlankDuration), 'TTC': (timeToContact), 'beta':(beta), 'theta':(theta)
		}

		os.chdir(dataPath  + '/MatFile')
		print 'Changed cwd to ==> ', os.getcwd()
		
		sio.savemat(self.rawMatFileName, self.rawDataStructure)
		os.chdir(dataPath)
		print 'Returned cwd to ==> ', os.getcwd()
		
		
		print self.rawMatFileName,' File Saved'

	def medianFilter(self):
		
		tempVar1 = self.scipy.signal.medfilt(self.rawMatFile['eyePOR_XY'], self.medianFilterLength)
		tempVar2 = self.scipy.signal.medfilt(self.rawMatFile['eyeGazePoint_XYZ'], self.medianFilterLength)
		tempVar3 = self.scipy.signal.medfilt(self.rawMatFile['rightPOR_XY'], self.medianFilterLength)
		tempVar4 = self.scipy.signal.medfilt(self.rawMatFile['rightGazePoint_XYZ'], self.medianFilterLength)
		tempVar5 = self.scipy.signal.medfilt(self.rawMatFile['leftPOR_XY'], self.medianFilterLength)
		tempVar6 = self.scipy.signal.medfilt(self.rawMatFile['leftGazePoint_XYZ'], self.medianFilterLength)
		
		

		self.dataStructure['processedData_tr'] = {'filteredEyePOR_XY' : tempVar1, 'filteredEyeGazePoint_XYZ' : tempVar2, 'filteredRightPOR_XY' : tempVar3,
		'filteredRightGazePoint_XYZ' : tempVar4, 'filteredLeftPOR_XY' : tempVar5, 'filteredLeftGazePoint_XYZ' : tempVar6}
		
	def organizeDataByTrial(self):
		#print 'Event Flag =', self.rawDataStructure['EventFlag']
		self.rawMatFile =  sio.loadmat(self.rawMatFileName)
		print 'Event Flag =', self.rawMatFile['EventFlag']
		sum = 0.0		
#		for i in range(len(self.rawMatFile['EventFlag'])):
#			sum = sum + self.rawMatFile['EventFlag'][i]
		print 'len', len(self.rawMatFile['EventFlag']), type(self.rawMatFile['EventFlag'][0]), sum
		return  
	def plot2D(self, x, y, figureTitle):
		return
	def plot3D(self, x, y, z, figureTitle):
		return
	def update_line1(self, num, sessionData, line):
		
		line.set_marker('p')
		line.set_data((sessionData['rawData_tr']['rightPupilPos_XYZ'][num,0], sessionData['rawData_tr']['rightPupilPos_XYZ'][num,1]))
		if num > 11445 :
			print 'F=', num
		return line,

	def update_line2(self, num, sessionData, line):
		
		line.set_marker('*')
		line.set_data((sessionData['rawData_tr']['leftPupilPos_XYZ'][num,0], sessionData['rawData_tr']['leftPupilPos_XYZ'][num,1]))
		if num > 11445 :
			print 'F=', num
		return line,


if __name__ == "__main__":

	
	dataPath = os.getcwd()
	frameNumber = 0
	textFileName = 'exp_data-2015-10-25-22-29'#'exp_data-2015-10-25-18-35'#'exp_data-2015-10-6-20-8'#'exp_data-2015-9-30-14-26'#'exp_data-2015-9-25-20-34'#'exp_data-2015-8-18-18-34'# 'exp_data-2015-4-24-13-34'
	mySessionData = sessionData()
	mySessionData.parseTextFile(textFileName)

	mySessionData.dataStructure = { 'rawData_tr' : None, 'processedData_tr' : None, 'expInfo' : None, 'dependentMeasures_tr': None }

	os.chdir(dataPath  + '/MatFile')
	print 'Changed cwd to ==> ', os.getcwd()
	mySessionData.dataStructure['rawData_tr'] = mySessionData.organizeDataByTrial()
	os.chdir(dataPath)
	print 'Returned cwd to ==> ', os.getcwd()
	
	
	#print 'Data =  ', mySessionData.dataStructure['rawData_tr']['rightPupilPos_XYZ'][1000:1020,2] #, sessionData['rawData_tr']['rightPupilPos_XYZ'][0][1], sessionData['rawData_tr']['rightPupilPos_XYZ'][0][2]
	

#	fig1 = plt.figure()
#	l1, = plt.plot([],[])
#	xMin = min(sessionData['rawData_tr']['rightPupilPos_XYZ'][:,0])
#	xMax = max(sessionData['rawData_tr']['rightPupilPos_XYZ'][:,0])
#	yMin = min(sessionData['rawData_tr']['rightPupilPos_XYZ'][:,1])
#	yMax = max(sessionData['rawData_tr']['rightPupilPos_XYZ'][:,1])
#	
#	plt.xlim(xMin, xMax)
#	plt.ylim(yMin, yMax)
#	plt.xlabel('X')
#	plt.title('Right PupilPos')
#	plt.grid(True)
#	line_ani = animation.FuncAnimation(fig1, update_line1, frames = 11448, fargs=(sessionData, l1), interval=14, blit=True)
#	plt.show()
	
	#print 'mat File Keys', matFile.keys()
	'''
	mat File Keys [

	'FrameTime', 'eyeTimeStamp', 'TrialType', 'invProMat', 'invViewMat', 'EventFlag', 'inCalibrateBool'
	'view_XYZ_Pos', 'Quat_Matrix', 'paddlePos_XYZ', 'paddleQUAT_WXYZ', 'ballPix_XYDist', 'ballPos_XYZ', 'ballVel_XYZ', 

	'IPD', 'IOD'
	'eyePOR_XY', 'eyeGazePoint_XYZ', 'eyeGazeDir_XYZ', 

	'rightPOR_XY', 'rightPupilRadius', 'rightPupilPos_XYZ', 'rightEyeLensDistance', 'rightGazePoint_XYZ', 'rightGazeDir_XYZ'
	'leftPOR_XY',  'leftPupilRadius', 'leftPupilPos_XYZ', 'leftEyeLensDistance', 'leftGazePoint_XYZ', 'leftGazeDir_XYZ',
	]
	'''
