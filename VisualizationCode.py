# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from PIL import Image as im
import os
import scipy.io as sio
import viz
import vizact
import vizshape
#import vizmatplottwo
import scipy
import vizmenu
import time
import math

#from gazeTools import gazeSphere
#from gazeTools import gazeVector

    

class visualization():

    """
    Visualization Class creates 3D or 2D representation of the Experiment Data
    """

    def __init__(self, textFileName):
        
        self.dataPath = os.getcwd()
        self.textFileName = textFileName
        self.frameNumber = 0
        self.roomLength = 70
        self.roomWidth = 40
        self.roomHeight = 20
        self.viewPointCounter = 0


    def setLighting(self):
        
        viz.MainView.getHeadLight().disable()
        #viz.MainView.get

        self.topLight = viz.addLight() 
        self.topLight.setPosition(-4,6,10)
        self.topLight.setEuler( 135, 45 ,0 )
        self.topLight.spread(350) 
        self.topLight.intensity(2) 

        self.bottomLight = viz.addLight() 
        self.bottomLight.setPosition(4,0,-10)
        self.bottomLight.setEuler( -45, -45 ,0 )
        self.bottomLight.spread(350) 
        self.bottomLight.intensity(2) 

    
    def createTheRoom(self):

        """<START OF DEFINING THE WORLD>"""
        self.setLighting()


        #creates the floor plane to be 35x80
        #rotated around the Z axis(perpendicular to Y axis)
        self.floor = vizshape.addPlane(size=(self.roomWidth, self.roomLength), axis=vizshape.AXIS_Y, cullFace=False)
        #makes the floor look like wood
        self.floor.texture(viz.addTexture('images/tile_wood.jpg'))
        #moves the floor +20 meters in the z direction
        self.floor.setPosition(0,0,0)
        #floor.alpha(0.8)

        #adds a ceiling(plane) that when facing the frontWall is to the camera's topside
        #wall is 35x80 meters**2
        #wall is rotated about the Z axis(perpendicular to Y axis)
        self.ceiling = vizshape.addPlane(size=(self.roomWidth, self.roomLength), axis=-vizshape.AXIS_Y, cullFace=False)
        #makes the ceiling appear Skyblue in color
        self.ceiling.color(viz.CYAN)
        #shifts the ceiling to rest on top of the four walls
        self.ceiling.setPosition(0,self.roomHeight,0)


        #If performance is an issue, set cullFace to True.

        #adds the wall(plane) farthest from the hand/person
        #35x10 meters**2
        #rotated around the X axis (perpendicular to Z axis)
        self.frontWall = vizshape.addPlane(size=(self.roomWidth, self.roomHeight), axis=-vizshape.AXIS_Z, cullFace = False)
        #moves the wall to match the edge of the 
        self.frontWall.setPosition(0,self.roomHeight/2,self.roomLength/2)
        #makes the wall appear white
        self.frontWall.color([0.6,0.6,0.6])
        self.wall_text_object = viz.addText3D( 'PerForM Lab',parent=self.frontWall,pos=[-3,1,-0.2])

        #adds a wall(plane) that when facing the frontWall is to the camera's back
        #wall is 35x10 meters**2
        #wall is rotated about the X axis(perpendicular to Z axis)
        self.backWall = vizshape.addPlane(size=(self.roomWidth, self.roomHeight), axis=vizshape.AXIS_Z, cullFace = False)
        #shifts the wall to match the edge of the floor
        self.backWall.setPosition(0,self.roomHeight/2,-self.roomLength/2)
        #makes the wall appear white
        self.backWall.color(viz.WHITE)

        #adds the wall(plane) that when facing the frontWall, is to the camera's left
        #wall is 80x10 meters**2
        #wall is rotated about the Y axis(perpendicular to X axis)
        self.leftWall = vizshape.addPlane(size=(self.roomLength, self.roomHeight), axis=-vizshape.AXIS_X, cullFace = False)
        #shifts the wall to match the edge of the floor
        self.leftWall.setPosition(self.roomWidth/2,self.roomHeight/2,0)
        #makes the wall appear white
        self.leftWall.color(viz.GRAY)

        #adds a wall(plane) that when facing the frontWall is to the camera's right
        #wall is 80x10 meters**2
        #wall is rotated about the Y axis(perpendicular to X axis)
        self.rightWall = vizshape.addPlane(size=(self.roomLength, self.roomHeight), axis=vizshape.AXIS_X, cullFace = False)
        #shifts the wall to match the edge of the floor
        self.rightWall.setPosition(-self.roomWidth/2,self.roomHeight/2,0)
        #makes the wall appear white
        self.rightWall.color(viz.GRAY)


        #add a meter marker at 0 meters along the z axis of the room
        #meter0 = vizshape.addPlane(size = (5,.3), axis = vizshape.AXIS_Y, cullFace = False)
        #meter0.setPosition(0,.3, 0)
        #makes the meter marker appear yellow
        #meter0.color(viz.WHITE)

        #adds a wall(plane) that when facing the frontWall is to the camera's back
        #wall is 35x10 meters**2
        #wall is rotated about the X axis(perpendicular to Z axis)
        self.ballPlane = vizshape.addPlane(size=(2,2), axis=vizshape.AXIS_Z, cullFace = False)
        #shifts the wall to match the edge of the floor
        self.ballPlane.setPosition(0,5,-20)
        #makes the wall appear white
        self.ballPlane.color(viz.WHITE)
        self.ballPlane.alpha(0.2)
        self.text_object = viz.addText3D( 'Target Plane',parent=self.ballPlane,pos=[-0.4,0.6,0])
        self.text_object.setScale(0.2,0.2,0.2)


        """</END OF DEFINING THE WORLD>"""

    def findIndex(Array,Value):
        
        Index =[]
        for index, number in enumerate(Array):
            if number == Value:
                Index.append(index)
            
        return Index

    def extractDataFromMatFile(self):
        
        os.chdir(self.dataPath  + '\\MatFile')
        print 'Changed cwd to ==> ', os.getcwd()
        self.rawMatFile = sio.loadmat('rawMat_' + self.textFileName +'.mat');
        os.chdir(self.dataPath)
        print 'Returned cwd to ==> ', os.getcwd()
        
        
        eventFlag
        #calibrationStatus = np.array(map(float,self.rawMatFile['calibrationStatus'][:]), dtype= float)
        rgx = map(float, self.rawMatFile['rightGazePoint_XYZ'][:,0])
        rgy = map(float, self.rawMatFile['rightGazePoint_XYZ'][:,1])
        rgz = map(float, self.rawMatFile['rightGazePoint_XYZ'][:,2])

        rightGazePoint_XYZ = np.array([rgx, rgy, rgz], dtype = float)

        rgx = map(float, self.rawMatFile['rightPupilPos_XYZ'][:,0])
        rgy = map(float, self.rawMatFile['rightPupilPos_XYZ'][:,1])
        rgz = map(float, self.rawMatFile['rightPupilPos_XYZ'][:,2])

        rightPupilPos_XYZ = np.array([rgx, rgy, rgz], dtype = float)

        rgx = map(float, self.rawMatFile['rightGazeDir_XYZ'][:,0])
        rgy = map(float, self.rawMatFile['rightGazeDir_XYZ'][:,1])
        rgz = map(float, self.rawMatFile['rightGazeDir_XYZ'][:,2])

        rightGazeDir_XYZ = np.array([rgx, rgy, rgz], dtype = float)
        
        lgx = map(float, self.rawMatFile['leftGazePoint_XYZ'][:,0])
        lgy = map(float, self.rawMatFile['leftGazePoint_XYZ'][:,1])
        lgz = map(float, self.rawMatFile['leftGazePoint_XYZ'][:,2])

        leftGazePoint_XYZ = np.array([lgx, lgy, lgz], dtype = float)

        lgx = map(float, self.rawMatFile['leftPupilPos_XYZ'][:,0])
        lgy = map(float, self.rawMatFile['leftPupilPos_XYZ'][:,1])
        lgz = map(float, self.rawMatFile['leftPupilPos_XYZ'][:,2])

        leftPupilPos_XYZ = np.array([lgx, lgy, lgz], dtype = float)

        lgx = map(float, self.rawMatFile['leftGazeDir_XYZ'][:,0])
        lgy = map(float, self.rawMatFile['leftGazeDir_XYZ'][:,1])
        lgz = map(float, self.rawMatFile['leftGazeDir_XYZ'][:,2])

        leftGazeDir_XYZ = np.array([lgx, lgy, lgz], dtype = float)

        ball_X = map(float, self.rawMatFile['ballPos_XYZ'][:,0])
        ball_Y = map(float, self.rawMatFile['ballPos_XYZ'][:,1])
        ball_Z = map(float, self.rawMatFile['ballPos_XYZ'][:,2])

        ball_Pos_XYZ = np.array([ball_X, ball_Y, ball_Z], dtype = float)

        ball_Vel_X = map(float, self.rawMatFile['ballVel_XYZ'][:,0])
        ball_Vel_Y = map(float, self.rawMatFile['ballVel_XYZ'][:,1])
        ball_Vel_Z = map(float, self.rawMatFile['ballVel_XYZ'][:,2])
        ball_Vel_XYZ = np.array([ball_Vel_X, ball_Vel_Y, ball_Vel_Z], dtype = float);

    #    Ball_Pix_X = map(float, self.rawMatFile['ballPix_XYDist'][:,0])
    #    Ball_Pix_Y = map(float, self.rawMatFile['ballPix_XYDist'][:,1])
    #    Ball_Pix_Z = map(float, self.rawMatFile['ballPix_XYDist'][:,2])
    #    Ball_Pix_XYZ = np.array([Ball_Pix_X, Ball_Pix_Y, Ball_Pix_Z], dtype = float);

        paddle_X = map(float, self.rawMatFile['paddlePos_XYZ'][:,0])
        paddle_Y = map(float, self.rawMatFile['paddlePos_XYZ'][:,1])
        paddle_Z = map(float, self.rawMatFile['paddlePos_XYZ'][:,2])
        paddle_Pos_XYZ = np.array([paddle_X, paddle_Y, paddle_Z], dtype = float);

        #Fix Me (Kamran)
        paddle_Quat_X = map(float, self.rawMatFile['paddleQUAT_WXYZ'][:,0])
        paddle_Quat_Y = map(float, self.rawMatFile['paddleQUAT_WXYZ'][:,1])
        paddle_Quat_Z = map(float, self.rawMatFile['paddleQUAT_WXYZ'][:,2])
        paddle_Quat_W = map(float, self.rawMatFile['paddleQUAT_WXYZ'][:,3])
        #Fix Me
        paddle_Quat_WXYZ = np.array([paddle_Quat_X, paddle_Quat_Y, paddle_Quat_Z, paddle_Quat_W ], dtype = float);

        view_X = map(float, self.rawMatFile['view_XYZ_Pos'][:,0])
        view_Y = map(float, self.rawMatFile['view_XYZ_Pos'][:,1])
        view_Z = map(float, self.rawMatFile['view_XYZ_Pos'][:,2])
        view_Pos_XYZ = np.array([view_X, view_Y, view_Z], dtype = float);

        #Fix Me (Kamran)
        view_Quat_X = map(float, self.rawMatFile['Quat_Matrix'][:,0])
        view_Quat_Y = map(float, self.rawMatFile['Quat_Matrix'][:,1])
        view_Quat_Z = map(float, self.rawMatFile['Quat_Matrix'][:,2])
        view_Quat_W = map(float, self.rawMatFile['Quat_Matrix'][:,3])
        #Fix Me
        view_Quat_WXYZ = np.array([view_Quat_X, view_Quat_Y, view_Quat_Z, view_Quat_W], dtype = float);

        frameTime = map(float, self.rawMatFile['FrameTime'])
    #    calibrationCounter = map(float, self.rawMatFile['calibrationCounter'])

        eventFlag = map(float, self.rawMatFile['EventFlag'])

        print 'Total Number of Frames = ', len(frameTime), '\n'
        print 'Total Number of Event Flags = ', len(eventFlag), '\n'
        print 'Ball Pos Size', ball_Pos_XYZ.shape
        print 'Ball Vel Size', ball_Vel_XYZ.shape
    #    print 'Ball Pix XYZ Size', Ball_Pix_XYZ.shape
    #    print 'Paddle Pos Size', Paddle_Pos_XYZ.shape
        print 'Paddle Quat Size', view_Quat_WXYZ.shape
        print 'View Pos Size', view_Pos_XYZ.shape
        print 'View Quat Size', view_Quat_WXYZ.shape

        #TrialStartIndex = FindIndex(EventFlag,1)
        #TrialEndIndex = FindIndex(EventFlag,3)
        #TrialNumber = len(TrialEndIndex)
        #print 'Number of Trials are=\n', TrialNumber
        #print 'Start Indexes =', TrialStartIndex
        #print 'End Indexes =', TrialEndIndex
    #    print 'calibrationStatus =', calibrationStatus
        self.rawDataStruct = {'rightGazePoint_XYZ' : rightGazePoint_XYZ, 'rightPupilPos_XYZ' : rightPupilPos_XYZ, 'rightGazeDir_XYZ' : rightGazeDir_XYZ,
                              'leftGazePoint_XYZ' : leftGazePoint_XYZ, 'leftPupilPos_XYZ' : leftPupilPos_XYZ, 'leftGazeDir_XYZ' : leftGazeDir_XYZ,
                                'ball_Pos_XYZ' : ball_Pos_XYZ, 'ball_Vel_XYZ' : ball_Vel_XYZ, 'paddle_Pos_XYZ' : paddle_Pos_XYZ, 'paddle_Quat_WXYZ' : paddle_Quat_WXYZ,
                                'view_Pos_XYZ' : view_Pos_XYZ, 'view_Quat_WXYZ' : view_Quat_WXYZ, 'frameTime' : frameTime, 'eventFlag' : eventFlag}

    def createVisualObjects(self):

        #creats a sphere(the ball) with radius of 5cm
        self.ball = vizshape.addSphere(radius = .05)
        #colors the ball red
        self.ball.color(viz.YELLOW)
        self.ball.visible(True)

        self.Origin = vizshape.addAxes()
        self.Origin.setPosition(-5.5,0.1,8)
    #    #creats a sphere(the ball) with radius of 5cm
    #    #Head = vizshape.addCone( radius = 0.5, height = 0.8)
    #    Head = vizshape.addArrow(1, color = viz.YELLOW_ORANGE)
    #    #colors the ball red
    #    Head.color(viz.PURPLE)
    #    Head.visible(True)
    #    Head.setScale(.2,.2,.3)

        #creats a sphere(the hand) with radius of 10cm
        self.Hand = vizshape.addCylinder( height = 0.02, radius = 0.2, axis = vizshape.AXIS_Z)
        #colors the hand red
        self.Hand.color(viz.RED)
        self.Hand.visible(True)

        self.IOD = 0.06
        
        # create a node3D leftEyeNode
        self.cyclopEyeNode = vizshape.addSphere(0.015, color = viz.GREEN)
        #cyclopEyeNode.visible(viz.OFF)
        self.cyclopEyeNode.setPosition(*self.rawDataStruct['view_Pos_XYZ'][:,0])
        self.cyclopEyeNode.setQuat(*self.rawDataStruct['view_Quat_WXYZ'][:,0])

        # create a node3D rightEyeNode
        self.rightEyeNode = vizshape.addSphere(0.015, color = viz.RED)
        #rightEyeNode.visible(viz.OFF)
        self.rightEyeNode.setParent(self.cyclopEyeNode)
        self.rightEyeNode.setPosition(self.IOD/2, 0, 0.0,viz.ABS_PARENT)
    #    right_sphere = gazeSphere(eyeTracker,viz.RIGHT_EYE,rightEyeNode,[clientWindowID],sphereColor=viz.ORANGE)
    #    rightGazeVector = gazeVector(eyeTracker,viz.RIGHT_EYE,rightEyeNode,[clientWindowID],gazeVectorColor=viz.ORANGE)
    #    right_sphere.toggleUpdate()
    #    rightGazeVector.toggleUpdate()
    #    right_sphere.node3D.alpha(0.7)    


        # create a node3D leftEyeNode
        self.leftEyeNode = vizshape.addSphere(0.015, color = viz.BLUE)
        #leftEyeNode.visible(viz.OFF)
        self.leftEyeNode.setParent(self.cyclopEyeNode)
        self.leftEyeNode.setPosition(-self.IOD/2, 0, 0.0,viz.ABS_PARENT)
    #    left_sphere = gazeSphere(eyeTracker,viz.LEFT_EYE,leftEyeNode,[clientWindowID],sphereColor=viz.YELLOW)
    #    leftGazeVector = gazeVector(eyeTracker,viz.LEFT_EYE,leftEyeNode,[clientWindowID],gazeVectorColor=viz.YELLOW)
    #    left_sphere.toggleUpdate()
    #    leftGazeVector.toggleUpdate()
    #    left_sphere.node3D.alpha(0.7)

        # Creating a Line to represent Gaze Vector
        viz.startLayer(viz.LINES)
        viz.vertex(0,0,0)
        viz.vertex(0,0,3)
        viz.vertexColor(viz.GREEN)
        self.eyeGazeVector = viz.endLayer() # Object will contain both points and lines
        self.eyeGazeVector.visible(True)
        self.eyeGazeVector.setParent(self.cyclopEyeNode)
        self.eyeGazeVector.pointSize(10)
        #rightGazeVector.setScale(5,5,5)
        self.eyeGazeSphere = vizshape.addSphere(0.02, color = viz.GREEN)
        self.eyeGazeSphere.setParent(self.cyclopEyeNode)


        # Creating a Line to represent Gaze Vector
        viz.startLayer(viz.LINES)
        viz.vertex(0,0,0)
        viz.vertex(0,0,3)
        viz.vertexColor(viz.RED)
        self.rightGazeVector = viz.endLayer() # Object will contain both points and lines
        self.rightGazeVector.visible(True)
        #rightGazeVector.setParent(rightEyeNode)
        self.rightGazeVector.pointSize(10)
        #rightGazeVector.setScale(5,5,5)
        self.rightGazeSphere = vizshape.addSphere(0.02, color = viz.RED)
        self.rightGazeSphere.setParent(self.rightEyeNode)

        # Creating a Line to represent Gaze Vector
        viz.startLayer(viz.LINES)
        viz.vertex(0,0,0)
        viz.vertex(0,0,3)
        viz.vertexColor(viz.BLUE)
        self.leftGazeVector = viz.endLayer() # Object will contain both points and lines
        self.leftGazeVector.visible(True)
        #leftGazeVector.setParent(leftEyeNode)
        #leftGazeVector.setScale(5,5,5)
        self.leftGazeSphere = vizshape.addSphere(0.02, color = viz.BLUE)
        self.leftGazeSphere.setParent(self.leftEyeNode)


        # Creating a Line to represent Eye-Ball Vector
        viz.startLayer(viz.LINES)
        viz.vertex(0,0,0)
        viz.vertex(0,0,3)
        viz.vertexColor(viz.YELLOW)
        self.EyeBallLine = viz.endLayer() # Object will contain both points and lines
        self.EyeBallLine.visible(True)
        #EyeBallLine.setScale(5,5,5)

        #self.male = viz.add('vcc_male.cfg') 
        #male.state(1) #looping idle animation
        #self.headBone = self.male.getBone('Bip01 Head')
        #self.headBone.lock()
        #self.male.setParent(self.cyclopEyeNode)
        #self.male.setPosition([0,-1.7,-0.08], viz.ABS_PARENT)
        #self.male.setPosition([-0.45, 0, 1.24])
        #male.setEuler([0,-5,0], viz.ABS_PARENT)
        #male.alpha(0.8)
        
    def dotproduct(self, v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))


    def length(self, v):
      return math.sqrt(self.dotproduct(v, v))


    def angle(self, v1, v2):
      return math.acos(self.dotproduct(v1, v2) / (self.length(v1) * self.length(v2)))


    def quaternion2Matrix(self, Q):
        Q = Q/np.linalg.norm(Q); # Ensure Q has unit norm
        
        # Set up convenience variables
        x = Q[0]; y = Q[1]; z = Q[2]; w = Q[3];
        w2 = w*w; x2 = x*x; y2 = y*y; z2 = z*z;
        xy = x*y; xz = x*z; yz = y*z;
        wx = w*x; wy = w*y; wz = w*z;
        
        M = np.array([[w2+x2-y2-z2 , 2*(xy - wz) , 2*(wy + xz) ,  0],
             [ 2*(wz + xy) , w2-x2+y2-z2 , 2*(yz - wx) ,  0 ],
             [ 2*(xz - wy) , 2*(wx + yz) , w2-x2-y2+z2 ,  0 ],
             [     0      ,       0     ,       0     ,  1 ]], dtype = float);
        return M;

    #def FindMinimumAngle(v1, v2, fr):
    #    
    #    global lEyeRotationMatrix, lEyeOffsetMatrix, View_Quat_WXYZ;
    #    Error = 2000;
    #    Angle = 0;
    #    Vector = v1
    #    for alfa in (np.linspace(-20*np.pi/180, 0*np.pi/180, num = 10)):
    #        lEyeRotationMatrix[0,:] = [np.cos(alfa), np.sin(alfa), 0, 0];
    #        lEyeRotationMatrix[1,:] = [-np.sin(alfa), np.cos(alfa), 0, 0];
    #        ViewRotation = Quaternion2Matrix(View_Quat_WXYZ[:,fr]);
    #        Result = lEyeOffsetMatrix.dot(v2);
    #        Result = lEyeRotationMatrix.dot(Result);
    #        v = ViewRotation.dot(Result);
    #        ErrorTemp = (np.abs(np.linalg.norm(np.cross(v1[0:3],v[0:3]))));
    #        
    #        if ErrorTemp < Error :
    #            Error = ErrorTemp
    #            Angle = alfa
    #            Vector = v
    #    #print 'Error,', Error,' alfa',Angle*180/np.pi
    #    return Angle,Vector

    def setRotationAngle(self, angle):
        print 'Screen Rotation Set to ', angle;
        self.alpha = angle*(np.pi/180);

    def onTimer(self, num):

        if self.frameNumber < len(self.rawDataStruct['frameTime']) - 1:
            #pointCounter[calibrationCounter[self.frameNumber]] = pointCounter[calibrationCounter[self.frameNumber]] + 1
            self.frameNumber = self.frameNumber + 1
        else:
            #self.frameNumber = 0
            print 'Data = ', errorArray
            MatFile = {'Data':errorArray}
            MatFileName = 'ErrorData' 
            sio.savemat(MatFileName + '.mat', MatFile)
            viz.quit()
        if (self.rawDataStruct['eventFlag'][self.frameNumber] == 1):
            self.PD = 0.8
            self.BD = 0.3
            self.PBD = 0.7
            self.timer = 0
    #    StartIndex = TrialStartIndex[counter]
    #    EndIndex   = TrialEndIndex[counter]
    #    for self.frameNumber in range(StartIndex, EndIndex):
        #print 'F=', self.frameNumber,'P=', Ball_Pos_XYZ[:,self.frameNumber],'Q=',View_Quat_WXYZ[:,self.frameNumber], '\n'
        self.ball.setPosition(*self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber]) # X[:,self.frameNumber])#
        self.Hand.setPosition(*self.rawDataStruct['paddle_Pos_XYZ'][:,self.frameNumber])
        self.Hand.setQuat(*self.rawDataStruct['paddle_Quat_WXYZ'][:,self.frameNumber])
        
        #self.headBone.setQuat(self.cyclopEyeNode.getQuat())
        #self.male.setEuler([0, 0, 0], viz.ABS_GLOBAL)

        
        self.V1 = self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber] - self.rawDataStruct['view_Pos_XYZ'][:,self.frameNumber]
        self.V1 = np.hstack((self.V1,1))
        self.V1.reshape(4,1)



        self.cyclopEyeNode.setPosition(*self.rawDataStruct['view_Pos_XYZ'][:,self.frameNumber])
        self.cyclopEyeNode.setQuat(*self.rawDataStruct['view_Quat_WXYZ'][:,self.frameNumber])
        
        
        self.rightGazePoint_XYZ = [ -self.rawDataStruct['rightGazeDir_XYZ'][0,self.frameNumber], self.rawDataStruct['rightGazeDir_XYZ'][1,self.frameNumber], self.rawDataStruct['rightGazeDir_XYZ'][2,self.frameNumber]]
        self.rightScale = np.linalg.norm(self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber] - self.rightEyeNode.getPosition(viz.ABS_GLOBAL))
        self.rightGazePoint_XYZ = np.multiply(self.rightGazePoint_XYZ, self.rightScale)
        self.rightGazeSphere.setPosition(self.rightGazePoint_XYZ[0], self.rightGazePoint_XYZ[1], self.rightGazePoint_XYZ[2], viz.ABS_PARENT)
        self.rightGazeVector.setVertex(0, self.rightEyeNode.getPosition(mode = viz.ABS_GLOBAL), viz.ABS_GLOBAL)
        self.rightGazeVector.setVertex(1, self.rightGazeSphere.getPosition(viz.ABS_GLOBAL), viz.ABS_GLOBAL)
        

        self.leftGazePoint_XYZ = [ -self.rawDataStruct['leftGazeDir_XYZ'][0,self.frameNumber], self.rawDataStruct['leftGazeDir_XYZ'][1,self.frameNumber], self.rawDataStruct['leftGazeDir_XYZ'][2,self.frameNumber]]
        self.leftScale = np.linalg.norm(self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber] - self.leftEyeNode.getPosition(viz.ABS_GLOBAL))
        self.leftGazePoint_XYZ = np.multiply(self.leftGazePoint_XYZ, self.leftScale)
        self.leftGazeSphere.setPosition(self.leftGazePoint_XYZ[0], self.leftGazePoint_XYZ[1], self.leftGazePoint_XYZ[2], viz.ABS_PARENT)
        self.leftGazeVector.setVertex(0, self.leftEyeNode.getPosition(mode = viz.ABS_GLOBAL), viz.ABS_GLOBAL)
        self.leftGazeVector.setVertex(1, self.leftGazeSphere.getPosition(viz.ABS_GLOBAL), viz.ABS_GLOBAL)
        
        self.EyeBallLine.setVertex(0, self.rawDataStruct['view_Pos_XYZ'][0,self.frameNumber], self.rawDataStruct['view_Pos_XYZ'][1,self.frameNumber], self.rawDataStruct['view_Pos_XYZ'][2,self.frameNumber], viz.ABS_GLOBAL)
        self.EyeBallLine.setVertex(1, self.rawDataStruct['ball_Pos_XYZ'][0,self.frameNumber], self.rawDataStruct['ball_Pos_XYZ'][1,self.frameNumber], self.rawDataStruct['ball_Pos_XYZ'][2,self.frameNumber], viz.ABS_GLOBAL)
        
        self.ballPlane.setPosition(*self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber])

        self.V1 = self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber] - self.cyclopEyeNode.getPosition()
        self.V2 =  self.leftGazePoint_XYZ - self.leftEyeNode.getPosition(viz.ABS_GLOBAL) + self.rightGazePoint_XYZ - self.rightEyeNode.getPosition(viz.ABS_GLOBAL) 

        self.V3 =  self.leftGazePoint_XYZ  + self.rightGazePoint_XYZ
        self.eyeGazeVector.setVertex(0,0,0, viz.ABS_PARENT)
        self.eyeGazeVector.setVertex(1, self.V3[0]/2.0, self.V3[1]/2.0, self.V3[2]/2.0, viz.ABS_PARENT)
        self.eyeGazeSphere.setPosition(self.eyeGazeVector.getVertex(1), viz.ABS_PARENT)

        self.V1 = self.rawDataStruct['ball_Pos_XYZ'][:,self.frameNumber] - self.cyclopEyeNode.getPosition(viz.ABS_GLOBAL)
        #V2 = eyeGazeSphere.getPosition(viz.ABS_GLOBAL) - cyclopEyeNode.getPosition(viz.ABS_GLOBAL)
        self.V2 = [a - b for a, b in zip(self.eyeGazeSphere.getPosition(viz.ABS_GLOBAL), self.cyclopEyeNode.getPosition(viz.ABS_GLOBAL))]
        # XYZ = [l1[idx][0] - l2[idx][0] , l1[idx][1] - l2[idx][1], l1[idx][2] - l2[idx][2]  for idx in range(len(l1))]
        errorAngle = np.multiply(self.angle(self.V2, self.V1), 180/np.pi)
        #print 'Angle = ', errorAngle
        #if (calibrationStatus[self.frameNumber] == 1):
        #errorArray[calibrationCounter[self.frameNumber], pointCounter[calibrationCounter[self.frameNumber]]] = errorAngle
        #viz.MainView.setPosition(*Ball_Pos_XYZ[:,self.frameNumber] + [0.0, 0.4, -1.5])
        #if (calibrationStatus[self.frameNumber] == 0):
        string = 'PerForM Lab'#'Before Calibration'
        #else:
        #    string = 'Eye Tracking in VR'#'After Calibration'
        self.wall_text_object.message(string)
        # HACK (KAMRAN)string2 = 'Error = %2.2f' %(errorAngle)+ u"\u00b0"
        # HACK (KAMRAN)string2 = 'AE = %.1f %c'%(errorAngle, u"\u00b0")
        # HACK (KAMRAN)self.text_object.message(string2)
    def updateViewPoint(self):
        
        if ( self.viewPointCounter == 0 ):
            viz.MainView.setPosition([2.552443504333496, 2.406543731689453, 1.4710679054260254])
            viz.MainView.setEuler([-90, 0.0, 0.0])
            self.viewPointCounter = 1
        elif( self.viewPointCounter == 1 ):
            viz.MainView.setPosition([0.6917849779129028, 2.0247855186462402, -0.20941883325576782])
            viz.MainView.setEuler([-22.73750877380371, -3.614262342453003, -0.09662231057882309])
            self.viewPointCounter = 2
        elif( self.viewPointCounter == 2 ):
            viz.MainView.setPosition([0.4033511281013489, 1.7814747095108032, -1.921712040901184])
            viz.MainView.setEuler([0.251657485961914, -0.0, 0.0])
            self.viewPointCounter = 0
        
        
        
if __name__ == '__main__':
    
   
    errorArray = np.array([],dtype = float)
    errorArray = np.zeros((27,80))
    pointCounter = np.zeros(27)
    viewPointFlag = False
    TimeInterval = 0.0633#0.01327800000001389 # 0.067
    nearH = 1.2497;
    nearW = 1.5622;
    viz.setMultiSample(4)
    viz.fov(60)
    viz.go()
    textFileName = 'exp_data-2015-7-21-20-51'# 'exp_data-2015-4-24-13-34'

    myVisualization = visualization(textFileName)
    myVisualization.extractDataFromMatFile()#'Exp_RawMat_calib_data-2015-5-12-21-45.mat');
    myVisualization.createTheRoom()
    myVisualization.createVisualObjects()
    #sets where the camera view is located
    viz.MainView.setPosition([0.4033511281013489, 1.7814747095108032, -1.921712040901184])
    
    #sets the angle at which the camera view is pointed
    viz.MainView.setEuler([0.251657485961914, -0.0, 0.0])
    #lEyeOffsetMatrix = np.array([[1, 0, 0, -0.03],[0, 1, 0, 0], [0, 0, 1, 0],[0, 0, 0, 1]], dtype = float)
    #lEyeRotationMatrix = np.eye(4, dtype = float);
    counter = 1
    myVisualization.setRotationAngle(5.5);
    #if the timer goes off go to the onTimer function
    viz.callback(viz.TIMER_EVENT,myVisualization.onTimer)
    viz.starttimer(1, TimeInterval, viz.FOREVER)
    # Go fullscreen on monitor 1
    viz.window.setFullscreenMonitor(2)
    vizact.onkeydown('v', myVisualization.updateViewPoint)

    
