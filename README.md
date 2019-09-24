# Digitsrecognizer
A python based machine learning application which predicts the digits based on the gestures shown in front of camera

This is a python based machine learning application in which I used Python openCV and various machine learning libraries for recognizing the gestures shown with a blue cap.This application contains three models which predict the gestures you have drawn using the cap in front of screen.
    I have used MNIST dataset which can be imported using keras.datasets.

 Prerequisites: 
  1. opencv 3.4.1 
  2. Keras
  3. Tensorflow
  4. Numpy
  
 Execution:
  1. First Run the file firstmodel.py using command
    >python firstmodel.py
  2. First Run the file secondmodel.py using command
    >python secondmodel.py
  3. First Run the file thirdmodel.py using command
    >python thirdmodel.py
    
 Once you execute the above files you will get three models saved with names firstmodel.h5,secondmodel.h5,thirdmodel.h5.
 
  4.Execute the recognizer.py,
    After executing this file you will see a window where you can draw the digits using a cap (use blue cap because I had found it easy to   detect blue object since it was there in OpenCV documentation) in front of camera and the three models predict the digit drawn in front of camera
    
![Output](/output.png)
