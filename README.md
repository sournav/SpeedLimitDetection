# SpeedLimitDetection
Gives a readout for speed limit signs using tensorflow (mobilenets), and tesseract OCR.</br>
# Instructions
These files should be saved in the object_detection folder of the models_master directory in tensorflow. </br>
See: https://github.com/tensorflow/models/tree/master/research/object_detection</br>
(Tested on Windows 7)
# Dependencies
numpy</br>
http://www.numpy.org/</br>
pytesseract</br>
https://pypi.python.org/pypi/pytesseract</br>
tensorflow (I suggest you use the GPU version of tensorflow if you have a GPU since processing is faster)</br>
https://www.tensorflow.org/</br>
PIL</br>
https://python-pillow.org/</br>
opencv</br>
https://opencv.org/</br>
matplotlib</br>
https://matplotlib.org/</br>
jupyter(This is optional, I've commented some iPython lines out that can be used with a notebook)</br>
http://jupyter.org/</br>
# Processing
The follwing will explain how an image is processed BEFORE it is fed into tesseract for optical character ecognition.</br>

![alt text](https://github.com/sournav/SpeedLimitDetection/blob/master/image1.jpg "Logo Title Text 1")

Once this is complete it becomes easier for the OCR software to process through the image and give a readout as to what the sign actually says
<p>That being said the tesseract in many cases could not give an accurate readout as to what the sign said. It only worked on certain images. Considering however that the intent of my program is to be used with a livestream video source such as a webcam, it may work better since there will be several images to process through thus giving it a wider sample space.</p>
