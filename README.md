# k-NN Classifier: Hot dog or not hot dog

In this project, k-Nearest Neighbors (k-NN) is used for image classification. We start by classifying images into two categories
(1) Hot dog, or (2) not hot dog (literally anything other than hot dogs). We see the results of how good we can classify images 
into either of those categories. We then run a series of images of hamburgers to see what they get classified as. This is
to demonstrate that the k-NN isn't really learning any specific features about the images, but rather just comparing distances.
We see that hamburgers can be easily misclassified as hot dogs.

This demonstrates that k-NN is a simple algorithm that it doesn't do any 'true learning' like we see in neural networks.

## Description
#### ArUco Markers:
ArUco markers are a type of fiducial marker. They can be used as reference points that are placed in the FOV of a camera.
They can be used for a number of things including camera calibration, object orientation, autonomous navigation, and many more.
More infor can be found here: https://docs.opencv.org/4.x/d9/d6a/group__aruco.html

#### ArUco Layout:
I have pre-generated 4 unique ArUco markers and outlined them in a rectangular fashion on a sheet of 8.5x11 printer paper.
The PDF of this can be found at 'aruco_pdf.pdf' in this directory. Feel free to print this out youself and give it a go!

#### Video Capture:
Once I printed out the PDF with the ArUco markers on it, I fixed it to a spot on one of my walls. I then recorded a
~30 second video of me walking around the piece of paper from different angles / distances. An example from this video can 
be seen here:
  
![](/gifs/AR_raw.gif)  

#### Weather Screenshot:
I wanted to overlay something ontop the ArUco markers. I chose to display the current weather conditions in my city. 
This is done using Selenium webdriver to open up the specified URL and screenshot the webpage. This can be seen in
line 30-42 in the 'ar_main.py' script. The automated webpage / screenshot process looks like this:

![](/gifs/selenium.gif)  

#### ArUco Detection:
Then using OpenCV techniques and functions, I can (attempt) to detect the ArUco markers in each frame of the video. All 4
markers are required to overlay an image. However, do to motion blur / shadowing in the video not all 4 are detected everytime.
I do use a 'cache' to remove any flickering that uses the previous points when all 4 markers were found until they are all
found again. The detection of the markers can be seen below:
  
![](/gifs/AR_marker_detection.gif)  

#### Image Overlay:
If all 4 markers are detected, then I overlay the weather image on the video image in the proper spot. This is done
by extracting the four outside corners of the ArUco markers and creating a transform matrix to transform the weather image
to the video. You'll see there are certain spots where this is 'choppy'. This is when one (or more) of the markers was not
detected in that frame. This can be made more smooth with some more advanced CV techniques if desired.

![](/gifs/AR_overlay.gif)  

## Getting Started

### Dependencies

See requirements.txt

### Installing

#### Clone the project:
```
git clone git@github.com:reedajohns/Video-Augmented-Reality-with-Aruco-Markers-OpenCV.git
```
#### Update proper paths:
Line 15 in 'ar_main.py': (Weater.com city, currently set to Minneapolis, MN) 
```
weather_URL = "https://weather.com/weather/today/l/f2a75f4d0ceadba8e629bb2bacb40414bb499eb922989f847a9fa0659bf127e3"
```

### Executing program

Open terminal and run command:
```
python ar_main.py --source PATH_TO_VIDEO
```
Where PATH_TO_VIDEO is your prerecorded video with the proper ArUco markers in it.

## Authors

Contributors names and contact info

ex. Reed Johnson (https://www.linkedin.com/in/reed-a-johnson/)

## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [Pyimagesearch](https://pyimagesearch.mykajabi.com/products/pyimagesearch-university-full-access-plan/categories/4665317/posts/15676936)