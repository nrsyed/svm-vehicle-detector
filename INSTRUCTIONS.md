# Instructions

It may be helpful to check out the examples in [examples.py](https://github.com/nrsyed/svm-vehicle-detector/blob/master/examples.py) and to read the function documentation in the source files to better understand usage and available options.

## Step-by-step instructions

1. **Extract features from positive and negative training samples with [```train.processFiles()```](https://github.com/nrsyed/svm-vehicle-detector/blob/master/train.py).**
   * All positive training samples should be in one directory.
   * All negative training samples should be in a different directory.
   * Sub-directories within each directory are fine. Just be sure to set keyword argument ```recurse=True```, e.g., ```processFiles(..., recurse=True)```, otherwise, the function will only grab images in the top level of the positive and negative sample directories.
   * Sample directories should contain only valid image files.
   * ```processFiles()``` returns a dict containing the processed files and feature descriptor parameters. This dict should be supplied to ```train.trainSVM()``` in the next step. If ```processFiles(..., output_file=True)```, the dict will be pickled and save to disk (a specific filename can be specified with ```processFiles(..., output_filename="myfile.pkl")```).

2. **Train the classifier with [```train.trainSVM()```](https://github.com/nrsyed/svm-vehicle-detector/blob/master/train.py).**
   * Supply ```trainSVM()``` with the dict returned by ```processFiles()``` above or with a pickle file saved by ```processFiles()```.
   * ```trainSVM() returns a dict containing the classifier and all relevant parameters. This dict should be supplied to the detector in the next step. If ```trainSVM(..., output_file=True)```, the dict will be pickled as in the previous step.

3. **Instantiate a [Detector](https://github.com/nrsyed/svm-vehicle-detector/blob/master/detector.py) and load the classifier dict (or pickle file) from ```trainSVM()``` in the previous step.**

Instantiating the detector and loading a classifier can be done as follows:

```python
# Instantiate detector and load classifier from dict.
detector = Detector().loadClassifier(classifier_data=classifier_dict)

# Alternatively, instantiate detector and load classifier from pickle file.
detector = Detector().loadClassifier(filepath="path/to/classifier.pkl")
```

Detector [sliding window](https://github.com/nrsyed/svm-vehicle-detector/blob/master/slidingwindow.py) parameters can also be set while instantiating the detector, if you don't wish to use the default settings:
```python
# Set desired sliding window parameters.
detector = Detector(init_size=(80, 80), x_overlap=0.6, y_range=(0.5, 0.9), ...)
```

4. **Run the detector on a video by creating an [OpenCV VideoCapture](https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture) object and supplying it to ```Detector.detectVideo()```.**

This may be done as follows:

```python
cap = cv2.VideoCapture("myvideo.mp4")
detector.detectVideo(video_capture=cap)
```

Other detector settings may be set while calling ```Detector.detectVideo()```, like the number of frames to sum, the heatmap threshold (an int between 0 and 255), the min bounding box size, the heatmap inset size, and whether to output the resulting video an an avi file. See [examples.py](https://github.com/nrsyed/svm-vehicle-detector/blob/master/examples.py) or [detector.py](https://github.com/nrsyed/svm-vehicle-detector/blob/master/detector.py) for more information.

## Sample terminal output

The following is what you might see printed to the terminal by running [```examples.example5()```](https://github.com/nrsyed/svm-vehicle-detector/blob/master/examples.py):

```
> $ python examples.py

Building file list...
8792 positive files and 8968 negative files found.

Converting images to YCrCb color space and extracting HOG, color histogram, spatial features from
channel(s) 0, 1, 2.

Features extracted from 17760 files in 14.7 seconds

Scaling features.

Shuffling samples into training, cross-validation, and test sets.

6594 samples in positive training set.
1758 samples in positive cross-validation set.
440 samples in positive test set.
8792 total positive samples.

6726 samples in negative training set.
1794 samples in negative cross-validation set.
448 samples in negative test set.
8968 total negative samples.

Loading sample data.
Training classifier...
Classifier trained in 5.1 s.

Val set false negatives: 16 / 1758 (99.1% accuracy)
Val set false positives: 41 / 1794 (97.715% accuracy)
Val set total misclassifications: 57 / 3552 (98.395% accuracy)

Augmenting training set with misclassified validation samples and retraining classifier.

Test set false negatives: 4 / 440 (99.1% accuracy)
Test set false positives: 4 / 448 (99.107% accuracy)
Test set total misclassifications: 8 / 888 (99.099% accuracy)

> $
```
