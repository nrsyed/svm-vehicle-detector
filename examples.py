import cv2
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.
pos_dir = "samples/vehicles"
neg_dir = "samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"

def example1():
    """
    Train a classifier and run it on a video using default settings
    without saving any files to disk.
    """

    # Extract HOG features from images in the sample directories and return
    # results and parameters in a dict.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
        hog_features=True)

    # Train SVM and return the classifier and parameters in a dict.
    # This function takes the dict from processFiles() as an input arg.
    classifier_data = trainSVM(feature_data=feature_data)

    # Instantiate a Detector object and load the dict from trainSVM().
    detector = Detector().loadClassifier(classifier_data=classifier_data)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)
    
    # Start the detector by supplying it with the VideoCapture object.
    # At this point, the video will be displayed, with bounding boxes
    # drawn around detected objects per the method detailed in README.md.
    detector.detectVideo(video_capture=cap)

def example2():
    """
    Extract features from sample images and save to pickle file.
    Load sample data to train classifier, then save classifier to pickle file.
    Run the classifier on a video by loading the classifier file, and write
    the resulting detection video to an avi file.
    """

    # Extract HOG features, color histogram features, and spatial features
    # from sample images, then save the data to a pickle file. Note that if an
    # output filepath isn't specified, a default timestamped filename will
    # be generated.
    feature_data_filename = "feature_data.pkl"
    processFiles(pos_dir, neg_dir, recurse=True, hog_features=True,
        hist_features=True, spatial_features=True, output_file=True,
        output_filename=feature_data_filename)

    # Load the pickle file produced by processFiles(), train the classifier,
    # then save the classifier data to a pickle file.
    classifier_data_filename = "classifier_data.pkl"
    trainSVM(filepath=feature_data_filename, output_file=True,
        output_filename=classifier_data_filename)

    # Instantiate a detector and load the classifier pickle file.
    detector = Detector().loadClassifier(filepath=classifier_data_filename)

    # Open a VideoCapture object for the video file.
    cap = cv2.VideoCapture(video_file)

    # Run the detector and save the resulting video to an avi file.
    detector.detectVideo(video_capture=cap, write=True)


def example3():
    """
    Extract features, train the classifier, run the detector using a
    variety of custom parameters.
    """
    
    # Extract features. Do not save to disk.
    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
        color_space="yuv", channels=[0, 2], hog_features=True,
        hist_features=False, spatial_features=True, hog_lib="sk",
        size=(128, 64), hog_bins=11, pix_per_cell=(16, 8),
        cells_per_block=(2,2), block_norm="L2", transform_sqrt=False,
        spatial_size=(64, 32))

    # Train a classifier and save it to disk, then use the returned dict
    # to instantiate and run a detector.
    classifier_data = trainSVM(feature_data=feature_data, loss="squared_hinge",
        penalty="l2", dual=False, fit_intercept=False, output_file=True,
        output_filename="example_classifier.pkl")

    detector = Detector(init_size=(128,64), x_overlap=0.75, y_step=0.02,
        x_range=(0.2, 0.85), y_range=(0.4, 0.9), scale=1.8)

    detector.loadClassifier(classifier_data=classifier_data)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=5, threshold=100,
        min_bbox=(50,50), draw_heatmap=False)

def example4():
    """
    Load an existing classifier and run it on a video with custom parameters.
    """

    detector = Detector(init_size=(64,64), x_overlap=0.3, y_step=0.015,
        x_range=(0.1, 0.9), scale=1.4)
    detector.loadClassifier(filepath="example_classifier.pkl")
    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=20, threshold=180,
        draw_heatmap_size=0.4)

def example5():
    """
    Train a classifier and run on video using parameters that seemed to work
    well for vehicle detection.
    """

    feature_data = processFiles(pos_dir, neg_dir, recurse=True,
        color_space="YCrCb", channels=[0, 1, 2], hog_features=True,
        hist_features=True, spatial_features=True, hog_lib="cv",
        size=(64,64), pix_per_cell=(8,8), cells_per_block=(2,2),
        hog_bins=20, hist_bins=16, spatial_size=(20,20))

    classifier_data = trainSVM(feature_data=feature_data, C=1000)

    detector = Detector(init_size=(90,90), x_overlap=0.7, y_step=0.01,
        x_range=(0.02, 0.98), y_range=(0.55, 0.89), scale=1.3)
    detector.loadClassifier(classifier_data=classifier_data)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(video_capture=cap, num_frames=9, threshold=120,
        draw_heatmap_size=0.3)
    

if __name__ == "__main__":
    #example1()
    #example2()
    #example3()
    #example4()
    example5()
