from datetime import datetime
import os
import pickle
import random
import time
import warnings
import cv2
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from descriptor import Descriptor

def processFiles(pos_dir, neg_dir, recurse=False, output_file=False,
        output_filename=None, color_space="bgr", channels=[0, 1, 2],
        hog_features=False, hist_features=False, spatial_features=False,
        hog_lib="cv", size=(64,64), hog_bins=9, pix_per_cell=(8,8),
        cells_per_block=(2,2), block_stride=None, block_norm="L1",
        transform_sqrt=True, signed_gradient=False, hist_bins=16,
        spatial_size=(16,16)):

    """
    Extract features from positive samples and negative samples.
    Store feature vectors in a dict and optionally save to pickle file.

    @param pos_dir (str): Path to directory containing positive samples.
    @param neg_dir (str): Path to directory containing negative samples.
    @param recurse (bool): Traverse directories recursively (else, top-level only).
    @param output_file (bool): Save processed samples to file.
    @param output_filename (str): Output file filename.
    @param color_space (str): Color space conversion.
    @param channels (list): Image channel indices to use.
    
    For remaining arguments, refer to Descriptor class:
    @see descriptor.Descriptor#__init__(...)

    @return feature_data (dict): Lists of sample features split into training,
        cross-validation, test sets; scaler object; parameters used to
        construct descriptor and process images.

    NOTE: OpenCV HOGDescriptor currently only supports 1-channel and 3-channel
    images, not 2-channel images.
    """

    if not (hog_features or hist_features or spatial_features):
        raise RuntimeError("No features selected (set hog_features=True, "
            + "hist_features=True, and/or spatial_features=True.)")

    pos_dir = os.path.abspath(pos_dir)
    neg_dir = os.path.abspath(neg_dir)

    if not os.path.isdir(pos_dir):
        raise FileNotFoundError("Directory " + pos_dir + " does not exist.")
    if not os.path.isdir(neg_dir):
        raise FileNotFoundError("Directory " + neg_dir + " does not exist.")

    print("Building file list...")
    if recurse:
        pos_files = [os.path.join(rootdir, file) for rootdir, _, files
            in os.walk(pos_dir) for file in files]
        neg_files = [os.path.join(rootdir, file) for rootdir, _, files
            in os.walk(neg_dir) for file in files]
    else:
        pos_files = [os.path.join(pos_dir, file) for file in
            os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, file))]
        neg_files = [os.path.join(neg_dir, file) for file in
            os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, file))]

    print("{} positive files and {} negative files found.\n".format(
        len(pos_files), len(neg_files)))

    # Get color space information.
    color_space = color_space.lower()
    if color_space == "gray":
        color_space_name = "grayscale"
        cv_color_const = cv2.COLOR_BGR2GRAY
        channels = [0]
    elif color_space == "hls":
        color_space_name = "HLS"
        cv_color_const = cv2.COLOR_BGR2HLS
    elif color_space == "hsv":
        color_space_name = "HSV"
        cv_color_const = cv2.COLOR_BGR2HSV
    elif color_space == "lab":
        color_space_name = "Lab"
        cv_color_const = cv2.COLOR_BGR2Lab
    elif color_space == "luv":
        color_space_name = "Luv"
        cv_color_const = cv2.COLOR_BGR2Luv
    elif color_space == "ycrcb" or color_space == "ycc":
        color_space_name = "YCrCb"
        cv_color_const = cv2.COLOR_BGR2YCrCb
    elif color_space == "yuv":
        color_space_name = "YUV"
        cv_color_const = cv2.COLOR_BGR2YUV
    else:
        color_space_name = "BGR"
        cv_color_const = -1

    # Get names of desired features.
    features = [feature_name for feature_name, feature_bool
        in zip(["HOG", "color histogram", "spatial"],
               [hog_features, hist_features, spatial_features])
        if feature_bool == True]

    feature_str  = features[0]
    for feature_name in features[1:]:
        feature_str += ", " + feature_name

    # Get information about channel indices.
    if len(channels) == 2 and hog_features and hog_lib == "cv":
        warnings.warn("OpenCV HOG does not support 2-channel images",
            RuntimeWarning)

    channel_index_str = str(channels[0])
    for ch_index in channels[1:]:
        channel_index_str += ", {}".format(ch_index)

    print("Converting images to " + color_space_name + " color space and " 
        + "extracting " + feature_str + " features from channel(s) "
        + channel_index_str + ".\n")

    # Store feature vectors for positive samples in list pos_features and
    # for negative samples in neg_features.
    pos_features = []
    neg_features = []
    start_time = time.time()

    # Get feature descriptor object to call on each sample.
    descriptor = Descriptor(hog_features=hog_features, hist_features=hist_features,
            spatial_features=spatial_features, hog_lib=hog_lib, size=size,
            hog_bins=hog_bins, pix_per_cell=pix_per_cell,
            cells_per_block=cells_per_block, block_stride=block_stride,
            block_norm=block_norm, transform_sqrt=transform_sqrt,
            signed_gradient=signed_gradient, hist_bins=hist_bins,
            spatial_size=spatial_size)

    # Iterate through files and extract features.
    for i, filepath in enumerate(pos_files + neg_files):
        image = cv2.imread(filepath)
        if cv_color_const > -1:
            image = cv2.cvtColor(image, cv_color_const)

        if len(image.shape) > 2:
            image = image[:,:,channels]

        feature_vector = descriptor.getFeatureVector(image)

        if i < len(pos_files):
            pos_features.append(feature_vector)
        else:
            neg_features.append(feature_vector)

    print("Features extracted from {} files in {:.1f} seconds\n".format(
        len(pos_features) + len(neg_features), time.time() - start_time))

    # Store the length of the feature vector produced by the descriptor.
    num_features = len(pos_features[0])

    # Instantiate scaler and scale features.
    print("Scaling features.\n")
    scaler = StandardScaler().fit(pos_features + neg_features)
    pos_features = scaler.transform(pos_features)
    neg_features = scaler.transform(neg_features)
    
    # Randomize lists of feature vectors. Split 75/20/5 into training,
    # cross-validation, and test sets.
    print("Shuffling samples into training, cross-validation, and test sets.\n")
    random.shuffle(pos_features)
    random.shuffle(neg_features)

    num_pos_train = int(round(0.75 * len(pos_features)))
    num_neg_train = int(round(0.75 * len(neg_features)))
    num_pos_val = int(round(0.2 * len(pos_features)))
    num_neg_val = int(round(0.2 * len(neg_features)))

    pos_train = pos_features[0 : num_pos_train]
    neg_train = neg_features[0 : num_neg_train]

    pos_val = pos_features[num_pos_train : (num_pos_train + num_pos_val)]
    neg_val = neg_features[num_neg_train : (num_neg_train + num_neg_val)]

    pos_test = pos_features[(num_pos_train + num_pos_val):]
    neg_test = neg_features[(num_neg_train + num_neg_val):]

    print("{} samples in positive training set.".format(len(pos_train)))
    print("{} samples in positive cross-validation set.".format(len(pos_val)))
    print("{} samples in positive test set.".format(len(pos_test)))
    print("{} total positive samples.\n".format(len(pos_train) +
        len(pos_val) + len(pos_test)))

    print("{} samples in negative training set.".format(len(neg_train)))
    print("{} samples in negative cross-validation set.".format(len(neg_val)))
    print("{} samples in negative test set.".format(len(neg_test)))
    print("{} total negative samples.\n".format(len(neg_train) +
        len(neg_val) + len(neg_test)))

    # Store sample data and parameters in dict.
    # Descriptor class object seems to produce errors when unpickling and
    # has been commented out below. The descriptor will be re-instantiated
    # by the Detector object later.
    feature_data = {
                    "pos_train": pos_train,
                    "neg_train": neg_train,
                    "pos_val": pos_val,
                    "neg_val": neg_val,
                    "pos_test": pos_test,
                    "neg_test": neg_test,
                    #"descriptor": descriptor,
                    "scaler": scaler,
                    "hog_features": hog_features,
                    "hist_features": hist_features,
                    "spatial_features": spatial_features,
                    "color_space": color_space,
                    "cv_color_const": cv_color_const,
                    "channels": channels,
                    "hog_lib": hog_lib,
                    "size": size,
                    "hog_bins": hog_bins,
                    "pix_per_cell": pix_per_cell,
                    "cells_per_block": cells_per_block,
                    "block_stride": block_stride,
                    "block_norm": block_norm,
                    "transform_sqrt": transform_sqrt,
                    "signed_gradient": signed_gradient,
                    "hist_bins": hist_bins,
                    "spatial_size": spatial_size,
                    "num_features": num_features
                    }

    # Pickle to file if desired.
    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                + "_data.pkl")

        pickle.dump(feature_data, open(output_filename, "wb"))
        print("Sample and parameter data saved to {}\n".format(output_filename))

    return feature_data

def trainSVM(filepath=None, feature_data=None, C=1,
        loss="squared_hinge", penalty="l2", dual=False, fit_intercept=False,
        output_file=False, output_filename=None):

    """
        Train a classifier from feature data extracted by processFiles().

        @param filepath (str): Path to feature data pickle file.
        @param feature_data (dict): Feature data dict returned by processFiles().
            NOTE: Either a file or dict may be supplied.
        @param output_file (bool): Save classifier and parameters to file.
        @param output_filename (str): Name of output file.

        For remaining arguments, @see sklearn.svm.LinearSVC()

        @return classifier_data (dict): Dict containing trained classifier and
            relevant training/processing feature parameters.
    """

    print("Loading sample data.")
    if filepath is not None:
        filepath = os.path.abspath(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError("File " + filepath + " does not exist.")
        feature_data = pickle.load(open(filepath, "rb"))
    elif feature_data is None:
        raise ValueError("Invalid feature data supplied.")

    # Train classifier on training set.
    pos_train = np.asarray(feature_data["pos_train"])
    neg_train = np.asarray(feature_data["neg_train"])
    pos_val = np.asarray(feature_data["pos_val"])
    neg_val = np.asarray(feature_data["neg_val"])
    pos_test = np.asarray(feature_data["pos_test"])
    neg_test = np.asarray(feature_data["neg_test"])

    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate(
        (np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))

    print("Training classifier...")
    start_time = time.time()
    classifier = svm.LinearSVC(C=C, loss=loss, penalty=penalty, dual=dual,
        fit_intercept=fit_intercept)
    classifier.fit(train_set, train_labels)
    print("Classifier trained in {:.1f} s.\n".format(time.time() - start_time))

    # Run classifier on cross-validation set.
    pos_val_predicted = classifier.predict(pos_val)
    neg_val_predicted = classifier.predict(neg_val)

    false_neg_val = np.sum(pos_val_predicted != 1)
    false_pos_val = np.sum(neg_val_predicted == 1)
    pos_predict_accuracy = 1 - (false_neg_val / float(pos_val.shape[0]))
    neg_predict_accuracy = 1 - (false_pos_val / float(neg_val.shape[0]))
    total_accuracy = 1 - ((false_neg_val + false_pos_val) /
        float(pos_val.shape[0] + neg_val.shape[0]))

    print("Val set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_neg_val, pos_val.shape[0], 100 * pos_predict_accuracy))
    print("Val set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_pos_val, neg_val.shape[0], 100 * neg_predict_accuracy))
    print("Val set total misclassifications: {} / {} ({:.3f}% accuracy)\n".format(
        false_neg_val + false_pos_val, pos_val.shape[0] + neg_val.shape[0],
        100 * total_accuracy))

    # Retrain classifier with misses from validation set. Run on test set.
    print("Augmenting training set with misclassified validation samples and "
        + "retraining classifier.\n")
    pos_train = np.vstack((pos_train, pos_val[pos_val_predicted != 1, :]))
    neg_train = np.vstack((neg_train, neg_val[neg_val_predicted == 1, :]))
    train_set = np.vstack((pos_train, neg_train))
    train_labels = np.concatenate(
        (np.ones(pos_train.shape[0],), np.zeros(neg_train.shape[0],)))

    classifier.fit(train_set, train_labels)

    pos_test_predicted = classifier.predict(pos_test)
    neg_test_predicted = classifier.predict(neg_test)

    false_neg_test = np.sum(pos_test_predicted != 1)
    false_pos_test = np.sum(neg_test_predicted == 1)
    pos_predict_accuracy = 1 - (false_neg_test / float(pos_test.shape[0]))
    neg_predict_accuracy = 1 - (false_pos_test / float(neg_test.shape[0]))
    total_accuracy = 1 - ((false_neg_test + false_pos_test) /
        float(pos_test.shape[0] + neg_test.shape[0]))

    print("Test set false negatives: {} / {} ({:.3}% accuracy)".format(
        false_neg_test, pos_test.shape[0], 100 * pos_predict_accuracy))
    print("Test set false positives: {} / {} ({:.3f}% accuracy)".format(
        false_pos_test, neg_test.shape[0], 100 * neg_predict_accuracy))
    print("Test set total misclassifications: {} / {} ({:.3f}% accuracy)".format(
        false_neg_test + false_pos_test, pos_test.shape[0] + neg_test.shape[0],
        100 * total_accuracy))

    # Store classifier data and parameters in new dict that excludes
    # sample data from feature_data dict.
    excludeKeys = ("pos_train", "neg_train", "pos_val", "neg_val",
        "pos_test", "neg_test")
    classifier_data = {key: val for key, val in feature_data.items()
        if key not in excludeKeys}
    classifier_data["classifier"] = classifier

    if output_file:
        if output_filename is None:
            output_filename = (datetime.now().strftime("%Y%m%d%H%M")
                + "_classifier.pkl")

        pickle.dump(classifier_data, open(output_filename, "wb"))
        print("\nSVM classifier data saved to {}".format(output_filename))

    return classifier_data
