import numpy as np
import cv2
import os

from numpy import pi as PI

import constants


def process_frame(frame):
    height, width, num_of_channels = np.shape(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[constants.Y_CROP: (height - constants.Y_CROP), constants.X_CROP: (width - constants.X_CROP)]
    processed_frame = cv2.resize(frame, dsize=(100, 100))
    return processed_frame


def calculate_flow_angle(new, old):
    a, b = new.ravel()
    c, d = old.ravel()
    angle = np.arctan((b-d)/(a-c+0.01))
    return angle


def angular_optical_flow(video):

    # foreground_model = cv2.createBackgroundSubtractorKNN()
    cap = cv2.VideoCapture(video)
    feature_vector = []

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=constants.number_of_feature_points, qualityLevel=0.1, minDistance=5, blockSize=5)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(10, 10), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (constants.number_of_feature_points, 1))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = process_frame(old_frame)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    previous_angles = None

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_gray)
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame_gray = process_frame(frame)

        # calculate optical flow
        try:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        except cv2.error:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        OpticalFlow_orientation = []

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new,good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            OpticalFlow_orientation.append(calculate_flow_angle(new, old))


        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        OpticalFlow_orientation = [x if x is not None else 0 for x in OpticalFlow_orientation]
        if previous_angles is None:
            previous_angles = np.zeros(shape=(len(OpticalFlow_orientation),))

        number_of_points_now = len(OpticalFlow_orientation)
        number_of_points_previously = len(previous_angles)
        considered_number_of_points = min(number_of_points_now, number_of_points_previously)

        change_in_angle = np.subtract(OpticalFlow_orientation[:considered_number_of_points], previous_angles[:considered_number_of_points])
        # print(change_in_angle)
        previous_angles = OpticalFlow_orientation

        bin_edges = [i*(PI/6) for i in range(-6, 7)]
        angle_histogram, bins = np.histogram(change_in_angle, bins=bin_edges)

        angle_histogram = angle_histogram / (np.max(angle_histogram) + 1)

        # frame_descriptor = np.hstack((intensity_histogram, orientation_histogram, improved_intensity_histogram, improved_orientation_histogram, angle_histogram))
        feature_vector.append(angle_histogram)

    cv2.destroyAllWindows()
    cap.release()
    # print(np.shape(np.array(feature_vector)))
    return np.array(feature_vector)


def save_features(features, folder, file, type):
    filename = file[0:-4] + 'features.dat'
    print(filename)
    if type == 'v':
        np.savetxt(folder+'Violent_angle/'+filename, features, fmt='%6.5f', delimiter=' ')
    else:
        np.savetxt(folder + 'Non_violent_angle/' + filename, features, fmt='%6.5f', delimiter=' ')


def process_videos(folder, save_folder):
    if not os.path.exists(save_folder + 'Non_violent_angle'):
        os.makedirs(save_folder + 'Non_violent_angle')
    if not os.path.exists(save_folder + 'Violent_angle'):
        os.makedirs(save_folder + 'Violent_angle')

    for f in os.listdir(folder):
        if str(f).endswith('avi'):
            features = angular_optical_flow(folder + str(f))
            save_features(features, save_folder, str(f), type='v')


def consolidate_data(folder, label):
    data = None
    for f in os.listdir(folder):

        if str(f).endswith('.dat') and not str(f).startswith('all_data'):
            # X = np.loadtxt(folder+str(f), delimiter=' ')
            print(str(f))
            try:
                X = np.genfromtxt(folder + str(f), delimiter=' ')
            except ValueError:
                continue
            if data is None:
                data = X
            else:
                data = np.vstack((data, X))

    np.savetxt(folder + 'all_data.dat', data, fmt='%6.5f', delimiter=' ')
    data_size = np.shape(data)[0]
    print(data_size)



if __name__ == '__main__':
    folder = 'D:/Data/Violent Crowd/Violence/'
    save_folder = 'D:/Data/Violent Crowd/Crowd_violence_dataset/Violent_angle/'
    # process_videos(folder, save_folder)
    consolidate_data(save_folder, 0)
