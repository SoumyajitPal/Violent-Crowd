import numpy as np
import cv2
import os

from numpy import pi as PI

import constants


def process_frame(frame):
    if frame is None:
        return None
    height, width, num_of_channels = np.shape(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[constants.Y_CROP: (height - constants.Y_CROP), constants.X_CROP: (width - constants.X_CROP)]
    processed_frame = cv2.resize(frame, dsize=(100, 100))
    return processed_frame


def weber_local_descriptors(pixel_coordinate, frame):
    neighbor_pixel_values = 0
    pixel_X, pixel_Y = np.uint8(pixel_coordinate)
    if pixel_X >= constants.FRAME_SIZE-1 or pixel_Y >= constants.FRAME_SIZE-1:
        return None, None
    current_pixel = frame[pixel_Y, pixel_X] + 1
    for i in range(pixel_X-1, pixel_X+2):
        for j in range(pixel_Y-1, pixel_Y+2):
            if 0 < i < constants.FRAME_SIZE and 0 < j < constants.FRAME_SIZE:
                neighbor_pixel_values += (frame[j, i] - current_pixel)/current_pixel

    intensity_gradient = np.arctan(neighbor_pixel_values)
    orientation_gradient = np.arctan((frame[pixel_Y-1, pixel_X] - frame[pixel_Y+1, pixel_X])/(frame[pixel_Y, pixel_X-1] - frame[pixel_Y, pixel_X+1] + 0.1))
    return intensity_gradient, orientation_gradient


def improved_weber_local_descriptor(pixel_coordinate, frame, learning_rate=0.5):
    neighbor_pixel_values = 0
    improved_wld_in_x = 0
    improved_wld_in_y = 0
    pixel_X, pixel_Y = np.uint8(pixel_coordinate)
    if pixel_X >= constants.FRAME_SIZE-1 or pixel_Y >= constants.FRAME_SIZE-1:
        return None, None, 0, 0
    current_pixel = frame[pixel_Y, pixel_X] + 1
    for i in range(pixel_X-1, pixel_X+2):
        for j in range(pixel_Y-1, pixel_Y+2):
            if 0 < i < constants.FRAME_SIZE and 0 < j < constants.FRAME_SIZE:
                weber_element = (frame[j, i] - current_pixel)/current_pixel
                if i != pixel_X and j != pixel_Y:
                    pixel_orientation = np.arctan((j - pixel_Y)/(i - pixel_X))
                else:
                    pixel_orientation = 0

                neighbor_pixel_values += weber_element
                improved_wld_in_x += weber_element * np.cos(pixel_orientation)
                improved_wld_in_y += weber_element * np.sin(pixel_orientation)

    improved_wld_in_x = np.arctan(improved_wld_in_x)
    improved_wld_in_y = np.arctan(improved_wld_in_y)

    intensity_gradient = np.arctan(neighbor_pixel_values)
    orientation_gradient = np.arctan((frame[pixel_Y-1, pixel_X] - frame[pixel_Y+1, pixel_X])/(frame[pixel_Y, pixel_X-1] - frame[pixel_Y, pixel_X+1] + 0.1))
    improved_wld_gradient = np.sqrt(improved_wld_in_x ** 2 + improved_wld_in_y ** 2)
    improved_wld_orientation = np.arctan(improved_wld_in_y/(improved_wld_in_x + 1))
    return intensity_gradient, orientation_gradient, improved_wld_gradient, improved_wld_orientation


def calculate_flow_angle(new, old):
    a, b = new.ravel()
    c, d = old.ravel()
    angle = np.arctan((b-d)/(a-c+0.01))
    return angle


def optical_flow(video):

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
    # old_gray = foreground_model.apply(old_gray)
    # background = foreground_model.apply(old_gray)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_gray)
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        frame_gray = process_frame(frame)
        # background = foreground_model.apply(frame_gray)
        # background = cv2.GaussianBlur(background, ksize=(5, 5), sigmaX=0.1)
        # ret, background = cv2.threshold(background, 180, 255, cv2.THRESH_BINARY)

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

        WLD_intensity = []
        WLD_orientation = []
        IWLD_intensity = []
        IWLD_orientation = []
        OpticalFlow_orientation = []

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new,good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # intensity_gradient, orientation_gradient = weber_local_descriptors((a, b), frame_gray)
            intensity_gradient, orientation_gradient, improved_intensity_gradient, improved_orientation_gradient = improved_weber_local_descriptor((a, b), frame_gray)
            if intensity_gradient is None:
                continue
            WLD_intensity.append(intensity_gradient)
            WLD_orientation.append(orientation_gradient)
            IWLD_intensity.append(improved_intensity_gradient)
            IWLD_orientation.append(improved_orientation_gradient)
            OpticalFlow_orientation.append(calculate_flow_angle(new, old))

            # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 1)
            # frame_gray = cv2.circle(frame_gray, (a, b), 1, color[i].tolist(), -1)
        # img = cv2.add(frame_gray, mask)
        # cv2.imshow('frame', img)
        # cv2.imshow('foreground', cv2.absdiff(background, frame_gray))
        # if cv2.waitKey(10) > 0:
        #     break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        WLD_intensity = [x if x is not None else 0 for x in WLD_intensity]
        WLD_orientation = [x if x is not None else 0 for x in WLD_orientation]
        IWLD_intensity = [x if x is not None else 0 for x in IWLD_intensity]
        IWLD_orientation = [x if x is not None else 0 for x in IWLD_orientation]
        OpticalFlow_orientation = [x if x is not None else 0 for x in OpticalFlow_orientation]

        bin_edges = [i*(PI/6) for i in range(-6, 7)]
        intensity_histogram, bins = np.histogram(WLD_intensity, bins=bin_edges)
        orientation_histogram, bins = np.histogram(WLD_orientation, bins=bin_edges)
        improved_intensity_histogram, bins = np.histogram(IWLD_intensity, bins=bin_edges)
        improved_orientation_histogram, bins = np.histogram(IWLD_orientation, bins=bin_edges)
        angle_histogram, bins = np.histogram(OpticalFlow_orientation, bins=bin_edges)

        intensity_histogram = intensity_histogram / (np.max(intensity_histogram) + 1)
        orientation_histogram = orientation_histogram / (np.max(orientation_histogram) + 1)
        improved_intensity_histogram = improved_intensity_histogram / (np.max(improved_intensity_histogram) + 1)
        improved_orientation_histogram = improved_orientation_histogram / (np.max(improved_orientation_histogram) + 1)
        angle_histogram = angle_histogram / (np.max(angle_histogram) + 1)

        frame_descriptor = np.hstack((intensity_histogram, orientation_histogram, improved_intensity_histogram, improved_orientation_histogram, angle_histogram))
        feature_vector.append(frame_descriptor)

    cv2.destroyAllWindows()
    cap.release()
    # print(np.shape(np.array(feature_vector)))
    return np.array(feature_vector)


def save_features(features, folder, file, type):
    filename = file[0:-4] + 'features.dat'
    print(filename)
    if type == 'v':
        np.savetxt(folder+'Violent_features_improved/'+filename, features, fmt='%6.5f', delimiter=' ')
    else:
        np.savetxt(folder + 'NonViolent_features_improved/' + filename, features, fmt='%6.5f', delimiter=' ')


def process_videos(folder):
    if not os.path.exists(folder + 'Violent_features_improved'):
        os.makedirs(folder + 'Violent_features_improved')
    if not os.path.exists(folder + 'NonViolent_features_improved'):
        os.makedirs(folder + 'NonViolent_features_improved')

    for f in os.listdir(folder):
        if str(f).endswith('avi'):
            features = optical_flow(folder + str(f))
            save_features(features, folder, str(f), type='nv')
            # if str(f).startswith('fi'):
            #     # Fighting Videos
            #     features = optical_flow(folder + str(f))
            #     save_features(features, folder, str(f), type='v')
            #     # continue
            # else:
            #     # Non fighting Videos
            #     features = optical_flow(folder + str(f))
            #     save_features(features, folder, str(f), type='nv')


if __name__ == '__main__':
    # folder = 'D:/Data/Violent Crowd/Violence/'
    # process_videos(folder)

    array_one = [30, 70, 44, 5, 100, 9, 74, 90, 97, 39, 46, 112, 42, 36, 43, 63, 93, 24, 114, 15, 68]
    array_two = [4, 29, 47, 69, 99, 113, 118, 119]
    array = array_one + array_two
    # array = [31, 120, 60, 58, 77, 48, 12, 68, 64, 97, 56, 44, 61, 70, 111, 45, 59, 55, 9]
    for n in array:
        features = optical_flow('D:/Data/Violent Crowd/NonViolence_normalized/video (' + str(n) + ').avi')
        save_features(features, 'D:/Data/Violent Crowd/Special Cases/', 'video (' + str(n) + ').avi', type='nv')

