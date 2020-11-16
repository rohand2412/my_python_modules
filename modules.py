#!/usr/bin/env python3
"""This script contains all of the modules used in the scripts in this OpencvTests directory"""

import copy
import cv2

class ColorTracker:
    """Tracks colors using customizable colorspace and has easy calibration with trackbars"""
    def __init__(self, channel_max_values, channel_names, window_detection_name, \
                 channel_bounds=None):
        self._window_detection_name = window_detection_name

        if not channel_bounds:
            channel_bounds = [(), (), ()]
        self._num_of_channels = 3
        self._channels = {}
        for i in range(self._num_of_channels):
            self._channels[channel_names[i]] = self._Channel(max_value=channel_max_values[i],
                                                             name=channel_names[i],
                                                             window_detection_name= \
                                                             self._window_detection_name,
                                                             bounds=channel_bounds[i])

    def create_trackbar(self):
        """Creates the trackbars used for easy calibration"""
        keys = list(self._channels)
        for i in range(self._num_of_channels):
            self._channels[keys[i]].create_trackbar()

    def processing(self, frame, iterations=2):
        """Thresholds, removes noise, and returns the contours"""
        keys = list(self._channels)
        frame_threshold = cv2.inRange(frame,
                                      (self._channels[keys[0]].get_low(),
                                       self._channels[keys[1]].get_low(),
                                       self._channels[keys[2]].get_low()),
                                      (self._channels[keys[0]].get_high(),
                                       self._channels[keys[1]].get_high(),
                                       self._channels[keys[2]].get_high()))
        frame_erode = cv2.erode(frame_threshold, None, iterations=iterations)
        frame_dilate = cv2.dilate(frame_erode, None, iterations=iterations)

        contours, _ = cv2.findContours(frame_dilate.copy(), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        return contours.copy()

    def get_channels(self):
        """Returns a deepcopy of the channels of the colorspace"""
        return copy.deepcopy(self._channels)

    class _Channel:
        """Allows individual manipulation of the channels"""
        def __init__(self, max_value, name, window_detection_name, bounds=()):
            self._max_value = max_value
            self._name = name
            self._low_name = "Low " + self._name
            self._high_name = "High " + self._name
            self._window_detection_name = window_detection_name

            if len(bounds) == 2:
                self._low = bounds[0]
                self._high = bounds[1]
            else:
                self._low = 0
                self._high = max_value

        def create_trackbar(self):
            """Generates trackbars for high and low bounds"""
            cv2.createTrackbar(self._low_name, self._window_detection_name, self._low,
                               self._max_value, self._on_low_thresh_trackbar)
            cv2.createTrackbar(self._high_name, self._window_detection_name, self._high,
                               self._max_value, self._on_high_thresh_trackbar)

        def _on_low_thresh_trackbar(self, trackbar_pos):
            """Callback on new position of lower bound trackbar"""
            self._low = min(self._high-1, trackbar_pos)
            cv2.setTrackbarPos(self._low_name, self._window_detection_name, self._low)

        def _on_high_thresh_trackbar(self, trackbar_pos):
            """Callback on new position of higher bound trackbar"""
            self._high = max(trackbar_pos, self._low+1)
            cv2.setTrackbarPos(self._high_name, self._window_detection_name, self._high)

        def get_low(self):
            """Returns lower bound"""
            return self._low

        def get_high(self):
            """Returns higher bound"""
            return self._high

        def get_max_value(self):
            """Returns max value of the channel"""
            return self._max_value
