import numpy as np
import csv
import os
from logo_analysis_testing import run_analysis_short
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import random
import itertools
from matplotlib.widgets import SpanSelector
import matplotlib.ticker as mtick


def find_thresholds_percentile(scores_file, set_percentile=97):
    """
    finds thresholds needed to flag images above desired percentile using flag_images
    scores_file: - csv file containing data for a single call of run_analysis
    set_percentile: - percentile of images desired above output thresholds
    return: - tuple containing thresholds for flag_images, (ssim, css, scs, template matching)
    """
    ssim_list = []
    css_list = []
    scs_list = []
    template_matching_list = []
    with open(scores_file, "rt") as file1:
        csv_file = csv.reader(file1)
        for row in csv_file:
            if row[0] == "Applicant Logo":
                continue
            ssim_list.append(float(row[2]))
            css_list.append(100 - float(row[3]))
            scs_list.append(float(row[4]))
            template_matching_list.append(float(row[5]))
        ssim_threshold = np.percentile(ssim_list, set_percentile)
        css_threshold = np.percentile(css_list, set_percentile)
        scs_threshold = np.percentile(scs_list, set_percentile)
        template_matching_threshold = np.percentile(template_matching_list, set_percentile)
        return float(ssim_threshold), 100 - float(css_threshold), float(scs_threshold), float(
            template_matching_threshold)


def find_images(flagged):
    """
    displays number of tests each image was flagged in, does not display images with zero flags
    flagged: - dictionary returned from flag_images
    return: - dictionary containing each flagged image, and the number of flags that image raised
    """
    for image in flagged.keys():
        flag_count_dict = {}
        for item in flagged.get(image).keys():
            flags = flagged.get(image).get(item)
            flag_count = 0
            if flags[0]:
                flag_count += 1
            if flags[1]:
                flag_count += 1
            if flags[2]:
                flag_count += 1
            if flags[3]:
                flag_count += 1
            flag_count_dict.update({item: flag_count})
    return flag_count_dict


def generate_histogram(file_location):
    """
    Generates histogram for distribution of scores within scores file at file_location for each test.
    file_location: - path to scores file
    return: - tuple containing numpy histogram for each test.
    """
    with open(file_location, "rt") as file1:
        ssim = []
        css = []
        scs = []
        tm = []
        csv_reader = csv.reader(file1)
        for row in csv_reader:
            for entry in row:
                if entry == "SSIM":
                    ssim_index = row.index(entry)
                if entry == "Color Similarity Score":
                    css_index = row.index(entry)
                if entry == "Shape Complexity Score":
                    scs_index = row.index(entry)
                if entry == "Template Matching":
                    tm_index = row.index(entry)
            if row[ssim_index] == "SSIM":
                continue
            ssim.append(float(row[ssim_index]))
            css.append(float(row[css_index]))
            scs.append(float(row[scs_index]))
            tm.append(float(row[tm_index]))
        ssim_histogram = np.histogram(ssim, bins=100, range=(-0.1, 1))
        css_histogram = np.histogram(css, bins=100, range=(0, 100))
        scs_histogram = np.histogram(scs, bins=100, range=(0, 1.01))
        tm_histogram = np.histogram(tm, bins=100, range=(-0.1, 1))
        return ssim_histogram, css_histogram, scs_histogram, tm_histogram



def generate_average_histogram(input_directory):
    """
    Creates average histogram for each test using scores files from input_directory.
    input_directory: - path to folder containing scores files
    return: - tuple containing numpy histograms for each test. (ssim, css, scs, template matching)
    """
    ssim_histograms = []
    css_histograms = []
    scs_histograms = []
    tm_histograms = []

    # Creating histogram for each scores file
    for file in tqdm(os.listdir(input_directory)):
        histogram_list = generate_histogram(input_directory + "/" + file)
        ssim_histograms.append(histogram_list[0])
        css_histograms.append(histogram_list[1])
        scs_histograms.append(histogram_list[2])
        tm_histograms.append(histogram_list[3])
    ssim_averages = []
    css_averages = []
    scs_averages = []
    tm_averages = []
    for i in range(100):
        ssim_sum = 0
        css_sum = 0
        scs_sum = 0
        tm_sum = 0
        for histogram in ssim_histograms:
            ssim_sum += histogram[0][i]
        for histogram in css_histograms:
            css_sum += histogram[0][i]
        for histogram in scs_histograms:
            scs_sum += histogram[0][i]
        for histogram in tm_histograms:
            tm_sum += histogram[0][i]
        ssim_averages.append(ssim_sum / len(ssim_histograms))
        css_averages.append(css_sum / len(css_histograms))
        scs_averages.append(scs_sum / len(scs_histograms))
        tm_averages.append(tm_sum / len(tm_histograms))
    return (ssim_averages/np.linalg.norm(ssim_averages)), (css_averages/np.linalg.norm(css_averages)), (scs_averages/np.linalg.norm(scs_averages)), (tm_averages/np.linalg.norm(tm_averages))


def minimize_thresholds(input_file, input_histograms, initial_ssim=1.00, initial_css=0.00, initial_scs=1.01,
                        initial_tm=1.00):
    """
    Finds the lowest possible thresholds to flag images in given file with least possible excess images flagged.
    input_file: - csv file containing scores for all images that must be flagged
    input_histograms: - average histogram of dataset, used to determine which thresholds to lower for greatest efficiency
    initial_###: - thresholds program will start with, defaults to most restrictive possible for each test
    return: - tuple containing minimized thresholds for each test - (SSIM, CSS, SCS, TM)
    """

    # Setting initial thresholds.
    ssim_threshold = initial_ssim
    css_threshold = initial_css
    scs_threshold = initial_scs
    tm_threshold = initial_tm

    # Opening input_file.
    with open(input_file, "rt") as file1:
        csv_reader = csv.reader(file1)

        # Iterating through each entry in file.
        for row in csv_reader:

            # Finding index of data for corresponding test.
            for entry in row:
                if entry == "SSIM":
                    ssim_index = row.index(entry)
                if entry == "Color Similarity Score":
                    css_index = row.index(entry)
                if entry == "Shape Complexity Score":
                    scs_index = row.index(entry)
                if entry == "Template Matching":
                    tm_index = row.index(entry)
            if row[ssim_index] == "SSIM":
                continue
            threshold_list = [ssim_threshold, css_threshold, scs_threshold, tm_threshold]

            # Reading data from row, storing as corresponding variables.
            ssim_score = float(row[ssim_index])
            css_score = float(row[css_index])
            scs_score = float(row[scs_index])
            tm_score = float(row[tm_index])
            scores_list = [ssim_score, css_score, scs_score, tm_score]

            # Determining required change in threshold to flag image.
            threshold_change = [False, False, False, False]
            for i in range(4):
                if i != 1:
                    if scores_list[i] < threshold_list[i]:
                        threshold_change[i] = (scores_list[i], threshold_list[i])
                else:
                    if scores_list[i] > threshold_list[i]:
                        threshold_change[i] = (threshold_list[i], scores_list[i])

            # Skipping row if threshold is already met.
            if False in threshold_change:
                # print("No change needed.")
                continue
            extra_images = []

            # Creating list defining edges of histogram bins.
            for i in range(4):
                added_images = 0
                if threshold_change[i]:
                    bins_crossed = []
                    if i == 0:
                        bin_edges = np.linspace(-0.1, 1, num=101)
                    if i == 1:
                        bin_edges = np.linspace(0, 100, num=101)
                    if i == 2:
                        bin_edges = np.linspace(0, 1.01, num=101)
                    if i == 3:
                        bin_edges = np.linspace(-0.1, 1, num=101)

                    # Setting bin edge to True if it is within the range of bin edges that would be added during threshold change.
                    for bin_edge in bin_edges:
                        bins_crossed.append((bin_edge > threshold_change[i][0]) & (bin_edge < threshold_change[i][1]))

                    # Converting list of bin edges to list of bins.
                    for j in range(100):
                        next_edge = bins_crossed[j + 1]
                        current_edge = bins_crossed[j]

                        # If both edges of bin are crossed, bin contents are added to added_images.
                        if current_edge == True and next_edge == True:
                            added_images += input_histograms[i][j]
                extra_images.append(added_images)
            # print(extra_images)

            # Checking which threshold change flags the least extra images on average, and changing threshold accordingly.
            if (extra_images[0] <= extra_images[1]) & (extra_images[0] <= extra_images[2]) & (
                    extra_images[0] <= extra_images[3]):
                # print("Changing SSIM threshold from " + str(ssim_threshold) + " to " + str(ssim_score) + ".")
                ssim_threshold = ssim_score
            elif (extra_images[1] <= extra_images[0]) & (extra_images[1] <= extra_images[2]) & (
                    extra_images[1] <= extra_images[3]):
                # print("Changing CSS threshold from " + str(css_threshold) + " to " + str(css_score) + ".")
                css_threshold = css_score
            elif (extra_images[2] <= extra_images[1]) & (extra_images[2] <= extra_images[0]) & (
                    extra_images[2] <= extra_images[3]):
                # print("Changing SCS threshold from " + str(scs_threshold) + " to " + str(scs_score) + ".")
                scs_threshold = scs_score
            elif (extra_images[3] <= extra_images[1]) & (extra_images[3] <= extra_images[2]) & (
                    extra_images[3] <= extra_images[0]):
                # print("Changing TM threshold from " + str(tm_threshold) + " to " + str(tm_score) + ".")
                tm_threshold = tm_score
            # print(ssim_threshold, css_threshold, scs_threshold, tm_threshold)
        return ssim_threshold, css_threshold, scs_threshold, tm_threshold


def find_thresholds(folder_loc, previous_loc, loocv_loc, initial_ssim=1.00, initial_css=0.00, initial_scs=1.01,
                    initial_tm=1.00):
    """
    Finds the best thresholds to flag all images within folder_loc as infringing.
    Requires a prepared directory to run:
    - Create an empty folder at "folder_loc"
    - Within empty folder create the folder "image_clones"
    - For each set of infringing images, create a folder in the "image_clones" folder with the exact name as the original image, then copy all infringing images into that folder.
    For example, the original Best Buy logo at previous_loc is titled "BBY.png", so the folder containing its clones must be titled "BBY.png".
    The original image must be among the images at "previous_loc"
    folder_loc: - path to folder created to contain files for this call of the function
    previous_loc: - path to folder containing dataset for applicants to be run against
    loocv_loc: - path to folder containing scores files from a LOOCV analysis of the dataset
    initial_###: - thresholds program will start with, defaults to most restrictive possible for each test
    return: - tuple containing minimized thresholds for each test - (SSIM, CSS, SCS, TM)
    """

    # Creating folder to contain output scores files
    if not os.path.exists(folder_loc + "/output_scores"):
        os.makedirs(folder_loc + "/output_scores")

    # Running short analysis for each folder of applicants
    for folder in os.listdir(folder_loc + "/" + "image_clones"):

        # Checking if analysis has already been run
        if (folder[:-4] + "_scores.csv") not in os.listdir(folder_loc + "/output_scores"):
            run_analysis_short(folder_loc + "/" + "image_clones" + "/" + folder, previous_loc,
                               folder_loc + "/output_scores", folder)

    # Creating average histogram for each test using data from loocv_loc
    avg_hst = generate_average_histogram(loocv_loc)
    current_ssim = initial_ssim
    current_css = initial_css
    current_scs = initial_scs
    current_tm = initial_tm

    # Finding the strictest thresholds for each score to flag every applicant
    paths = os.listdir(folder_loc + "/output_scores")
    scores_list = [None]
    while len(paths) > 0:
        scores_list = list()
        for path in paths:
            # print(score)
            new_thresholds = minimize_thresholds(folder_loc + "/output_scores/" + path, avg_hst,
                                                 initial_ssim=current_ssim,
                                                 initial_css=current_css, initial_scs=current_scs,
                                                 initial_tm=current_tm)

            threshold_list = [1.00, 0.00, 1.01, 1.00]

            scores_list.append([new_thresholds[0], new_thresholds[1], new_thresholds[2], new_thresholds[3]])
            sum_list = []
        for score in scores_list:
            threshold_change = [False, False, False, False]
            for i in range(4):
                if i != 1:
                    if score[i] < threshold_list[i]:
                        threshold_change[i] = (score[i], threshold_list[i])
                else:
                    if score[i] > threshold_list[i]:
                        threshold_change[i] = (threshold_list[i], score[i])

            # Skipping row if threshold is already met.
            extra_images = []

            # Creating list defining edges of histogram bins.
            for i in range(4):
                added_images = 0
                if threshold_change[i]:
                    bins_crossed = []
                    if i == 0:
                        bin_edges = np.linspace(-0.1, 1, num=101)
                    if i == 1:
                        bin_edges = np.linspace(0, 100, num=101)
                    if i == 2:
                        bin_edges = np.linspace(0, 1.01, num=101)
                    if i == 3:
                        bin_edges = np.linspace(-0.1, 1, num=101)

                    # Setting bin edge to True if it is within the range of bin edges that would be added during threshold change.
                    for bin_edge in bin_edges:
                        bins_crossed.append((bin_edge > threshold_change[i][0]) & (bin_edge < threshold_change[i][1]))

                    # Converting list of bin edges to list of bins.
                    for j in range(100):
                        next_edge = bins_crossed[j + 1]
                        current_edge = bins_crossed[j]

                        # If both edges of bin are crossed, bin contents are added to added_images.
                        if current_edge == True and next_edge == True:
                            added_images += avg_hst[i][j]
                extra_images.append(added_images)
            sum = 0
            for item in extra_images:
                sum += item
            sum_list.append(sum)
        print("")
        print("")
        print("")
        print(sum_list)
        print(scores_list)
        remove_index = sum_list.index(np.min(sum_list))
        current_ssim = scores_list[remove_index][0]
        current_css = scores_list[remove_index][1]
        current_scs = scores_list[remove_index][2]
        current_tm = scores_list[remove_index][3]
        if len(paths) == 1:
            print("\n")
            print("final thresholds: ")
            print(scores_list[0])
            print("average flagged images: ")
            print(sum_list[0])
        sum_list.pop(remove_index)
        scores_list.pop(remove_index)
        paths.pop(remove_index)