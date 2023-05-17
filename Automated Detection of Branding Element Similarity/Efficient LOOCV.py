import multiprocessing
import numpy as np
from experiments.optimized_experiments import logo_comparison_efficient
from experiments.optimized_experiments.logo_efficient import Logo
import os
import pandas as pd
from tqdm import tqdm
import psutil
import pickle
import lz4.frame
import time


def run_loocv(previous_loc, output_loc, num_threads=psutil.cpu_count(), worker=None):
    """
    Runs a LOOCV analysis of the image files in the previous location.
    reserve_thread: - If False program will use all CPU threads, if number entered program will leave that many threads unused.
    """

    # Reading logo names from directory
    previous_logo_names = os.listdir(previous_loc)
    for file in previous_logo_names:
        if len(file) > 10:
            previous_logo_names.remove(file)

    previous_logos = list()
    previous_logos.sort()

    # Creating and compressing Logo objects with required attributes
    for i in tqdm(previous_logo_names):
        new_logo = Logo(previous_loc + '/' + i)
        new_logo.color_detect()
        if not hasattr(new_logo, 'contour_count'):
            new_logo.shape_analysis()
        previous_logos.append(lz4.frame.compress(pickle.dumps(new_logo)))

    # Dividing workload based on available CPU threads
    previous_logos.sort()
    if worker is not None:
        task = worker.split(" of ")
        worker_lists = np.array_split(range(len(previous_logos)), int(task[1]))
        this_worker_list = worker_lists[(int(task[0]) - 1)]
        applicant_lists = np.array_split(this_worker_list, num_threads)
    else:
        applicant_lists = np.array_split(range(len(previous_logos)), num_threads)


    # Initializing process for each thread
    for item in applicant_lists:
        p = multiprocessing.Process(target=run_thread, args=[previous_logos, item, output_loc])
        p.start()


def run_thread(previous_logos, applicant_logo_index, output_loc):
    """
    Runs a LOOCV analysis of the image files in the previous location.
    num_threads: - number of CPU threads to utilize, defaults to all available
    worker: - denotes which worker this is when running with multiple machines, use format "# of #", ex: "3 of 7"
    defaults to None
    """
    applicant_list = list()
    previous_list = list()
    ssim_score_list = list()
    color_score_list = list()
    tm_score_list = list()

    # Iterating through assigned applicant Logos and running LOOCV
    for applicant in tqdm(applicant_logo_index):

        # Decompressing applicant logo
        new_applicant = pickle.loads(lz4.frame.decompress(previous_logos[applicant]))
        removed_logo = new_applicant
        sc_df = pd.DataFrame()
        for previous in previous_logos:

            # Decompressing previous logo
            new_previous = pickle.loads(lz4.frame.decompress(previous))
            if new_previous.name != new_applicant.name:
                # Comparing logo attributes
                ssim = logo_comparison_efficient.logo_ssim(new_applicant, new_previous)
                color = logo_comparison_efficient.calculate_color_similarity(new_applicant, new_previous)
                tm_score = logo_comparison_efficient.logo_contains(new_applicant, new_previous)

                # Adding scores to lists
                applicant_list.append(new_applicant.name)
                previous_list.append(new_previous.name)
                ssim_score_list.append(ssim)
                color_score_list.append(color)
                tm_score_list.append(tm_score)
        sc_df = pd.concat([sc_df, logo_comparison_efficient.calculate_logo_shape_complexity_similarity(new_applicant,
                                                                                                       previous_logos)])

        data_df = pd.DataFrame({'Applicant Logo': applicant_list,
                                'Previous Logo': previous_list,
                                'SSIM': ssim_score_list,
                                'Color Similarity Score': color_score_list,
                                'Template Matching': tm_score_list})
        sc_df.columns = ['Applicant Logo', 'Previous Logo', 'Shape Complexity Score']
        data_df = data_df.merge(sc_df, how='inner', on=['Applicant Logo', 'Previous Logo'])
        data_df = data_df[
            ['Applicant Logo', 'Previous Logo', 'SSIM', 'Color Similarity Score', 'Shape Complexity Score',
             'Template Matching']]
        data_df.to_csv(f'{output_loc}/{str(removed_logo)[:-4]}_scores.csv', index=False)


if __name__ == "__main__":
    machine = input("which machine is this?")
    run_loocv("data/previous_logos-1", "test_files/test_results/efficient_loocv_scores", worker=machine)
    time.sleep(500)
    print("done")


