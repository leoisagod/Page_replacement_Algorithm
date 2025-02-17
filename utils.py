################################################################################
# Purpose: Advanced OS Homework 1 - Page Replacement Algorithm
# Author: M133040019 Hsin-Yu, Fu
# Date: 2024/10/14
#                                                          .
#                                                         ":"
#                                                       ___:____     |"\/"|
#                                                     ,'        `.    \  /
#                                                     |  O        \___/  |
#                                                   ~^~^~^~^~^~^~^~^~^~^~^~^~                                   
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import queue
from collections import deque
import os

################################################################################
# Type: Data
# Function: generate the first reference string - random
# Input: reference string value range(y-axis) and length of the dataset(x-axis)
# Return: int list
#     size: (120000)
################################################################################
def gen_data_random(range_d=1200, size=120000):
    data = np.random.randint(1, range_d+1, size)
    return list(data)

################################################################################
# Type: Data
# Function: generate the second reference string - locality
# Input: reference string value range(y-axis) and length of the dataset(x-axis)
#     proc_len_top: upper bound to the length of a process
#     proc_len_bot: lower bound to the length of a process
#     proc_top:  upper bound to the reference range of a process
#     proc_bot:  lower bound to the reference range of a process
# Return: int list
#     size: (120000)
################################################################################
def gen_data_local(range_d=1200, size=120000, proc_len_top=1/100, proc_len_bot=1/200, proc_top=100, proc_bot=5):
    # procedure call reference string length, proc_len: list containing all segmentation length
    proc_len = list(map(int, np.random.uniform(proc_len_bot, proc_len_top, int(1/proc_len_bot)) * size))

    data = []
    # 1. generate locality range from 1 to x (x = [5, 50]), length: proc_len
    # 2. place the locality segment start to [1, 1200-x]
    while len(data) < size:
        proc_range = np.random.randint(proc_bot, proc_top)
        proc_data = np.random.randint(1, proc_range, proc_len.pop()) + np.random.randint(0, (range_d -  proc_range))
        data.extend(proc_data)
    return data[:size]
################################################################################
# Type: Data
# Function: generate the self design reference string - Guassian-distributied 
#     locality with noise
# Description: Reshape the uniform distribution in original locality spot into
#     Guassian distribution. Moreover, add random noise to the dataset
# Input: reference string value range(y-axis) and length of the dataset(x-axis)
#     proc_len_top: upper bound to the length of a process
#     proc_len_bot: lower bound to the length of a process
#     proc_top:  upper bound to the reference range of a process
#     proc_bot:  lower bound to the reference range of a process
#     std_dev: standard deviation of the Gaussian distribution formula
# Return: int list
#     size: (120000)
################################################################################
def gen_data_GLN(range_d=1200, size=120000, proc_len_top=1/100, proc_len_bot=1/200, proc_top=100, proc_bot=5, std_dev=1.):
    # procedure call reference string length, proc_len: list containing all segmentation length
    proc_len = list(map(int, np.random.uniform(proc_len_bot, proc_len_top, int(1/proc_len_bot)) * size))
    count=0
    data = []
    while len(data) < size:
        count+=1
        # generate locality range from 1 to x (x = [5, 50])
        proc_range = np.random.randint(proc_bot, proc_top)
        # normal distribution centered at proc_range/2
        normal_values = np.random.normal(loc=proc_range//2, scale=std_dev, size=proc_len.pop())
        # clipped to range
        clipped_values = np.clip(normal_values, 1, proc_range)
        # round to integer;  place the locality segment start to [1, 1200-x]
        proc_data = np.rint(clipped_values).astype(int) + np.random.randint(0, (range_d -  proc_range))
        if count<=3:
            #plot_hist(proc_data,bins=proc_range)
            pass
        data.extend(proc_data)
    data=add_noise(data[:size])
    return data
################################################################################
# Type: Data
# Function: add noise to the input dataset
# Input: dataset (list or array)
#     range_d: range of the noise
#     noise_prob: likelihood of a noise point generation
# Return: noisy dataset (shape like input data)
################################################################################
def add_noise(data=[], range_d=1200, noise_prob=0.1):
    for i, ele in enumerate(data):
        if np.random.rand() < noise_prob:
            data[i] = np.random.randint(1, range_d)
    return data

################################################################################
# Type: Hardware
# Function: Model a proccess running and signaling to modify/reference bit
# Input: frame_number, original bitstring, usage probability, rate limit
# Return: page_faults, disk_write, interrupt (int, int, int)
################################################################################
def hw_set_bit(frame_num=10, bitstring=[], rate_limit = 0.25, prob_threshold = 0.05, rand_seq=np.zeros(0)):
    # seed
    if rand_seq.shape != frame_num:
        rand_seq = np.random.rand(frame_num)
    # limit the read/write rate, moedeling sparsity in real-world page usage
    if (np.count_nonzero(bitstring)/frame_num) < rate_limit:
        # reference_bits set to 1: read | write
        bitstring[rand_seq > prob_threshold] = 1
    return bitstring

################################################################################
# Type: ALGO
# Function: Page Replacement Algorithm: FIFO
# Input: dataset, frame size
# Return: page_faults, disk_write, interrupt (int, int, int)
################################################################################
def FIFO(data=[], frame_num=10):
    # total number of page fault, disk write and interrupt
    page_faults = 0
    disk_write = 0
    interrupt = 0
    # page frames, initialize to -1
    frames = np.full(frame_num, -1)
    # reference & modify bits of frames, initialize to 0
    reference_bits = np.zeros(frame_num)
    modify_bits = np.zeros(frame_num)
    # Queue to record first reference in frames
    fifo = queue.Queue(maxsize=frame_num)
    
    for reference in data:
        # page control
        if reference not in frames:
            page_faults+=1
            # page fault: interrupt
            interrupt+=1
            # if still empty frames, put reference to the first empty frame
            if -1 in frames:
                frames[np.where(frames==-1)[0][0]] = reference
            # find the page to replace if no empty frames: first element in Queue
            else:
                frame_to_write=np.where(frames==fifo.get(timeout=3))[0][0]
                frames[frame_to_write] = reference
                # modified: write back to disk + interrupt
                if modify_bits[frame_to_write] == 1:
                    disk_write+=1
                    modify_bits[frame_to_write] = 0
                    interrupt+=1
                # reference bit reset
                reference_bits[frame_to_write] = 0
            # add the reference to queue
            fifo.put(reference,timeout=3)
        else:
            # re-reference: usually used
            # hardware set: no need interrupt
            reference_bits[np.where(frames==reference)[0][0]]=1
        # process & hardware behavior modeling
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, reference_bits, 0.9, 0.05, rand_seq)
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, modify_bits, 1, 0.5, rand_seq)
        
    return page_faults, disk_write, interrupt

################################################################################
# Type: ALGO
# Function: Page Replacement Algorithm: Optimal
# Input: dataset, frame size
# Return: page_faults, disk_write, interrupt (int, int, int)
################################################################################
def optimal(data=[], frame_num=10):
    # total number of page fault, disk write and interrupt
    page_faults = 0
    disk_write = 0
    interrupt = 0
    # page frames, initialize to -1
    frames = np.full(frame_num, -1)
    # reference & modify bits of frames, initialize to 0
    reference_bits = np.zeros(frame_num)
    modify_bits = np.zeros(frame_num)
    # distances between pages to it's next reference
    dist=[]

    for idx, reference in enumerate(data):
        # update dist: -1 since propageted 1
        dist = [x-1 for x in dist]

        if reference not in frames:
            page_faults+=1
            # page fault: interrupt
            interrupt+=1

            # if still empty frames, put reference to the first empty frame
            if -1 in frames:
                frames[np.where(frames==-1)[0][0]] = reference
                # update dist: the distance from next reference value
                try:
                    dist.append(data[idx+1:].index(reference) + 1)
                except ValueError:
                    dist.append(999999999)

            # find the page to replace if no empty frames: page with largest distance to data
            # replace dist refer to the new reference
            else:
                frame_to_write = dist.index(max(dist))
                frames[frame_to_write] = reference
                try:
                    dist[frame_to_write] = data[idx+1:].index(reference) + 1
                except ValueError:
                    dist[frame_to_write] = 999999999
                # modified: write back to disk + interrupt
                if modify_bits[frame_to_write] == 1:
                    disk_write+=1
                    modify_bits[frame_to_write] = 0
                    interrupt+=1
                # reference bit reset
                reference_bits[frame_to_write] = 0
        # update dist: No page fault but need to find next reference value
        else:
            # re-reference: usually used
            # hardware set: no need interrupt
            reference_bits[np.where(frames==reference)[0][0]]=1
            try:
                dist[np.where(frames==reference)[0][0]] = data[idx+1:].index(reference) + 1
            except ValueError:
                dist[np.where(frames==reference)[0][0]] = 999999999

        # process & hardware behavior modeling
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, reference_bits, 0.9, 0.05, rand_seq)
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, modify_bits, 1, 0.5, rand_seq)
    return page_faults, disk_write, interrupt

################################################################################
# Type: ALGO
# Function: Page Replacement Algorithm: Enhanced Second Chance
# Input: dataset, frame size
# Return: page_faults, disk_write, interrupt (int, int, int)
################################################################################
def Enhanced_SC(data=[], frame_num=10):
    # total number of page fault, disk write and interrupt
    page_faults = 0
    disk_write = 0
    interrupt = 0
    # page frames, initialize to -1
    frames = np.full(frame_num, -1)
    # Queue to record first reference in frames, initialize to 0
    circularQ = deque(maxlen=frame_num)
    circularQ.extend(np.zeros(frame_num))
    # reference & modify bits of frames, initialize to 0
    reference_bits = np.zeros(frame_num)
    modify_bits = np.zeros(frame_num)

    for reference in data:
        if reference not in frames:
            page_faults+=1
            # page fault: interrupt
            interrupt+=1

            # if still empty frames, put reference to the first empty frame
            if -1 in frames:
                frames[np.where(frames==-1)[0][0]] = reference
                # add the reference to queue: q<x
                circularQ.popleft()
                circularQ.append(reference)
            else:
                for turn in range(frame_num*4):
                    # check the first element of queue: x<q, (ref, mod)
                    frame_check = np.where(frames==circularQ[0])[0][0]
                    # (0, 0) not used & modified : replace immediately
                    if reference_bits[frame_check] == 0 and modify_bits[frame_check] == 0:
                        # replace and queue rotate left
                        frames[frame_check] = reference
                        circularQ[0] = reference
                        circularQ.rotate(-1)
                        break
                    # (0, 1) modified but not used: replace in second iter
                    elif turn >= frame_num and reference_bits[frame_check] == 0 and modify_bits[frame_check] == 1:
                        # replace and queue rotate left
                        frames[frame_check] = reference
                        circularQ[0] = reference
                        circularQ.rotate(-1)
                        # modified == 1: write back to disk
                        modify_bits[frame_check] = 0
                        disk_write+=1
                        interrupt+=1
                        break
                    # (1, 0) recently used but clean: replace in third iter
                    elif turn >= frame_num*2 and reference_bits[frame_check] == 1 and modify_bits[frame_check] == 0:
                        # replace and queue rotate left
                        frames[frame_check] = reference
                        circularQ[0] = reference
                        circularQ.rotate(-1)
                        # reset reference bit
                        reference_bits[frame_check] = 0
                        break
                    # (1, 1) reset bits and write back
                    elif turn >= frame_num*3 and reference_bits[frame_check] == 1 and modify_bits[frame_check] == 1:
                        # replace and queue rotate left
                        frames[frame_check] = reference
                        circularQ[0] = reference
                        circularQ.rotate(-1)
                        # reset reference bit
                        reference_bits[frame_check] = 0
                        # modified == 1: write back to disk
                        modify_bits[frame_check] = 0
                        disk_write+=1
                        interrupt+=1
                        break
                    else:
                        circularQ.rotate(-1)               
        else:
            # re-reference: usually used
            # hardware set: no need interrupt
            reference_bits[np.where(frames==reference)[0][0]]=1
            pass
        
        # process & hardware behavior modeling
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, reference_bits, 0.9, 0.05, rand_seq)
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, modify_bits, 1, 0.5, rand_seq)

    return page_faults, disk_write, interrupt

################################################################################
# Type: Math
# Function: Calculate the distance between frames' data and the reference point
# Input: frames, reference point
# Return: distance (integer list)
################################################################################
def cal_distance(frames, reference=0):
    dist = []
    for i in range(len(frames)):
        dist.append(abs(frames[i] - reference))
    return dist

################################################################################
# Type: ALGO
# Function: Page Replacement Algorithm: Self design - Farest Neighbor
# Discription: For every incoming page fault, we have a reference r. For every
#    frames f, we calculate the distance d_f of it's data between r. Select the
#    vitim frame with the largest d_f. Since the distance is the largest, the
#    author called it Farest Neighbor Algorithm.
# Input: dataset, frame size
# Return: page_faults, disk_write, interrupt (int, int, int)
################################################################################
def FN(data=[], frame_num=10):
    # total number of page fault, disk write and interrupt
    page_faults = 0
    disk_write = 0
    interrupt = 0
    # page frames, initialize to -1
    frames = np.full(frame_num, -1)
    # reference & modify bits of frames, initialize to 0
    reference_bits = np.zeros(frame_num)
    modify_bits = np.zeros(frame_num)
    # distance
    dist=[]
    
    for reference in data:
        # page control
        if reference not in frames:
            page_faults+=1
            # page fault: interrupt
            interrupt+=1
            # if still empty frames, put reference to the first empty frame
            if -1 in frames:
                frames[np.where(frames==-1)[0][0]] = reference
            # find the page to replace if no empty frames: first element in Queue
            else:
                dist = cal_distance(frames, reference)
                frame_to_write = dist.index(max(dist))
                frames[frame_to_write] = reference
                # modified: write back to disk + interrupt
                if modify_bits[frame_to_write] == 1:
                    disk_write+=1
                    modify_bits[frame_to_write] = 0
                    interrupt+=1
                # reference bit reset
                reference_bits[frame_to_write] = 0
            
        else:
            # re-reference: usually used
            # hardware set: no need interrupt
            reference_bits[np.where(frames==reference)[0][0]]=1
        # process & hardware behavior modeling
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, reference_bits, 0.9, 0.05, rand_seq)
        rand_seq = np.random.rand(frame_num)
        hw_set_bit(frame_num, modify_bits, 1, 0.5, rand_seq)
        
    return page_faults, disk_write, interrupt


################################################################################
# Type: Plot
# Function: plot the dataset
# Input: dataset to be plotted
# Return: None
################################################################################
def plot_dataset(data=[], size=120000):
    plt.figure(figsize=(12, 6))  # Set the figure size
    plt.scatter(range(1, size+1), data, s=0.05)  # Reduce the line width for large datasets
    plt.title('Locality dataset')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.show()

################################################################################
# Type: Plot
# Function: plot the histogram of a data
# Input: dataset to be plotted
#     bins: bins in plt.hist()
# Return: None
################################################################################
def plot_hist(data=[], bins=None):
    plt.hist(data, bins=bins, edgecolor='black', align='left')
    plt.title('Histogram of one locality with Approximately Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

################################################################################
# Type: Plot
# Function: plot a comparison results of 4 algorithms in one plot
# Input: dataset to be plotted
#     index: target comparison cost- page fault(0); interrupt(2); disk write(1)
# Return: None
################################################################################
def plot_all_algo(data=[], index=0):
    FIFO_result=[]
    optimal_result=[]
    Enhanced_SC_result=[]
    FN_result=[]
    for i in range(10, 110, 10):
        FIFO_result.append(FIFO(data, i)[index])
        optimal_result.append(optimal(data, i)[index])
        Enhanced_SC_result.append(Enhanced_SC(data, i)[index])
        FN_result.append(FN(data, i)[index])

    plt.figure(figsize=(10, 10))

    # Plot each list as a separate line
    frames = list(range(10, 110, 10))
    plt.plot(frames, FIFO_result, label="FIFO")
    plt.plot(frames, optimal_result, label="optimal")
    plt.plot(frames, Enhanced_SC_result, label="Enhanced_SC")
    plt.plot(frames, FN_result, label="FN")

    # Add labels and title
    plt.xlabel("Frames")
    if index == 0:
        plt.ylabel("Page FaultS")
    elif index == 1:
        plt.ylabel("Disk writes")
    elif index == 2:
        plt.ylabel("Interrupts")
    plt.title("Page Replacement Alogrithm Compare")

    # Show legend
    plt.legend()

    # Show the plot
    plt.show()
################################################################################
# Type: Plot
# Function: plot the algorithm result with respect to target data
# Input: data_name: dataset to be plotted
#     algo: page replacement algorithm to be plot
#     data: if provide, use this data instead of generating a new one
# Return: None
################################################################################
def plot_algo_string(data_name='random', algo='FIFO', data=None):
    result_pagefault=[]
    result_interrupt=[]
    result_diskwrite=[]
    if algo == "FIFO":
        A=FIFO
    elif algo == "OPT":
        A=optimal
    elif algo == "ESC":
        A=Enhanced_SC
    elif algo == "FN":
        A=FN
    else:
        raise Exception("No match algo names")
    if data == None:
        if data_name=="random":
            D = gen_data_random()
        elif data_name=="locality":
            D = gen_data_local()
        elif data_name=="GLN":
            D = gen_data_GLN()
        else:
            raise Exception("No match data names")
    else:
        D = data
    for i in range(10, 110, 10):
        result=A(D, i)
        result_pagefault.append(result[0])
        result_interrupt.append(result[2])
        result_diskwrite.append(result[1])

    plt.figure(figsize=(10, 10))

    # Plot each list as a separate line
    frames = list(range(10, 110, 10))
    plt.plot(frames, result_pagefault, color='g', marker='<', mfc='b', label="PAGE FAULT")
    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Page Faults")
    plt.title("Algorithm: " + algo + "; Data:" + data_name)
    # Show legend
    plt.legend()
    # Show the plot
    plt.show()
    #plt.savefig(data_name+algo+"pagefault", bbox_inches='tight')
    # plt.clf()
    plt.plot(frames, result_interrupt, color='g', marker='<', mfc='b', label="INTERRUPT(TIMES)")
    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Interrupts")
    plt.title("Algorithm: " + algo + "; Data:" + data_name)
    # Show legend
    plt.legend()
    # Show the plot
    plt.show()
    # plt.savefig(data_name+algo+"interrupt", bbox_inches='tight')
    # plt.clf()
    plt.plot(frames, result_diskwrite, color='g', marker='<', mfc='b', label="DISK WRITE(PAGES)")
    # Add labels and title
    plt.xlabel("Frames")
    plt.ylabel("Disk writes")
    plt.title("Algorithm: " + algo + "; Data:" + data_name)
    # Show legend
    plt.legend()
    # Show the plot
    plt.show()
    # plt.savefig(data_name+algo+"disk", bbox_inches='tight')
    # plt.clf()
if __name__ == '__main__':
    #plot_dataset(gen_data_random())
    #plot_dataset(gen_data_local())
    
    # data=[7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1]
    # print(FIFO(data, 3))
    # data2=[1,2,3,4,1,2,5,1,2,3,4,5]
    # print(FIFO(data2, 3))
    # print(FIFO(data2, 4))

    # print(FIFO(gen_data_local(), 100))

    #data=[7,0,1,2,0,3,0,4,2,3,0,3,2,1,2,0,1,7,0,1]
    #print(optimal(data, 3))
    print(os.getcwd())
    path="C:\\Users\\WCMCLAB\\Desktop\\AdvanceOS\\HW1\\fig"
    os.chdir(path)
    print(os.getcwd())
    DATA=gen_data_GLN(std_dev=10)
    plot_algo_string(data_name='GLN', algo='FIFO', data=DATA)
    plot_algo_string(data_name='GLN', algo='OPT', data=DATA)
    plot_algo_string(data_name='GLN', algo='ESC', data=DATA)
    plot_algo_string(data_name='GLN', algo='FN', data=DATA)
    
    print(FIFO(DATA, 10))
    print(optimal(DATA, 10))
    print(Enhanced_SC(DATA, 10))
    print(FN(DATA, 10))
    print('---')
    
    print(FIFO(DATA, 50))
    print(optimal(DATA, 50))
    print(Enhanced_SC(DATA, 50))
    print(FN(DATA, 50))
    print('---')

    print(FIFO(DATA, 100))
    print(optimal(DATA, 100))
    print(Enhanced_SC(DATA, 100))
    print(FN(DATA, 100))
    print('---')
