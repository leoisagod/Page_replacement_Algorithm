from utils import *

if __name__ == '__main__':
    print("################################################################################")
    print("#         Advanced Operating System HW1 - Page Replacement Algorithm           #")
    print("################################################################################\n")
    d_option = int(input("Choose a reference string:\n1) random  2) locality  3) Gaussian-distributed locality\n"))
    if d_option == 1:
        D=gen_data_random()
    elif d_option == 2:
        D=gen_data_local()
    elif d_option == 3:
        D=gen_data_GLN(std_dev=2.5)
    else:
        raise("Invalid option!")
    print("plotting the dataset...\n")
    plot_dataset(D)

    a_option = int(input("Choose an algorithm\n1) FIFO  2) optimal  3) ESC  4) Farthest neighbor  5) ALL\n"))
    print("plotting the result... this may take a while\n")
    if a_option == 1:
        plot_algo_string(algo='FIFO', data=D)
    elif a_option == 2:
        plot_algo_string(algo='OPT', data=D)
    elif a_option == 3:
        plot_algo_string(algo='ESC', data=D)
    elif a_option == 4:
        plot_algo_string(algo='FN', data=D)
    elif a_option == 5:
        plot_all_algo(data=D, index=0)
        plot_all_algo(data=D, index=2)
        plot_all_algo(data=D, index=1)
    else:
        raise("Invalid option!")
