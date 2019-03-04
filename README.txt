Taha Uygun-tuygu3-Randomized Optimization-Assignment 2
Jonathan Tay's code was used as a reference. All rights to Jonathan Tay
This code can be found at: https://github.com/hymntaha/assignment2

Here is the intro how to use directories

Important Notes
1) This project uses a modified version of ABAGAIL, located in the ABAGAIL sub-folder
2) The folders NNOUTPUT, CONTPEAKS, FLIPFLOP and TSP must be created in the same folder as the Jython code before running it.
3) The files m_test.csv, m_trg.csv and m_val.csv  must be in the same folder as the NN*.py files
4) In the main directory you need to run jython example.py command for each python files.

Code Files:
1) NN0.py - Code for backpropagation training of neural network
2) NN1.py - Code for Randomized Hill Climbing training of neural network
3) NN2.py - Code for Simulated Annealing training of neural network
4) NN3.py - Code for Genetic Algorithm training of neural network
5) continuouspeaks.py - Code to use Randomized Optimisation to solve the Continuous Peaks problem
5) flip flop.py - Code to use Randomized Optimisation to solve the Flip Flop problem
6) tsp.py - Code to use Randomized Optimisation to solve the Traveling Salesman Problem

There are also a number of folders
1) Datasets - contains the code to generate the datasets for this assignment from the original files from the UCI ML Repository
2) NNOUTPUT - output folder for the Neural Network experiments
3) CONTPEAKS - output folder for the Continuous Peaks experiments
4) FLIPFLOP - output folder for the Flip Flop experiments
5) TSP - output folder for the Traveling Salesman Problem experiments
6) ABAGAIL - folder with source, ant build file, and jar for ABAGAIL

Data Files
1) m_test.csv - The test set
2) m_trg.csv - The training set
3) m_val.csv - The validation set

In order to generate plots, you need to run plot_nn.py and plot_opt via python 3. plot_nn was used for neural network plots and plot_opt for optimization problem.

To download the dataset:
https://archive.ics.uci.edu/ml/datasets/wine+quality

To generate the data files from the original data, run the parse_data.py code and the DUMPER.py code in the Datasets folder. The data files should then be moved one level up, to reside in the same directory as the assignment 2 code files.
The data file code was written in Python 3.5, using Pandas 0.18.0 and sklearn 0.19.1

Java code was built with ant 1.10.1 on java 1.8.0_121. 
The code files in the code files section were written in Jython 2.7.0. 

Plotting code was  written in Python 3.5, using Pandas 0.18.0 and matplotlib 1.5.1

Within the output folders, the data files are csv files (with .txt extensions). The file names correspond to experiments:
<ALGORITHM><PARAMETERS>_LOG<_TRIAL NUMBER>.txt

In the data folders, there are also summary files (ave.txt and table.xlsx). These contain summaries of the data.


