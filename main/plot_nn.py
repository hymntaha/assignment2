# Import Data
import pandas as pd
import matplotlib.pyplot as plt

CH = dict()
CH['title'] = 'White Wine'
CH['RHC_data'] = pd.read_csv('./NN_OUTPUT/RHC_LOG.csv',error_bad_lines=False)
CH['SA_data'] = pd.read_csv('./NN_OUTPUT/SA0.55_LOG.csv',error_bad_lines=False)
CH['GA_data'] = pd.read_csv('./NN_OUTPUT/GA_100_20_20_LOG.csv',error_bad_lines=False)
CH['BP_data'] = pd.read_csv('./NN_OUTPUT/BACKPROP_LOG.csv')

problems = [CH]

def learning_curve(prob):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.gca()
    plt.title(prob['title'] + ' Learning Curve (Accuracy)')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    ax.plot('iteration', 'acc_tst', data=prob['RHC_data'].iloc[:1200], label="RHC", color="deeppink")
    ax.plot('iteration', 'acc_tst', data=prob['SA_data'].iloc[:1200], label="SA", color="deepskyblue")
    ax.plot('iteration', 'acc_tst', data=prob['GA_data'].iloc[:1200], label="GA", color="lime")
    ax.plot('iteration', 'acc_tst', data=prob['BP_data'].iloc[:1200], label="BP", color="darkred")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=4, fancybox=True, shadow=True)


    plt.savefig('./OUTPUT/Complexity_Accuracy-{}.png'.format(prob['title']),bbox_inches="tight")
    plt.show()

    acc = dict()
    acc['RHA'] = round(prob['RHC_data'].acc_tst.iloc[-1],3)
    acc['SA'] = round(prob['SA_data'].acc_tst.iloc[-1],3)
    acc['GA']= round(prob['GA_data'].acc_tst.iloc[-1],3)
    acc['BP'] = round(prob['BP_data'].acc_tst.iloc[-1],3)
    print(acc)

for problem in problems:
    learning_curve(problem)

    import matplotlib.pyplot as plt

def learning_curve_err(prob):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.subplot()
    plt.title(prob['title'] + ' Learning Curve (Mean Squared Error)')
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.grid()
    ax.plot('iteration', 'MSE_tst', data=prob['RHC_data'].iloc[:1200], label="RHC", color="deeppink")
    ax.plot('iteration', 'MSE_tst', data=prob['SA_data'].iloc[:1200], label="SA", color="deepskyblue")
    ax.plot('iteration', 'MSE_tst', data=prob['GA_data'].iloc[:1200], label="GA", color="lime")
    ax.plot('iteration', 'MSE_tst', data=prob['BP_data'].iloc[:1200], label="BP", color="darkred")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=4, fancybox=True, shadow=True)


    plt.savefig('./OUTPUT/LC_Error-{}.png'.format(prob['title']),bbox_inches="tight")
    plt.show()
    error = dict()
    error['RHA'] = round(prob['RHC_data'].MSE_tst.iloc[-1],3)
    error['SA'] = round(prob['SA_data'].MSE_tst.iloc[-1],3)
    error['GA']= round(prob['GA_data'].MSE_tst.iloc[-1],3)
    error['BP'] = round(prob['BP_data'].MSE_tst.iloc[-1],3)
    print(error)

for problem in problems:
    learning_curve_err(problem)

    import matplotlib.pyplot as plt

def learning_curve_time(prob):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.subplot()
    plt.title(prob['title'] + ' Learning Curve (Computation Time)')
    plt.xlabel("Iterations")
    plt.ylabel("Computation Time (s)")
    plt.grid()
    ax.plot('iteration', 'elapsed', data=prob['RHC_data'].iloc[:1200], label="RHC", color="deeppink")
    ax.plot('iteration', 'elapsed', data=prob['SA_data'].iloc[:1200], label="SA", color="deepskyblue")
    ax.plot('iteration', 'elapsed', data=prob['GA_data'].iloc[:1200], label="GA", color="lime")
    ax.plot('iteration', 'elapsed', data=prob['BP_data'].iloc[:1200], label="BP", color="darkred")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=4, fancybox=True, shadow=True)

    plt.savefig('./OUTPUT/LC_CompTime-{}.png'.format(prob['title']),bbox_inches="tight")
    plt.show()
    time = dict()
    time['RHA'] = round(prob['RHC_data'].elapsed.iloc[-1],3)
    time['SA'] = round(prob['SA_data'].elapsed.iloc[-1],3)
    time['GA']= round(prob['GA_data'].elapsed.iloc[-1],3)
    time['BP'] = round(prob['BP_data'].elapsed.iloc[-1],3)
    print(time)

for problem in problems:
    learning_curve_time(problem)

    GAc = dict()
GAc['title'] = 'White Wine-Genetic Algorithm'
GAc['saveas'] = 'GA'
GAc['xlabel'] = 'Complexity'
GAc['files'] = dict()
GAc['files']['100-10-10'] = pd.read_csv('./NN_OUTPUT/GA_100_10_10_LOG.csv')
GAc['files']['100-10-20'] = pd.read_csv('./NN_OUTPUT/GA_100_10_20_LOG.csv')
GAc['files']['100-20-10'] = pd.read_csv('./NN_OUTPUT/GA_100_20_10_LOG.csv')
GAc['files']['100-20-20'] = pd.read_csv('./NN_OUTPUT/GA_100_20_20_LOG.csv')
GAc['files']['50-10-10'] = pd.read_csv('./NN_OUTPUT/GA_50_10_10_LOG.csv')
GAc['files']['50-10-20'] = pd.read_csv('./NN_OUTPUT/GA_50_10_20_LOG.csv')
GAc['files']['50-20-10'] = pd.read_csv('./NN_OUTPUT/GA_50_20_10_LOG.csv')
GAc['files']['50-20-20'] = pd.read_csv('./NN_OUTPUT/GA_50_20_20_LOG.csv')

SAc = dict()
SAc['title'] = 'White Wine-SimulatedAnnealing'
SAc['saveas'] = 'SA'
SAc['xlabel'] = 'Complexity'
SAc['files'] = dict()
SAc['files']['0.15'] = pd.read_csv('./NN_OUTPUT/SA0.15_LOG.csv',error_bad_lines=False)
SAc['files']['0.35'] = pd.read_csv('./NN_OUTPUT/SA0.35_LOG.csv',error_bad_lines=False)
SAc['files']['0.55'] = pd.read_csv('./NN_OUTPUT/SA0.55_LOG.csv',error_bad_lines=False)
SAc['files']['0.7'] = pd.read_csv('./NN_OUTPUT/SA0.7_LOG.csv')
SAc['files']['0.95'] = pd.read_csv('./NN_OUTPUT/SA0.95_LOG.csv',error_bad_lines=False)


problemsc = [GAc,SAc]

def acc_comp(problem):
    complexities = []
    acc_vals = []
    acc_c = dict()
    for key in problem['files']:
        df = problem['files'][key]
        end_acc_val = round(df.acc_tst.iloc[-1],3)
        complexities.append(key)
        acc_vals.append(end_acc_val)
        acc_c[key] = end_acc_val

    fig = plt.figure()
    fig.set_size_inches(10, 3)
    plt.bar(complexities, acc_vals, align='center', width = 0.1, color='darkmagenta')
    #plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.xlabel(problem['xlabel'])
    plt.grid()
    plt.title(problem['title']+' Accuracy Vs. Complexity Graph')
    plt.ylim(min(acc_vals)*.9, max(acc_vals)*1.1)
    plt.savefig('./OUTPUT/Accuracy_Hist-{}.png'.format(problem['saveas']),bbox_inches="tight")
    plt.show()
    print(acc_c)

for problem in problemsc:
    acc_comp(problem)

    def acc_comp_curve(problem):
        fig = plt.figure()
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.subplot()
    plt.title(problem['title'] + ' Learning/Complexity Curve (Accuracy)')
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid()
    for key in problem['files']:
        ax.plot('iteration', 'acc_trg', data=problem['files'][key].iloc[:2000], label=key)

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=5, fancybox=True, shadow=True)


    plt.savefig('./OUTPUT/Accuracy-{}.png'.format(problem['title']),bbox_inches="tight")
    plt.show()


for problem in problemsc:
    acc_comp_curve(problem)


def err_comp(problem):
    complexities = []
    err_vals = []
    err_c = dict()
    for key in problem['files']:
        df = problem['files'][key]
        end_err_val = round(df.MSE_tst.iloc[-1],3)
        complexities.append(key)
        err_vals.append(end_err_val)
        err_c[key] = end_err_val

    fig = plt.figure()
    fig.set_size_inches(10, 3)
    plt.bar(complexities, err_vals, align='center',  width = 0.1,  color='darkmagenta')
    plt.ylabel('Mean Squared Error')
    plt.xlabel(problem['xlabel'])
    plt.grid()
    plt.title(problem['title']+' Mean Squared Error Vs. Complexity Graph')
    plt.ylim(min(err_vals)*.9, max(err_vals)*1.1)
    plt.savefig('./OUTPUT/CC_Error_Hist-{}.png'.format(problem['saveas']),bbox_inches="tight")
    plt.show()
    print(err_c)

for problem in problemsc:
    err_comp(problem)

def time_comp(problem):
    complexities = []
    time_vals = []
    time_c = dict()
    for key in problem['files']:
        df = problem['files'][key]
        end_time_val = round(df.elapsed.iloc[-1],2)
        complexities.append(key)
        time_vals.append(end_time_val)
        time_c[key] = end_time_val

    fig = plt.figure()
    fig.set_size_inches(10, 3)
    plt.bar(complexities, time_vals, align='center', width = 0.1, color='darkmagenta')
    plt.ylabel('Computation Time (s)')
    plt.xlabel(problem['xlabel'])
    plt.grid()
    plt.title(problem['title']+' Computation Time Vs. Complexity Graph')
    plt.ylim(min(time_vals)*.9, max(time_vals)*1.1)
    plt.savefig('./OUTPUT/ComputationTime_Hist-{}.png'.format(problem['saveas']),bbox_inches="tight")
    plt.show()
    print(time_c)

for problem in problemsc:
    time_comp(problem)

######################### Training vs Test #############################################
########################################################################################
########################################################################################
#############################################################################################


import matplotlib.pyplot as plt
import pandas


def learning(iterations, train_scores, test_scores, title):

    plt.figure()
    plt.title(title)
    plt.ylim((.3, 1.01))

    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.plot(iterations, train_scores, color="deeppink",
             label="Training")

    plt.plot(iterations, test_scores, color="blue",
             label="Test")

    plt.legend(loc="best")
    return plt

def curves(iterations, valLabels, valIndex, title, yLabel):
    plt.figure()
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(yLabel)

    plt.grid()

    colors = ['deeppink', 'deepskyblue', 'darkred', 'lime', 'black']
    for i in range(len(valLabels)):
        val = valLabels[i][valIndex]
        label = valLabels[i][-1]
        color = colors[i]

        plt.plot(iterations, val, color=color,
                 label=label)

    plt.legend(loc="best")
    return plt

def timing(iterations, timeDuration, title):

    plt.figure()
    plt.title(title)

    plt.xlabel("Iterations")
    plt.ylabel("Complexity")

    plt.grid()

    plt.plot(iterations, timeDuration, 'o-', color="dodgerblue",
             label="Time")

    plt.legend(loc="best")
    return plt

def plot(dataset_path, title):
    df = pandas.read_csv(dataset_path)
    plot = learning(df['iteration'], df['acc_tst'], df['acc_trg'], title)
    plot.savefig('OUTPUT/nn_training/' + title.replace(' ', '_').replace('.', 'pt').lower() + '.png')
    plot.close()

    plot = timing(df['iteration'], df['elapsed'], title + ' Training Time')
    plot.savefig('OUTPUT/nn_training/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.png')
    plot.close()

def dataframes_load(path):
    return pandas.read_csv(path)

def plot_peaks(title):
    valLabels = []

    df1 = dataframes_load('CONTPEAKS/CONTPEAKS_RHC_1_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'RHC'))

    df2 = dataframes_load('CONTPEAKS/CONTPEAKS_SA0.55_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'SA'))

    df3 = dataframes_load('CONTPEAKS/CONTPEAKS_GA100_50_50_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'GA'))

    df4 = dataframes_load('CONTPEAKS/CONTPEAKS_MIMIC100_50_0.9_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.png')

    plot = curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.png')

    plot = curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.png')

    plot.close()

def plot_tsp(title):
    valLabels = []

    df1 = dataframes_load('TSP/TSP_RHC_1_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'Randomized Hill Climbing'))

    df2 = dataframes_load('TSP/TSP_SA0.55_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'Simulated Annealing'))

    df3 = dataframes_load('TSP/TSP_GA100_30_30_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'Genetic Algorithm'))

    df4 = dataframes_load('TSP/TSP_MIMIC100_50_0.5_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.png')

    plot = curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.png')

    plot = curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.png')

    plot.close()

def plot_flipflop(title):
    valLabels = []

    df1 = dataframes_load('FLIPFLOP/FLIPFLOP_RHC_1_LOG.txt')
    valLabels.append((df1['fitness'], df1['time'], df1['fevals'], 'Randomized Hill Climbing'))

    df2 = dataframes_load('FLIPFLOP/FLIPFLOP_SA0.15_1_LOG.txt')
    valLabels.append((df2['fitness'], df2['time'], df2['fevals'], 'Simulated Annealing'))

    df3 = dataframes_load('FLIPFLOP/FLIPFLOP_GA100_30_30_1_LOG.txt')
    valLabels.append((df3['fitness'], df3['time'], df3['fevals'], 'Genetic Algorithm'))

    df4 = dataframes_load('FLIPFLOP/FLIPFLOP_MIMIC100_50_0.5_1_LOG.txt')
    valLabels.append((df4['fitness'], df4['time'], df4['fevals'], 'MIMIC'))

    plot = curves(df1['iterations'], valLabels, 0, title + ' Fitness', 'Fitness Function')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_fitness.png')

    plot = curves(df1['iterations'], valLabels, 1, title + ' Time', 'Training Time (s)')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_time.png')

    plot = curves(df1['iterations'], valLabels, 2, title + ' Evals', 'Function Evals')
    plot.savefig('OUTPUT_OPT/OPTIMIZE/' + title.replace(' ', '_').replace('.', 'pt').lower() + '_evals.png')

    plot.close()

plot('NN_OUTPUT/BACKPROP_LOG.csv', 'Backprop NN')

plot('NN_OUTPUT/RHC_LOG.csv', 'Randomized Hill Climbing NN')

plot('NN_OUTPUT/SA0.15_LOG.csv', 'Simulated Annealing Cooling .15 NN')
plot('NN_OUTPUT/SA0.35_LOG.csv', 'Simulated Annealing Cooling .35 NN')
plot('NN_OUTPUT/SA0.55_LOG.csv', 'Simulated Annealing Cooling .55 NN')
plot('NN_OUTPUT/SA0.7_LOG.csv', 'Simulated Annealing Cooling .7 NN')
plot('NN_OUTPUT/SA0.95_LOG.csv', 'Simulated Annealing Cooling .95 NN')

plot('NN_OUTPUT/GA_50_10_10_LOG.csv', 'Pop 50, Genetic 10 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_50_10_20_LOG.csv', 'Pop 50, Genetic 10 Mate, 20 Mutate NN')
plot('NN_OUTPUT/GA_50_20_10_LOG.csv', 'Pop 50, Genetic 20 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_50_20_20_LOG.csv', 'Pop 50, Genetic 20 Mate, 20 Mutate NN')
plot('NN_OUTPUT/GA_100_10_10_LOG.csv', 'Pop 100, Genetic 10 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_100_10_20_LOG.csv', 'Pop 100, Genetic 10 Mate, 20 Mutate NN')
plot('NN_OUTPUT/GA_100_20_10_LOG.csv', 'Pop 100, Genetic 20 Mate, 10 Mutate NN')
plot('NN_OUTPUT/GA_100_20_20_LOG.csv', 'Pop 100, Genetic 20 Mate, 20 Mutate NN')

plot_peaks('Continous Peaks')
plot_tsp('Traveling Salesman')
plot_flipflop('Flip Flop')
