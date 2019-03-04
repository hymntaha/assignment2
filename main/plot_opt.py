import pandas as pd
import matplotlib.pyplot as plt

FF = dict()
FF['title'] = 'FlipFlop'
FF['RHC_data'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_RHC_1_LOG.txt')
FF['SA_data'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_SA0.55_1_LOG.txt')
FF['GA_data'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_GA100_50_50_1_LOG.txt')
FF['MIMIC_data'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.9_1_LOG.txt')

TS = dict()
TS['title'] = 'Traveling Salesman'
TS['RHC_data'] = pd.read_csv('./TSP/TSP_RHC_1_LOG.txt')
TS['SA_data'] = pd.read_csv('./TSP/TSP_SA0.55_2_LOG.txt')
TS['GA_data'] = pd.read_csv('./TSP/TSP_GA100_50_50_2_LOG.txt')
TS['MIMIC_data'] = pd.read_csv('./TSP/TSP_MIMIC100_50_0.9_2_LOG.txt')

### N=100
# CP = dict()
# CP['title'] = 'Continuous_Peaks'
# CP['RHC_data'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_RHC_1_LOG.txt')
# CP['SA_data'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.55_1_LOG.txt')
# CP['GA_data'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_GA100_50_50_1_LOG.txt')
# CP['MIMIC_data'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_MIMIC100_50_0.7_1_LOG.txt')

### N=200
CP = dict()
CP['title'] = 'Continuous_Peaks'
CP['RHC_data'] = pd.read_csv('./CONTPEAKS1/CONTPEAKS_RHC_1_LOG.txt')
CP['SA_data'] = pd.read_csv('./CONTPEAKS1/CONTPEAKS_SA0.55_1_LOG.txt')
CP['GA_data'] = pd.read_csv('./CONTPEAKS1/CONTPEAKS_GA100_50_50_1_LOG.txt')
CP['MIMIC_data'] = pd.read_csv('./CONTPEAKS1/CONTPEAKS_MIMIC100_50_0.7_1_LOG.txt')

problems = [FF,TS,CP]


def learning_curve(prob):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.subplot()
    plt.title(prob['title'] + ' Learning Graph')
    plt.grid()
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Score")

    ax.plot('iterations', 'fitness', data=prob['RHC_data'], label="RHC", color="mediumblue")
    ax.plot('iterations', 'fitness', data=prob['SA_data'], label="SA", color="darkred")
    ax.plot('iterations', 'fitness', data=prob['GA_data'], label="GA", color="lime")
    ax.plot('iterations', 'fitness', data=prob['MIMIC_data'], label="MIMIC", color="magenta")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=4, fancybox=True, shadow=True)


    plt.savefig('./OUTPUT_OPT/N_200_Fitness-{}.png'.format(prob['title']),bbox_inches="tight")
    plt.show()
    time = dict()
    time['RHA'] = round(prob['RHC_data'].fitness.iloc[-1],3)
    time['SA'] = round(prob['SA_data'].fitness.iloc[-1],3)
    time['GA']= round(prob['GA_data'].fitness.iloc[-1],3)
    time['MIMIC'] = round(prob['MIMIC_data'].fitness.iloc[-1],3)
    print(time)

for problem in problems:
    learning_curve(problem)

def comp_curve(prob):
    fig = plt.figure()
    fig.set_size_inches(10, 5)
    ax = plt.subplot()
    plt.title(prob['title'] + ' Computation Time Graph')
    plt.grid()
    plt.xlabel("Iterations")
    plt.ylabel("Computation Time (s)")

    ax.plot('iterations', 'time', data=prob['RHC_data'], label="RHC", color="mediumblue")
    ax.plot('iterations', 'time', data=prob['SA_data'], label="SA", color="darkred")
    ax.plot('iterations', 'time', data=prob['GA_data'], label="GA", color="lime")
    ax.plot('iterations', 'time', data=prob['MIMIC_data'], label="MIMIC", color="magenta")

    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -.3),
                       ncol=4, fancybox=True, shadow=True)


    plt.savefig('./OUTPUT_OPT/CompTime-{}.png'.format(prob['title']),bbox_inches="tight")
    plt.show()

    time = dict()
    time['RHA'] = round(prob['RHC_data'].time.iloc[-1],3)
    time['SA'] = round(prob['SA_data'].time.iloc[-1],3)
    time['GA']= round(prob['GA_data'].time.iloc[-1],3)
    time['MIMIC'] = round(prob['MIMIC_data'].time.iloc[-1],3)
    print(time)


for problem in problems:
    comp_curve(problem)

flipflop = dict()
flipflop['title'] = 'FlipFlop - MIMIC - 3000 Iters. -'
flipflop['saveas'] = 'FlipFlop'
flipflop['xlabel'] = 'Complexity'
flipflop['files'] = dict()
flipflop['files']['100-50-0.1'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.1_1_LOG.txt')
flipflop['files']['100-50-0.3'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.3_1_LOG.txt')
flipflop['files']['100-50-0.5'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.5_1_LOG.txt')
flipflop['files']['100-50-0.7'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.7_1_LOG.txt')
flipflop['files']['100-50-0.9'] = pd.read_csv('./FLIPFLOP/FLIPFLOP_MIMIC100_50_0.9_1_LOG.txt')

traveling = dict()
traveling['title'] = 'Traveling Saleman - Genetic Algorithm - 3000 Iters. -'
traveling['saveas'] = 'Traveling Salesman'
traveling['xlabel'] = 'Complexity'
traveling['files'] = dict()
traveling['files']['100-10-10'] = pd.read_csv('./TSP/TSP_GA100_10_10_1_LOG.txt')
traveling['files']['100-10-30'] = pd.read_csv('./TSP/TSP_GA100_10_30_1_LOG.txt')
traveling['files']['100-10-50'] = pd.read_csv('./TSP/TSP_GA100_10_50_1_LOG.txt')
traveling['files']['100-30-10'] = pd.read_csv('./TSP/TSP_GA100_30_10_1_LOG.txt')
traveling['files']['100-30-30'] = pd.read_csv('./TSP/TSP_GA100_30_30_1_LOG.txt')
traveling['files']['100-30-50'] = pd.read_csv('./TSP/TSP_GA100_30_50_1_LOG.txt')
traveling['files']['100-50-10'] = pd.read_csv('./TSP/TSP_GA100_50_10_1_LOG.txt')
traveling['files']['100-50-30'] = pd.read_csv('./TSP/TSP_GA100_50_30_1_LOG.txt')
traveling['files']['100-50-50'] = pd.read_csv('./TSP/TSP_GA100_50_50_1_LOG.txt')

peak = dict()
peak['title'] = 'Continuous Peaks - Simulated Annealing - 3000 Iters. -'
peak['saveas'] = 'Continuous_Peaks'
peak['xlabel'] = 'Complexity'
peak['files'] = dict()
peak['files']['0.15'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.15_1_LOG.txt')
peak['files']['0.35'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.35_1_LOG.txt')
peak['files']['0.55'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.55_1_LOG.txt')
peak['files']['0.75'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.75_1_LOG.txt')
peak['files']['0.95'] = pd.read_csv('./CONTPEAKS/CONTPEAKS_SA0.95_1_LOG.txt')

problemsc = [flipflop,traveling,peak]

def fit_comp(problem):
    complexities = []
    fit_vals = []
    fit_c = dict()
    for key in problem['files']:
        df = problem['files'][key]
        end_fit_val = df.fitness.iloc[-1]
        complexities.append(key)
        fit_vals.append(end_fit_val)
        fit_c[key] = round(end_fit_val,3)

    fig = plt.figure()
    fig.set_size_inches(10, 5)
    plt.bar(complexities, fit_vals, align='center', width = 0.1, color='darkmagenta')

    plt.ylabel('Fitness Score')
    plt.xlabel(problem['xlabel'])
    plt.title(problem['title']+' Fitness Score Vs. Complexity Graph')
    plt.ylim(min(fit_vals)*.9, max(fit_vals)*1.1)
    plt.savefig('./OUTPUT_OPT/Complexity_Fitness-{}.png'.format(problem['saveas']),bbox_inches="tight")
    plt.show()
    print(fit_c)

for problem in problemsc:
    fit_comp(problem)

def time_comp(problem):
    complexities = []
    time_vals = []
    time_c = dict()
    for key in problem['files']:
        df = problem['files'][key]
        end_time_val = df.time.iloc[-1]
        complexities.append(key)
        time_vals.append(end_time_val)
        time_c[key] = round(end_time_val,3)

    fig = plt.figure()
    fig.set_size_inches(10, 5)
    plt.bar(complexities, time_vals, align='center', width = 0.1, color='darkmagenta')
    plt.ylabel('Computational Time (s)')
    plt.xlabel(problem['xlabel'])
    plt.title(problem['title']+' Computation Time Vs Complexity Graph')
    plt.ylim(min(time_vals)*.9, max(time_vals)*1.1)
    plt.savefig('./OUTPUT_OPT/Complexity_ComputationTime-{}.png'.format(problem['saveas']),bbox_inches="tight")
    plt.show()
    print(time_c)

for problem in problemsc:
    time_comp(problem)
