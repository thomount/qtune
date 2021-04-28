import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rcParams

from configs import parse_cmd_args

def draw_lines(filelist, metric_pos, metric_name):
    '''
    multiple lines on the same metric (y) with increasing iterations (x)
    :param filelist:
    :param metric_pos:
    :param metric_name:
    :return: 1 (succeed)/0 (fail)
    '''

    ''' Load Data: [QTune] '''
    col_list = ["Iteration", metric_name]
    df = pd.read_csv("training-results/"+filelist[0], usecols=col_list, sep = "\t")
    print(df[col_list[0]])
    x_qtune = list(df[col_list[0]])
    x_qtune = [int(x) for x in x_qtune]
    y_qtune = list(df[col_list[1]])
    y_qtune = [float(y) for y in y_qtune]

    ''' Load Data: [Random] '''
    col_list = ["Iteration", metric_name]
    df = pd.read_csv("training-results/"+filelist[1], usecols=col_list, sep = "\t")
    x_random = list(df[col_list[0]])
    x_random = [int(x) for x in x_random]
    y_random = list(df[col_list[1]])
    y_random = [float(y) for y in y_random]

    ''' figure drawing '''
    mpl.rcdefaults()
    rcParams.update({
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 15,
        #     'figure.autolayout': True,
        'figure.subplot.hspace': 0.45,
        'figure.subplot.wspace': 0.22,
        #     'mathtext.fontset': 'cm',
    })

    fig = plt.figure()

    qid = 1
    ax = fig.add_subplot(1, 1, qid)


    rf = np.array(y_qtune)
    dt = np.array(y_random)

    x = np.arange(1, max(len(x_random), len(x_qtune))+5)
    # y = np.arange(0.0, 1.0)

    l1, = plt.plot(x_qtune, rf[:len(x_qtune)], marker='D', ms=3, linewidth=2)
    l2, = plt.plot(x_random, dt[:len(x_random)], marker='X', ms=3, linewidth=2)

    ax.text(0.5, -0.36,
            f"({chr(ord('a') + qid - 1)}) $D_{{ {qid} }}$",
            horizontalalignment='center', transform=ax.transAxes, fontsize=15, family='serif',
            )
    ax.set_xticks(np.arange(0, len(x), len(x)/10))
    if metric_name == 'Latency':
        y_range = max(max(y_qtune),max(y_random))+5
    elif metric_name == 'throughput':
        y_range = max(max(y_qtune), max(y_random)) + 1000

    ax.set_yticks(np.arange(0, y_range, y_range/10))
    ax.set_ylim(0, y_range)
    ax.set_xlim(0, len(x))
    ax.set_xlabel('#-Iterations')
    ax.set_ylabel('Performance')

    #     legend = plt.legend([l1, l2, l3, l4], ['IQS', 'AQS', 'LTR', 'SSY'], loc = 'lower right', markerscale=1, ncol = 2, fontsize=16, framealpha=0.5,prop={'size': 16})

    #     plt.tight_layout()

    # plt.savefig(fname='fig/hybrid'+str(qid)+'.pdf', dpi=600, quality=95, transparent=True)

    fig.legend([l1, l2], ['QTune', 'Random'],
               loc='upper center', ncol=4,
               handlelength=3,
               columnspacing=6.,
               bbox_to_anchor=(0., 0.98, 1., .0),
               bbox_transform=plt.gcf().transFigure,
               fontsize=10,
               )
    # plt.savefig("/Users/ztypl/Documents/Paper/exploration/sigmod 2020/figs/experiment/decision.pdf",bbox_inches = 'tight',
    #    pad_inches = 0.)
    # plt.show()

    plt.savefig('training-results/training.png')

    return 1

if __name__ == '__main__':
    argus = parse_cmd_args()
    mark = draw_lines(argus['linelist'], 3, 'Latency')
    if mark:
        print('Successfully update figure!')
    else:
        print('Fail to update figure!')

