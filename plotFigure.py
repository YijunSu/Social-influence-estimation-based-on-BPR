# coding=utf-8

import pylab as pl
import matplotlib.lines as mline

figure_1_x = []
figure_1_y = []
figure_2_x = []
figure_2_y = []
figure_3_x = [10, 20, 30, 40, 50]
figure_3_y = [0.532384927484, 0.526696289197, 0.4912337458, 0.5137537654232, 0.5223737246]

iter_num = 5000
with open('factor_num&auc.txt') as f:
    for line in f:
        result = line.split()
        figure_1_x.append(int(result[0]))
        figure_1_y.append(float(result[1]))


with open('factor_num_and_auc_event.txt') as f:
    for line in f:
        result = line.split()
        figure_2_x.append(int(result[0]))
        figure_2_y.append(float(result[1]))

plot_1 = pl.plot(figure_1_x, figure_1_y, 'ro')
pl.title('BPR without neighbor')
pl.xlabel('factor num')
pl.ylabel('AUC')

pl.xlim(0, 60)
pl.ylim(0.0, 1.0)

pl.show()

plot_2 = pl.plot(figure_2_x, figure_2_y, 'go')
pl.title('BPR with event neighbor')
pl.xlabel('factor num')
pl.ylabel('AUC')

pl.xlim(0, 60)
pl.ylim(0.0, 1.0)

pl.show()

plot_1_line = pl.plot(figure_1_x, figure_1_y, 'r')
plot_2_line = pl.plot(figure_2_x, figure_2_y, 'g')
plot_3_line = pl.plot(figure_3_x, figure_3_y, 'b')
pl.title('model compare')
pl.xlabel('factor num')
pl.ylabel('AUC')

pl.xlim(0, 60)
pl.ylim(0.0, 1.0)

red_line = mline.Line2D([], [], color='red', label='BPR without neighbor')
green_line = mline.Line2D([], [], color='green', label='BPR with event neighbor')
blue_line = mline.Line2D([], [], color='blue', label='MF with event neighbor')
pl.legend([plot_1_line, plot_2_line, plot_3_line], handles=[red_line, green_line, blue_line])
pl.show()
