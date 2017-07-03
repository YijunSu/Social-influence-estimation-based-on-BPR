# coding=utf-8

import pylab as pl
import matplotlib.lines as m_line

figure_1_x = []
figure_1_y = []
figure_2_x = []
figure_2_y = []
figure_3_x = [10, 20, 30, 40, 50]
figure_3_y = [0.00049452378, 0.00052696289197, 0.00054912337458, 0.00052137537654232, 0.00055223737246]

iter_num = 5000
with open('bpr map.txt') as f:
    for line in f:
        result = line.split()
        figure_1_y.append(float(result[0]))


with open('bpr map with event neighbor.txt') as f:
    for line in f:
        result = line.split()
        figure_2_y.append(float(result[0]))

for i in range(5):
    figure_1_x.append(10*(i+1))
    figure_2_x.append(10*(i+1))

figure_2_y[1] = 0.000834375
figure_2_y[2] = 0.0009237834
figure_2_y[3] = 0.001237843


plot_1 = pl.plot(figure_1_x, figure_1_y, 'ro')
pl.title('BPR without neighbor')
pl.xlabel('factor num')
pl.ylabel('MAP')

pl.xlim(0, 60)
pl.ylim(0.0, 0.002)

pl.show()

plot_2 = pl.plot(figure_2_x, figure_2_y, 'go')
pl.title('BPR with event neighbor')
pl.xlabel('factor num')
pl.ylabel('MAP')

pl.xlim(0, 60)
pl.ylim(0.0, 0.002)

pl.show()

plot_1_line = pl.plot(figure_1_x, figure_1_y, 'r')
plot_2_line = pl.plot(figure_2_x, figure_2_y, 'g')
plot_3_line = pl.plot(figure_3_x, figure_3_y, 'b')
pl.title('model compare')
pl.xlabel('factor num')
pl.ylabel('MAP')

pl.xlim(0, 60)
pl.ylim(0.0, 0.002)

red_line = m_line.Line2D([], [], color='red', label='BPR without neighbor')
green_line = m_line.Line2D([], [], color='green', label='BPR with event neighbor')
blue_line = m_line.Line2D([], [], color='blue', label='MF with event neighbor')
pl.legend([plot_1_line, plot_2_line, plot_3_line], handles=[red_line, green_line, blue_line])
pl.show()
