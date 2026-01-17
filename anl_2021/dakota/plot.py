#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt

# input: fileName xcol
def main():
    xCol = sys.argv[-1]
    files = sys.argv[1:-1]
    print(files)

    for file in files:
        draw(file, xCol)

def draw(fileName, xCol):
    with open(fileName, 'r') as f:
        jar = f.readlines()

    #xcol = 2 + 1
    xcol = int(xCol) + 1
    ycol = 10 + 1

    xtag = jar[0].split()[xcol]
    ytag = jar[0].split()[ycol]

    x = []; y = []
    for line in jar[1:]:
        tmp = line.split()
        #if True:
        if float(tmp[ycol]) < 100:
            x.append(float(tmp[xcol]))
            y.append(float(tmp[ycol]))

    #hor = [62.667 for _ in x]
    #plt.plot(x, hor, c='r')
    #plt.scatter(x, y)
    #plt.xlabel(xtag)
    #plt.ylabel(ytag)
    #if int(xCol) > 2:
    #    plt.xscale('log')
    #plt.tight_layout()
    #plt.show()

    plt.hist(y, density=True)
    plt.plot([63, 63], [0, 0.06])
    plt.xlabel('Fuel Gas Swelling')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
