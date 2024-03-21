# Current working code

import random as rnd
import matplotlib.pyplot as plt
import toml
import numpy as np

def readconfig(file:str):
    # Load the TOML file
    with open(file, 'r') as config_file:
        config = toml.load(config_file)
    
    # Plotting
    labels = config['plotting']['labels']
    color_pts = config['plotting']['color_pts']
    color_prts = config['plotting']['color_prts']
    marker = config['plotting']['marker']
    font = config['plotting']['font']
    figsize = config['plotting']['figsize']
    saver = config['plotting']['saver']
    savetitle = config['plotting']['savetitle']
    dpi = config['plotting']['dpi']
    
    return labels, color_pts, color_prts, marker, font, figsize, saver, savetitle, dpi




def plot_structure(points:list, parts:list, labels, color_pts, color_prts, marker, font, figsize, saver, savetitle, dpi):

    scp = 0  # to scale points
    scl = 0

    if len(points[0]) > 150:
        scp = 2
        if len(points[0]) > 300:
            scp = 4
            if len(points[0]) > 700:
                scp = 6
                scl = 0.5

    plt.figure(figsize=figsize)
    # plot parts
    for h in range(len(parts[0])):

        plt.plot((points[0][parts[0][h]-1], points[0][parts[1][h]-1]), 
                (points[1][parts[0][h]-1], points[1][parts[1][h]-1]),  color = color_prts, marker=marker, linewidth=2-scl, markersize=9-scp, fillstyle='none')   # 'k-'

    # plot points
    for h in range(len(points[0])):
        plt.plot(points[0][h], points[1][h], marker=marker, color = color_pts, linewidth=4-scl, markersize=7-scp)
        if labels:
            plt.text(points[0][h], (points[1][h]), f"  {h+1}", fontfamily= font)

    plt.title("Graph showing the paths different decisions take you to", pad=12)
    if saver:
        plt.savefig(savetitle, dpi=dpi)
    plt.show()




def move4(posin:list, choice:int):
    posfin = posin[:]
    if choice == 0:
        posfin[0] = posfin[0] + 1    # moves to the right

    elif choice == 1:
        posfin[0] = posfin[0] + 1
        posfin[1] = posfin[1] + 1    # moves up

    elif choice == 2:
        posfin[0] = posfin[0] + 1
        posfin[1] = posfin[1] - 1    # moves down

    return posfin




def builderXi(a:list, v:list, s:list, p:int, x0, x1, x2, x3):      # a is list points, v is list parts, p is current point 
    
    l = len(a[0])
    if l != p:
        xval = a[0][p]
        pesos = [x0(xval), x1(xval), x2(xval), x3(xval)]   # [0.3, 24, 7, 6] Implement a function that updates these weights, depending on the x value (not on the amount of points)  
        opciones = [0, 1, 2, 3]  # 0,
        branches = rnd.choices(opciones, weights=pesos, k=1)[0]

        if branches == 0:
            s[p] = 1     

        elif branches == 1:
            choice = rnd.randint(0,2)
            s[p] = 1
            b11 = move4(([a[0][p],a[1][p]]), choice)
            ruun = True
            i = 0
            exists = 0
            while ruun:
                if a[0][i]==b11[0] and a[1][i]==b11[1]:
                    exists = 1
                    ruun = False

                elif i >= len(a[0])-1:
                    ruun = False
                i +=1
            if exists:
                v[0].append(p+1)
                v[1].append(i)
            else:
                a[0].append(b11[0])
                a[1].append(b11[1])
                v[0].append(p+1)
                v[1].append(l+1)
                s.append(0)

        elif branches == 2:
            choice =  rnd.randint(0,2)
            if choice == 0:
                f1, f2 = 0, 1
            elif choice == 1:
                f1, f2 = 0, 2
            elif choice == 2:
                f1, f2 = 1, 2
            s[p] = 1

            b21 = move4(([a[0][p],a[1][p]]), f1)
            ruun21 = True
            i = 0
            exists21 = 0
            while ruun21:
                if a[0][i]==b21[0] and a[1][i]==b21[1]:
                    exists21 = 1
                    ruun21 = False
                elif i >= len(a[0])-1:
                    ruun21 = False
                i +=1
            if exists21:
                v[0].append(p+1)
                v[1].append(i)  #+exists21)
            else:
                a[0].append(b21[0])
                a[1].append(b21[1])
                v[0].append(p+1)
                v[1].append(l+1-exists21)
                s.append(0)

            b22 = move4(([a[0][p],a[1][p]]), f2)
            ruun22 = True
            i = 0
            exists22 = 0
            while ruun22:
                if a[0][i]==b22[0] and a[1][i]==b22[1]:
                    exists22 = 1
                    ruun22 = False
                elif i >= len(a[0])-1:
                    ruun22 = False
                i +=1
            if exists22:
                v[0].append(p+1)
                v[1].append(i) 
            else:
                a[0].append(b22[0])
                a[1].append(b22[1])
                v[0].append(p+1)
                v[1].append(l+2-exists21-exists22)
                s.append(0)

        elif branches == 3:  
            s[p] = 1

            b31 = move4(([a[0][p],a[1][p]]), 0) 
            ruun31 = True
            i = 0
            exists31 = 0
            while ruun31:
                if a[0][i]==b31[0] and a[1][i]==b31[1]:
                    exists31 = 1
                    ruun31 = False
                elif i >= len(a[0])-1:
                    ruun31 = False
                i +=1
            if exists31:
                v[0].append(p+1)
                v[1].append(i)
            else:
                a[0].append(b31[0]) 
                a[1].append(b31[1]) 
                v[0].append(p+1)
                v[1].append(l+1-exists31) 
                s.append(0)

            b32 = move4(([a[0][p],a[1][p]]), 1) 
            ruun32 = True 
            i = 0
            exists32 = 0
            while ruun32: 
                if a[0][i]==b32[0] and a[1][i]==b32[1]: 
                    exists32 = 1
                    ruun32 = False 
                elif i >= len(a[0])-1:
                    ruun32 = False 
                i +=1
            if exists32:
                v[0].append(p+1)
                v[1].append(i)
            else:
                a[0].append(b32[0]) 
                a[1].append(b32[1]) 
                v[0].append(p+1)
                v[1].append(l+2-exists31+exists32) 
                s.append(0)

            b33 = move4(([a[0][p],a[1][p]]), 2) 
            ruun33 = True 
            i = 0
            exists33 = 0
            while ruun33: 
                if a[0][i]==b33[0] and a[1][i]==b33[1]: 
                    exists33 = 1
                    ruun33 = False 
                elif i >= len(a[0])-1:
                    ruun33 = False 
                i +=1
            if exists33:
                v[0].append(p+1)
                v[1].append(i)
            else:
                a[0].append(b33[0]) 
                a[1].append(b33[1]) 
                v[0].append(p+1)
                v[1].append(l+3-exists31-exists32-exists33) 
                s.append(0)
        run = True
    else:
        run = False
    return a, v, s, run




def building(epoch:int, visualizeweights=False):
    #epoch is limit to stop it from running forever, the weight of the 0-branches choice, is the current limiter

    #Creation of objects
    pts = [[0],[0]]
    prts = [[],[]]
    stems = [0]

    # Functions definition --------------------------------- Only one should be active at any given time
    #------------------------------------------------------
    # These weights are working alright so far.
    spreader = 0.75     
    def x0(x):
        return (-1+x*0.1)*spreader
    def x1(x):
        return 9+x*0
    def x2(x):
        return 5+x*0
    def x3(x):
        return (5-x*0.2)*spreader*1.5
    # ------------------------------------------------------
    # def x0(x):
    #     return 1+x*0.2
    # def x1(x):
    #     return x*0.2+20
    # def x2(x):
    #     return 22-x*0.5
    # def x3(x):
    #     return 25-x*0.8
    #------------------------------------------------------
    # def x0(x):
    #     return 0.1+x*0.09
    # def x1(x):
    #     return 6+x*0
    # def x2(x):
    #     return 4+x*0
    # def x3(x):
    #     return 2*abs(np.sin(0.1*x))
    # ------------------------------------------------------
    # def x0(x):
    #     return abs(np.sin(np.pi/8*x))
    # def x1(x):
    #     return abs(np.sin(np.pi/2*x))
    # def x2(x):
    #     return abs(np.cos(x/2))
    # def x3(x):
    #     return abs(np.sin(x/2))
    # ------------------------------------------------------

    run = True
    while stems.count(0)>0 and run:

        loc = stems.index(0)
        pts, prts, stems, run = builderXi(pts, prts, stems, loc, x0, x1, x2, x3)
        #Stop from going to infinity
        if len(pts[0])>epoch:
            run = False

    if visualizeweights:    #implement a visualization of weights use w.r.t x
        xaxis = np.arange(0, max(pts[0]))
        y0 = x0(xaxis)
        y1 = x1(xaxis)
        y2 = x2(xaxis)
        y3 = x3(xaxis)
        plt.title("Variation of the weights with x value of points")
        plt.plot(xaxis, y0, label="y0")
        plt.plot(xaxis, y1, label="y1")
        plt.plot(xaxis, y2, label="y2")
        plt.plot(xaxis, y3, label="y3")
        plt.legend()
        plt.show()

    return pts, prts




if __name__ == '__main__':
    
    labels, color_pts, color_prts, marker, font, figsize, saver, savetitle, dpi= readconfig('Parameters_Decisiontree.toml')

    pts, prts = building(epoch = 1200, visualizeweights=False)

    plot_structure(pts, prts, labels, color_pts, color_prts, marker, font, figsize, saver, savetitle, dpi)
