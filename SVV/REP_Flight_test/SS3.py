# Subsystem 3: the Equations of motion and matrices calculation
# It will calculate the A, B, C, D matrices needed for the State-Space system

import numpy as np
import Cit_par24 as c
from scipy.linalg import eig

# Variable to alternate between two methods of creating the matrices, 
jordismethod = c.jordismethod     # if false, it runs alternate method (using the inverses)

if jordismethod:
    V = c.V0
    Cma = c.Cma
    Cmde  = c.Cmde

    # Define system matrices
    # Symmetric case has 1 input
    A_s = np.array([[V/c.c*c.CXu/(2*c.muc), V/c.c*c.CXa/(2*c.muc),V/c.c*c.CZ0/(2*c.muc), 0],[V/c.c*c.CZu/(2*c.muc-c.CZadot), V/c.c*c.CZa/(2*c.muc-c.CZadot), -V/c.c*c.CX0/(2*c.muc-c.CZadot), V/c.c*(2*c.muc+c.CZq)/(2*c.muc-c.CZadot)],[0,0,0,V/c.c],[V/c.c*(c.Cmu+c.CZu*c.Cmadot/(2*c.muc-c.CZadot))/(2*c.muc*c.KY2), V/c.c*(Cma+c.CZa*c.Cmadot/(2*c.muc-c.CZadot))/(2*c.muc*c.KY2), -V/c.c*(c.CX0*(c.Cmadot)/(2*c.muc-c.CZadot))/(2*c.muc*c.KY2),V/c.c*(c.Cmq+c.Cmadot*(2*c.muc+c.CZq)/(2*c.muc-c.CZadot))/(2*c.muc*c.KY2)]])  # State transition matrix
    B_s = np.array([[V/c.c*c.CXde/(2*c.muc)],[V/c.c*c.CZde/(2*c.muc-c.CZadot)],[0],[V/c.c*(Cmde+c.CZde*c.Cmadot/(2*c.muc-c.CZadot))/(2*c.muc*c.KY2)]])                # Control input matrix
    C_s = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])        # Output matrix
    D_s = np.array([[0],[0],[0],[0]])                                # Feedforward matrix

    # Asymmetric case has 2 inputs
    A_a = np.array([[V/c.b*c.CYb/(2*c.mub),V/c.b*c.CL/(2*c.mub),V/c.b*c.CYp/(2*c.mub), V/c.b*(c.CYr-4*c.mub)/(2*c.mub)],[0,0,2*V/c.b, 0],[V/c.b*(c.Clb*c.KZ2+c.Cnb*c.KXZ)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), 0 , V/c.b*(c.Clp*c.KZ2+c.Cnp*c.KXZ)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), V/c.b*(c.Clr*c.KZ2+c.Cnr*c.KXZ)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2))],[V/c.b*(c.Clb*c.KXZ+c.Cnb*c.KX2)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), 0, V/c.b*(c.Clp*c.KXZ+c.Cnp*c.KX2)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), V/c.b*(c.Clr*c.KXZ+c.Cnr*c.KX2)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2))]])
    B_a = np.array([[0, V/c.b*c.CYdr/(2*c.mub)],[0,0],[V/c.b*(c.Clda*c.KZ2+c.Cnda*c.KXZ)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), V/c.b*(c.Cldr*c.KZ2+c.Cndr*c.KXZ)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)) ],[V/c.b*(c.Clda*c.KXZ+c.Cnda*c.KX2)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2)), V/c.b*(c.Cldr*c.KXZ+c.Cndr*c.KX2)/(4*c.mub*(c.KX2*c.KZ2-c.KXZ**2))]])
    C_a = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])        # Output matrix
    D_a = np.array([[0,0],[0,0],[0,0],[0,0]])                        # Feedforward matrix

else:
    # Symmetric case
    C1_s = (c.c/c.V0)*np.array([[-2*c.muc, 0, 0, 0], [0, c.CZadot-2*c.muc, 0, 0], [0, 0, -1, 0], [0, c.Cmadot, 0, -2*c.muc*c.KY2]])
    C2_s = np.array([[c.CXu, c.CXa, c.CZ0, c.CXq], [c.CZu, c.CZa, -c.CX0, c.CZq+2*c.muc], [0, 0, 0, 1], [c.Cmu, c.Cma, 0, c.Cmq]]) #Jordi made a change here, maybe check!
    C3_s = -1*np.array([[-c.CXde], [-c.CZde], [0], [-c.Cmde]])
    A_s = -np.linalg.inv(C1_s) @ C2_s
    B_s = -np.linalg.inv(C1_s) @ C3_s
    C_s = np.identity(4)
    D_s = np.zeros((4, 1))


    # Asymmetric case
    C1_as = (c.b/c.V0)*np.array([[c.CYbdot-2*c.mub, 0, 0, 0], [0, -1/2, 0, 0], [0, 0, -4*c.mub*c.KX2, 4*c.mub*c.KXZ], [c.Cnbdot, 0, 4*c.mub*c.KXZ, -4*c.mub*c.KZ2]])
    C2_as = np.array([[c.CYb, c.CL, c.CYp, c.CYr-4*c.mub], [0, 0, 1, 0], [c.Clb, 0, c.Clp, c.Clr], [c.Cnb, 0, c.Cnp, c.Cnr]])
    C3_as = np.array([[-c.CYda, -c.CYdr], [0, 0], [-c.Clda, -c.Cldr], [-c.Cnda, -c.Cndr]])

    A_a = -np.linalg.inv(C1_as) @ C2_as
    B_a = -np.linalg.inv(C1_as) @ C3_as
    C_a = np.identity(4)
    D_a = np.zeros((4, 2))
