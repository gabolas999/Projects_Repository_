# Parameters for the Citation 550 - Linear simulation
# Used for all subsystems

from numpy import pi, genfromtxt
import numpy as np
import utilities as u


# Data to import
# ========================================================================================================================
# Get values produced in SS1 and SS2. Stored in csv files to avoid circular dependencies. They were produced in other subsystems.
SS1 = genfromtxt("data/SS1.csv")     # cla, e, cd0, a0
SS2 = genfromtxt("data/SS2.csv")     # Cma, Cmde

name_Excel  = 'data/PFD_08-08-2021fl1.xlsx'
name_FTIS   = 'data/FTISxprt-20210808_143959.mat'
# ========================================================================================================================




# Parameters to change
# ========================================================================================================================
# Chosen manoeuvre
manoeuvre = "Phugoid"    
# Options ↓↓↓
# Symmetric: "Phugoid", "Short period"
# Asymmetric: "Dutch Roll", "Aper. Roll", "Dutch Roll YD", "Spiral "

# Hardcoded durations of manoeuvres
manoeuvre_durations = [129, 22, 34, 138, 24, 130]  #phugoid 129, dutch roll 23, aperoll 25, short period 138, dutch roll yd 13, spiral100.

# Plotting
graph_sepparate = False
graph_superposed = True

jordismethod = False
# ========================================================================================================================




# Definition of the output for each manoeuvre                               
if manoeuvre == "Phugoid":
    relevant_manoeuvre_output = "Pitch Rate, Pitch Angle and Velocity"
elif manoeuvre == "Dutch Roll":
    relevant_manoeuvre_output = "Sideslip, Roll Rate and Yaw Rate"
elif manoeuvre == "Aper. Roll":
    relevant_manoeuvre_output = "Roll Rate"
elif manoeuvre == "Short period":
    relevant_manoeuvre_output = "Pitch Rate"
elif manoeuvre == "Dutch Roll YD":
    relevant_manoeuvre_output = "Sideslip, Roll Rate and Yaw Rate"
elif manoeuvre == "Spiral ":
    relevant_manoeuvre_output = "Sideslip, Roll Rate and Yaw Rate"
else:
    print("AAAAAAAAAAAAA NAME OF MANOEUVRE IS WRONG")


# Data from the FTIS measurements for the selected manoeuvre
static, dynamic_input, dynamic_output, time_array  = u.FTIS_imports(name_Excel, name_FTIS, manoeuvre, manoeuvre_durations)


# Account for calibration error in the sensors for the data
if manoeuvre=="Dutch Roll" or manoeuvre=="Aper. Roll" or manoeuvre=="Dutch Roll YD" or manoeuvre=="Spiral ":
    dynamic_output[1,:] = dynamic_output[1,:] - dynamic_output[1,0] # Roll
    dynamic_output[2,:] = dynamic_output[2,:] - dynamic_output[2,0] # Yaw rate
    dynamic_output[3,:] = dynamic_output[3,:] - dynamic_output[3,0] # Yaw rate
    dynamic_input[0, :] = dynamic_input[0, :]-dynamic_input[0, 0]   # Aileron
    dynamic_input[1, :] = dynamic_input[1, :]-dynamic_input[1, 0]   # Rudder

elif manoeuvre == "Short period" or manoeuvre == "Phugoid":
    dynamic_input[:] = dynamic_input[:]-dynamic_input[0]   # Elevator


hp0    =  static[0]*u.ft2m          # pressure altitude in the stationary flight condition right before manoeuvre[m]
V0     =  static[1]
alpha0 =  static[2]
th0    =  static[3]
FU     =  static[4]

# Aircraft mass
read_excel = u.read_excel("data/PFD_08-08-2021fl1.xlsx")
take_off_mass = 9197*u.lbs2kg + np.sum(read_excel['payload'])+read_excel['fuel']*u.lbs2kg
m      =  take_off_mass - FU*u.lbs2kg        # mass [kg]


# Rest of variables are pre-defined ↓↓↓↓↓----------------------------------------------
# Aerodynamic properties
CLa    =  SS1[0]         # Slope of CL-alpha curve [1/deg]
e      =  SS1[1]         # Oswald factor [ ]
CD0    =  SS1[2]         # Zero lift drag coefficient [ ]
a0     =  SS1[-1]        # Angle of attack at zero lift  [deg]

# Longitudinal stability
Cma    =  SS2[0]         # longitudinal stabilty [ ]
Cmde   =  SS2[1]         # elevator effectiveness [ ]

# Aircraft geometry
S      = 30.00	         # wing area [m^2]
Sh     = 0.2 * S         # stabiliser area [m^2]
Sh_S   = Sh / S	         # [ ]
lh     = 0.71 * 5.968    # tail length [m]
c      = 2.0569	         # mean aerodynamic cord [m]
lh_c   = lh / c	         # [ ]
b      = 15.911	         # wing span [m]                       # THIS IS HARDCODED INTO UTILITIES, IN FUNCTION FTIS_IMPORTS() if changed here, change there as well.
bh     = 5.791	         # stabiliser span [m]
A      = b ** 2 / S      # wing aspect ratio [ ]
Ah     = bh ** 2 / Sh    # stabiliser aspect ratio [ ]
Vh_V   = 1	             # [ ]
ih     = -2 * pi / 180   # stabiliser angle of incidence [rad]

# Constant values concerning atmosphere and gravity
rho0   = 1.2250          # air density at sea level [kg/m^3] 
lambda1 = -0.0065        # temperature gradient in ISA [K/m]
Temp0  = 288.15          # temperature at sea level in ISA [K]
R      = 287.05          # specific gas constant [m^2/sec^2K]
g      = 9.81            # [m/sec^2] (gravity constant)

# air density [kg/m^3]  
rho    = rho0 * ((1+(lambda1 * hp0 / Temp0)))** (-((g / (lambda1*R)) + 1))
W      = m * g            # [N]       (aircraft weight)

# Constant values concerning aircraft inertia
muc    = m / (rho * S * c)
mub    = m / (rho * S * b)
KX2    = 0.019
KZ2    = 0.042
KXZ    = 0.002
KY2    = 1.25 * 1.114

# Aerodynamic constants
Cmac   = 0                      # Moment coefficient about the aerodynamic centre [ ]
CNwa   = CLa                    # Wing normal force slope [ ]
CNha   = 2 * pi * Ah / (Ah + 2) # Stabiliser normal force slope [ ]
depsda = 4 / (A + 2)            # Downwash gradient [ ]

# Lift and drag coefficient
CL = 2 * W / (rho * V0 ** 2 * S)              # Lift coefficient [ ]
CD = CD0 + (CLa * alpha0) ** 2 / (pi * A * e) # Drag coefficient [ ]

# Stability derivatives
CX0    = W * np.sin(th0) / (0.5 * rho * V0 ** 2 * S)
CXu    = -0.09500
CXa    = +0.47966		# Positive, see FD lecture notes) 
CXadot = +0.08330
CXq    = -0.28170
CXde   = -0.03728

CZ0    = -W * np.cos(th0) / (0.5 * rho * V0 ** 2 * S)      # same as alpha_trim
CZu    = -0.37616
CZa    = -5.74340
CZadot = -0.00350
CZq    = -5.66290
CZde   = -0.69612

Cm0    = +0.0297
Cmu    = +0.06990
Cmadot = +0.17800
Cmq    = -8.79415
CmTc   = -0.0064

CYb    = -0.7500
CYbdot =  0     
CYp    = -0.0304
CYr    = +0.8495
CYda   = -0.0400
CYdr   = +0.2300

Clb    = -0.10260
Clp    = -0.71085
Clr    = +0.23760
Clda   = -0.23088
Cldr   = +0.03440

Cnb    =  +0.1348
Cnbdot =   0     
Cnp    =  -0.0602
Cnr    =  -0.2061
Cnda   =  -0.0120
Cndr   =  -0.0939