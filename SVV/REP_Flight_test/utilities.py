# Utilities file, miscellaneous functions are defined here and can be used throughout the project subsystems

import math as m
import numpy as np
import pandas as pd
import scipy
import scipy.signal
import scipy.io     # needed

# CONVERSION FACTORS AND CONSTANTS
g = 9.80665
lbs2kg = 0.453592
kts2m_s = 0.514444
inch2m = 0.0254
deg2rad = np.pi/180
ft2m = 0.3048
lbshr2kgs = 0.000125998

sgc = 287  ## [J/kgK]
temp0 = 288.15  ## [K]
p0 = 101325  ## [Pa]
atr = -0.0065  ## [K/m]
astr1 = 0.001  ## [K/m]
astr2 = 0.0028  ## [K/m]
ames1 = -0.0028  ## [K/m]
ames2 = -0.002  ## [K/m]
rho0 = 1.225 #[kg/m3]
gamma = 1.4 # [Need to change]


def ET(time):
    return time.hour * 3600 + time.minute * 60 + time.second

# Read DATA from excel
def read_excel(filename):
    dataframe = pd.read_excel(filename)
    payload = dataframe.iloc[6:15, 7].to_numpy()
    pax = dataframe.iloc[6:15, 3].to_numpy()
    pos_change = np.array([dataframe.iloc[68, 2], dataframe.iloc[69, 2], dataframe.iloc[69, 7]])
    fuel = dataframe.iloc[16, 3]
    cl_cd1 = dataframe.iloc[26:33, 3:10].to_numpy()
    cl_cd2 = dataframe.iloc[42:49, 3:10].to_numpy()
    elevator_trim = dataframe.iloc[57:64, 3:13].to_numpy()
    cg_shift = dataframe.iloc[73:75, 3:13].to_numpy()
    eig = dataframe.iloc[81:83, :10].to_numpy()
    eigenmotion = {
        eig[0, 0]: ET(eig[0, 3]),
        eig[0, 4]: ET(eig[0, 6]),
        eig[0, 7]: ET(eig[0, -1]),
        eig[1, 0]: ET(eig[1, 3]),
        eig[1, 4]: ET(eig[1, 6]),
        eig[1, 7]: ET(eig[1, -1])
    }
    data = {
        "payload": payload,
        "pax": pax,
        "pos_change": pos_change,
        "fuel": fuel,
        "cl_cd1": cl_cd1,
        "cl_cd2": cl_cd2,
        "stat_p2": elevator_trim,
        "cg_shift": cg_shift, 
        "eigenmotion": eigenmotion
    }
    return data


# Get temperature depending on altitude (h) using International Standard Atmosphere
def temp(h):
    if h >= 0 and h <= 11000:
        T1 = temp0 + atr*h
    elif h>11000 and h <=20000:
        T1 = 216.65
    elif h > 20000 and h <= 32000:
        T1 = 216.65 + astr1 * (h - 20000)
    elif h > 32000 and h <= 47000:
        T1 = (216.65 + astr1*(32000 - 20000))+ astr2*(h - 32000)
    elif h > 47000 and h <= 51000:
        T1 = (216.65+astr1*(32000-20000))+ astr2*(47000 - 32000)
    elif h > 51000 and h <= 71000:
        T1 = (216.65+astr1*(32000-20000))+ astr2*(47000 - 32000)+ ames1 * (h-51000)
    elif h > 71000 and h <= 86000:
        T1 = ((216.65+astr1*(32000-20000))+ astr2*(47000 - 32000)+ ames1 * (71000-51000)) + ames2*(h - 71000)        
    return T1

# Get pressure depending on altitude (h) using International Standard Atmosphere
def pressure(h):       
    if h >= 0 and h <= 11000:
        p1 = p0 *(temp(h)/(temp0))**(-g/(sgc * atr))
    elif h>11000 and h <=20000:
        p1 = (p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(h)))*(h-11000))
    elif h > 20000 and h <= 32000:
        p1 = ((p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(20000)))*(20000-11000))) * (temp(h)/216.65)**(-g/(sgc * astr1))
    elif h > 32000 and h <= 47000:
        p1 = (((p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(20000)))*(20000-11000))) * (temp(32000)/216.65)**(-g/(sgc * astr1))) * (temp(h)/temp(32000))**(-g/(sgc * astr2))
    elif h > 47000 and h <= 51000:
        p1 = ((((p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(20000)))*(20000-11000))) * (temp(32000)/216.65)**(-g/(sgc * astr1))) * (temp(47000)/temp(32000))**(-g/(sgc * astr2))) * m.exp(-(g/(sgc*temp(h)))*(h-47000))
    elif h > 51000 and h <= 71000:
        p1 = (((((p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(20000)))*(20000-11000))) * (temp(32000)/216.65)**(-g/(sgc * astr1))) * (temp(47000)/temp(32000))**(-g/(sgc * astr2))) * m.exp(-(g/(sgc*temp(h)))*(51000-47000))) * (temp(h)/temp(51000))**(-g/(sgc * ames1))
    elif h > 71000 and h <= 86000:
        p1 = ((((((p0 * (temp(11000)/(temp0))**(-g/(sgc * atr))) * m.exp(-(g/(sgc*temp(20000)))*(20000-11000))) * (temp(32000)/216.65)**(-g/(sgc * astr1))) * (temp(47000)/temp(32000))**(-g/(sgc * astr2))) * m.exp(-(g/(sgc*temp(h)))*(51000-47000))) * (temp(71000)/temp(51000))**(-g/(sgc * ames1)))* (temp(h)/temp(71000))**(-g/(sgc * ames2))
    return p1


def density(h):
    rho = pressure(h) / (sgc * temp(h))
    return rho


def reduce_Airspeed(hp, vc, TAT):
    # Converting indicated airspeed (= calibrated speed V_c), to equivalent airspeed (V_e). this reduces atmospheric variables to ISA values; Look flight dynamics assignment, page 8
    # VIAS = Vc, which is transformed into Ve (equivalent), 
    sgc = 287       # [J/kgK]
    p0 = 101325     # [Pa]
    rho0 = 1.225    #[kg/m3]
    gamma = 1.4
    rho = density(hp) 
    p = pressure(hp)

    # Mach number
    M =  m.sqrt(  (2 / (gamma - 1)) * ((1 + (p0 / p) * ((1 + (gamma - 1) / (2 * gamma) * (rho0 / p0 * vc**2))**(gamma / (gamma - 1)) - 1))**((gamma-1)/gamma )-1) )

    #correction the measured total air temperature for ram rise
    T = TAT/(1+ (gamma-1)/2*M**2)
    a = m.sqrt(gamma*sgc*T)
    v_t = M*a
    v_e= v_t*m.sqrt(rho/rho0)
    return v_e, v_t, M


def filter(sig, smooth_window, polynom):
    #Filtering any signal
    sig_f = scipy.signal.savgol_filter(sig, smooth_window, polynom, axis=0)
    return sig_f


# Function that imports 
def FTIS_imports(name_Excel, name_FTIS, manoeuvre, manoeuvre_durations:list):
    time_manoeuvre = read_excel(name_Excel)["eigenmotion"][manoeuvre]

    duration_manoeuvre = manoeuvre_durations 
    if manoeuvre=="Phugoid":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[0]
    elif manoeuvre=="Dutch Roll":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[1]
    elif manoeuvre=="Aper. Roll":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[2]
    elif manoeuvre=="Short period":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[3]
    elif manoeuvre=="Dutch Roll YD":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[4]
    elif manoeuvre=="Spiral ":
        endtime_manoeuvre = time_manoeuvre + duration_manoeuvre[5]
    else:
        print("AAAAAAAAAAAAA NAME OF MANOEUVRE IS WRONG")


    # Load the MATLAB file (FTIS measured data)
    mat = scipy.io.loadmat(name_FTIS, simplify_cells=True)
    flightdata = mat['flightdata']


    # Defining the starting point, and duration of current manoeuvre
    time_manoeuvre_c = time_manoeuvre-9-6   #5←← seconds margin to get the stationary conditions, 9 seconds bc the FTIS data is shifted
    endtime_manoeuvre_c = endtime_manoeuvre-9
    time_manoeuvre_c_Hz = time_manoeuvre_c*10
    endtime_manoeuvre_c_Hz = endtime_manoeuvre_c*10

    hp0 = flightdata['Dadc1_alt']['data'][time_manoeuvre_c_Hz]
    V0 = flightdata['Dadc1_cas']['data'][time_manoeuvre_c_Hz]
    Vt0 = flightdata['Dadc1_tas']['data'][time_manoeuvre_c_Hz] # True airspeed stationary conditions
    AOA0 = flightdata['vane_AOA']['data'][time_manoeuvre_c_Hz]
    pitch0= flightdata['Ahrs1_Pitch']['data'][time_manoeuvre_c_Hz]
    FU0 = flightdata['rh_engine_FU']['data'][time_manoeuvre_c_Hz] + flightdata['lh_engine_FU']['data'][time_manoeuvre_c_Hz]

    # Inputs
    d_e = flightdata['delta_e']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]  #* deg2rad #(in the end no need to convert to radians)
    d_a = flightdata['delta_a']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]  #* deg2rad 
    d_r = flightdata['delta_r']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]  #* deg2rad 

    time_array = flightdata['time']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]


    #outputs
    hp = flightdata['Dadc1_alt']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]
    tat = flightdata['Dadc1_tat']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]
    pitch_rate = flightdata['Ahrs1_bPitchRate']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] * 2.0569 / Vt0 #* deg2rad
    aoa = (flightdata['vane_AOA']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] - AOA0)        # transformation reference frame

    # Filter angle of attack
    aoa = filter(aoa, smooth_window=7, polynom=1)
    
    roll = flightdata['Ahrs1_Roll']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] #* deg2rad
    roll_rate = flightdata['Ahrs1_bRollRate']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] * 15.911 / (2*Vt0) #* deg2rad  #* 15.911 / (2*Vt0)  #(reduction we didn't use in the end)
    yaw_rate = flightdata['Ahrs1_bYawRate']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] * 15.911 / (2*Vt0)  #*deg2rad   #* 15.911 / (2*Vt0)  
    pitch = (flightdata['Ahrs1_Pitch']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] - pitch0) #* deg2rad    #transformation reference frame 
    V = flightdata['Dadc1_cas']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] #calibrated
    Vtas = flightdata['Dadc1_tas']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz] #true airspeed
    u_hat = (Vtas - Vt0) / Vt0
    True_Heading = flightdata['Fms1_trueHeading']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]

    dt = 0.1 #0.1s as it is 10 Hz

    longitude = flightdata['Gps_long']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]
    latitude = flightdata['Gps_lat']['data'][time_manoeuvre_c_Hz:endtime_manoeuvre_c_Hz]

    dlatitude = np.diff(latitude)/dt
    dlongitude = np.diff(longitude)/dt

    dlatitude = np.hstack((dlatitude, dlatitude[-1]))
    dlongitude = np.hstack((dlongitude, dlongitude[-1]))

    sideslip = np.arctan(dlongitude/dlatitude) - True_Heading   # not used
    # To get yaw, integrate yaw rate
    yaw=[]
    for i in range(time_array.shape[0]):
        if i==0:
            i = 1
        yaw.append(scipy.integrate.simpson(yaw_rate[0:i], dx = dt, axis=-1, even='avg'))  # Integrating yaw rate to get sideslip, assuming the heading doesn't change during the manoeuvre
        # It assumes the change in yaw is directly change in sideslip.
    yaw=np.array(yaw)

    # Reducing the airspeed
    ve = []
    for ix in range(hp.shape[0]):
        ve1, v_t1, M1 = reduce_Airspeed(hp[ix]*ft2m, V[ix]*kts2m_s, tat[ix]+273.15)
        ve.append(ve1)
    ve = np.array(ve)
    # finished reducing airspeed

    if manoeuvre=="Phugoid" or manoeuvre=="Short period":
        dynamic_input = d_e*deg2rad
        dynamic_output = u_hat
        dynamic_output = np.vstack((dynamic_output, aoa*deg2rad))
        dynamic_output = np.vstack((dynamic_output, pitch*deg2rad))
        dynamic_output = np.vstack((dynamic_output, pitch_rate*deg2rad))

    elif manoeuvre=="Dutch Roll" or manoeuvre=="Aper. Roll" or manoeuvre=="Dutch Roll YD" or manoeuvre=="Spiral ":
        dynamic_input = d_a*deg2rad
        dynamic_input = np.vstack((dynamic_input, d_r*deg2rad))
        dynamic_output = yaw*deg2rad
        dynamic_output = np.vstack((dynamic_output, roll*deg2rad))
        dynamic_output = np.vstack((dynamic_output, roll_rate*deg2rad))
        dynamic_output = np.vstack((dynamic_output, yaw_rate*deg2rad))

    static = [hp0, ve[0], AOA0*deg2rad, pitch0*deg2rad, FU0]
    return static, dynamic_input, dynamic_output, time_array


if __name__ == '__main__':
    print(read_excel('PFD_08-08-2021fl1.xlsx')["eigenmotion"])