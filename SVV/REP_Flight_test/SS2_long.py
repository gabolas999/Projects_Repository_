# Subsystem 2: Longitudinal stability values.
# It will calculate the Cm_alpha and Cm_delta values. 

import utilities as u
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv

# Files to use
name_Excel  = 'data/PFD_08-08-2021fl1.xlsx'
SS1 = np.genfromtxt("data/SS1.csv")
CLa = SS1[0]    # [1/deg]
a0 = SS1[-1]    # [deg]

def fit_fuel_data():      # Hard code the conversion of Citation II fuel moments with respect to the datum line first row is mass (POUNDS), second row is MOMENT/100 ARM VARIES (INCH-POUNDS)
    Fuelm_arm = np.array([[100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5008],
                          [298.16,591.18,879.08,1165.42,1448.40,1732.53,2014.80,2298.84,2581.92,2866.30,3150.18,3434.52,3718.52,4003.23,4287.76,4572.24,4856.56,5141.16,5425.64,5709.90,5994.04,6278.47,6562.82,6846.96,7131.00,7415.33,7699.60,7984.34,8269.06,8554.05,8839.04,9124.80,9410.62,9696.97,9983.40,10270.08,10556.84,10843.87,11131.00,11418.20,11705.50,11993.31,12281.18,12569.04,12856.86,13144.73,13432.48,13720.56,14008.46,14320.34]])
    
    # Identifying the parameters that match the curve for a selected function, in this case the linear_fit_b0
    params, cov = curve_fit(linear_fit_b0, Fuelm_arm[0, :], 100*Fuelm_arm[1, :])
    return params[0]    # m = slope -> [inch]

def xcg_ramp(W_bem, x_bem, W_pay, x_pay, W_fuel, m, verbose=False, mac=False):
    moment_fuel = linear_fit_b0(W_fuel, m)
    moment_bem = W_bem*x_bem
    moment_payload = W_pay @ x_pay.T
    xcg = (moment_bem+moment_fuel+moment_payload)/(W_bem + np.sum(W_pay) + W_fuel)

    if verbose:
        print(f"x_cg for W_ramp: {100*(xcg-x_lemac)/cbar} [%c_bar]")
    if mac:
        return 100*((xcg-x_lemac)/cbar)
    return xcg

def xcg_correct(xcg_ramp, W_ramp, W_fuel_used, m, W_p, dx_p, verbose=False, mac=False):
    """
    Long Explanation:
        dM due to fuel burn: dM_f = m * (W_fuel - W_fuel_used) - m * W_fuel = -m * W_fuel_used
            -> since M_fuel_current = m * W_fuel_current (linear_fit_b0 function)
            -> dM_f = -linear_fit_b0(W_fuel_used, m)

        dM due to person moving: dM_p: W_p*dx_p -> dx_p: change in the moment arm of person

        Then, the formula for new x_cg is:
            xcg = (M_ramp + dM)/(W_ramp - W_fuel_used) -> dM = dM_f + dM_p
                = M_ramp/(W_ramp - W_fuel_used) + dM/(W_ramp - W_fuel_used)
                = W_ramp/(W_ramp-W_fuel_used) * (xcg_ramp + dM/W_ramp) -> xcg_ramp=M_ramp/W_ramp
                = 1/(1-W_fuel_used/W_ramp) * (xcg_ramp + dM/W_ramp)
                = 1/(1-W_fuel_used/W_ramp) * (xcg_ramp + (dM_f + dM_p)/W_ramp)
                = 1/(1-W_fuel_used/W_ramp) * (xcg_ramp + (-m*W_fuel_used + W_p*dx_p)/W_ramp),
                    where -m*W_fuel_used = -linear_fit_b0(W_fuel_used, m) = dM_f

        !!! When you want to correct it only for fuel burn, use dx_p = 0    
    """
    xcg = (1/(1-(W_fuel_used/W_ramp))) * (xcg_ramp + ((-linear_fit_b0(W_fuel_used, m) + W_p*dx_p)/W_ramp))

    if verbose:
        print(f"x_cg current: {100*(xcg-x_lemac)/cbar} [%c_bar]")
    if mac:
        return 100*((xcg-x_lemac)/cbar)
    return xcg

def delta_alpha(p2, plot=False):
    alpha = p2[:, 2]
    delta = p2[:, 3]
    params, cov = curve_fit(linear_fit, alpha, delta)
    m, b = params

    if plot:
        a0, aN, N = (2, 8, 50)
        plt.figure()
        plt.xlabel(r"$\alpha$ [deg]")
        plt.ylabel(r"$\delta_e$ [deg]")
        plt.plot(np.linspace(a0, aN, N), b + m*np.linspace(a0, aN, N), color="r", label="approximation")
        plt.scatter(alpha, delta, label="data")
        plt.legend()
        plt.grid()
        plt.show()
    return m

# Implement different formulas
def Cm_delta(delta_delta_e, C_N, delta_x_cg, c_bar):
    return (-1 / delta_delta_e) * C_N * (delta_x_cg / c_bar)

def Cm_alpha(ddelta_dalpha, Cm_delta):
    return -Cm_delta*ddelta_dalpha

def C_N(C_N_alpha, alpha, alpha0):
    return C_N_alpha*(alpha-alpha0)

def linear_fit(x, m, b):
    # Fit the data to the equation: y = m*x + b
    return b + m*x

def linear_fit_b0(x, m):
    # Fit the data to the equation: y = m*x
    return m*x

def plot_xcg_fuel(xcg_ramp, W_ramp, W_fuel, m_slope, N, use_mac=True):
    W_fuel_used_array = np.linspace(0, W_fuel, N)
    xcg_data = np.zeros(N)
    for i in range(N):
        xcg_data[i] = xcg_correct(xcg_ramp, W_ramp, W_fuel_used_array[i], m_slope, 0, 0, mac=use_mac)

    plt.figure()
    plt.title("Centre of gravity variation due to fuel burn")
    
    if use_mac:
        plt.ylabel(r"$x_{c.g.} [\% \bar{c}]$")
    else:
        plt.ylabel(r"$x_{c.g.}$ [inch]")

    plt.xlabel(r"$W_{f, used} [\% W_f]$")
    plt.plot(100*W_fuel_used_array/W_fuel, xcg_data)
    plt.grid()
    plt.show()

if __name__ == "__main__":      # Only run this code when you want to update the SS2.csv values.

    data = u.read_excel(name_Excel)
    p2 = data["stat_p2"]                # Elevator trim curve measurements
    cg = data["cg_shift"]               # measurements before and after the cg shift
    W_pay = data["payload"]/u.lbs2kg    # (array consisting of all payload weights) [lbs]
    W_fuel = data["fuel"]               # Fuel weight [lbs]
    pax = data["pax"]                   # name of the passengers
    pos_change = data["pos_change"]     # Name of the person moving, initial and final location
    
    # Data manually recorded from the flight
    W_bem = 9197.0                      # [lbs]
    x_bem = 291.28                      # [inch]
    cbar = 80.98                        # [inch]
    x_lemac = 261.45                    # [inch]

    ddelta_dalpha = delta_alpha(p2)     # [-]
    
    m_slope = fit_fuel_data()           # slope of the M_fuel - W_fuel curve [inch]
    x_pay = np.array([131,131,170,214,214,251,251,288,288])  # tDistance to datum feature of: two pilots, coordinator, row 1, row 2, row 3. [inch]
    W_ramp = W_bem + np.sum(W_pay) + W_fuel     # [lbs]

    # Maping
    pax_pay = dict(zip(pax, W_pay))             # maps passenger to corresponding weight
    seat_arm = dict(zip(['Cockpit', 'Cockpit', 'coordinator', '1L', '1R', '2L', '2R', '3L', '3R'], x_pay))    # maps seat to its moment arm
    
    xcg0 = xcg_ramp(W_bem, x_bem, W_pay, x_pay, W_fuel, m_slope)    # Initial cg location

    fuel_used = cg[:, -2]      # Fuel used before and after cg shift
    alpha = cg[0, 2]           # alpha before the cg shift (use before due to assumption)
    delta_de = cg[-1, 3] - cg[0, 3]     # Change in the elevator deflection after the cg shift

    CN = C_N(CLa, alpha, a0)                                        # [-]
    xcg1 = xcg_correct(xcg0, W_ramp, fuel_used[0], m_slope, 0, 0)   # [inch]
    xcg2 = xcg_correct(xcg0, W_ramp, fuel_used[1], m_slope, pax_pay[pos_change[0]], seat_arm[pos_change[-1]]-seat_arm[pos_change[1]])                       # [inch]
    delta_xcg = xcg2 - xcg1                                         # [inch]
    Cmdelta = Cm_delta(delta_de*u.deg2rad, CN, delta_xcg, cbar)     #*u.deg2rad #(next to delta_de)  # [-] #can someone check this deg2rad
    Cmalpha = Cm_alpha(ddelta_dalpha, Cmdelta)                      # [-]
    
    filename = "data/SS2.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(np.array([[Cmalpha], [Cmdelta]]))
    
    print(f"Cm_alpha: {Cmalpha}\nCm_delta: {Cmdelta}\nSaved into '{filename}'")
    # plot_xcg_fuel(xcg0, W_ramp, W_fuel, m_slope, 100)
