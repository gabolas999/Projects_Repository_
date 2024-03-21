# The state space simulation, which creates the ss systems, and inputs the data obtained from the FTIS measurements 

import matplotlib.pyplot as plt
import control as ctrl
from SS3 import A_s, B_s, C_s, D_s, A_a, B_a, C_a, D_a
import Cit_par24 as cit
import utilities as u

# Create state-space system
sys_sym = ctrl.ss(A_s, B_s, C_s, D_s)
sys_asym = ctrl.ss(A_a, B_a, C_a, D_a)

# Perform simulation, by inserting the inputs into the system
if cit.manoeuvre == "Phugoid" or cit.manoeuvre == "Short period":
    t, yout = ctrl.forced_response(sys_sym, T=cit.time_array, U=cit.dynamic_input)
else:
    t, yout = ctrl.forced_response(sys_asym, T=cit.time_array, U=cit.dynamic_input)


# Plotting the graphs ------------------------------------------------------------------------------------------------------
graph_sepparate  =  cit.graph_sepparate
graph_superposed =  cit.graph_superposed


if graph_sepparate:
    figure, axis = plt.subplots(2, 4)
    figure.set_size_inches(13,6)
    if cit.manoeuvre == "Phugoid" or cit.manoeuvre == "Short period":
        axis[0, 0].plot(t, cit.dynamic_output[0, :])
        axis[0, 0].set_title("Velocity FTIS")

        axis[1, 0].plot(t, yout[0])
        axis[1, 0].set_title("Velocity Simulated")

        axis[0, 1].plot(t, cit.dynamic_output[1, :])
        axis[0, 1].set_title("AOA FTIS")

        axis[1, 1].plot(t, yout[1])
        axis[1, 1].set_title("AOA Simulated")

        axis[0, 2].plot(t, cit.dynamic_output[2, :])
        axis[0, 2].set_title("Pitch FTIS")

        axis[1, 2].plot(t, yout[2])
        axis[1, 2].set_title("Pitch Simulated")

        axis[0, 3].plot(t, cit.dynamic_output[3, :])
        axis[0, 3].set_title("Pitch Rate FTIS")

        axis[1, 3].plot(t, yout[3])
        axis[1, 3].set_title("Pitch Rate Simulated")
    else:
        axis[0, 0].plot(t, cit.dynamic_output[0, :])
        axis[0, 0].set_title("Sideslip FTIS")

        axis[1, 0].plot(t, yout[0])
        axis[1, 0].set_title("Sideslip Simulated")

        axis[0, 1].plot(t, cit.dynamic_output[1, :])
        axis[0, 1].set_title("Roll FTIS")

        axis[1, 1].plot(t, yout[1])
        axis[1, 1].set_title("Roll Simulated")

        axis[0, 2].plot(t, cit.dynamic_output[2, :])
        axis[0, 2].set_title("Roll Rate FTIS")

        axis[1, 2].plot(t, yout[2])
        axis[1, 2].set_title("Roll Rate Simulated")

        axis[0, 3].plot(t, cit.dynamic_output[3, :])
        axis[0, 3].set_title("Yaw Rate FTIS")

        axis[1, 3].plot(t, yout[3])
        axis[1, 3].set_title("Yaw Rate Simulated")
    plt.suptitle(cit.manoeuvre + " Relevant Outputs: " +  cit.relevant_manoeuvre_output, fontsize=16)
    plt.show()


if graph_superposed:
    figure, axis = plt.subplots(5, 1)
    figure.set_size_inches(8,10)

    if cit.manoeuvre == "Phugoid" or cit.manoeuvre == "Short period":
        axis[0].plot(t, cit.dynamic_output[0, :], label="Velocity FTIS")   # * 100
        axis[0].plot(t, yout[0], label="Velocity Simulated", color='k')
        axis[0].set_title("Velocity")
        axis[0].legend(loc='upper right')
        axis[0].grid()

        axis[1].plot(t, cit.dynamic_output[1, :], label="AOA FTIS")     # *42.8
        axis[1].plot(t, yout[1], label = "AOA Simulated", color='k')
        axis[1].set_title("AOA")
        axis[1].legend(loc='upper right')
        axis[1].grid()

        axis[2].plot(t, cit.dynamic_output[2, :], label="Pitch FTIS")    # *1428
        axis[2].plot(t, yout[2], label="Pitch Simulated", color='k')
        axis[2].set_title("Pitch")
        axis[2].legend(loc='upper right')
        axis[2].grid()

        axis[3].plot(t, cit.dynamic_output[3, :], label = "Pitch Rate FTIS")      #*100
        axis[3].plot(t, yout[3], label="Pitch Rate Simulated", color='k')
        axis[3].set_title("Pitch Rate")
        axis[3].legend(loc='upper right')
        axis[3].grid()

        axis[4].plot(t, u.filter(cit.dynamic_input, 20, 1), label = "Elevator input FTIS")      # Input is filtered 
        axis[4].set_title("Elevator input")
        axis[4].legend(loc='upper right')
        axis[4].grid()

    else:
        # axis[0].plot(t, cit.dynamic_output[0, :], label="Sideslip FTIS")
        axis[0].plot(t, yout[0], label="Sideslip Simulated", color='k')
        axis[0].set_title("Sideslip")
        axis[0].legend(loc='upper right')
        axis[0].grid()

        axis[1].plot(t, cit.dynamic_output[1, :], label="Roll FTIS")
        axis[1].plot(t, yout[1], label="Roll Simulated", color='k')
        axis[1].set_title("Roll")
        axis[1].legend(loc='upper right')
        axis[1].grid()

        axis[2].plot(t, cit.dynamic_output[2, :], label="Roll Rate FTIS")
        axis[2].plot(t, yout[2], label="Roll Rate Simulated", color='k')
        axis[2].set_title("Roll Rate")
        axis[2].legend(loc='upper right')
        axis[2].grid()

        axis[3].plot(t, cit.dynamic_output[3, :], label="Yaw Rate FTIS")
        axis[3].plot(t, yout[3], label="Yaw Rate Simulated", color='k')
        axis[3].set_title("Yaw Rate")
        axis[3].legend(loc='upper right')
        axis[3].grid()

        axis[4].plot(t, cit.dynamic_input[0, :], label="Aileron input FTIS")
        axis[4].plot(t, cit.dynamic_input[1, :], label="Rudder input FTIS", color='k')
        axis[4].set_title("Aileron and Rudder input")
        axis[4].legend(loc='upper right')
        axis[4].grid()

    plt.suptitle(cit.manoeuvre + " Relevant Outputs: " +  cit.relevant_manoeuvre_output, fontsize=16)
    plt.show()