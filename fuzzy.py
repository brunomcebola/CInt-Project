import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

"""
Data:
    temp1
    temp2
    temp3
    light1
    light2
    light3
    co2
    pir1
    pir2
"""


def our_fuzzy(data):
    # Generate universe variables

    # temp_sensor1 = np.arange(0.0, 1.0001, 0.0001)
    # temp_sensor2 = np.arange(0.0, 1.0001, 0.0001)
    # temp_sensor3 = np.arange(0.0, 1.0001, 0.0001)

    light_sensor1 = np.arange(0.0, 750, 1)
    light_sensor2 = np.arange(0.0, 750, 1)
    light_sensor3 = np.arange(0.0, 750, 1)

    # co2_sensor = np.arange(0.0, 1.0001, 0.0001)

    pir_sensor1 = np.arange(0.0, 1.001, 0.001)
    pir_sensor2 = np.arange(0.0, 1.001, 0.001)


    # Temperature Fuzzy Membership Functions
    """
    
    temp_low_1 = fuzz.trimf(temp_sensor1, [0, 0, 0.5])
    temp_mid_1 = fuzz.trimf(temp_sensor1, [0, .5, 1.0])
    temp_high_1 = fuzz.trimf(temp_sensor1, [0.5, 1.0, 1.0])

    temp_low_2 = fuzz.trimf(temp_sensor2, [0, 0, 0.5])
    temp_mid_2 = fuzz.trimf(temp_sensor2, [0, .5, 1.0])
    temp_high_2 = fuzz.trimf(temp_sensor2, [0.5, 1.0, 1.0])

    temp_low_3 = fuzz.trimf(temp_sensor3, [0, 0, 0.5])
    temp_mid_3 = fuzz.trimf(temp_sensor3, [0, .5, 1.0])
    temp_high_3 = fuzz.trimf(temp_sensor3, [0.5, 1.0, 1.0])
    
    """
    # Light Fuzzy Membership Functions

    light_low_1 = fuzz.trimf(light_sensor1, [0, 0, 375])
    light_mid_1 = fuzz.trimf(light_sensor1, [0, 375, 750])
    light_high_1 = fuzz.trimf(light_sensor1, [375, 750, 750])

    light_low_2 = fuzz.trimf(light_sensor2, [0, 0, 375])
    light_mid_2 = fuzz.trimf(light_sensor2, [0, 375, 750])
    light_high_2 = fuzz.trimf(light_sensor2, [375, 750, 750])

    light_low_3 = fuzz.trimf(light_sensor3, [0, 0, 375])
    light_mid_3 = fuzz.trimf(light_sensor3, [0, 375, 750])
    light_high_3 = fuzz.trimf(light_sensor3, [375, 750, 750])

    # PIR Fuzzy Membership Functions

    pir_no_presence1 = fuzz.trimf(pir_sensor1, [0, 0, 0.5])
    pir_presence1 = fuzz.trimf(pir_sensor1, [0, 1.0, 1.0])

    pir_no_presence2 = fuzz.trimf(pir_sensor2, [0, 0, 0.5])
    pir_presence2 = fuzz.trimf(pir_sensor2, [0, 1.0, 1.0])

    # CO2 Fuzzy Membership Functions
    """ 
    co2_low = fuzz.trimf(co2_sensor, [0, 0, 0.5])
    co2_mid = fuzz.trimf(co2_sensor, [0, .5, 1.0])
    co2_high = fuzz.trimf(co2_sensor, [0.5, 1.0, 1.0])
    """

    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#

    # We need the activation of our fuzzy membership functions at these values.
    
    # Temperature
    """ 

    temp1_level_lo = fuzz.interp_membership(temp_sensor1, temp_low_1, data[0])
    temp1_level_md = fuzz.interp_membership(temp_sensor1, temp_mid_1, data[0])
    temp1_level_hi = fuzz.interp_membership(temp_sensor1, temp_high_1, data[0])

    temp2_level_lo = fuzz.interp_membership(temp_sensor1, temp_low_2, data[1])
    temp2_level_md = fuzz.interp_membership(temp_sensor1, temp_mid_2, data[1])
    temp2_level_hi = fuzz.interp_membership(temp_sensor1, temp_high_2, data[1])
    
    temp3_level_lo = fuzz.interp_membership(temp_sensor3, temp_low_3, data[2])
    temp3_level_md = fuzz.interp_membership(temp_sensor3, temp_mid_3, data[2])
    temp3_level_hi = fuzz.interp_membership(temp_sensor3, temp_high_3, data[2])

    """

    #Light

    light1_level_lo = fuzz.interp_membership(light_sensor1, light_low_1, data[3])
    light1_level_md = fuzz.interp_membership(light_sensor1, light_mid_1, data[3])
    light1_level_hi = fuzz.interp_membership(light_sensor1, light_high_1, data[3])

    light2_level_lo = fuzz.interp_membership(light_sensor2, light_low_2, data[4])
    light2_level_md = fuzz.interp_membership(light_sensor2, light_mid_2, data[4])
    light2_level_hi = fuzz.interp_membership(light_sensor2, light_high_2, data[4])

    light3_level_lo = fuzz.interp_membership(light_sensor3, light_low_3, data[5])
    light3_level_md = fuzz.interp_membership(light_sensor3, light_mid_3, data[5])
    light3_level_hi = fuzz.interp_membership(light_sensor3, light_high_3, data[5])

    # CO2
    """ 

    co2_level_lo = fuzz.interp_membership(co2_sensor, co2_low, data[6])
    co2_level_md = fuzz.interp_membership(co2_sensor, co2_mid, data[6])
    co2_level_hi = fuzz.interp_membership(co2_sensor, co2_high, data[6])

    """

    # PIR

    pir1_level_np = fuzz.interp_membership(pir_sensor1, pir_no_presence1, data[7])
    pir1_level_p = fuzz.interp_membership(pir_sensor1, pir_presence1, data[7])

    pir2_level_np = fuzz.interp_membership(pir_sensor2, pir_no_presence2, data[8])
    pir2_level_p = fuzz.interp_membership(pir_sensor2, pir_presence2, data[8])

    #------------------------------------------------------------------------------#
    #------------------------------------------------------------------------------#

    # Now we take our rules and apply them. Rule 1 concerns bad food OR service.
    
    
    pir_active_rule_np = np.min(pir1_level_np, pir2_level_np)
    pir_active_rule_p = np.max(pir1_level_p, pir2_level_p)


    
    # The OR operator means we take the maximum of these two.
    active_rule1 = np.fmax(qual_level_lo, serv_level_lo)

    # Now we apply this by clipping the top off the corresponding output
    # membership function with `np.fmin`
    tip_activation_lo = np.fmin(active_rule1, tip_lo)  # removed entirely to 0

    # For rule 2 we connect acceptable service to medium tipping
    tip_activation_md = np.fmin(serv_level_md, tip_md)

    # For rule 3 we connect high service OR high food with high tipping
    active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
    tip_activation_hi = np.fmin(active_rule3, tip_hi)
    tip0 = np.zeros_like(x_tip)

    """ # Visualize this
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.fill_between(x_tip, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
    ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.fill_between(x_tip, tip0, tip_activation_md, facecolor='g', alpha=0.7)
    ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_tip, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
    ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.set_title('Output membership activity')

    # Turn off top/right axes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout() """

    #-----------------------------------------------------------------------#
    #-----------------------------------------------------------------------#


    # Aggregate all three output membership functions together
    aggregated = np.fmax(tip_activation_lo,
                        np.fmax(tip_activation_md, tip_activation_hi))

    # Calculate defuzzified result
    tip = fuzz.defuzz(x_tip, aggregated, 'centroid')
    tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # for plot

    # Visualize this
    fig, ax0 = plt.subplots(figsize=(8, 3))

    ax0.plot(x_tip, tip_lo, 'b', linewidth=0.5, linestyle='--', )
    ax0.plot(x_tip, tip_md, 'g', linewidth=0.5, linestyle='--')
    ax0.plot(x_tip, tip_hi, 'r', linewidth=0.5, linestyle='--')
    ax0.fill_between(x_tip, tip0, aggregated, facecolor='Orange', alpha=0.7)
    ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
    ax0.set_title('Aggregated membership and result (line)')

    # Turn off top/right axes
    for ax in (ax0,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()