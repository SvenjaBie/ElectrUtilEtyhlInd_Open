import pyomo.environ as pm
import pandas as pd
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle
from datetime import datetime


# optimisation function definition. CHP is connected to new storage equipment and electricity can be sold to grid
def optimisation_run_CHP_feed_in(price_el_hourly, price_ng_orig, price_EUA_orig, CO2_emiss_grid_hourly, amp_values,
                             variability_values, gr_cap, hours, disc_rate, capex_data, min_load_CHP):
    """This function optimizes the heat and electricity generation system for industrial processes. An economic optimisation
 chooses the optimal combination and size of new technologies (electric boiler, thermal energy storage, battery storage,
 electrolyser, hydrogen storage and hydrogen boiler) and the use of natural gas use with an existing CHP plant,
 taking into account the price for CO2 emission-allowances for a single chosen year. In this version, selling
 electricity back to the grid is possible only for the CHP. """

    # ------------------------------------- input DATA preperation ---------------------------------------------------
    # define the resolution of the optimisation problem in hours
    time_step = 1  # in hours

    # resample natural gas price data to get prices with the same resolution of the optimisation problem
    # original data contains one price per day
    price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    # calculate and display the mean and variance of the prices used for the optimisation
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # prepare electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    # use index from natural gas price data
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    # calculate and display the mean and variance of the prices used for the optimisation
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    ## option to manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()
    # define factor by which volatility should be amplified
    amp = amp_values
    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])
            # # removing negative prices  # if done here, mean price of price curves increase with increasing k
            # price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        # removing negative prices  # if done here, mean price of price curves are all the same. TODO: revise!
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)
        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # Plot new price data with increased variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    # prepare EUA price data like done for natural gas price data
    price_EUA_hourly = price_EUA_orig.resample('{}h'.format(time_step)).ffill()
    EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
    price_EUA_hourly.index = price_ng_hourly.index
    price_EUA_hourly_mean = price_EUA_hourly.mean()
    price_EUA_hourly_mean_hours = price_EUA_hourly['Price'].iloc[:hours].mean()
    price_EUA_hourly_var_hours = price_EUA_hourly['Price'].iloc[:hours].var()
    print("Mean EUA price is " + str(price_EUA_hourly_mean_hours), ". The variance is " +
          str(price_EUA_hourly_var_hours))

    # # calculate cost for using natural gas as price for the gas + cost for CO2 emissions
    price_EUA_hourly_MWh = price_EUA_hourly * 0.2  # eur/ton * 0.2 ton(CO2)/MWh(natural gas) = eur/MWh(natural gas)
    price_NG_use_hourly = pd.DataFrame({'Cost of using natural gas': None}, index=price_ng_hourly.index)
    price_NG_use_hourly['Cost of using natural gas'] = price_EUA_hourly_MWh['Price'] + price_ng_hourly['Open']
    # calculate and display mean and variance of the resulting price for using natural gas
    price_NG_use_hourly_mean_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].mean()
    price_NG_use_hourly_var_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].var()
    print("Mean price for using NG [MWh] is " + str(price_NG_use_hourly_mean_hours), ". The variance is " +
          str(price_NG_use_hourly_var_hours))

    # # figure to display electricity and gas price(s) together
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original electricity price data', color='b')
    # price_ng_hourly['Open'].plot(x=price_ng_hourly.index, label='Original natural gas price data', color='r')
    # plt.ylabel("EUR/MWh")
    # plt.legend(loc='upper left')
    # ax = price_EUA_hourly['Price'].plot(x=price_EUA_hourly.index, secondary_y=True, label='Original CO2 emission cost data', color='g', linewidth=1.5)
    # ax.set_ylabel("EUR/ton")
    # plt.xlabel("Date")
    # plt.title("Price data for electricity (day-ahead market), natural gas (Dutch TTF market), and CO2 emission allowances (ETS)")
    # plt.legend(loc='upper right')
    # plt.show()

    # # plot prices for clarification
    # plt.plot(price_el_hourly.iloc[:hours, 0], label='Electricity price', color='b', marker='o',
    #             markersize=0.75)
    # plt.plot(price_NG_use_hourly.iloc[:hours, 0],
    #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
    #             markersize=0.75)
    # plt.ylabel("EUR/MWh")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()

    # # for hour many hours of the operational time is electricity cheaper than natural gas?
    # # calculate difference el_price - natural_gas_use
    # price_difference = price_el_hourly.iloc[:hours, 0] - price_NG_use_hourly.iloc[:hours, 0]
    # # how often is price difference < 0?
    # negative_hours = pd.Series(data=price_difference[price_difference < 0],
    #                            index=price_difference[price_difference < 0].index)
    # print(len(negative_hours))

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # price_ng_hourly_short = pd.DataFrame(price_ng_hourly.iloc[0:hours])
    # price_EUA_hourly_short = pd.DataFrame(price_EUA_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short
    # price_ng_hourly = price_ng_hourly_short
    # price_EUA_hourly = price_EUA_hourly_short

    # for running the optimisation for electricity prices with increased variability
    variability = variability_values

    # create respective dictionary that the model loops over and which stores the results of all runs
    looping_variable = variability
    # define the processes for which the model should run
    processes = ['Olefins', 'Ethylene oxide']  # process names
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'available area': None, 'results': {},
                                                                'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}
    # print(el_price_scenario_dict)

    # -------------------------------------- Start the model runs (in loop) --------------------------------------------
    for count, amp in enumerate(looping_variable):
        print("Current variability amplitude is: ", amp)
        # select the processes for which the model should run and define their heat and power demand in [MW] and
        # available space in [m^2]. Add their optimised utility demand if heat integration has been done
        # Olefins
        el_price_scenario_dict['Olefins']['non-optimized'][amp][
            'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # Ethylene Oxide
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
            'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # # Ethylbenzene
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
        #     'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # # Ethylene Glycol
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
        #     'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # # PET
        # el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        # el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        # el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        # loop over each process
        for process in processes:
            print("Current process is: ", process)
            # loop over their utility demand data (optimised after heat integration or non-optimised without doing a
            # heat integration study)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                # define the process parameters
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION ----------------------------------------------------------------
                # Definitions of constraints

                def heat_balance(m, time):  # heat demand at t has to be equal to the sum of the heat produced at t
                    return H_dem == m.H_ElB_CP[time] + m.H_CHP_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time]

                def el_balance(m, time):  # power demand at t has to be equal to the sum of the power produced or bought at t
                    return P_dem == m.P_gr_CP[time] + m.P_CHP_CP[time] + m.P_bat_CP[time]

                def ElB_balance(m, time):  # energy conversion of electric boiler
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time]) * eta_ElB

                def ElB_size(m, time):  # definition of boiler capacity
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] <= m.ElB_cap

                def CHP_ng_P_conversion(m, time):  # gas to power conversion of CHP
                    return m.NG_CHP_in[time] == (
                            m.P_CHP_CP[time] + m.P_CHP_excess[time] + m.P_CHP_bat[time] + m.P_CHP_gr[time]) / eta_CHP_el

                def CHP_ng_H_conversion(m, time):  # gas to heat conversion of CHP
                    return m.NG_CHP_in[time] == (
                            m.H_CHP_CP[time] + m.H_CHP_TES[time] + m.H_CHP_excess[time]) / eta_CHP_th

                def CHP_max_H(m, time):  # generated heat cannot exceed capacity
                    return m.H_CHP_CP[time] + m.H_CHP_TES[time] + m.H_CHP_excess[time] <= CHP_cap * eta_CHP_th

                def CHP_min_H(m, time):  # operation has to be above the minimal load
                    return m.H_CHP_CP[time] + m.H_CHP_TES[time] + m.H_CHP_excess[time] \
                           >= CHP_cap * eta_CHP_th * CHP_cap_min

                def bat_soe(m, time):  # calculating the state of energy of the battery
                    if time == 0:
                        return m.bat_soe[time] == 0
                    else:
                        return m.bat_soe[time] == m.bat_soe[time - 1] + \
                               eta_bat * time_step * (m.P_gr_bat[time - 1] + m.P_CHP_bat[time - 1]) - \
                               1 / eta_bat * time_step * (m.P_bat_CP[time - 1] + m.P_bat_ElB[time - 1]
                                                          + m.P_bat_H2E[time - 1])

                def bat_in(m, time):  # limiting the charging with c-rate and use binary to prevent simultaeous charging and discharging
                    return (m.P_gr_bat[time] + m.P_CHP_bat[time]) <= \
                           m.bat_cap / eta_bat * crate_bat / time_step * m.b1[time]

                def bat_out(m, time):  # limiting the discharging with c-rate and use binary to prevent simultaeous charging and discharging
                    if time == 0:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) == 0
                    else:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) \
                               <= m.bat_cap * eta_bat * crate_bat / time_step * (1 - m.b1[time])

                def bat_size(m, time):  # define battery capacity
                    return m.bat_soe[time] <= m.bat_cap

                def TES_soe(m, time):  # calculating the state of energy of the TES
                    if time == 0:
                        return m.TES_soe[time] == 0
                    else:
                        return m.TES_soe[time] == m.TES_soe[time - 1] + (m.H_CHP_TES[time - 1] + m.H_ElB_TES[time - 1]
                                                                         - m.H_TES_CP[time - 1] / eta_TES_A) * time_step

                def TES_in(m, time):  # limiting the charging with c-rate and use binary to prevent simultaeous charging and discharging
                    return m.H_CHP_TES[time] + m.H_ElB_TES[time] <= m.TES_cap * crate_TES / time_step * m.b2[time]

                def TES_out(m, time):  # limiting the discharging with c-rate and use binary to prevent simultaeous charging and discharging
                    if time == 0:
                        return m.H_TES_CP[time] == 0
                    else:
                        return m.H_TES_CP[time] <= m.TES_cap * eta_TES_A * crate_TES / time_step * (1 - m.b2[time])

                def TES_size(m, time):  # define TES capacity
                    return m.TES_soe[time] <= m.TES_cap

                def H2S_soe(m, time):  # calculate state of energy of the hydrogen storage
                    if time == 0:
                        return m.H2S_soe[time] == 0
                    else:
                        return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] -
                                                                         m.H2_H2S_H2B[time - 1] / eta_H2S) * time_step

                def H2S_in(m, time):  # implement binary to prevent simultaneous charging and discharging
                    return m.H2_H2E_H2S[time] <= m.H2S_cap / time_step * m.b4[time]

                def H2S_out(m, time):  # implement binary to prevent simultaneous charging and discharging
                    if time == 0:
                        return m.H2_H2S_H2B[time] == 0
                    else:
                        return m.H2_H2S_H2B[time] <= m.H2S_cap * eta_H2S / time_step * (1 - m.b4[time])

                def H2S_size(m, time):  # define hydrogen storage capacity
                    return m.H2S_soe[time] <= m.H2S_cap

                def H2B_balance(m, time):  # hydrogen to heat conversion of hyrogen boiler
                    return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

                def H2B_size(m, time):  # define hydrogen boiler size
                    return m.H_H2B_CP[time] <= m.H2B_cap

                def H2E_balance(m, time):  # power to hydrogen conversion of electrolyzer
                    return (m.P_gr_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time]

                def H2E_size(m, time):  # define capacity of electrolyzer
                    return m.P_gr_H2E[time] + m.P_bat_H2E[time] <= m.H2E_cap

                def spat_dem(m, time):  # allows to intall new units until the footprint exceeds the available space
                    return m.bat_cap * bat_areaftpr + m.ElB_cap * ElB_areaftpr + m.TES_cap * TES_areaftpr_A + \
                           m.H2E_cap * H2E_areaftpr + m.H2B_cap * H2B_areaftpr + m.H2S_cap * H2S_areaftpr \
                           <= available_area

                def max_grid_power_in(m, time):  # limit power inflow through the grid connection and use binary to prevent simultaneous bidirectional use
                    return m.P_gr_CP[time] + m.P_gr_ElB[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] <= gr_connection * \
                           m.b3[time]

                def max_grid_power_out(m, time):  # limit power outflow through the grid connection and use binary to prevent simultaneous bidirectional use
                    return m.P_CHP_gr[time] <= gr_connection * (1 - m.b3[time])

                def minimize_total_costs(m):  # define the total cost of the system
                    # cost of electricity: Price * (consumption - feed-in)
                    # cost of using natural gas: Consumption * (price of NG + cost for CO2 emission allowances)
                    return sum(
                        (price_el_hourly.iloc[time, count] + 0) * time_step * (
                                    m.P_gr_ElB[time] + m.P_gr_CP[time] +
                                    m.P_gr_bat[time] + m.P_gr_H2E[time])
                        - price_el_hourly.iloc[time, count] * time_step * m.P_CHP_gr[time]
                        + m.NG_CHP_in[time] * time_step * (price_ng_hourly.iloc[time, 0] +
                                                           price_EUA_hourly.iloc[time, 0] * EF_ng)
                        for time in m.T) \
                           + \
                           m.bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                           m.ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                           m.TES_cap * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                           m.H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                           m.H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                           m.H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

                m = pm.ConcreteModel()

                # define SETS
                m.T = pm.RangeSet(0, hours - 1)  # 'duration' of the optimisation

                # define CONSTANTS
                # Electric boiler constants
                c_ElB = capex_data['c_ElB']  # CAPEX for electric boiler, 70000 eur/MW
                ElB_lifetime = 20  # lifetime of electric boiler, years
                ElB_areaftpr = 30  # spatial requirements, m^2/MW
                eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]
                # CHP constants
                eta_CHP_el = 0.3  # Electric efficiency of CHP [%]
                eta_CHP_th = 0.4  # Thermal efficiency of CHP [%]
                CHP_cap = H_dem / eta_CHP_th  # Thermal capacity (LPS) CHP, [MW]
                CHP_cap_min = min_load_CHP  # minimal load factor, [% of Pnom]
                # rr_CHP = 0.03  # maximum ramp rate for CHP, in [% of Pnom/min]
                # Battery constants
                eta_bat = 0.95  # Battery (dis)charging efficiency
                c_bat = capex_data['c_bat']  # CAPEX for battery per eur/MWh, 338e3 USD --> 314.15e3 eur (12.07.23)
                bat_lifetime = 15  # lifetime of battery
                bat_areaftpr = 10  # spatial requirement, [m^2/MWh]
                crate_bat = 0.7  # C rate of battery, 0.7 kW/kWh, [-]
                # TES constants
                c_TES_A = capex_data['c_TES']  # CAPEX for sensible heat storage
                # c_TES_B = 14000  # CAPEX for heat storage including heater, [UDS/MWh]
                # c_TES_C = 60000  # CAPEX for latent heat storage [eur/MWh]
                TES_lifetime = 25  # heat storage lifetime, [years]
                eta_TES_A = 0.9  # discharge efficiency [-]
                eta_TES_C = 0.98  # discharge efficiency [-]
                TES_areaftpr_A = 5  # spatial requirement TES, [m^2/MWh]
                TES_areaftpr_B = 7  # spatial requirement TES (configuration B), [m^2/MWh]
                crate_TES = 0.5  # C rate of TES, 0.5 kW/kWh, [-]  #TODO: revise this number
                # Hydrogen equipment constants
                eta_H2S = 0.9  # charge efficiency hydrogen storage [-], accounting for fugitive losses
                eta_H2B = 0.92  # conversion efficiency hydrogen boiler [-]
                eta_H2E = 0.69  # conversion efficiency electrolyser [-]
                c_H2S = capex_data['c_H2S']  #10000  # CAPEX for hydrogen storage per MWh, [eur/MWh]
                c_H2B = capex_data['c_H2B']  #35000  # CAPEX for hydrogen boiler per MW, [eur/MW]
                c_H2E = capex_data['c_H2E']  #700e3  # CAPEX for electrolyser per MW, [eur/MW]
                H2S_lifetime = 20  # lifetime hydrogen storage, [years]
                H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
                H2E_lifetime = 15  # lifetime electrolyser, [years]
                H2E_areaftpr = 100  # spatial requirement electrolyser, [m^2/MW]
                H2B_areaftpr = 5  # spatial requirement hydrogen boiler, [m^2/MW]
                H2S_areaftpr = 10  # spatial requirement hydrogen storage, [m^2/MWh]
                # other constants
                EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)
                gr_connection = gr_cap * (P_dem + H_dem) / (
                        eta_bat * eta_bat * eta_H2E * eta_H2S * eta_H2B)  # connection capacity required to fully
                # electrify the system in the 'worst case' conversion chain

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # define VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.NG_CHP_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake, MWh
                m.P_CHP_CP = pm.Var(m.T, bounds=(0, None))  # Power generated from CHP to core process, MW
                m.P_CHP_bat = pm.Var(m.T, bounds=(0, None))  # Power from CHP to battery, MW
                m.P_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess power from CHP, MW
                m.P_CHP_gr = pm.Var(m.T, bounds=(0, None))  # Power from CHP to grid, MW
                m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
                m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
                m.H_CHP_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
                m.H_CHP_TES = pm.Var(m.T, bounds=(0, None))  # Heat from CHP to TES, MW
                m.H_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat from CHP, MW
                m.H_ElB_TES = pm.Var(m.T, bounds=(0, None))  # Heat from electric boiler to TES, MW
                m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
                m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
                m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
                m.P_bat_CP = pm.Var(m.T, bounds=(0, None))  # discharging power batter to core process, MW
                m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
                m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
                m.bat_cap = pm.Var(bounds=(0, None))  # Battery capacity, MWh
                m.ElB_cap = pm.Var(bounds=(0, None))  # electric boiler capacity, MW
                m.TES_cap = pm.Var(bounds=(0, None))  # TES capacity, MWh
                m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
                m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
                m.H2S_cap = pm.Var(bounds=(0, None))  # hydrogen storage capacity, MWh
                m.H2B_cap = pm.Var(bounds=(0, None))  # hydrogen boiler capacity, MW
                m.H2E_cap = pm.Var(bounds=(0, None))  # electrolyser capacity, MW
                m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
                m.H2_H2S_H2B = pm.Var(m.T,
                                      bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
                m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
                m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
                m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the battery
                m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the TES
                m.b3 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous bi-directional use of the
                # grid connection
                m.b4 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous bi-directional use of the
                # hydrogen tank

                # add CONSTRAINTS to the model
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # CHP constraints
                m.CHP_ng_P_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_P_conversion)
                m.CHP_ng_H_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_H_conversion)
                m.CHP_H_max_constraint = pm.Constraint(m.T, rule=CHP_max_H)
                m.CHP_H_min_constraint = pm.Constraint(m.T, rule=CHP_min_H)
                # electric boiler constraint
                m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
                m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
                # battery constraints
                m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
                m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out)
                m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
                m.bat_size_constraint = pm.Constraint(m.T, rule=bat_size)
                # TES constraints
                m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
                m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
                m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
                m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)
                # hydrogen constraints
                m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
                m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
                m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
                m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
                m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
                m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
                m.H2S_discharge_constraint = pm.Constraint(m.T, rule=H2S_out)
                m.H2S_charge_constraint = pm.Constraint(m.T, rule=H2S_in)
                # # spatial constraint
                # m.spat_dem_constraint = pm.Constraint(m.T, rule=spat_dem)
                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)
                m.max_grid_power_out_constraint = pm.Constraint(m.T, rule=max_grid_power_out)

                # add OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')

                # Solve optimization problem
                opt = pm.SolverFactory('gurobi')  # use gurobi solvers
                #opt.options["MIPGap"] = 0.02  # define the optimality gap that should be reached
                results = opt.solve(m, tee=True)  # solve the problem

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results in dataframe
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from electric boiler to process'] = pm.value(m.H_ElB_CP[:])
                result['Heat from electric boiler to TES'] = pm.value(m.H_ElB_TES[:])
                result['Electricity from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
                result['Heat from CHP to process'] = pm.value(m.H_CHP_CP[:])
                result['Heat from CHP to TES'] = pm.value(m.H_CHP_TES[:])
                result['Heat excess from CHP'] = pm.value(m.H_CHP_excess[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Electricity from CHP to process'] = pm.value(m.P_CHP_CP[:])
                result['Electricity from CHP to battery'] = pm.value(m.P_CHP_bat[:])
                result['Electricity from CHP to grid'] = pm.value(m.P_CHP_gr[:])
                result['Electricity excess from CHP'] = pm.value(m.P_CHP_excess[:])
                result['Battery SOE'] = pm.value(m.bat_soe[:])
                result['Electricity from battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
                result['Electricity from battery to process'] = pm.value(m.P_bat_CP[:])
                result['Electricity from grid to battery'] = pm.value(m.P_gr_bat[:])
                result['natural gas consumption [MWh]'] = pm.value(m.NG_CHP_in[:])
                result['Heat from TES to process'] = pm.value(m.H_TES_CP[:])
                result['Electricity from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
                result['Heat from hydrogen boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Electricity from battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
                result['Heat from H2 boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Hydrogen from electrolyser to H2 boiler'] = pm.value(m.H2_H2E_H2B[:])
                result['Hydrogen from electrolyser to storage'] = pm.value(m.H2_H2E_H2S[:])
                result['Hydrogen from storage to H2 boiler'] = pm.value(m.H2_H2S_H2B[:])
                result['Binary 1: Battery'] = pm.value(m.b1[:])
                result['Binary 2: TES'] = pm.value(m.b2[:])
                result['Binary 3: Grid connection'] = pm.value(m.b3[:])
                result['Binary 4: H2S'] = pm.value(m.b4[:])

                # calculate more results
                CHP_gen_CP = sum(
                    result['Heat from CHP to process'] + result['Electricity from CHP to process'])
                # check if grid capacity constraint is hit
                grid_P_out = result['Electricity from grid to process'] + \
                             result['Electricity from grid to electric boiler'] + \
                             result['Electricity from grid to battery'] + \
                             result['Electricity from grid to electrolyser']
                grid_P_out_max = max(grid_P_out)
                Grid_gen = result['Electricity from grid to process'].sum() + \
                           result['Electricity from grid to electric boiler'].sum() + \
                           result['Electricity from grid to battery'].sum() + \
                           result['Electricity from grid to electrolyser'].sum()
                ElB_gen_CP = result['Heat from electric boiler to process'].sum()
                # Battery_gen = result['Discharging battery'].sum()
                # Excess_heat = result['Excess heat from CHP(ng)'].sum()
                # Excess_electricity = result['Excess electricity from CHP'].sum()
                # Calculate scope 1 CO2 emissions
                CO2_emissions = result[
                                    'natural gas consumption [MWh]'].sum() * EF_ng * time_step  # [MWh]*[ton/MWh] = [ton]
                Cost_EUA = sum(price_EUA_hourly.iloc[time, 0] * time_step * pm.value(m.NG_CHP_in[time]) * EF_ng
                               for time in m.T)

                # Calculate Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process'],
                                                'Grid to electric boiler': result[
                                                    'Electricity from grid to electric boiler'],
                                                'Grid to battery': result['Electricity from grid to battery'],
                                                'Grid to electrolyser': result[
                                                    'Electricity from grid to electrolyser']})
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - (result['Heat from electric boiler to process'] +
                                                     result['Heat from CHP to process'] +
                                                     result['Heat from TES to process'] +
                                                     result['Heat from hydrogen boiler to process']))
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from battery to process']
                                                                  + result['Electricity from CHP to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                # display total cost and installed capacities
                print("Objective = ", pm.value(m.objective))
                # print("Investment cost battery per MWh, USD = ", c_bat)
                # print("Investment cost electric boiler per MW, USD = ", c_ElB)
                # print("Investment cost TES per MWh, USD = ", c_TES_A)
                print("Battery capacity =", pm.value(m.bat_cap))
                print("Electric boiler capacity =", pm.value(m.ElB_cap))
                print("TES capacity =", pm.value(m.TES_cap))
                print("electrolyser capacity =", pm.value(m.H2E_cap))
                print("Hydrogen boiler capacity =", pm.value(m.H2B_cap))
                print("Hydrogen storage capacity =", pm.value(m.H2S_cap))
                print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

                # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.bat_cap) > 0:
                    battery_discharge_sum = result['Electricity from battery to electrolyser'] + \
                                            result['Electricity from battery to electric boiler'] + \
                                            result['Electricity from battery to process']
                    battery_charge_sum = result['Electricity from grid to battery'] + \
                                         result['Electricity from CHP to battery']
                    battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
                    for i in range(0, len(battery_charge_sum)):
                        if battery_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if battery_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                              battery_discharge_sum[i]
                    print("Number of hours of simultaneous battery charging and discharging: ",
                          len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF TES capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.TES_cap) > 0:
                    TES_discharge_sum = result['Heat from TES to process']
                    TES_charge_sum = result['Heat from CHP to TES'] + \
                                     result['Heat from electric boiler to TES']
                    TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
                    for i in range(0, len(TES_charge_sum)):
                        if TES_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if TES_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                          TES_discharge_sum[i]
                    print("Number of hours of simultaneous TES charging and discharging: ",
                          len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF H2S capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.H2S_cap) > 0:
                    H2S_discharge_sum = result['Hydrogen from storage to H2 boiler']
                    H2S_charge_sum = result['Hydrogen from electrolyser to storage']
                    H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
                    for i in range(0, len(H2S_charge_sum)):
                        if H2S_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if H2S_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                          H2S_discharge_sum[i]
                    print("Number of hours of simultaneous H2S charging and discharging: ",
                          len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0]))

                # Check if grid connection is simultaneously used in a bidirectional manner
                hours_with_simultaneous_gridcon_use = pd.Series(index=grid_P_out.index)
                for i in range(0, len(grid_P_out)):
                    if grid_P_out[i] > 0.00001:  # because using 0 led to rounding errors
                        if pm.value(m.P_CHP_gr[i]) > 0.00001:  # because using 0 led to rounding errors
                            hours_with_simultaneous_gridcon_use[i] = grid_P_out[i] + pm.value(m.P_CHP_gr[i])
                print("Number of hours of simultaneous bidirectional grid connection use: ",
                      len(hours_with_simultaneous_gridcon_use[hours_with_simultaneous_gridcon_use > 0]))
                #
                # # energy flows and prices in one figure for analysis
                # fig, axs = plt.subplots(2, sharex=True)
                # # # grid flows
                # axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                #             color='lightcoral', marker='.')
                # # all flows from the CHP to the process and from the CHP to the grid
                # axs[0].plot(result['Electricity from CHP to process'], label='Electricity from CHP to process',
                #             color='brown', marker='.')
                # axs[0].plot(result['Heat from CHP to process'], label='Heat from CHP to process', color='red',
                #             marker='.')
                # axs[0].plot(result['Electricity excess from CHP'], label='Excess electricity from CHP', color='black',
                #             marker='s')
                # axs[0].plot(result['Heat excess from CHP'], label='Excess Heat from CHP', color='coral', marker='s')
                # axs[0].plot(result['Electricity from CHP to grid'], marker='*',
                #             label='Electricity from CHP to grid', color='chocolate')
                # # battery flows
                # if pm.value(m.bat_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to battery'], label='Electricity from grid to battery',
                #                 color='gold', marker='.')
                #     axs[0].plot(result['Electricity from CHP to battery'], label='Electricity from CHP to battery',
                #                 color='tan', marker='.')
                #     axs[0].plot(result['Electricity from battery to process'],
                #                 label='Electricity from battery to process', color='darkkhaki', marker='s')
                #     if pm.value(m.ElB_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electric boiler'],
                #                     label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
                #     if pm.value(m.H2E_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electrolyser'],
                #                     label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
                #     #axs[0].plot(result['Battery SOE'], label='Battery SOE', marker='2')
                # # # electric boiler flows
                # if pm.value(m.ElB_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electric boiler'],
                #                 label='Electricity from grid to electric boiler', color='seagreen', marker='.')
                #     axs[0].plot(result['Heat from electric boiler to process'],
                #                 label='Heat from electric boiler to process', color='turquoise', marker='.')
                #     if pm.value(m.TES_cap) > 0:
                #         axs[0].plot(result['Heat from electric boiler to TES'],
                #                     label='Heat from electric boiler to TES',
                #                     color='lime', marker='.')
                # # TES flows
                # if pm.value(m.TES_cap) > 0:
                #     axs[0].plot(result['Heat from TES to process'], label='Heat from TES to process',
                #                 color='deepskyblue', marker='.')
                #     axs[0].plot(result['Heat from CHP to TES'], label='Heat from CHP to TES',
                #                 marker='.')
                # # # Hydrogen flows
                # if pm.value(m.H2E_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electrolyser'],
                #                 label='Electricity from grid to electrolyser', color='royalblue', marker='.')
                #     axs[0].plot(result['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
                #                 color='blueviolet', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
                #                 label='Hydrogen from electrolyser to H2 boiler', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to storage'], color='fuchsia',
                #                 label='Hydrogen from electrolyser to storage', marker='.')
                #     axs[0].plot(result['Hydrogen from storage to H2 boiler'], color='deeppink',
                #                 label='Hydrogen from storage to H2 boiler', marker='.')
                # axs[0].axhline(y=gr_connection, color='grey', linestyle='--', label='Grid connection capacity')
                # axs[0].set_ylabel("MW")
                # axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
                #
                # # plot prices for clarification
                # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                #             markersize=0.75)
                # axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
                #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
                #             markersize=0.75)
                # axs[1].set_ylabel("EUR/MWh")
                # axs[1].legend()
                # # ax2 = axs[1].twinx()
                # # ax2.plot(price_EUA_hourly.iloc[:hours, 0], label='CO2 emission cost', color='g', marker='o',
                # #          markersize=0.75)
                # # ax2.set_ylabel("EUR/ton")
                # # ax2.legend(loc='upper right')
                # plt.xlabel("Date")
                # plt.show()

                # Add aggregated data to dictionary containing the results of all model runs
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                    pm.value(m.ElB_cap) * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                    pm.value(m.TES_cap) * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                    pm.value(m.H2E_cap) * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                    pm.value(m.H2B_cap) * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                    pm.value(m.H2S_cap) * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat + \
                    pm.value(m.ElB_cap) * c_ElB + \
                    pm.value(m.TES_cap) * c_TES_A + \
                    pm.value(m.H2E_cap) * c_H2E + \
                    pm.value(m.H2B_cap) * c_H2B + \
                    pm.value(m.H2S_cap) * c_H2S
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = CO2_emissions
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = Cost_EUA
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = \
                    pm.value(m.bat_cap) * bat_areaftpr + pm.value(m.ElB_cap) * ElB_areaftpr + \
                    pm.value(m.TES_cap) * TES_areaftpr_A + pm.value(m.H2E_cap) * H2E_areaftpr + \
                    pm.value(m.H2B_cap) * H2B_areaftpr + pm.value(m.H2S_cap) * H2S_areaftpr
                el_price_scenario_dict[process][run][amp]['results']['CHP gen to CP'] = CHP_gen_CP
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to CP'] = \
                    result['Heat from CHP to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to TES'] = \
                    result['Heat from CHP to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP excess heat gen'] = \
                    result['Heat excess from CHP'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to CP'] = \
                    result['Electricity from CHP to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to battery'] = \
                    result['Electricity from CHP to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP excess power gen'] = \
                    result['Electricity excess from CHP'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to grid'] = \
                    result['Electricity from CHP to grid'].sum()
                el_price_scenario_dict[process][run][amp]['results'][
                    'Simultaneous bidirectional use of grid connection [hours]'] \
                    = len(hours_with_simultaneous_gridcon_use[
                              hours_with_simultaneous_gridcon_use > 0])
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = Grid_gen
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = \
                    result['natural gas consumption [MWh]'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = \
                    result['Electricity from grid to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = \
                    result['Electricity from grid to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = \
                    result['Electricity from grid to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = ElB_gen_CP
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to TES'] = \
                    result['Heat from electric boiler to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = pm.value(m.ElB_cap)
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = pm.value(m.bat_cap)
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = \
                    result['Electricity from battery to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = \
                    result['Electricity from battery to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = \
                    result['Electricity from battery to electrolyser'].sum()
                if pm.value(m.bat_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours Battery'] \
                        = len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0])
                else:
                    el_price_scenario_dict[process][run][amp]['results']['Simultaeous charging and discharging hours'] \
                        = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = pm.value(m.TES_cap)
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = \
                    result['Heat from TES to process'].sum()
                if pm.value(m.TES_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours TES'] \
                        = len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = pm.value(m.H2E_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = pm.value(m.H2B_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = pm.value(m.H2S_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = result[
                    'Heat from H2 boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = \
                    result['Hydrogen from electrolyser to H2 boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = \
                    result['Hydrogen from electrolyser to storage'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = \
                    result['Hydrogen from storage to H2 boiler'].sum()
                if pm.value(m.H2S_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours H2S'] \
                        = len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = disc_rate
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = \
                    CO2_emissions / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = Cost_EUA / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # add 'result' dataframe with energy flows to the dict
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                #print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")

    return el_price_scenario_dict

# optimisation function definition. GB instead of CHP.
def optimisation_run_GB(price_el_hourly, price_ng_orig, price_EUA_orig, CO2_emiss_grid_hourly, amp_values,
                             variability_values, gr_cap, hours, disc_rate, capex_data, min_load_GB):
    """This function optimizes the heat and electricity generation system for industrial processes. An economic optimisation
 chooses the optimal combination and size of new technologies (electric boiler, thermal energy storage, battery storage,
 electrolyser, hydrogen storage and hydrogen boiler) and the use of natural gas use with an existing gas boiler,
 taking into account the price for CO2 emission-allowances for a single chosen year. """

    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 1  # in hours

    # TODO: Implement warning if dataset contains NaNs (for all input data)
    # natural gas price data
    price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    ## manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()

    # define factor by which volatility should be amplified
    amp = amp_values

    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])
            # # removing negative prices  # if done here, mean price of price curves increase with increasing k
            # price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        # removing negative prices  # if done here, mean price of price curves are all the same. TODO: revise!
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)

        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # figure variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    # EUA price data
    price_EUA_hourly = price_EUA_orig.resample('{}h'.format(time_step)).ffill()
    EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
    price_EUA_hourly.index = price_ng_hourly.index
    price_EUA_hourly_mean = price_EUA_hourly.mean()
    price_EUA_hourly_mean_hours = price_EUA_hourly['Price'].iloc[:hours].mean()
    price_EUA_hourly_var_hours = price_EUA_hourly['Price'].iloc[:hours].var()
    print("Mean EUA price is " + str(price_EUA_hourly_mean_hours), ". The variance is " +
          str(price_EUA_hourly_var_hours))

    # # calculate cost for using natural gas as price for the gas + cost for CO2 emissions
    price_EUA_hourly_MWh = price_EUA_hourly * 0.2  # eur/ton * 0.2 ton(CO2)/MWh(natural gas) = eur/MWh(natural gas)
    price_NG_use_hourly = pd.DataFrame({'Cost of using natural gas': None}, index=price_ng_hourly.index)
    price_NG_use_hourly['Cost of using natural gas'] = price_EUA_hourly_MWh['Price'] + price_ng_hourly['Open']
    price_NG_use_hourly_mean_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].mean()
    price_NG_use_hourly_var_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].var()
    print("Mean price for using NG [MWh] is " + str(price_NG_use_hourly_mean_hours), ". The variance is " +
          str(price_NG_use_hourly_var_hours))

    # # figure to display electricity and gas price(s) together
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original electricity price data', color='b')
    # price_ng_hourly['Open'].plot(x=price_ng_hourly.index, label='Original natural gas price data', color='r')
    # plt.ylabel("EUR/MWh")
    # plt.legend(loc='upper left')
    # ax = price_EUA_hourly['Price'].plot(x=price_EUA_hourly.index, secondary_y=True, label='Original CO2 emission cost data', color='g', linewidth=1.5)
    # ax.set_ylabel("EUR/ton")
    # plt.xlabel("Date")
    # plt.title("Price data for electricity (day-ahead market), natural gas (Dutch TTF market), and CO2 emission allowances (ETS)")
    # plt.legend(loc='upper right')
    # plt.show()

    # # plot prices for clarification
    # plt.plot(price_el_hourly.iloc[:hours, 0], label='Electricity price', color='b', marker='o',
    #             markersize=0.75)
    # plt.plot(price_NG_use_hourly.iloc[:hours, 0],
    #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
    #             markersize=0.75)
    # plt.ylabel("EUR/MWh")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()

    # # for hour many hours of the operational time is electricity cheaper than natural gas?
    # # calculate difference el_price - natural_gas_use
    # price_difference = price_el_hourly.iloc[:hours, 0] - price_NG_use_hourly.iloc[:hours, 0]
    # # how often is price difference < 0?
    # negative_hours = pd.Series(data=price_difference[price_difference < 0],
    #                            index=price_difference[price_difference < 0].index)
    # print(len(negative_hours))

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # price_ng_hourly_short = pd.DataFrame(price_ng_hourly.iloc[0:hours])
    # price_EUA_hourly_short = pd.DataFrame(price_EUA_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short
    # price_ng_hourly = price_ng_hourly_short
    # price_EUA_hourly = price_EUA_hourly_short

    # for electricity
    variability = variability_values

    # create respective dictionary
    looping_variable = variability
    processes = ['Ethylbenzene', 'Ethylene glycol','PET']  # 'Olefins', 'Ethylene oxide', 'Ethylbenzene', 'Ethylene glycol', 'PET']  # process names
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'available area': None, 'results': {},
                                                                'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}
    # print(el_price_scenario_dict)

    # for amp in variability:
    for count, amp in enumerate(looping_variable):
        print("Current variability amplitude is: ", amp)
        # # Olefins
        # el_price_scenario_dict['Olefins']['non-optimized'][amp][
        #     'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        # el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        # el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # # Ethylene Oxide
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
        #     'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # Ethylbenzene
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
            'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # Ethylene Glycol
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
            'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # PET
        el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_ElB_CP[time] + m.H_GB_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time] + m.P_bat_CP[time]

                def ElB_balance(m, time):
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time]) * eta_ElB

                def ElB_size(m, time):
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] <= m.ElB_cap


                def GB_balance(m, time):
                    return m.NG_GB_in[time] == (m.H_GB_CP[time] + m.H_GB_TES[time] + m.H_GB_excess[time]) / eta_GB

                def GB_max_H(m, time):
                    return m.H_GB_CP[time] + m.H_GB_TES[time] + m.H_GB_excess[time] <= GB_cap * eta_GB

                def GB_min_H(m, time):
                    return m.H_GB_CP[time] + m.H_GB_TES[time] + m.H_GB_excess[time] \
                           >= GB_cap * eta_GB * GB_cap_min

                def bat_soe(m, time):
                    if time == 0:
                        return m.bat_soe[time] == 0
                    else:
                        return m.bat_soe[time] == m.bat_soe[time - 1] + \
                               eta_bat * time_step * m.P_gr_bat[time - 1] - \
                               1 / eta_bat * time_step * (m.P_bat_CP[time - 1] + m.P_bat_ElB[time - 1]
                                                          + m.P_bat_H2E[time - 1])

                # Use Big M method to avoid simultaneous charging and discharging of the battery
                # TODO: revise crate
                def bat_in(m, time):
                    return m.P_gr_bat[time] <= \
                           m.bat_cap / eta_bat * crate_bat / time_step * m.b1[time]

                # TODO: revise crate
                def bat_out(m, time):
                    if time == 0:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) == 0
                    else:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) \
                               <= m.bat_cap * eta_bat * crate_bat / time_step * (1 - m.b1[time])

                def bat_size(m, time):
                    return m.bat_soe[time] <= m.bat_cap

                def TES_soe(m, time):
                    if time == 0:
                        return m.TES_soe[time] == 0
                    else:
                        return m.TES_soe[time] == m.TES_soe[time - 1] + (m.H_GB_TES[time - 1] + m.H_ElB_TES[time - 1]
                                                                         - m.H_TES_CP[time - 1] / eta_TES_A) * time_step

                # Use Big M method to avoid simultaneous charging and discharging of the TES
                # TODO: revise crate!
                def TES_in(m, time):
                    return m.H_GB_TES[time] + m.H_ElB_TES[time] <= m.TES_cap * crate_TES / time_step * m.b2[time]

                # TODO: revise crate!
                def TES_out(m, time):
                    if time == 0:
                        return m.H_TES_CP[time] == 0
                    else:
                        return m.H_TES_CP[time] <= m.TES_cap * eta_TES_A * crate_TES / time_step * (1 - m.b2[time])

                def TES_size(m, time):
                    return m.TES_soe[time] <= m.TES_cap

                def H2S_soe(m, time):
                    if time == 0:
                        return m.H2S_soe[time] == 0
                    else:
                        return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] -
                                                                         m.H2_H2S_H2B[time - 1] / eta_H2S) * time_step

                def H2S_in(m, time):
                    return m.H2_H2E_H2S[time] <= m.H2S_cap / time_step * m.b3[time]

                def H2S_out(m, time):
                    if time == 0:
                        return m.H2_H2S_H2B[time] == 0
                    else:
                        return m.H2_H2S_H2B[time] <= m.H2S_cap * eta_H2S / time_step * (1 - m.b3[time])

                def H2S_size(m, time):
                    return m.H2S_soe[time] <= m.H2S_cap

                def H2B_balance(m, time):
                    return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

                def H2B_size(m, time):
                    return m.H_H2B_CP[time] <= m.H2B_cap

                def H2E_balance(m, time):
                    return (m.P_gr_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time]

                def H2E_size(m, time):
                    return m.P_gr_H2E[time] + m.P_bat_H2E[time] <= m.H2E_cap

                def spat_dem(m, time):
                    return m.bat_cap * bat_areaftpr + m.ElB_cap * ElB_areaftpr + m.TES_cap * TES_areaftpr_A + \
                           m.H2E_cap * H2E_areaftpr + m.H2B_cap * H2B_areaftpr + m.H2S_cap * H2S_areaftpr \
                           <= available_area

                def max_grid_power_in(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_gr_CP[time] + m.P_gr_ElB[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] <= gr_connection


                def minimize_total_costs(m):  # 1.0000000000001
                    # cost of electricity: Price * (consumption - feed-in)
                    # cost of using natural gas: Consumption * (price of NG + cost for CO2 emission allowances)
                    return sum(
                        (price_el_hourly.iloc[time, count] + 0) * time_step * (
                                    m.P_gr_ElB[time] + m.P_gr_CP[time] + m.P_gr_bat[time] + m.P_gr_H2E[time])
                        + m.NG_GB_in[time] * time_step * (price_ng_hourly.iloc[time, 0] +
                                                           price_EUA_hourly.iloc[time, 0] * EF_ng)
                        for time in m.T) \
                           + \
                           m.bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                           m.ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                           m.TES_cap * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                           m.H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                           m.H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                           m.H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                # Electric boiler
                c_ElB = capex_data['c_ElB']  # CAPEX for electric boiler, 70000 eur/MW
                ElB_lifetime = 20  # lifetime of electric boiler, years
                ElB_areaftpr = 30  # spatial requirements, m^2/MW
                eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]

                # GB
                eta_GB = 0.9  # Thermal efficiency of GB [%]
                GB_cap = H_dem / eta_GB  # Thermal capacity (LPS) CHP, [MW]
                GB_cap_min = min_load_GB  # minimal load factor, [% of Pnom]

                # Battery constants
                eta_bat = 0.95  # Battery (dis)charging efficiency
                c_bat = capex_data['c_bat']  # CAPEX for battery per eur/MWh, 338e3 USD --> 314.15e3 eur (12.07.23)
                bat_lifetime = 15  # lifetime of battery
                bat_areaftpr = 10  # spatial requirement, [m^2/MWh]
                crate_bat = 0.7  # C rate of battery, 0.7 kW/kWh, [-]

                # TES constants
                c_TES_A = capex_data['c_TES']  # CAPEX for sensible heat storage
                # c_TES_B = 14000  # CAPEX for heat storage including heater, [UDS/MWh]
                # c_TES_C = 60000  # CAPEX for latent heat storage [eur/MWh]
                TES_lifetime = 25  # heat storage lifetime, [years]
                eta_TES_A = 0.9  # discharge efficiency [-]
                eta_TES_C = 0.98  # discharge efficiency [-]
                TES_areaftpr_A = 5  # spatial requirement TES, [m^2/MWh]
                TES_areaftpr_B = 7  # spatial requirement TES (configuration B), [m^2/MWh]
                crate_TES = 0.5  # C rate of TES, 0.5 kW/kWh, [-]  #TODO: revise this number

                # Hydrogen equipment constants
                eta_H2S = 0.9  # charge efficiency hydrogen storage [-], accounting for fugitive losses
                eta_H2B = 0.92  # conversion efficiency hydrogen boiler [-]
                eta_H2E = 0.69  # conversion efficiency electrolyser [-]
                c_H2S = capex_data['c_H2S']  # CAPEX for hydrogen storage per MWh, [eur/MWh]
                c_H2B = capex_data['c_H2B']  # CAPEX for hydrogen boiler per MW, [eur/MW]
                c_H2E = capex_data['c_H2E']  # CAPEX for electrolyser per MW, [eur/MW]
                H2S_lifetime = 20  # lifetime hydrogen storage, [years]
                H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
                H2E_lifetime = 15  # lifetime electrolyser, [years]
                H2E_areaftpr = 100  # spatial requirement electrolyser, [m^2/MW]
                H2B_areaftpr = 5  # spatial requirement hydrogen boiler, [m^2/MW]
                H2S_areaftpr = 10  # spatial requirement hydrogen storage, [m^2/MWh]

                # other
                EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)
                # Todo: Discuss grid connection assumption
                gr_connection = gr_cap * (P_dem + H_dem) / (
                        eta_bat * eta_bat * eta_H2E * eta_H2S * eta_H2B)  # 'worst case' conversion chain

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.NG_GB_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake, MWh
                m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
                m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
                m.H_GB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
                m.H_GB_TES = pm.Var(m.T, bounds=(0, None))  # Heat from CHP to TES, MW
                m.H_GB_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat from CHP, MW
                m.H_ElB_TES = pm.Var(m.T, bounds=(0, None))  # Heat from electric boiler to TES, MW
                m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
                m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
                m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
                m.P_bat_CP = pm.Var(m.T, bounds=(0, None))  # discharging power batter to core process, MW
                m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
                m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
                m.bat_cap = pm.Var(bounds=(0, None))  # Battery capacity, MWh
                m.ElB_cap = pm.Var(bounds=(0, None))  # electric boiler capacity, MW
                m.TES_cap = pm.Var(bounds=(0, None))  # TES capacity, MWh
                m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
                m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
                m.H2S_cap = pm.Var(bounds=(0, None))  # hydrogen storage capacity, MWh
                m.H2B_cap = pm.Var(bounds=(0, None))  # hydrogen boiler capacity, MW
                m.H2E_cap = pm.Var(bounds=(0, None))  # electrolyser capacity, MW
                m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
                m.H2_H2S_H2B = pm.Var(m.T,
                                      bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
                m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
                m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
                m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the battery
                m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the TES
                m.b3 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the H2 storage

                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # CHP constraints
                m.GB_balance_constraint = pm.Constraint(m.T, rule=GB_balance)
                m.GB_H_max_constraint = pm.Constraint(m.T, rule=GB_max_H)
                m.GB_H_min_constraint = pm.Constraint(m.T, rule=GB_min_H)
                # electric boiler constraint
                m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
                m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
                # battery constraints
                m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
                m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out)
                m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
                m.bat_size_constraint = pm.Constraint(m.T, rule=bat_size)
                # TES constraints
                m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
                m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
                m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
                m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)

                # hydrogen constraints
                m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
                m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
                m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
                m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
                m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
                m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
                m.H2S_discharge_constraint = pm.Constraint(m.T, rule=H2S_out)
                m.H2S_charge_constraint = pm.Constraint(m.T, rule=H2S_in)

                # # spatial constraint
                # m.spat_dem_constraint = pm.Constraint(m.T, rule=spat_dem)

                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')

                # Solve optimization problem
                # reduce the optimality gap that should be reached
                opt = pm.SolverFactory('gurobi')
                #opt.options["MIPGap"] = 0.02
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from electric boiler to process'] = pm.value(m.H_ElB_CP[:])
                result['Heat from electric boiler to TES'] = pm.value(m.H_ElB_TES[:])
                result['Electricity from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
                result['Heat from GB to process'] = pm.value(m.H_GB_CP[:])
                result['Heat from GB to TES'] = pm.value(m.H_GB_TES[:])
                result['Heat excess from GB'] = pm.value(m.H_GB_excess[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Battery SOE'] = pm.value(m.bat_soe[:])
                result['Electricity from battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
                result['Electricity from battery to process'] = pm.value(m.P_bat_CP[:])
                result['Electricity from grid to battery'] = pm.value(m.P_gr_bat[:])
                result['natural gas consumption [MWh]'] = pm.value(m.NG_GB_in[:])
                result['Heat from TES to process'] = pm.value(m.H_TES_CP[:])
                result['Electricity from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
                result['Heat from hydrogen boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Electricity from battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
                result['Heat from H2 boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Hydrogen from electrolyser to H2 boiler'] = pm.value(m.H2_H2E_H2B[:])
                result['Hydrogen from electrolyser to storage'] = pm.value(m.H2_H2E_H2S[:])
                result['Hydrogen from storage to H2 boiler'] = pm.value(m.H2_H2S_H2B[:])
                result['Binary 1: Battery'] = pm.value(m.b1[:])
                result['Binary 2: TES'] = pm.value(m.b2[:])


                # check if grid capacity constraint is hit
                grid_P_out = result['Electricity from grid to process'] + \
                             result['Electricity from grid to electric boiler'] + \
                             result['Electricity from grid to battery'] + \
                             result['Electricity from grid to electrolyser']
                grid_P_out_max = max(grid_P_out)
                Grid_gen = result['Electricity from grid to process'].sum() + \
                           result['Electricity from grid to electric boiler'].sum() + \
                           result['Electricity from grid to battery'].sum() + \
                           result['Electricity from grid to electrolyser'].sum()
                ElB_gen_CP = result['Heat from electric boiler to process'].sum()
                # Battery_gen = result['Discharging battery'].sum()
                # Excess_heat = result['Excess heat from CHP(ng)'].sum()
                # Excess_electricity = result['Excess electricity from CHP'].sum()
                CO2_emissions = result[
                                    'natural gas consumption [MWh]'].sum() * EF_ng * time_step  # [MWh]*[ton/MWh] = [ton]
                Cost_EUA = sum(price_EUA_hourly.iloc[time, 0] * time_step * pm.value(m.NG_GB_in[time]) * EF_ng
                               for time in m.T)

                # Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process'],
                                                'Grid to electric boiler': result[
                                                    'Electricity from grid to electric boiler'],
                                                'Grid to battery': result['Electricity from grid to battery'],
                                                'Grid to electrolyser': result[
                                                    'Electricity from grid to electrolyser']})
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - (result['Heat from electric boiler to process'] +
                                                     result['Heat from GB to process'] +
                                                     result['Heat from TES to process'] +
                                                     result['Heat from hydrogen boiler to process']))
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from battery to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))
                # print("Investment cost battery per MWh, USD = ", c_bat)
                # print("Investment cost electric boiler per MW, USD = ", c_ElB)
                # print("Investment cost TES per MWh, USD = ", c_TES_A)
                print("Battery capacity =", pm.value(m.bat_cap))
                print("Electric boiler capacity =", pm.value(m.ElB_cap))
                print("TES capacity =", pm.value(m.TES_cap))
                print("electrolyser capacity =", pm.value(m.H2E_cap))
                print("Hydrogen boiler capacity =", pm.value(m.H2B_cap))
                print("Hydrogen storage capacity =", pm.value(m.H2S_cap))
                print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

                # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.bat_cap) > 0:
                    battery_discharge_sum = result['Electricity from battery to electrolyser'] + \
                                            result['Electricity from battery to electric boiler'] + \
                                            result['Electricity from battery to process']
                    battery_charge_sum = result['Electricity from grid to battery']
                    battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
                    for i in range(0, len(battery_charge_sum)):
                        if battery_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if battery_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                              battery_discharge_sum[i]
                    print("Number of hours of simultaneous battery charging and discharging: ",
                          len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF TES capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.TES_cap) > 0:
                    TES_discharge_sum = result['Heat from TES to process']
                    TES_charge_sum = result['Heat from GB to TES'] + result['Heat from electric boiler to TES']
                    TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
                    for i in range(0, len(TES_charge_sum)):
                        if TES_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if TES_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                          TES_discharge_sum[i]
                    print("Number of hours of simultaneous TES charging and discharging: ",
                          len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF H2S capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.H2S_cap) > 0:
                    H2S_discharge_sum = result['Hydrogen from storage to H2 boiler']
                    H2S_charge_sum = result['Hydrogen from electrolyser to storage']
                    H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
                    for i in range(0, len(H2S_charge_sum)):
                        if H2S_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if H2S_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                          H2S_discharge_sum[i]
                    print("Number of hours of simultaneous H2S charging and discharging: ",
                          len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0]))


                # # energy flows and prices in one figure for analysis
                # fig, axs = plt.subplots(2, sharex=True)
                # # # grid flows
                # axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                #             color='lightcoral', marker='.')
                # # all flows from the GB to the process
                # axs[0].plot(result['Heat from GB to process'], label='Heat from GB to process', color='red',
                #             marker='.')
                # axs[0].plot(result['Heat excess from GB'], label='Excess Heat from GB', color='coral', marker='s')
                # # battery flows
                # if pm.value(m.bat_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to battery'], label='Electricity from grid to battery',
                #                 color='gold', marker='.')
                #     axs[0].plot(result['Electricity from battery to process'],
                #                 label='Electricity from battery to process', color='darkkhaki', marker='s')
                #     if pm.value(m.ElB_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electric boiler'],
                #                     label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
                #     if pm.value(m.H2E_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electrolyser'],
                #                     label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
                #     #axs[0].plot(result['Battery SOE'], label='Battery SOE', marker='2')
                # # # electric boiler flows
                # if pm.value(m.ElB_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electric boiler'],
                #                 label='Electricity from grid to electric boiler', color='seagreen', marker='.')
                #     axs[0].plot(result['Heat from electric boiler to process'],
                #                 label='Heat from electric boiler to process', color='turquoise', marker='.')
                #     if pm.value(m.TES_cap) > 0:
                #         axs[0].plot(result['Heat from electric boiler to TES'],
                #                     label='Heat from electric boiler to TES',
                #                     color='lime', marker='.')
                # # TES flows
                # if pm.value(m.TES_cap) > 0:
                #     axs[0].plot(result['Heat from TES to process'], label='Heat from TES to process',
                #                 color='deepskyblue', marker='.')
                #     axs[0].plot(result['Heat from GB to TES'], label='Heat from GB to TES',
                #                 marker='.')
                # # # Hydrogen flows
                # if pm.value(m.H2E_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electrolyser'],
                #                 label='Electricity from grid to electrolyser', color='royalblue', marker='.')
                #     axs[0].plot(result['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
                #                 color='blueviolet', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
                #                 label='Hydrogen from electrolyser to H2 boiler', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to storage'], color='fuchsia',
                #                 label='Hydrogen from electrolyser to storage', marker='.')
                #     axs[0].plot(result['Hydrogen from storage to H2 boiler'], color='deeppink',
                #                 label='Hydrogen from storage to H2 boiler', marker='.')
                # axs[0].axhline(y=gr_connection, color='grey', linestyle='--', label='Grid connection capacity')
                # axs[0].set_ylabel("MW")
                # axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
                #
                # # plot prices for clarification
                # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                #             markersize=0.75)
                # axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
                #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
                #             markersize=0.75)
                # axs[1].set_ylabel("EUR/MWh")
                # axs[1].legend()
                # # ax2 = axs[1].twinx()
                # # ax2.plot(price_EUA_hourly.iloc[:hours, 0], label='CO2 emission cost', color='g', marker='o',
                # #          markersize=0.75)
                # # ax2.set_ylabel("EUR/ton")
                # # ax2.legend(loc='upper right')
                # plt.xlabel("Date")
                # plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                    pm.value(m.ElB_cap) * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                    pm.value(m.TES_cap) * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                    pm.value(m.H2E_cap) * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                    pm.value(m.H2B_cap) * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                    pm.value(m.H2S_cap) * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat + \
                    pm.value(m.ElB_cap) * c_ElB + \
                    pm.value(m.TES_cap) * c_TES_A + \
                    pm.value(m.H2E_cap) * c_H2E + \
                    pm.value(m.H2B_cap) * c_H2B + \
                    pm.value(m.H2S_cap) * c_H2S
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = CO2_emissions
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = Cost_EUA
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = \
                    pm.value(m.bat_cap) * bat_areaftpr + pm.value(m.ElB_cap) * ElB_areaftpr + \
                    pm.value(m.TES_cap) * TES_areaftpr_A + pm.value(m.H2E_cap) * H2E_areaftpr + \
                    pm.value(m.H2B_cap) * H2B_areaftpr + pm.value(m.H2S_cap) * H2S_areaftpr
                el_price_scenario_dict[process][run][amp]['results']['GB heat gen to CP'] = \
                    result['Heat from GB to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['GB heat gen to TES'] = \
                    result['Heat from GB to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['GB excess heat gen'] = \
                    result['Heat excess from GB'].sum()
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = Grid_gen
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = \
                    result['natural gas consumption [MWh]'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = \
                    result['Electricity from grid to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = \
                    result['Electricity from grid to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = \
                    result['Electricity from grid to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = ElB_gen_CP
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to TES'] = \
                    result['Heat from electric boiler to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = pm.value(m.ElB_cap)
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = pm.value(m.bat_cap)
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = \
                    result['Electricity from battery to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = \
                    result['Electricity from battery to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = \
                    result['Electricity from battery to electrolyser'].sum()
                if pm.value(m.bat_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours Battery'] \
                        = len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0])
                else:
                    el_price_scenario_dict[process][run][amp]['results']['Simultaeous charging and discharging hours'] \
                        = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = pm.value(m.TES_cap)
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = \
                    result['Heat from TES to process'].sum()
                if pm.value(m.TES_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours TES'] \
                        = len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = pm.value(m.H2E_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = pm.value(m.H2B_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = pm.value(m.H2S_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = result[
                    'Heat from H2 boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = \
                    result['Hydrogen from electrolyser to H2 boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = \
                    result['Hydrogen from electrolyser to storage'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = \
                    result['Hydrogen from storage to H2 boiler'].sum()
                if pm.value(m.H2S_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours H2S'] \
                        = len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = disc_rate
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = \
                    CO2_emissions / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = Cost_EUA / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # add 'result' dataframe with energy flows to the dict
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                #print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")

    return el_price_scenario_dict



# ______________________Scheduling optimisation for benchmark utility system_____________________________________________
def benchmark_CHP_scheduling_optimisation(price_el_hourly, price_ng_orig, price_EUA_orig, CO2_emiss_grid_hourly,
                                             amp_values, variability_values, hours, gr_cap, CHP_min_load):
    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 1  # in hours
    # TODO: Implement warning if dataset contains NaNs (for all input data)
    # natural gas price data
    price_ng_hourly = price_ng_orig.resample('{}H'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    ## manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()

    # define factor by which volatility should be amplified
    amp = amp_values

    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])


        # removing negative prices  # mean price of price curves are not(!) the same
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)

        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # figure variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    # EUA price data
    price_EUA_hourly = price_EUA_orig.resample('{}H'.format(time_step)).ffill()
    EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
    price_EUA_hourly.index = price_ng_hourly.index
    price_EUA_hourly_mean = price_EUA_hourly.mean()
    price_EUA_hourly_mean_hours = price_EUA_hourly['Price'].iloc[:hours].mean()
    price_EUA_hourly_var_hours = price_EUA_hourly['Price'].iloc[:hours].var()
    print("Mean EUA price is " + str(price_EUA_hourly_mean_hours), ". The variance is " +
          str(price_EUA_hourly_var_hours))

    # # figure to display electricity and gas price(s) together
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original electricity price data', color='b')
    # price_ng_hourly['Open'].plot(x=price_ng_hourly.index, label='Original natural gas price data', color='r')
    # plt.ylabel("EUR/MWh")
    # plt.legend(loc='upper left')
    # ax = price_EUA_hourly['Price'].plot(x=price_EUA_hourly.index, secondary_y=True, label='Original CO2 emission cost data', color='g', linewidth=1.5)
    # ax.set_ylabel("EUR/ton")
    # plt.xlabel("Date")
    # plt.title("Price data for electricity (day-ahead market), natural gas (Dutch TTF market), and CO2 emission allowances (ETS)")
    # plt.legend(loc='upper right')
    # plt.show()

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # price_ng_hourly_short = pd.DataFrame(price_ng_hourly.iloc[0:hours])
    # price_EUA_hourly_short = pd.DataFrame(price_EUA_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short
    # price_ng_hourly = price_ng_hourly_short
    # price_EUA_hourly = price_EUA_hourly_short

    # create respective dictionary
    looping_variable = variability_values
    processes = ['Olefins benchmark', 'Ethylene oxide benchmark'] #, 'Ethylbenzene benchmark', 'Ethylene glycol benchmark', 'PET benchmark']
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'results': {}, 'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}
    print(el_price_scenario_dict)

    # for amp in variability:
    for count, amp in enumerate(looping_variable):
        print("Current scenario is: ", amp)
        # Olefins
        el_price_scenario_dict['Olefins benchmark']['non-optimized'][amp][
            'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        el_price_scenario_dict['Olefins benchmark']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        # Ethylene Oxide
        # el_price_scenario_dict[variability]['E1']['optimized']['power demand'] =
        el_price_scenario_dict['Ethylene oxide benchmark']['non-optimized'][amp][
            'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        # el_price_scenario_dict[variability]['E1']['optimized']['heat demand'] =
        el_price_scenario_dict['Ethylene oxide benchmark']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        # # Ethylbenzene
        # # el_price_scenario_dict[variability]['E2']['optimized']['power demand'] =
        # el_price_scenario_dict['Ethylbenzene benchmark']['non-optimized'][amp][
        #     'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        # # el_price_scenario_dict[variability]['E2']['optimized']['heat demand'] =
        # el_price_scenario_dict['Ethylbenzene benchmark']['non-optimized'][amp][
        #     'heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        # # Ethylene Glycol
        # # el_price_scenario_dict[variability]['E3']['optimized']['power demand'] =
        # el_price_scenario_dict['Ethylene glycol benchmark']['non-optimized'][amp][
        #     'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        # # el_price_scenario_dict[variability]['E3']['optimized']['heat demand'] =
        # el_price_scenario_dict['Ethylene glycol benchmark']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        # # PET
        # # el_price_scenario_dict[variability]['E6']['optimized']['power demand'] =
        # el_price_scenario_dict['PET benchmark']['non-optimized'][amp][
        #     'power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        # # el_price_scenario_dict[variability]['E6']['optimized']['heat demand'] =
        # el_price_scenario_dict['PET benchmark']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_CHP_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time] + m.P_CHP_CP[time]

                def CHP_ng_P_conversion(m, time):
                    return m.NG_CHP_in[time] == (m.P_CHP_CP[time] + m.P_CHP_excess[time] + m.P_CHP_gr[time]) / \
                           eta_CHP_el

                def CHP_ng_H_conversion(m, time):
                    return m.NG_CHP_in[time] == (m.H_CHP_CP[time] + m.H_CHP_excess[time]) / eta_CHP_th

                def CHP_max_H(m, time):
                    return m.H_CHP_CP[time] + m.H_CHP_excess[time] <= CHP_cap * eta_CHP_th

                def CHP_min_H(m, time):
                    return m.H_CHP_CP[time] + m.H_CHP_excess[time] >= CHP_cap * eta_CHP_th * CHP_min_load

                def max_grid_power_in(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_gr_CP[time] <= gr_connection * m.b1[time]

                def max_grid_power_out(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_CHP_gr[time] <= gr_connection * (1 - m.b1[time])

                def minimize_total_costs(m):
                    return sum(price_el_hourly.iloc[time, count] * time_step * (m.P_gr_CP[time] - m.P_CHP_gr[time])
                               + price_ng_hourly.iloc[time, 0] * time_step * m.NG_CHP_in[time]
                               + price_EUA_hourly.iloc[time, 0] * time_step * m.NG_CHP_in[time] * EF_ng
                               for time in m.T)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)

                eta_CHP_el = 0.3  # Electric efficiency of CHP [%]
                eta_CHP_th = 0.4  # Thermal efficiency of CHP [%]
                CHP_cap = H_dem / eta_CHP_th  # Thermal capacity (LPS) CHP, [MW]

                # Todo: Discuss grid connection assumption
                gr_connection = gr_cap * CHP_cap * eta_CHP_el  # allows to sell all electricity generated by CHP

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.NG_CHP_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake, MWh
                m.P_CHP_CP = pm.Var(m.T, bounds=(0, None))  # Power generated from CHP to core process, MW
                m.P_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess power generated by CHP, MW
                m.P_CHP_gr = pm.Var(m.T, bounds=(0, None))  # Power from CHP to grid, MW
                m.H_CHP_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
                m.H_CHP_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat generated by CHP, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous bi-directional use of the
                # grid connection

                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # CHP constraints
                m.CHP_ng_P_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_P_conversion)
                m.CHP_ng_H_conversion_constraint = pm.Constraint(m.T, rule=CHP_ng_H_conversion)
                m.CHP_max_H_constraint = pm.Constraint(m.T, rule=CHP_max_H)
                m.CHP_min_H_constraint = pm.Constraint(m.T, rule=CHP_min_H)
                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)
                m.max_grid_power_out_constraint = pm.Constraint(m.T, rule=max_grid_power_out)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')  # what does this last part do?

                # Solve optimization problem
                opt = pm.SolverFactory('gurobi')
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from CHP to process'] = pm.value(m.H_CHP_CP[:])
                result['Heat excess from CHP'] = pm.value(m.H_CHP_excess[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Electricity from CHP to process'] = pm.value(m.P_CHP_CP[:])
                result['Electricity excess from CHP'] = pm.value(m.P_CHP_excess[:])
                result['Electricity from CHP to grid'] = pm.value(m.P_CHP_gr[:])
                result['natural gas consumption [MWh]'] = pm.value(m.NG_CHP_in[:])
                CHP_gen_CP = sum(
                    result['Heat from CHP to process'] + result['Electricity from CHP to process'])
                Grid_use = result['Electricity from grid to process'].sum()
                # Grid_gen_CP = result['Electricity from grid to core process'].sum()
                CO2_emissions = result['natural gas consumption [MWh]'].sum() * EF_ng * time_step  # [ton]
                Cost_EUA = sum(price_EUA_hourly.iloc[time, 0] * time_step * pm.value(m.NG_CHP_in[time]) * EF_ng
                               for time in m.T)

                # Scope 2 CO2 emissions
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(result['Electricity from grid to process'],
                                                                    axis='index')  # leads to [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(result['Heat demand process'] - result['Heat from CHP to process'])
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from CHP to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))

                # # plot power and heat flows
                # result['Electricity from grid to process'].plot(linestyle='--', color='darkviolet')
                # result['Electricity from CHP to process'].plot(color='gold')
                # result['Heat from CHP to process'].plot(color='red')
                # result['Electricity excess from CHP'].plot(color='black')  # should be zero
                # result['Electricity from CHP to grid'].plot(linestyle='-.')
                # plt.legend()
                # # plot prices for clarification
                # price_el_hourly.plot()
                # price_ng_hourly.plot()
                # plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = 0
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = CO2_emissions
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = Cost_EUA
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required space'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP gen to CP'] = CHP_gen_CP
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to CP'] = \
                    result['Heat from CHP to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess heat gen'] = \
                    result['Heat excess from CHP'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to CP'] = \
                    result['Electricity from CHP to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to battery'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess power gen'] = \
                    result['Electricity excess from CHP'].sum()
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to grid'] = \
                    result['Electricity from CHP to grid'].sum()
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = \
                    result['natural gas consumption [MWh]'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = Grid_use
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = \
                    CO2_emissions / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = Cost_EUA / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = 0
                # energy flows
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")
    return el_price_scenario_dict

# optimisation function definition. GB instead of CHP.
def benchmark_GB_scheduling_optimisation(price_el_hourly, price_ng_orig, price_EUA_orig, CO2_emiss_grid_hourly,
                                         amp_values, variability_values, hours, GB_cap_min):
    """This function optimizes the heat and electricity generation system for industrial processes. An economic optimisation
 chooses the optimal combination and size of new technologies (electric boiler, thermal energy storage, battery storage,
 electrolyser, hydrogen storage and hydrogen boiler) and the use of natural gas use with an existing gas boiler,
 taking into account the price for CO2 emission-allowances for a single chosen year. """

    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 1  # in hours

    # TODO: Implement warning if dataset contains NaNs (for all input data)
    # natural gas price data
    price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # electricity price data
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    ## manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()

    # define factor by which volatility should be amplified
    amp = amp_values

    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])
            # # removing negative prices  # if done here, mean price of price curves increase with increasing k
            # price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        # removing negative prices  # if done here, mean price of price curves are all the same. TODO: revise!
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)

        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # figure variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    # EUA price data
    price_EUA_hourly = price_EUA_orig.resample('{}h'.format(time_step)).ffill()
    EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
    price_EUA_hourly.index = price_ng_hourly.index
    price_EUA_hourly_mean = price_EUA_hourly.mean()
    price_EUA_hourly_mean_hours = price_EUA_hourly['Price'].iloc[:hours].mean()
    price_EUA_hourly_var_hours = price_EUA_hourly['Price'].iloc[:hours].var()
    print("Mean EUA price is " + str(price_EUA_hourly_mean_hours), ". The variance is " +
          str(price_EUA_hourly_var_hours))

    # # calculate cost for using natural gas as price for the gas + cost for CO2 emissions
    price_EUA_hourly_MWh = price_EUA_hourly * 0.2  # eur/ton * 0.2 ton(CO2)/MWh(natural gas) = eur/MWh(natural gas)
    price_NG_use_hourly = pd.DataFrame({'Cost of using natural gas': None}, index=price_ng_hourly.index)
    price_NG_use_hourly['Cost of using natural gas'] = price_EUA_hourly_MWh['Price'] + price_ng_hourly['Open']
    price_NG_use_hourly_mean_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].mean()
    price_NG_use_hourly_var_hours = price_NG_use_hourly['Cost of using natural gas'].iloc[:hours].var()
    print("Mean price for using NG [MWh] is " + str(price_NG_use_hourly_mean_hours), ". The variance is " +
          str(price_NG_use_hourly_var_hours))

    # # figure to display electricity and gas price(s) together
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original electricity price data', color='b')
    # price_ng_hourly['Open'].plot(x=price_ng_hourly.index, label='Original natural gas price data', color='r')
    # plt.ylabel("EUR/MWh")
    # plt.legend(loc='upper left')
    # ax = price_EUA_hourly['Price'].plot(x=price_EUA_hourly.index, secondary_y=True, label='Original CO2 emission cost data', color='g', linewidth=1.5)
    # ax.set_ylabel("EUR/ton")
    # plt.xlabel("Date")
    # plt.title("Price data for electricity (day-ahead market), natural gas (Dutch TTF market), and CO2 emission allowances (ETS)")
    # plt.legend(loc='upper right')
    # plt.show()

    # # plot prices for clarification
    # plt.plot(price_el_hourly.iloc[:hours, 0], label='Electricity price', color='b', marker='o',
    #             markersize=0.75)
    # plt.plot(price_NG_use_hourly.iloc[:hours, 0],
    #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
    #             markersize=0.75)
    # plt.ylabel("EUR/MWh")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()

    # # for hour many hours of the operational time is electricity cheaper than natural gas?
    # # calculate difference el_price - natural_gas_use
    # price_difference = price_el_hourly.iloc[:hours, 0] - price_NG_use_hourly.iloc[:hours, 0]
    # # how often is price difference < 0?
    # negative_hours = pd.Series(data=price_difference[price_difference < 0],
    #                            index=price_difference[price_difference < 0].index)
    # print(len(negative_hours))

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # price_ng_hourly_short = pd.DataFrame(price_ng_hourly.iloc[0:hours])
    # price_EUA_hourly_short = pd.DataFrame(price_EUA_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short
    # price_ng_hourly = price_ng_hourly_short
    # price_EUA_hourly = price_EUA_hourly_short

    # for electricity
    variability = variability_values

    # create respective dictionary
    looping_variable = variability
    processes = ['Ethylbenzene', 'Ethylene glycol',
                 'PET']  # 'Olefins', 'Ethylene oxide', 'Ethylbenzene', 'Ethylene glycol', 'PET']  # process names
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'available area': None, 'results': {},
                                                                'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}
    # print(el_price_scenario_dict)

    # for amp in variability:
    for count, amp in enumerate(looping_variable):
        print("Current variability amplitude is: ", amp)
        # # Olefins
        # el_price_scenario_dict['Olefins']['non-optimized'][amp][
        #     'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        # el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        # el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # # Ethylene Oxide
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
        #     'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # Ethylbenzene
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
            'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # Ethylene Glycol
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
            'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # PET
        el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_GB_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time]

                def GB_balance(m, time):
                    return m.NG_GB_in[time] == (m.H_GB_CP[time] + m.H_GB_excess[time]) / eta_GB

                def GB_max_H(m, time):
                    return m.H_GB_CP[time] + m.H_GB_excess[time] <= GB_cap * eta_GB

                def GB_min_H(m, time):
                    return m.H_GB_CP[time] + m.H_GB_excess[time] >= GB_cap * eta_GB * GB_cap_min

                def minimize_total_costs(m):
                    # cost of electricity: Price * (consumption - feed-in)
                    # cost of using natural gas: Consumption * (price of NG + cost for CO2 emission allowances)
                    return sum(price_el_hourly.iloc[time, count] * time_step * m.P_gr_CP[time] +
                               m.NG_GB_in[time] * time_step * (price_ng_hourly.iloc[time, 0] +
                                                               price_EUA_hourly.iloc[time, 0] * EF_ng) for time in m.T)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                # GB
                eta_GB = 0.9  # Thermal efficiency of GB [%]
                GB_cap = H_dem / eta_GB  # Thermal capacity (LPS) CHP, [MW]
                #GB_cap_min = 0.5  # minimal load factor, [% of Pnom]

                # other
                EF_ng = 0.2  # emission factor natural gas, tCO2/MWh(CH4)

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.NG_GB_in = pm.Var(m.T, bounds=(0, None))  # natural gas intake, MWh
                m.H_GB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from CHP (natural gas), MW
                m.H_GB_excess = pm.Var(m.T, bounds=(0, None))  # Excess heat from CHP, MW

                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # CHP constraints
                m.GB_balance_constraint = pm.Constraint(m.T, rule=GB_balance)
                m.GB_H_max_constraint = pm.Constraint(m.T, rule=GB_max_H)
                m.GB_H_min_constraint = pm.Constraint(m.T, rule=GB_min_H)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')

                # Solve optimization problem
                # reduce the optimality gap that should be reached
                opt = pm.SolverFactory('gurobi')
                #opt.options["MIPGap"] = 0.02
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from GB to process'] = pm.value(m.H_GB_CP[:])
                result['Heat excess from GB'] = pm.value(m.H_GB_excess[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['natural gas consumption [MWh]'] = pm.value(m.NG_GB_in[:])


                CO2_emissions = result[
                                    'natural gas consumption [MWh]'].sum() * EF_ng * time_step  # [MWh]*[ton/MWh] = [ton]
                Cost_EUA = sum(price_EUA_hourly.iloc[time, 0] * time_step * pm.value(m.NG_GB_in[time]) * EF_ng
                               for time in m.T)

                # Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process']})
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - result['Heat from GB to process'])
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - result['Electricity from grid to process'])
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))

                # # energy flows and prices in one figure for analysis
                # fig, axs = plt.subplots(2, sharex=True)
                # # # grid flows
                # axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                #             color='lightcoral', marker='.')
                # # all flows from the GB to the process
                # axs[0].plot(result['Heat from GB to process'], label='Heat from GB to process', color='red',
                #             marker='.')
                # axs[0].plot(result['Heat excess from GB'], label='Excess Heat from GB', color='coral', marker='s')
                # 
                # axs[0].set_ylabel("MW")
                # axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
                # 
                # # plot prices for clarification
                # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                #             markersize=0.75)
                # axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
                #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
                #             markersize=0.75)
                # axs[1].set_ylabel("EUR/MWh")
                # axs[1].legend()
                # # ax2 = axs[1].twinx()
                # # ax2.plot(price_EUA_hourly.iloc[:hours, 0], label='CO2 emission cost', color='g', marker='o',
                # #          markersize=0.75)
                # # ax2.set_ylabel("EUR/ton")
                # # ax2.legend(loc='upper right')
                # plt.xlabel("Date")
                # plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = 0
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = CO2_emissions
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = Cost_EUA
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = 0
                el_price_scenario_dict[process][run][amp]['results']['GB heat gen to CP'] = \
                    result['Heat from GB to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['GB heat gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['GB excess heat gen'] = \
                    result['Heat excess from GB'].sum()
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = \
                    result['natural gas consumption [MWh]'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = \
                    el_price_scenario_dict[process][run][amp]['results']['grid to CP']
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = 0
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = 0
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = 1000000000000
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = 0
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = P_dem

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = \
                    CO2_emissions / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = Cost_EUA / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # add 'result' dataframe with energy flows to the dict
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                #print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")

    return el_price_scenario_dict



# ----------------------------------------------------------------------------------------------------------------------
def optimisation_run_fully_electrified(price_el_hourly, price_ng_orig, CO2_emiss_grid_hourly, amp_values,
                                       variability_values, gr_cap, hours, disc_rate):
    """This function optimizes a fully electrified heat and electricity generation system for industrial processes from
    the ethylene industry. An economic optimisation chooses the optimal combination and size of technologies (electric
    boiler, thermal energy storage, battery storage, electrolyser, hydrogen storage and hydrogen boiler), based on the
    resulting total cost (Capex + Opex) for a single chosen year. In this version, selling electricity back to the grid
    is not possible. """

    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 1  # in hours

    # TODO: Implement warning if dataset contains NaNs (for all input data)
    # natural gas price data
    price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # electricity price data TODO: change the way the index is defined
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    ## manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()

    # define factor by which volatility should be amplified
    amp = amp_values

    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])
            # # removing negative prices  # if done here, mean price of price curves increase with increasing k
            # price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        # removing negative prices  # if done here, mean price of price curves are all the same. TODO: revise!
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)

        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # figure variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # # plot prices for clarification
    # plt.plot(price_el_hourly.iloc[:hours, 0], label='Electricity price', color='b', marker='o',
    #             markersize=0.75)
    # plt.ylabel("EUR/MWh")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short

    # for electricity
    variability = variability_values

    # create respective dictionary
    looping_variable = variability
    processes = ['Olefins', 'Ethylene oxide', 'Ethylbenzene', 'Ethylene glycol',
                 'PET']  # 'Olefins', 'Ethylene oxide', 'Ethylbenzene', 'Ethylene glycol', 'PET']  # process names
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'available area': None, 'results': {},
                                                                'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}

    # for amp in variability:
    for count, amp in enumerate(looping_variable):
        print("Current variability amplitude is: ", amp)
        # Olefins
        el_price_scenario_dict['Olefins']['non-optimized'][amp][
            'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # Ethylene Oxide
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
            'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # Ethylbenzene
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
            'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # Ethylene Glycol
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
            'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # PET
        el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_ElB_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time] + m.P_bat_CP[time]

                def ElB_balance(m, time):
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time]) * eta_ElB

                def ElB_size(m, time):
                    return m.H_ElB_CP[time] + m.H_ElB_TES[time] <= m.ElB_cap

                def bat_soe(m, time):
                    if time == 0:
                        return m.bat_soe[time] == 0
                    else:
                        return m.bat_soe[time] == m.bat_soe[time - 1] + \
                               eta_bat * time_step * m.P_gr_bat[time - 1] - \
                               1 / eta_bat * time_step * (m.P_bat_CP[time - 1] + m.P_bat_ElB[time - 1]
                                                          + m.P_bat_H2E[time - 1])

                # Use Big M method to avoid simultaneous charging and discharging of the battery
                # TODO: revise crate
                def bat_in(m, time):
                    return m.P_gr_bat[time] <= m.bat_cap / eta_bat * crate_bat / time_step * m.b1[time]

                # TODO: revise crate
                def bat_out(m, time):
                    if time == 0:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) == 0
                    else:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time]) \
                               <= m.bat_cap * eta_bat * crate_bat / time_step * (1 - m.b1[time])

                def bat_size(m, time):
                    return m.bat_soe[time] <= m.bat_cap

                def TES_soe(m, time):
                    if time == 0:
                        return m.TES_soe[time] == 0
                    else:
                        return m.TES_soe[time] == m.TES_soe[time - 1] \
                               + (m.H_ElB_TES[time - 1] - m.H_TES_CP[time - 1] / eta_TES_A) * time_step

                # Use Big M method to avoid simultaneous charging and discharging of the TES
                # TODO: revise crate!
                def TES_in(m, time):
                    return m.H_ElB_TES[time] <= m.TES_cap * crate_TES / time_step * m.b2[time]

                # TODO: revise crate!
                def TES_out(m, time):
                    if time == 0:
                        return m.H_TES_CP[time] == 0
                    else:
                        return m.H_TES_CP[time] <= m.TES_cap * eta_TES_A * crate_TES / time_step * (1 - m.b2[time])

                def TES_size(m, time):
                    return m.TES_soe[time] <= m.TES_cap

                def H2S_soe(m, time):
                    if time == 0:
                        return m.H2S_soe[time] == 0
                    else:
                        return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] -
                                                                         m.H2_H2S_H2B[time - 1] / eta_H2S) * time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have
                # def H2S_in(m, time):
                #     return m.H2_H2E_H2S[time] <= m.H2S_cap * crate_H2S / time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have?
                # def H2S_out(m, time):
                #     if time == 0:
                #         return m.H2_H2S_H2B[time] == 0
                #     else:
                #         return m.H2_H2S_H2B[time] <= m.H2S_cap * crate_H2S * eta_H2S / time_step

                def H2S_size(m, time):
                    return m.H2S_soe[time] <= m.H2S_cap

                def H2B_balance(m, time):
                    return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

                def H2B_size(m, time):
                    return m.H_H2B_CP[time] <= m.H2B_cap

                def H2E_balance(m, time):
                    return (m.P_gr_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time]

                def H2E_size(m, time):
                    return m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time] <= m.H2E_cap

                def spat_dem(m, time):
                    return m.bat_cap * bat_areaftpr + m.ElB_cap * ElB_areaftpr + m.TES_cap * TES_areaftpr_A + \
                           m.H2E_cap * H2E_areaftpr + m.H2B_cap * H2B_areaftpr + m.H2S_cap * H2S_areaftpr \
                           <= available_area

                def max_grid_power_in(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_gr_CP[time] + m.P_gr_ElB[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] <= gr_connection

                def minimize_total_costs(m):
                    return sum(price_el_hourly.iloc[time, count] * time_step * (m.P_gr_ElB[time] + m.P_gr_CP[time] +
                                                                                m.P_gr_bat[time] + m.P_gr_H2E[time])
                               for time in m.T) \
                           + \
                           m.bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                           m.ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                           m.TES_cap * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                           m.H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                           m.H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                           m.H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                # Electric boiler
                c_ElB = 70000  # CAPEX for electric boiler, 70000 eur/MW
                ElB_lifetime = 20  # lifetime of electric boiler, years
                ElB_areaftpr = 30  # spatial requirements, m^2/MW
                eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]

                # Battery constants
                eta_bat = 0.95  # Battery (dis)charging efficiency
                c_bat = 300e3  # CAPEX for battery per eur/MWh, 338e3 USD --> 314.15e3 eur (12.07.23)
                bat_lifetime = 15  # lifetime of battery
                bat_areaftpr = 10  # spatial requirement, [m^2/MWh]
                crate_bat = 0.7  # C rate of battery, 0.7 kW/kWh, [-]

                # TES constants
                c_TES_A = 23000  # CAPEX for sensible heat storage
                c_TES_B = 14000  # CAPEX for heat storage including heater, [UDS/MWh]
                c_TES_C = 60000  # CAPEX for latent heat storage [eur/MWh]
                TES_lifetime = 25  # heat storage lifetime, [years]
                eta_TES_A = 0.9  # discharge efficiency [-]
                eta_TES_C = 0.98  # discharge efficiency [-]
                TES_areaftpr_A = 5  # spatial requirement TES, [m^2/MWh]
                TES_areaftpr_B = 7  # spatial requirement TES (configuration B), [m^2/MWh]
                crate_TES = 0.5  # C rate of TES, 0.5 kW/kWh, [-]  #TODO: revise this number

                # Hydrogen equipment constants
                eta_H2S = 0.9  # charge efficiency hydrogen storage [-], accounting for fugitive losses
                eta_H2B = 0.92  # conversion efficiency hydrogen boiler [-]
                eta_H2E = 0.69  # conversion efficiency electrolyser [-]
                c_H2S = 10000  # CAPEX for hydrogen storage per MWh, [eur/MWh]
                c_H2B = 35000  # CAPEX for hydrogen boiler per MW, [eur/MW]
                c_H2E = 700e3  # CAPEX for electrolyser per MW, [eur/MW]
                H2S_lifetime = 20  # lifetime hydrogen storage, [years]
                H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
                H2E_lifetime = 15  # lifetime electrolyser, [years]
                H2E_areaftpr = 100  # spatial requirement electrolyser, [m^2/MW]
                H2B_areaftpr = 5  # spatial requirement hydrogen boiler, [m^2/MW]
                H2S_areaftpr = 10  # spatial requirement hydrogen storage, [m^2/MWh]

                # other
                # Todo: Discuss grid connection assumption
                gr_connection = gr_cap * (P_dem + H_dem) / (
                        eta_bat * eta_bat * eta_H2E * eta_H2S * eta_H2B)  # 'worst case' conversion chain

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
                m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
                m.H_ElB_TES = pm.Var(m.T, bounds=(0, None))  # Heat from electric boiler to TES, MW
                m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
                m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
                m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
                m.P_bat_CP = pm.Var(m.T, bounds=(0, None))  # discharging power batter to core process, MW
                m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
                m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
                m.bat_cap = pm.Var(bounds=(0, None))  # Battery capacity, MWh
                m.ElB_cap = pm.Var(bounds=(0, None))  # electric boiler capacity, MW
                m.TES_cap = pm.Var(bounds=(0, None))  # TES capacity, MWh
                m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
                m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
                m.H2S_cap = pm.Var(bounds=(0, None))  # hydrogen storage capacity, MWh
                m.H2B_cap = pm.Var(bounds=(0, None))  # hydrogen boiler capacity, MW
                m.H2E_cap = pm.Var(bounds=(0, None))  # electrolyser capacity, MW
                m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
                m.H2_H2S_H2B = pm.Var(m.T,
                                      bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
                m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
                m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
                m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the battery
                m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the TES


                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # electric boiler constraint
                m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
                m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
                # battery constraints
                m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
                m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out)
                m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
                m.bat_size_constraint = pm.Constraint(m.T, rule=bat_size)
                # TES constraints
                m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
                m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
                m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
                m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)
                # hydrogen constraints
                m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
                m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
                m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
                m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
                m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
                m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
                # m.H2S_discharge_constraint = pm.Constraint(m.T, rule=H2S_out)
                # m.H2S_charge_constraint = pm.Constraint(m.T, rule=H2S_in)
                # spatial constraint
                m.spat_dem_constraint = pm.Constraint(m.T, rule=spat_dem)
                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')  # what does this last part do?

                # Solve optimization problem
                opt = pm.SolverFactory('gurobi')
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from electric boiler to process'] = pm.value(m.H_ElB_CP[:])
                result['Heat from electric boiler to TES'] = pm.value(m.H_ElB_TES[:])
                result['Electricity from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Battery SOE'] = pm.value(m.bat_soe[:])
                result['Electricity from battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
                result['Electricity from battery to process'] = pm.value(m.P_bat_CP[:])
                result['Electricity from grid to battery'] = pm.value(m.P_gr_bat[:])
                result['Heat from TES to process'] = pm.value(m.H_TES_CP[:])
                result['Electricity from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
                result['Heat from hydrogen boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Electricity from battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
                result['Heat from H2 boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Hydrogen from electrolyser to H2 boiler'] = pm.value(m.H2_H2E_H2B[:])
                result['Hydrogen from electrolyser to storage'] = pm.value(m.H2_H2E_H2S[:])
                result['Hydrogen from storage to H2 boiler'] = pm.value(m.H2_H2S_H2B[:])
                # check if grid capacity constraint is hit
                grid_P_out = result['Electricity from grid to process'] + \
                             result['Electricity from grid to electric boiler'] + \
                             result['Electricity from grid to battery'] + \
                             result['Electricity from grid to electrolyser']
                grid_P_out_max = max(grid_P_out)
                Grid_gen = result['Electricity from grid to process'].sum() + \
                           result['Electricity from grid to electric boiler'].sum() + \
                           result['Electricity from grid to battery'].sum() + \
                           result['Electricity from grid to electrolyser'].sum()
                ElB_gen_CP = result['Heat from electric boiler to process'].sum()
                # Battery_gen = result['Discharging battery'].sum()
                # Excess_heat = result['Excess heat from CHP(ng)'].sum()

                # Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process'],
                                                'Grid to electric boiler': result[
                                                    'Electricity from grid to electric boiler'],
                                                'Grid to battery': result['Electricity from grid to battery'],
                                                'Grid to electrolyser': result[
                                                    'Electricity from grid to electrolyser']})
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - (result['Heat from electric boiler to process'] +
                                                     result['Heat from TES to process'] +
                                                     result['Heat from hydrogen boiler to process']))
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from battery to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))
                # print("Investment cost battery per MWh, USD = ", c_bat)
                # print("Investment cost electric boiler per MW, USD = ", c_ElB)
                # print("Investment cost TES per MWh, USD = ", c_TES_A)
                print("Battery capacity =", pm.value(m.bat_cap))
                print("Electric boiler capacity =", pm.value(m.ElB_cap))
                print("TES capacity =", pm.value(m.TES_cap))
                print("electrolyser capacity =", pm.value(m.H2E_cap))
                print("Hydrogen boiler capacity =", pm.value(m.H2B_cap))
                print("Hydrogen storage capacity =", pm.value(m.H2S_cap))
                print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

                # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.bat_cap) > 0:
                    battery_discharge_sum = result['Electricity from battery to electrolyser'] + \
                                            result['Electricity from battery to electric boiler'] + \
                                            result['Electricity from battery to process']
                    battery_charge_sum = result['Electricity from grid to battery']
                    battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
                    for i in range(0, len(battery_charge_sum)):
                        if battery_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if battery_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                              battery_discharge_sum[i]
                    print("Number of hours of simultaneous battery charging and discharging: ",
                          len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF TES capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.TES_cap) > 0:
                    TES_discharge_sum = result['Heat from TES to process']
                    TES_charge_sum = result['Heat from electric boiler to TES']
                    TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
                    for i in range(0, len(TES_charge_sum)):
                        if TES_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if TES_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                          TES_discharge_sum[i]
                    print("Number of hours of simultaneous TES charging and discharging: ",
                          len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF H2S capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.H2S_cap) > 0:
                    H2S_discharge_sum = result['Hydrogen from storage to H2 boiler']
                    H2S_charge_sum = result['Hydrogen from electrolyser to storage']
                    H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
                    for i in range(0, len(H2S_charge_sum)):
                        if H2S_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if H2S_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                          H2S_discharge_sum[i]
                    print("Number of hours of simultaneous H2S charging and discharging: ",
                          len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0]))


                # # energy flows and prices in one figure for analysis
                # fig, axs = plt.subplots(2, sharex=True)
                # # # grid flows
                # axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                #             color='lightcoral', marker='.')
                # # battery flows
                # if pm.value(m.bat_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to battery'], label='Electricity from grid to battery',
                #                 color='gold', marker='.')
                #     axs[0].plot(result['Electricity from battery to process'],
                #                 label='Electricity from battery to process', color='darkkhaki', marker='s')
                #     if pm.value(m.ElB_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electric boiler'],
                #                     label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
                #     if pm.value(m.H2E_cap) > 0:
                #         axs[0].plot(result['Electricity from battery to electrolyser'],
                #                     label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
                #     #axs[0].plot(result['Battery SOE'], label='Battery SOE', marker='2')
                # # # electric boiler flows
                # if pm.value(m.ElB_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electric boiler'],
                #                 label='Electricity from grid to electric boiler', color='seagreen', marker='.')
                #     axs[0].plot(result['Heat from electric boiler to process'],
                #                 label='Heat from electric boiler to process', color='turquoise', marker='.')
                #     if pm.value(m.TES_cap) > 0:
                #         axs[0].plot(result['Heat from electric boiler to TES'],
                #                     label='Heat from electric boiler to TES',
                #                     color='lime', marker='.')
                # # TES flows
                # if pm.value(m.TES_cap) > 0:
                #     axs[0].plot(result['Heat from TES to process'], label='Heat from TES to process',
                #                 color='deepskyblue', marker='.')
                #
                # # # Hydrogen flows
                # if pm.value(m.H2E_cap) > 0:
                #     axs[0].plot(result['Electricity from grid to electrolyser'],
                #                 label='Electricity from grid to electrolyser', color='royalblue', marker='.')
                #     axs[0].plot(result['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
                #                 color='blueviolet', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
                #                 label='Hydrogen from electrolyser to H2 boiler', marker='.')
                #     axs[0].plot(result['Hydrogen from electrolyser to storage'], color='fuchsia',
                #                 label='Hydrogen from electrolyser to storage', marker='.')
                #     axs[0].plot(result['Hydrogen from storage to H2 boiler'], color='deeppink',
                #                 label='Hydrogen from storage to H2 boiler', marker='.')
                # axs[0].axhline(y=gr_connection, color='grey', linestyle='--', label='Grid connection capacity')
                # axs[0].set_ylabel("MW")
                # axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
                #
                # # plot prices for clarification
                # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                #             markersize=0.75)
                # axs[1].set_ylabel("EUR/MWh")
                # axs[1].legend()
                # # ax2 = axs[1].twinx()
                # # ax2.plot(price_EUA_hourly.iloc[:hours, 0], label='CO2 emission cost', color='g', marker='o',
                # #          markersize=0.75)
                # # ax2.set_ylabel("EUR/ton")
                # # ax2.legend(loc='upper right')
                # plt.xlabel("Date")
                # plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                    pm.value(m.ElB_cap) * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                    pm.value(m.TES_cap) * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                    pm.value(m.H2E_cap) * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                    pm.value(m.H2B_cap) * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                    pm.value(m.H2S_cap) * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat + \
                    pm.value(m.ElB_cap) * c_ElB + \
                    pm.value(m.TES_cap) * c_TES_A + \
                    pm.value(m.H2E_cap) * c_H2E + \
                    pm.value(m.H2B_cap) * c_H2B + \
                    pm.value(m.H2S_cap) * c_H2S
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = \
                    pm.value(m.bat_cap) * bat_areaftpr + pm.value(m.ElB_cap) * ElB_areaftpr + \
                    pm.value(m.TES_cap) * TES_areaftpr_A + pm.value(m.H2E_cap) * H2E_areaftpr + \
                    pm.value(m.H2B_cap) * H2B_areaftpr + pm.value(m.H2S_cap) * H2S_areaftpr
                el_price_scenario_dict[process][run][amp]['results']['CHP gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess heat gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to battery'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess power gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to grid'] = 0
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = Grid_gen
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = \
                    result['Electricity from grid to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = \
                    result['Electricity from grid to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = \
                    result['Electricity from grid to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = ElB_gen_CP
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to TES'] = \
                    result['Heat from electric boiler to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = pm.value(m.ElB_cap)
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = pm.value(m.bat_cap)
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = \
                    result['Electricity from battery to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = \
                    result['Electricity from battery to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = \
                    result['Electricity from battery to electrolyser'].sum()
                if pm.value(m.bat_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours Battery'] \
                        = len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0])
                else:
                    el_price_scenario_dict[process][run][amp]['results']['Simultaeous charging and discharging hours'] \
                        = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = pm.value(m.TES_cap)
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = \
                    result['Heat from TES to process'].sum()
                if pm.value(m.TES_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours TES'] \
                        = len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = pm.value(m.H2E_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = pm.value(m.H2B_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = pm.value(m.H2S_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = result[
                    'Heat from H2 boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = \
                    result['Hydrogen from electrolyser to H2 boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = \
                    result['Hydrogen from electrolyser to storage'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = \
                    result['Hydrogen from storage to H2 boiler'].sum()
                if pm.value(m.H2S_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours H2S'] \
                        = len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = disc_rate
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # energy flows
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")

    return el_price_scenario_dict

# ----------------------------------------------------------------------------------------------------------------------
def optimisation_run_fully_electrified_ESCAPE(price_el_hourly, price_ng_orig, CO2_emiss_grid_hourly, amp_values,
                                       variability_values, hours, disc_rate):
    """This function optimizes a fully electrified heat and electricity generation system for industrial processes from
    the ethylene industry. An economic optimisation chooses the optimal combination and size of technologies (electric
    boiler, thermal energy storage, battery storage, electrolyser, hydrogen storage and hydrogen boiler), based on the
    resulting total cost (Capex + Opex) for a single chosen year. In this version, selling electricity back to the grid
    is not possible. """

    # ------------------------------------- input DATA pre-treatment --------------------------------------------------------
    time_step = 1  # in hours

    # TODO: Implement warning if dataset contains NaNs (for all input data)
    # natural gas price data
    price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
    ng_row_NaN = price_ng_hourly[price_ng_hourly.isna().any(axis=1)]
    price_ng_hourly_mean_hours = price_ng_hourly['Open'].iloc[:hours].mean()
    price_ng_hourly_var_hours = price_ng_hourly['Open'].iloc[:hours].var()
    print("Mean natural gas price is " + str(price_ng_hourly_mean_hours), ". The variance is " +
          str(price_ng_hourly_var_hours))

    # electricity price data TODO: change the way the index is defined
    el_row_NaN = price_el_hourly[price_el_hourly.isna().any(axis=1)]  # indicates row with NaN value
    price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
    price_el_hourly.index = price_ng_hourly.index
    price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
    price_el_hourly_mean_hours = price_el_hourly['Original data'].iloc[:hours].mean()
    price_el_hourly_var_hours = price_el_hourly['Original data'].iloc[:hours].var()
    print("Mean electricity price is " + str(price_el_hourly_mean_hours), ". The variance is " +
          str(price_el_hourly_var_hours))

    # check if CO2 intensity data does not contain NaNs
    CO2_row_NaN = CO2_emiss_grid_hourly[CO2_emiss_grid_hourly.isna().any(axis=1)]  # indicates row with NaN value

    ## manipulate electricity price data to increase the amplitude of the price variation
    # get average price
    price_el_hourly_mean = price_el_hourly.mean()

    # define factor by which volatility should be amplified
    amp = amp_values

    # check if amp contains values and manipulate the variability accordingly
    if len(amp) > 0:
        # generate new price profiles and sort their values from high to low to plot price duration curves
        for k in amp:
            print("Current k is: ", k)
            colname = ("amplified by " + "%.3f") % k  # add new price data as additional columns to dataframe
            price_el_hourly[str(colname)] = price_el_hourly_mean.iloc[0] + k * (
                    price_el_hourly['Original data'] -
                    price_el_hourly_mean.iloc[0])
            # # removing negative prices  # if done here, mean price of price curves increase with increasing k
            # price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0

        # removing negative prices  # if done here, mean price of price curves are all the same. TODO: revise!
        price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0
        # plotting mean
        print("Mean of original curve without negative prices: " + str(price_el_hourly['Original data'].mean()))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            # removing negative prices
            price_el_hourly.loc[price_el_hourly[str(colname)] < 0, str(colname)] = 0
            # plotting mean
            print("Mean of " + str(k) + "-curve without negative prices: " + str(price_el_hourly[colname].mean()))

        ## plot price duration curves for the period considered in the optimisation
        # sort values from high to low and add new column to dataframe
        price_el_hourly_sorted_df = \
            pd.DataFrame(price_el_hourly['Original data'].iloc[:hours].sort_values(ascending=False))
        for k in amp:
            colname = ("amplified by " + "%.3f") % k
            price_el_hourly_sorted_df[str(colname)] = \
                price_el_hourly[str(colname)].iloc[:hours].sort_values(ascending=False)

        # remove the index
        price_el_hourly_sorted_df = price_el_hourly_sorted_df.reset_index(drop=True)
        # plot the values
        fig, ax = plt.subplots()
        ax.plot(price_el_hourly_sorted_df)
        ax.set_ylabel("EUR/MWh", fontsize=16)
        ax.set_xlabel("Hours", fontsize=16, weight='bold')
        ax.tick_params(axis='y', labelsize=18, width=4)
        ax.tick_params(axis='x', labelsize=18, width=4)
        # TODO: Update legend entries
        # ax.legend(['Original data', 'Amplitude increased by 5%', 'Amplitude increased by 10%',
        #            'Amplitude increased by 15%', 'Amplitude increased by 20%'], fontsize=16)
        plt.show()

    # # remove negative prices and replace them by 0 if optimisation should be run without negative prices
    # else:
    #     price_el_hourly.loc[price_el_hourly['Original data'] < 0, 'Original data'] = 0

    # # figure variability
    # fig, ax = plt.subplots()
    # price_el_hourly['Original data'].plot(x=price_el_hourly.index, label='Original data', color='k')
    # for j in amp:
    #     colname = ("amp " + "%.3f") % j
    #     price_el_hourly[str(colname)].plot(x=price_el_hourly.index, label=str(colname), alpha=0.25)
    # plt.axhline(y=price_el_hourly_mean.iloc[0], color='tab:gray', linestyle='--', label='Mean')
    # plt.legend(fontsize=15)
    # plt.ylabel("EUR/MWh", fontsize=15)
    # plt.xlabel("", fontsize=15)
    # ax.tick_params(axis='y', labelsize=15)
    # ax.tick_params(axis='x', labelsize=15)
    # #plt.title("Electricity prices (Dutch Day-Ahead market) with increased variability")
    # plt.show()

    # # plot prices for clarification
    # plt.plot(price_el_hourly.iloc[:hours, 0], label='Electricity price', color='b', marker='o',
    #             markersize=0.75)
    # plt.ylabel("EUR/MWh")
    # plt.xlabel("Date")
    # plt.legend()
    # plt.show()

    # ----------------------------- Dictionaries to run optimisation for each process ----------------------------------
    # --------------------------------(with non-optimised and optimised values) ----------------------------------------

    # # Create subsets for the price data, if the optimisation should be run for fewer hours
    # price_el_hourly_short = pd.DataFrame(price_el_hourly.iloc[0:hours])
    # # replace full data sets by short data sets (to avoid changing code below)
    # price_el_hourly = price_el_hourly_short

    # for electricity
    variability = variability_values

    # create respective dictionary
    looping_variable = variability
    processes = ['Olefins']#, 'Olefins', 'Ethylene oxide', 'Ethylbenzene', 'Ethylene glycol', 'PET']  # process names
    el_price_scenario_dict = {process: {'non-optimized': {amp: {'power demand': None, 'heat demand': None,
                                                                'available area': None, 'results': {},
                                                                'energy flows': {}}
                                                          for amp in looping_variable}}
                              for process in processes}

    # for amp in variability:
    for count, amp in enumerate(['original']):
        print("Current variability amplitude is: ", amp)
        # Olefins
        el_price_scenario_dict['Olefins']['non-optimized'][amp][
            'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # # Ethylene Oxide
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
        #     'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # # Ethylbenzene
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
        #     'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # # Ethylene Glycol
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
        #     'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # # PET
        # el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        # el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        # el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_ElB_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time] + m.P_bat_CP[time]

                def ElB_balance(m, time):
                    return m.H_ElB_CP[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time]) * eta_ElB

                def ElB_size(m, time):
                    return m.H_ElB_CP[time] <= m.ElB_cap

                def bat_soe(m, time):
                    if time == 0:
                        return m.bat_soe[time] == 0
                    else:
                        return m.bat_soe[time] == m.bat_soe[time - 1] + \
                               eta_bat * time_step * m.P_gr_bat[time - 1] - \
                               1 / eta_bat * time_step * (m.P_bat_CP[time - 1] + m.P_bat_ElB[time - 1]
                                                          + m.P_bat_H2E[time - 1] + m.P_bat_TES[time - 1])

                # Use Big M method to avoid simultaneous charging and discharging of the battery
                # TODO: revise crate
                def bat_in(m, time):
                    return m.P_gr_bat[time] <= m.bat_cap / eta_bat * crate_bat / time_step * m.b1[time]

                # TODO: revise crate
                def bat_out(m, time):
                    if time == 0:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_TES[time]) == 0
                    else:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_TES[time]) \
                               <= m.bat_cap * eta_bat * crate_bat / time_step * (1 - m.b1[time])

                def bat_size(m, time):
                    return m.bat_soe[time] <= m.bat_cap

                def TES_soe(m, time):
                    if time == 0:
                        return m.TES_soe[time] == 0
                    else:
                        return m.TES_soe[time] == m.TES_soe[time - 1] \
                               + ((m.P_bat_TES[time - 1] + m.P_gr_TES[time - 1]) * eta_TES_B
                                  - m.H_TES_CP[time - 1]) * time_step

                # Use Big M method to avoid simultaneous charging and discharging of the TES TODO: test with and without
                # TODO: revise crate!
                def TES_in(m, time):
                    return (m.P_bat_TES[time] + m.P_gr_TES[time]) * eta_TES_B <= m.TES_cap * crate_TES / time_step #* m.b2[time]

                # TODO: revise crate!
                def TES_out(m, time):
                    if time == 0:
                        return m.H_TES_CP[time] == 0  #TODO: Test if e-boiler is only built because of this assumption. Try without!
                    else:
                        return m.H_TES_CP[time] <= m.TES_cap * eta_TES_A * crate_TES / time_step #* (1 - m.b2[time])

                def TES_size(m, time):
                    return m.TES_soe[time] <= m.TES_cap

                def H2S_soe(m, time):
                    if time == 0:
                        return m.H2S_soe[time] == 0
                    else:
                        return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] -
                                                                         m.H2_H2S_H2B[time - 1] / eta_H2S) * time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have
                # def H2S_in(m, time):
                #     return m.H2_H2E_H2S[time] <= m.H2S_cap * crate_H2S / time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have?
                # def H2S_out(m, time):
                #     if time == 0:
                #         return m.H2_H2S_H2B[time] == 0
                #     else:
                #         return m.H2_H2S_H2B[time] <= m.H2S_cap * crate_H2S * eta_H2S / time_step

                def H2S_size(m, time):
                    return m.H2S_soe[time] <= m.H2S_cap

                def H2B_balance(m, time):
                    return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

                def H2B_size(m, time):
                    return m.H_H2B_CP[time] <= m.H2B_cap

                def H2E_balance(m, time):
                    return (m.P_gr_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time]

                def H2E_size(m, time):
                    return m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time] <= m.H2E_cap

                def spat_dem(m, time):
                    return m.bat_cap * bat_areaftpr + m.ElB_cap * ElB_areaftpr + m.TES_cap * TES_areaftpr_B + \
                           m.H2E_cap * H2E_areaftpr + m.H2B_cap * H2B_areaftpr + m.H2S_cap * H2S_areaftpr \
                           <= available_area

                def max_grid_power_in(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_gr_CP[time] + m.P_gr_ElB[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] + \
                           m.P_gr_TES[time] <= gr_connection

                def minimize_total_costs(m):
                    return sum(price_el_hourly.iloc[time, count] * time_step * (m.P_gr_ElB[time] + m.P_gr_CP[time] +
                                                                                m.P_gr_bat[time] + m.P_gr_H2E[time] +
                                                                                m.P_gr_TES[time])
                               for time in m.T) \
                           + \
                           m.bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                           m.ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                           m.TES_cap * c_TES_B * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                           m.H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                           m.H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                           m.H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                # Electric boiler
                c_ElB = 70000  # CAPEX for electric boiler, 70000 eur/MW
                ElB_lifetime = 20  # lifetime of electric boiler, years
                ElB_areaftpr = 30  # spatial requirements, m^2/MW
                eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]

                # Battery constants
                eta_bat = 0.95  # Battery (dis)charging efficiency
                c_bat = 300e3  # CAPEX for battery per eur/MWh, 338e3 USD --> 314.15e3 eur (12.07.23)
                bat_lifetime = 15  # lifetime of battery
                bat_areaftpr = 10  # spatial requirement, [m^2/MWh]
                crate_bat = 0.7  # C rate of battery, 0.7 kW/kWh, [-]

                # TES constants
                c_TES_A = 23000  # CAPEX for sensible heat storage
                c_TES_B = 14000  # CAPEX for heat storage including heater, [UDS/MWh]
                c_TES_C = 60000  # CAPEX for latent heat storage [eur/MWh]
                TES_lifetime = 25  # heat storage lifetime, [years]
                eta_TES_A = 0.9  # discharge efficiency [-]
                eta_TES_B = 0.96  # charge efficiency [-]
                eta_TES_C = 0.98  # discharge efficiency [-]
                TES_areaftpr_A = 5  # spatial requirement TES, [m^2/MWh]
                TES_areaftpr_B = 5  # spatial requirement TES (configuration B), [m^2/MWh]
                crate_TES = 0.5  # C rate of TES, 0.5 kW/kWh, [-]  #TODO: revise this number

                # Hydrogen equipment constants
                eta_H2S = 0.9  # charge efficiency hydrogen storage [-], accounting for fugitive losses
                eta_H2B = 0.92  # conversion efficiency hydrogen boiler [-]
                eta_H2E = 0.69  # conversion efficiency electrolyser [-]
                c_H2S = 10000  # CAPEX for hydrogen storage per MWh, [eur/MWh]
                c_H2B = 35000  # CAPEX for hydrogen boiler per MW, [eur/MW]
                c_H2E = 700e3  # CAPEX for electrolyser per MW, [eur/MW]
                H2S_lifetime = 20  # lifetime hydrogen storage, [years]
                H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
                H2E_lifetime = 15  # lifetime electrolyser, [years]
                H2E_areaftpr = 100  # spatial requirement electrolyser, [m^2/MW]
                H2B_areaftpr = 5  # spatial requirement hydrogen boiler, [m^2/MW]
                H2S_areaftpr = 10  # spatial requirement hydrogen storage, [m^2/MWh]

                # other
                # Todo: Discuss grid connection assumption
                gr_connection = 400  # 'worst case' conversion chain

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
                m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
                m.P_gr_TES = pm.Var(m.T, bounds=(0, None))  # grid to TES, MW
                m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
                m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
                m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
                m.P_bat_CP = pm.Var(m.T, bounds=(0, None))  # discharging power batter to core process, MW
                m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
                m.P_bat_TES = pm.Var(m.T, bounds=(0, None))  # discharging power batter to TES, MW
                m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
                m.bat_cap = pm.Var(bounds=(0, None))  # Battery capacity, MWh
                m.ElB_cap = pm.Var(bounds=(0, None))  # electric boiler capacity, MW
                m.TES_cap = pm.Var(bounds=(0, None))  # TES capacity, MWh
                m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
                m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
                m.H2S_cap = pm.Var(bounds=(0, None))  # hydrogen storage capacity, MWh
                m.H2B_cap = pm.Var(bounds=(0, None))  # hydrogen boiler capacity, MW
                m.H2E_cap = pm.Var(bounds=(0, None))  # electrolyser capacity, MW
                m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
                m.H2_H2S_H2B = pm.Var(m.T,
                                      bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
                m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
                m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
                m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the battery
                # m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # # of the TES
                # m.b3 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous bidirectional use of the
                # # grid connection


                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # electric boiler constraint
                m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
                m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
                # battery constraints
                m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
                m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out)
                m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
                m.bat_size_constraint = pm.Constraint(m.T, rule=bat_size)
                # TES constraints
                m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
                m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
                m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
                m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)
                # hydrogen constraints
                m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
                m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
                m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
                m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
                m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
                m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
                # m.H2S_discharge_constraint = pm.Constraint(m.T, rule=H2S_out)
                # m.H2S_charge_constraint = pm.Constraint(m.T, rule=H2S_in)
                # spatial constraint
                m.spat_dem_constraint = pm.Constraint(m.T, rule=spat_dem)
                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')  # what does this last part do?

                # Solve optimization problem
                opt = pm.SolverFactory('gurobi')
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from electric boiler to process'] = pm.value(m.H_ElB_CP[:])
                result['Electricity from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Battery SOE'] = pm.value(m.bat_soe[:])
                result['Electricity from battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
                result['Electricity from battery to process'] = pm.value(m.P_bat_CP[:])
                result['Electricity from grid to battery'] = pm.value(m.P_gr_bat[:])
                result['Electricity from grid to TES'] = pm.value(m.P_gr_TES[:])
                result['Electricity from battery to TES'] = pm.value(m.P_bat_TES[:])
                result['Heat from TES to process'] = pm.value(m.H_TES_CP[:])
                result['Electricity from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
                result['Heat from hydrogen boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Electricity from battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
                result['Heat from H2 boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Hydrogen from electrolyser to H2 boiler'] = pm.value(m.H2_H2E_H2B[:])
                result['Hydrogen from electrolyser to storage'] = pm.value(m.H2_H2E_H2S[:])
                result['Hydrogen from storage to H2 boiler'] = pm.value(m.H2_H2S_H2B[:])
                # check if grid capacity constraint is hit
                grid_P_out = result['Electricity from grid to process'] + \
                             result['Electricity from grid to electric boiler'] + \
                             result['Electricity from grid to battery'] + \
                             result['Electricity from grid to electrolyser'] + \
                             result['Electricity from grid to TES']
                grid_P_out_max = max(grid_P_out)
                Grid_gen = result['Electricity from grid to process'].sum() + \
                           result['Electricity from grid to electric boiler'].sum() + \
                           result['Electricity from grid to battery'].sum() + \
                           result['Electricity from grid to electrolyser'].sum() + \
                           result['Electricity from grid to TES'].sum()


                # Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process'],
                                                'Grid to electric boiler': result[
                                                    'Electricity from grid to electric boiler'],
                                                'Grid to battery': result['Electricity from grid to battery'],
                                                'Grid to electrolyser': result[
                                                    'Electricity from grid to electrolyser'],
                                                'Grid to TES': result['Electricity from grid to TES']
                                                })
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - (result['Heat from electric boiler to process'] +
                                                     result['Heat from TES to process'] +
                                                     result['Heat from hydrogen boiler to process']))
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from battery to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))
                # print("Investment cost battery per MWh, USD = ", c_bat)
                # print("Investment cost electric boiler per MW, USD = ", c_ElB)
                # print("Investment cost TES per MWh, USD = ", c_TES_A)
                print("Battery capacity =", pm.value(m.bat_cap))
                print("Electric boiler capacity =", pm.value(m.ElB_cap))
                print("TES capacity =", pm.value(m.TES_cap))
                print("electrolyser capacity =", pm.value(m.H2E_cap))
                print("Hydrogen boiler capacity =", pm.value(m.H2B_cap))
                print("Hydrogen storage capacity =", pm.value(m.H2S_cap))
                print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

                # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.bat_cap) > 0:
                    battery_discharge_sum = result['Electricity from battery to electrolyser'] + \
                                            result['Electricity from battery to electric boiler'] + \
                                            result['Electricity from battery to process'] + \
                                            result['Electricity from battery to TES']
                    battery_charge_sum = result['Electricity from grid to battery']
                    battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
                    for i in range(0, len(battery_charge_sum)):
                        if battery_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if battery_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                              battery_discharge_sum[i]
                    print("Number of hours of simultaneous battery charging and discharging: ",
                          len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF TES capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.TES_cap) > 0:
                    TES_discharge_sum = result['Heat from TES to process']
                    TES_charge_sum = result['Electricity from grid to TES'] + result['Electricity from battery to TES']
                    TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
                    for i in range(0, len(TES_charge_sum)):
                        if TES_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if TES_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                          TES_discharge_sum[i]
                    print("Number of hours of simultaneous TES charging and discharging: ",
                          len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF H2S capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if pm.value(m.H2S_cap) > 0:
                    H2S_discharge_sum = result['Hydrogen from storage to H2 boiler']
                    H2S_charge_sum = result['Hydrogen from electrolyser to storage']
                    H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
                    for i in range(0, len(H2S_charge_sum)):
                        if H2S_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if H2S_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                          H2S_discharge_sum[i]
                    print("Number of hours of simultaneous H2S charging and discharging: ",
                          len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0]))


                # energy flows and prices in one figure for analysis
                fig, axs = plt.subplots(2, sharex=True)
                # # grid flows
                axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                            color='lightcoral', marker='.')
                # battery flows
                if pm.value(m.bat_cap) > 0:
                    axs[0].plot(result['Electricity from grid to battery'], label='Electricity from grid to battery',
                                color='gold', marker='.')
                    axs[0].plot(result['Electricity from battery to process'],
                                label='Electricity from battery to process', color='darkkhaki', marker='s')
                    if pm.value(m.ElB_cap) > 0:
                        axs[0].plot(result['Electricity from battery to electric boiler'],
                                    label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
                    if pm.value(m.H2E_cap) > 0:
                        axs[0].plot(result['Electricity from battery to electrolyser'],
                                    label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
                    if pm.value(m.TES_cap) > 0:
                        axs[0].plot(result['Electricity from battery to TES'],
                                    label='Electricity from battery to TES', marker='s')
                    #axs[0].plot(result['Battery SOE'], label='Battery SOE', marker='2')
                # # electric boiler flows
                if pm.value(m.ElB_cap) > 0:
                    axs[0].plot(result['Electricity from grid to electric boiler'],
                                label='Electricity from grid to electric boiler', color='seagreen', marker='.')
                    axs[0].plot(result['Heat from electric boiler to process'],
                                label='Heat from electric boiler to process', color='turquoise', marker='.')

                # TES flows
                if pm.value(m.TES_cap) > 0:
                    axs[0].plot(result['Electricity from grid to TES'], label='Electricity from grid to TES',
                                marker='.')
                    axs[0].plot(result['Heat from TES to process'], label='Heat from TES to process',
                                color='deepskyblue', marker='.')

                # # Hydrogen flows
                if pm.value(m.H2E_cap) > 0:
                    axs[0].plot(result['Electricity from grid to electrolyser'],
                                label='Electricity from grid to electrolyser', color='royalblue', marker='.')
                    axs[0].plot(result['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
                                color='blueviolet', marker='.')
                    axs[0].plot(result['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
                                label='Hydrogen from electrolyser to H2 boiler', marker='.')
                    axs[0].plot(result['Hydrogen from electrolyser to storage'], color='fuchsia',
                                label='Hydrogen from electrolyser to storage', marker='.')
                    axs[0].plot(result['Hydrogen from storage to H2 boiler'], color='deeppink',
                                label='Hydrogen from storage to H2 boiler', marker='.')
                axs[0].axhline(y=gr_connection, color='grey', linestyle='--', label='Grid connection capacity')
                axs[0].set_ylabel("MW")
                axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')

                # plot prices for clarification
                axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                            markersize=0.75)
                axs[1].set_ylabel("EUR/MWh")
                axs[1].legend()
                plt.xlabel("Date")
                plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                    pm.value(m.ElB_cap) * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                    pm.value(m.TES_cap) * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                    pm.value(m.H2E_cap) * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                    pm.value(m.H2B_cap) * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                    pm.value(m.H2S_cap) * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = \
                    pm.value(m.bat_cap) * c_bat + \
                    pm.value(m.ElB_cap) * c_ElB + \
                    pm.value(m.TES_cap) * c_TES_A + \
                    pm.value(m.H2E_cap) * c_H2E + \
                    pm.value(m.H2B_cap) * c_H2B + \
                    pm.value(m.H2S_cap) * c_H2S
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = \
                    pm.value(m.bat_cap) * bat_areaftpr + pm.value(m.ElB_cap) * ElB_areaftpr + \
                    pm.value(m.TES_cap) * TES_areaftpr_A + pm.value(m.H2E_cap) * H2E_areaftpr + \
                    pm.value(m.H2B_cap) * H2B_areaftpr + pm.value(m.H2S_cap) * H2S_areaftpr
                el_price_scenario_dict[process][run][amp]['results']['CHP gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess heat gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to battery'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess power gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to grid'] = 0
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = Grid_gen
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = \
                    result['Electricity from grid to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = \
                    result['Electricity from grid to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = \
                    result['Electricity from grid to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to TES'] = \
                    result['Electricity from grid to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = \
                    result['Heat from electric boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = pm.value(m.ElB_cap)
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = pm.value(m.bat_cap)
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = \
                    result['Electricity from battery to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = \
                    result['Electricity from battery to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = \
                    result['Electricity from battery to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to TES'] = \
                    result['Electricity from battery to TES'].sum()
                if pm.value(m.bat_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours Battery'] \
                        = len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0])
                else:
                    el_price_scenario_dict[process][run][amp]['results']['Simultaeous charging and discharging hours'] \
                        = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = pm.value(m.TES_cap)
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = \
                    result['Heat from TES to process'].sum()
                if pm.value(m.TES_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours TES'] \
                        = len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = pm.value(m.H2E_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = pm.value(m.H2B_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = pm.value(m.H2S_cap)
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = result[
                    'Heat from H2 boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = \
                    result['Hydrogen from electrolyser to H2 boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = \
                    result['Hydrogen from electrolyser to storage'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = \
                    result['Hydrogen from storage to H2 boiler'].sum()
                if pm.value(m.H2S_cap) > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours H2S'] \
                        = len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = disc_rate
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # energy flows
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

                #print(current_process_dict)

                # # storing the results  # TODO: Update filename
                # filename = f'el_scenario_dict_{run}_{process}_{amp}'
                # with open(filename, 'ab') as process_dict_file:
                #     pickle.dump(el_price_scenario_dict, process_dict_file)
                # print("Finished saving el_price_scenario_dict")
    # for amp in variability:
    for count, amp in enumerate(['amp 1.025', 'amp 1.050', 'amp 1.075', 'amp 1.100', 'amp 1.125', 'amp 1.150',
                      'amp 1.175', 'amp 1.200']):
        print("Current variability amplitude is: ", amp)
        # Olefins
        el_price_scenario_dict['Olefins']['non-optimized'][amp][
            'power demand'] = 37.6690 + 1.38483E+02  # MW, power + cooling
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['heat demand'] = 180.8466  # MW, LPS
        el_price_scenario_dict['Olefins']['non-optimized'][amp]['available area'] = 75000  # in [m^2]
        # # Ethylene Oxide
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp][
        #     'power demand'] = 5.132 + 15.0363  # MW, power + cooling
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['heat demand'] = 30.0683  # MW, LPS
        # el_price_scenario_dict['Ethylene oxide']['non-optimized'][amp]['available area'] = 75000
        # # Ethylbenzene
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp][
        #     'power demand'] = 0.2991 + 0.5965  # MW, power + cooling
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['heat demand'] = 2.3019 + 41.0574  # MW, MPS + HPS
        # el_price_scenario_dict['Ethylbenzene']['non-optimized'][amp]['available area'] = 75000
        # # Ethylene Glycol
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp][
        #     'power demand'] = 1.0610 + 1.1383  # MW, power + cooling
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['heat demand'] = 44.3145  # MW , MPS
        # el_price_scenario_dict['Ethylene glycol']['non-optimized'][amp]['available area'] = 75000
        # # PET
        # el_price_scenario_dict['PET']['non-optimized'][amp]['power demand'] = 0.6659 + 0.4907  # MW, power + coolin
        # el_price_scenario_dict['PET']['non-optimized'][amp]['heat demand'] = 24.48670  # MW, HPS
        # el_price_scenario_dict['PET']['non-optimized'][amp]['available area'] = 80000

        for process in processes:
            print("Current process is: ", process)
            for run in ['non-optimized']:
                print("Current run is: ", run)
                current_process_dict = el_price_scenario_dict[process][run][amp]
                P_dem = current_process_dict['power demand']
                H_dem = current_process_dict['heat demand']
                available_area = current_process_dict['available area']

                # ------------------ START OPTIMISATION --------------------------------------------------------------------
                # Definitions

                def heat_balance(m, time):
                    return H_dem == m.H_ElB_CP[time] + m.H_TES_CP[time] + m.H_H2B_CP[time]

                def el_balance(m, time):
                    return P_dem == m.P_gr_CP[time] + m.P_bat_CP[time]

                def ElB_balance(m, time):
                    return m.H_ElB_CP[time] == (m.P_gr_ElB[time] + m.P_bat_ElB[time]) * eta_ElB

                def ElB_size(m, time):
                    return m.H_ElB_CP[time] <= ElB_cap

                def bat_soe(m, time):
                    if time == 0:
                        return m.bat_soe[time] == 0
                    else:
                        return m.bat_soe[time] == m.bat_soe[time - 1] + \
                               eta_bat * time_step * m.P_gr_bat[time - 1] - \
                               1 / eta_bat * time_step * (m.P_bat_CP[time - 1] + m.P_bat_ElB[time - 1]
                                                          + m.P_bat_H2E[time - 1] + m.P_bat_TES[time - 1])

                # Use Big M method to avoid simultaneous charging and discharging of the battery
                # TODO: revise crate
                def bat_in(m, time):
                    return m.P_gr_bat[time] <= bat_cap / eta_bat * crate_bat / time_step * m.b1[time]

                # TODO: revise crate
                def bat_out(m, time):
                    if time == 0:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_TES[time]) == 0
                    else:
                        return (m.P_bat_CP[time] + m.P_bat_ElB[time] + m.P_bat_H2E[time] + m.P_bat_TES[time]) \
                               <= bat_cap * eta_bat * crate_bat / time_step * (1 - m.b1[time])

                def bat_size(m, time):
                    return m.bat_soe[time] <= bat_cap

                def TES_soe(m, time):
                    if time == 0:
                        return m.TES_soe[time] == 0
                    else:
                        return m.TES_soe[time] == m.TES_soe[time - 1] \
                               + ((m.P_bat_TES[time - 1] + m.P_gr_TES[time - 1]) * eta_TES_B
                                  - m.H_TES_CP[time - 1]) * time_step

                # Use Big M method to avoid simultaneous charging and discharging of the TES TODO: test with and without
                # TODO: revise crate!
                def TES_in(m, time):
                    return (m.P_bat_TES[time] + m.P_gr_TES[time]) * eta_TES_B <= TES_cap * crate_TES / time_step #* m.b2[time]

                # TODO: revise crate!
                def TES_out(m, time):
                    if time == 0:
                        return m.H_TES_CP[time] == 0  #TODO: Test if e-boiler is only built because of this assumption. Try without!
                    else:
                        return m.H_TES_CP[time] <= TES_cap * eta_TES_A * crate_TES / time_step #* (1 - m.b2[time])

                def TES_size(m, time):
                    return m.TES_soe[time] <= TES_cap

                def H2S_soe(m, time):
                    if time == 0:
                        return m.H2S_soe[time] == 0
                    else:
                        return m.H2S_soe[time] == m.H2S_soe[time - 1] + (m.H2_H2E_H2S[time - 1] -
                                                                         m.H2_H2S_H2B[time - 1] / eta_H2S) * time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have
                # def H2S_in(m, time):
                #     return m.H2_H2E_H2S[time] <= m.H2S_cap * crate_H2S / time_step

                # # TODO: add C-rat, otherwise this constraint is not necesssary to have?
                # def H2S_out(m, time):
                #     if time == 0:
                #         return m.H2_H2S_H2B[time] == 0
                #     else:
                #         return m.H2_H2S_H2B[time] <= m.H2S_cap * crate_H2S * eta_H2S / time_step

                def H2S_size(m, time):
                    return m.H2S_soe[time] <= H2S_cap

                def H2B_balance(m, time):
                    return (m.H2_H2E_H2B[time] + m.H2_H2S_H2B[time]) * eta_H2B == m.H_H2B_CP[time]

                def H2B_size(m, time):
                    return m.H_H2B_CP[time] <= H2B_cap

                def H2E_balance(m, time):
                    return (m.P_gr_H2E[time] + m.P_bat_H2E[time]) * eta_H2E == m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time]

                def H2E_size(m, time):
                    return m.H2_H2E_H2B[time] + m.H2_H2E_H2S[time] <= H2E_cap


                def max_grid_power_in(m, time):  # total power flow from grid to plant is limited to x MW
                    return m.P_gr_CP[time] + m.P_gr_ElB[time] + m.P_gr_bat[time] + m.P_gr_H2E[time] + \
                           m.P_gr_TES[time] <= gr_connection

                def minimize_total_costs(m):
                    return sum(price_el_hourly.iloc[time, count] * time_step * (m.P_gr_ElB[time] + m.P_gr_CP[time] +
                                                                                m.P_gr_bat[time] + m.P_gr_H2E[time] +
                                                                                m.P_gr_TES[time])
                               for time in m.T) \
                           + \
                           bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                           ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                           TES_cap * c_TES_B * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                           H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                           H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                           H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)

                m = pm.ConcreteModel()

                # SETS
                m.T = pm.RangeSet(0, hours - 1)

                # CONSTANTS
                # Electric boiler
                c_ElB = 70000  # CAPEX for electric boiler, 70000 eur/MW
                ElB_lifetime = 20  # lifetime of electric boiler, years
                ElB_areaftpr = 30  # spatial requirements, m^2/MW
                eta_ElB = 0.99  # Conversion ratio electricity to steam for electric boiler [%]
                ElB_cap = el_price_scenario_dict[process][run]['original']['results'][
                    'ElB size']

                # Battery constants
                eta_bat = 0.95  # Battery (dis)charging efficiency
                c_bat = 300e3  # CAPEX for battery per eur/MWh, 338e3 USD --> 314.15e3 eur (12.07.23)
                bat_lifetime = 15  # lifetime of battery
                bat_areaftpr = 10  # spatial requirement, [m^2/MWh]
                crate_bat = 0.7  # C rate of battery, 0.7 kW/kWh, [-]
                bat_cap = el_price_scenario_dict[process][run]['original']['results']['Battery size']

                # TES constants
                c_TES_A = 23000  # CAPEX for sensible heat storage
                c_TES_B = 14000  # CAPEX for heat storage including heater, [UDS/MWh]
                c_TES_C = 60000  # CAPEX for latent heat storage [eur/MWh]
                TES_lifetime = 25  # heat storage lifetime, [years]
                eta_TES_A = 0.9  # discharge efficiency [-]
                eta_TES_B = 0.96  # charge efficiency [-]
                eta_TES_C = 0.98  # discharge efficiency [-]
                TES_areaftpr_A = 5  # spatial requirement TES, [m^2/MWh]
                TES_areaftpr_B = 5  # spatial requirement TES (configuration B), [m^2/MWh]
                crate_TES = 0.5  # C rate of TES, 0.5 kW/kWh, [-]  #TODO: revise this number
                TES_cap = el_price_scenario_dict[process][run]['original']['results']['TES size']

                # Hydrogen equipment constants
                eta_H2S = 0.9  # charge efficiency hydrogen storage [-], accounting for fugitive losses
                eta_H2B = 0.92  # conversion efficiency hydrogen boiler [-]
                eta_H2E = 0.69  # conversion efficiency electrolyser [-]
                c_H2S = 10000  # CAPEX for hydrogen storage per MWh, [eur/MWh]
                c_H2B = 35000  # CAPEX for hydrogen boiler per MW, [eur/MW]
                c_H2E = 700e3  # CAPEX for electrolyser per MW, [eur/MW]
                H2S_lifetime = 20  # lifetime hydrogen storage, [years]
                H2B_lifetime = 20  # lifetime hydrogen boiler, [years]
                H2E_lifetime = 15  # lifetime electrolyser, [years]
                H2E_areaftpr = 100  # spatial requirement electrolyser, [m^2/MW]
                H2B_areaftpr = 5  # spatial requirement hydrogen boiler, [m^2/MW]
                H2S_areaftpr = 10  # spatial requirement hydrogen storage, [m^2/MWh]
                H2E_cap = el_price_scenario_dict[process][run]['original']['results']['electrolyser size']
                H2B_cap = el_price_scenario_dict[process][run]['original']['results']['Hydrogen boiler size']
                H2S_cap = el_price_scenario_dict[process][run]['original']['results']['Hydrogen storage size']

                # other
                # Todo: Discuss grid connection assumption
                gr_connection = 400  # 'worst case' conversion chain

                param_NaN = math.isnan(sum(m.component_data_objects(ctype=type)))

                # VARIABLES
                m.P_gr_CP = pm.Var(m.T, bounds=(0, None))  # Power taken from grid for electricity demand, MW
                m.H_ElB_CP = pm.Var(m.T, bounds=(0, None))  # Heat generated from electricity, MW
                m.P_gr_ElB = pm.Var(m.T, bounds=(0, None))  # grid to el. boiler, MW
                m.P_gr_TES = pm.Var(m.T, bounds=(0, None))  # grid to TES, MW
                m.H_TES_CP = pm.Var(m.T, bounds=(0, None))  # Heat from TES to core process, MW
                m.TES_soe = pm.Var(m.T, bounds=(0, None))  # state of energy TES, MWh
                m.P_gr_bat = pm.Var(m.T, bounds=(0, None))  # max charging power batter, MW
                m.P_bat_CP = pm.Var(m.T, bounds=(0, None))  # discharging power batter to core process, MW
                m.P_bat_ElB = pm.Var(m.T, bounds=(0, None))  # discharging power batter to electric boiler, MW
                m.P_bat_TES = pm.Var(m.T, bounds=(0, None))  # discharging power batter to TES, MW
                m.bat_soe = pm.Var(m.T, bounds=(0, None))  # State of energy of battery
                m.H_H2B_CP = pm.Var(m.T, bounds=(0, None))  # Heat flow from hydrogen boiler to core process, MW
                m.H2S_soe = pm.Var(m.T, bounds=(0, None))  # state of energy hydrogen storage, MWh
                m.H2_H2E_H2S = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen storage, MWh
                m.H2_H2S_H2B = pm.Var(m.T,
                                      bounds=(0, None))  # hydrogen flow from hydrogen storage to hydrogen boiler, MWh
                m.H2_H2E_H2B = pm.Var(m.T, bounds=(0, None))  # hydrogen flow from electrolyser to hydrogen boiler, MWh
                m.P_gr_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from grid to electrolyser, MW
                m.P_bat_H2E = pm.Var(m.T, bounds=(0, None))  # power flow from battery to electrolyser, MW
                m.b1 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # of the battery
                # m.b2 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous charging and discharging
                # # of the TES
                # m.b3 = pm.Var(m.T, within=pm.Binary)  # binary variable to avoid simultaneous bidirectional use of the
                # # grid connection


                # CONSTRAINTS
                # balance supply and demand
                m.heat_balance_constraint = pm.Constraint(m.T, rule=heat_balance)
                m.P_balance_constraint = pm.Constraint(m.T, rule=el_balance)
                # electric boiler constraint
                m.ElB_size_constraint = pm.Constraint(m.T, rule=ElB_size)
                m.ElB_balance_constraint = pm.Constraint(m.T, rule=ElB_balance)
                # battery constraints
                m.bat_soe_constraint = pm.Constraint(m.T, rule=bat_soe)
                m.bat_out_maxP_constraint = pm.Constraint(m.T, rule=bat_out)
                m.bat_in_constraint = pm.Constraint(m.T, rule=bat_in)
                m.bat_size_constraint = pm.Constraint(m.T, rule=bat_size)
                # TES constraints
                m.TES_discharge_constraint = pm.Constraint(m.T, rule=TES_out)
                m.TES_charge_constraint = pm.Constraint(m.T, rule=TES_in)
                m.TES_soe_constraint = pm.Constraint(m.T, rule=TES_soe)
                m.TES_size_constraint = pm.Constraint(m.T, rule=TES_size)
                # hydrogen constraints
                m.H2S_soe_constraint = pm.Constraint(m.T, rule=H2S_soe)
                m.H2B_balance_constraint = pm.Constraint(m.T, rule=H2B_balance)
                m.H2E_balance_constraint = pm.Constraint(m.T, rule=H2E_balance)
                m.H2S_size_constraint = pm.Constraint(m.T, rule=H2S_size)
                m.H2B_size_constraint = pm.Constraint(m.T, rule=H2B_size)
                m.H2E_size_constraint = pm.Constraint(m.T, rule=H2E_size)
                # m.H2S_discharge_constraint = pm.Constraint(m.T, rule=H2S_out)
                # m.H2S_charge_constraint = pm.Constraint(m.T, rule=H2S_in)
                # grid constraints
                m.max_grid_power_in_constraint = pm.Constraint(m.T, rule=max_grid_power_in)

                # OBJECTIVE FUNCTION
                m.objective = pm.Objective(rule=minimize_total_costs,
                                           sense=pm.minimize,
                                           doc='Define objective function')  # what does this last part do?

                # Solve optimization problem
                opt = pm.SolverFactory('gurobi')
                results = opt.solve(m, tee=True)

                # ------------------ OPTIMISATION END --------------------------------------------------------------------------

                # Collect results
                result = pd.DataFrame(index=price_ng_hourly.index[0:hours])
                result['Heat demand process'] = H_dem
                result['Power demand process'] = P_dem
                result['Heat from electric boiler to process'] = pm.value(m.H_ElB_CP[:])
                result['Electricity from grid to electric boiler'] = pm.value(m.P_gr_ElB[:])
                result['Electricity from grid to process'] = pm.value(m.P_gr_CP[:])
                result['Battery SOE'] = pm.value(m.bat_soe[:])
                result['Electricity from battery to electric boiler'] = pm.value(m.P_bat_ElB[:])
                result['Electricity from battery to process'] = pm.value(m.P_bat_CP[:])
                result['Electricity from grid to battery'] = pm.value(m.P_gr_bat[:])
                result['Electricity from grid to TES'] = pm.value(m.P_gr_TES[:])
                result['Electricity from battery to TES'] = pm.value(m.P_bat_TES[:])
                result['Heat from TES to process'] = pm.value(m.H_TES_CP[:])
                result['Electricity from grid to electrolyser'] = pm.value(m.P_gr_H2E[:])
                result['Heat from hydrogen boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Electricity from battery to electrolyser'] = pm.value(m.P_bat_H2E[:])
                result['Heat from H2 boiler to process'] = pm.value(m.H_H2B_CP[:])
                result['Hydrogen from electrolyser to H2 boiler'] = pm.value(m.H2_H2E_H2B[:])
                result['Hydrogen from electrolyser to storage'] = pm.value(m.H2_H2E_H2S[:])
                result['Hydrogen from storage to H2 boiler'] = pm.value(m.H2_H2S_H2B[:])
                # check if grid capacity constraint is hit
                grid_P_out = result['Electricity from grid to process'] + \
                             result['Electricity from grid to electric boiler'] + \
                             result['Electricity from grid to battery'] + \
                             result['Electricity from grid to electrolyser'] + \
                             result['Electricity from grid to TES']
                grid_P_out_max = max(grid_P_out)
                Grid_gen = result['Electricity from grid to process'].sum() + \
                           result['Electricity from grid to electric boiler'].sum() + \
                           result['Electricity from grid to battery'].sum() + \
                           result['Electricity from grid to electrolyser'].sum() + \
                           result['Electricity from grid to TES'].sum()


                # Scope 2 CO2 emissions
                grid_use_hourly = pd.DataFrame({'Grid to CP': result['Electricity from grid to process'],
                                                'Grid to electric boiler': result[
                                                    'Electricity from grid to electric boiler'],
                                                'Grid to battery': result['Electricity from grid to battery'],
                                                'Grid to electrolyser': result[
                                                    'Electricity from grid to electrolyser'],
                                                'Grid to TES': result['Electricity from grid to TES']
                                                })
                total_grid_use_hourly = grid_use_hourly.sum(axis=1)
                scope_2_CO2 = (CO2_emiss_grid_hourly.div(1000)).mul(total_grid_use_hourly, axis='index')  # leads to
                # [ton/MWh] * [MWh] = ton
                scope_2_CO2.rename(columns={'Carbon Intensity gCO2eq/kWh (direct)': 'Carbon Emissions [ton] (direct)'},
                                   inplace=True)
                total_scope_2_CO2 = scope_2_CO2['Carbon Emissions [ton] (direct)'].sum()

                # control: H_CP==H_dem and P_CP==P_dem?
                control_H = sum(
                    result['Heat demand process'] - (result['Heat from electric boiler to process'] +
                                                     result['Heat from TES to process'] +
                                                     result['Heat from hydrogen boiler to process']))
                # - result['Excess heat from CHP(ng)']
                control_P = sum(result['Power demand process'] - (result['Electricity from grid to process'] +
                                                                  result['Electricity from battery to process']))
                print("control_H =", control_H)
                print("control_P =", control_P)
                print("Objective = ", pm.value(m.objective))
                # print("Investment cost battery per MWh, USD = ", c_bat)
                # print("Investment cost electric boiler per MW, USD = ", c_ElB)
                # print("Investment cost TES per MWh, USD = ", c_TES_A)
                print("Battery capacity =", bat_cap)
                print("Electric boiler capacity =", ElB_cap)
                print("TES capacity =", TES_cap)
                print("electrolyser capacity =", H2E_cap)
                print("Hydrogen boiler capacity =", H2B_cap)
                print("Hydrogen storage capacity =", H2S_cap)
                print("Grid capacity: ", gr_connection, "Max. power flow from grid: ", grid_P_out_max)

                # IF battery capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if bat_cap > 0:
                    battery_discharge_sum = result['Electricity from battery to electrolyser'] + \
                                            result['Electricity from battery to electric boiler'] + \
                                            result['Electricity from battery to process'] + \
                                            result['Electricity from battery to TES']
                    battery_charge_sum = result['Electricity from grid to battery']
                    battery_hours_with_simultaneous_charging_and_discharging = pd.Series(index=battery_charge_sum.index)
                    for i in range(0, len(battery_charge_sum)):
                        if battery_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if battery_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                battery_hours_with_simultaneous_charging_and_discharging[i] = battery_charge_sum[i] + \
                                                                                              battery_discharge_sum[i]
                    print("Number of hours of simultaneous battery charging and discharging: ",
                          len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF TES capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if TES_cap > 0:
                    TES_discharge_sum = result['Heat from TES to process']
                    TES_charge_sum = result['Electricity from grid to TES'] + result['Electricity from battery to TES']
                    TES_hours_with_simultaneous_charging_and_discharging = pd.Series(index=TES_charge_sum.index)
                    for i in range(0, len(TES_charge_sum)):
                        if TES_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if TES_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                TES_hours_with_simultaneous_charging_and_discharging[i] = TES_charge_sum[i] + \
                                                                                          TES_discharge_sum[i]
                    print("Number of hours of simultaneous TES charging and discharging: ",
                          len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0]))

                # IF H2S capacity is installed, how many hours does the battery charge and discharge simultaneously?
                if H2S_cap > 0:
                    H2S_discharge_sum = result['Hydrogen from storage to H2 boiler']
                    H2S_charge_sum = result['Hydrogen from electrolyser to storage']
                    H2S_hours_with_simultaneous_charging_and_discharging = pd.Series(index=H2S_charge_sum.index)
                    for i in range(0, len(H2S_charge_sum)):
                        if H2S_charge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                            if H2S_discharge_sum[i] > 0.00001:  # because using 0 led to rounding errors
                                H2S_hours_with_simultaneous_charging_and_discharging[i] = H2S_charge_sum[i] + \
                                                                                          H2S_discharge_sum[i]
                    print("Number of hours of simultaneous H2S charging and discharging: ",
                          len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0]))


                # energy flows and prices in one figure for analysis
                fig, axs = plt.subplots(2, sharex=True)
                # # grid flows
                axs[0].plot(result['Electricity from grid to process'], label='Electricity from grid to process',
                            color='lightcoral', marker='.')
                # battery flows
                if bat_cap > 0:
                    axs[0].plot(result['Electricity from grid to battery'], label='Electricity from grid to battery',
                                color='gold', marker='.')
                    axs[0].plot(result['Electricity from battery to process'],
                                label='Electricity from battery to process', color='darkkhaki', marker='s')
                    if ElB_cap > 0:
                        axs[0].plot(result['Electricity from battery to electric boiler'],
                                    label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
                    if H2E_cap > 0:
                        axs[0].plot(result['Electricity from battery to electrolyser'],
                                    label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
                    if TES_cap > 0:
                        axs[0].plot(result['Electricity from battery to TES'],
                                    label='Electricity from battery to TES', marker='s')
                    #axs[0].plot(result['Battery SOE'], label='Battery SOE', marker='2')
                # # electric boiler flows
                if ElB_cap > 0:
                    axs[0].plot(result['Electricity from grid to electric boiler'],
                                label='Electricity from grid to electric boiler', color='seagreen', marker='.')
                    axs[0].plot(result['Heat from electric boiler to process'],
                                label='Heat from electric boiler to process', color='turquoise', marker='.')

                # TES flows
                if TES_cap > 0:
                    axs[0].plot(result['Electricity from grid to TES'], label='Electricity from grid to TES',
                                marker='.')
                    axs[0].plot(result['Heat from TES to process'], label='Heat from TES to process',
                                color='deepskyblue', marker='.')

                # # Hydrogen flows
                if H2E_cap > 0:
                    axs[0].plot(result['Electricity from grid to electrolyser'],
                                label='Electricity from grid to electrolyser', color='royalblue', marker='.')
                    axs[0].plot(result['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
                                color='blueviolet', marker='.')
                    axs[0].plot(result['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
                                label='Hydrogen from electrolyser to H2 boiler', marker='.')
                    axs[0].plot(result['Hydrogen from electrolyser to storage'], color='fuchsia',
                                label='Hydrogen from electrolyser to storage', marker='.')
                    axs[0].plot(result['Hydrogen from storage to H2 boiler'], color='deeppink',
                                label='Hydrogen from storage to H2 boiler', marker='.')
                axs[0].axhline(y=gr_connection, color='grey', linestyle='--', label='Grid connection capacity')
                axs[0].set_ylabel("MW")
                axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')

                # plot prices for clarification
                axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
                            markersize=0.75)
                axs[1].set_ylabel("EUR/MWh")
                axs[1].legend()
                plt.xlabel("Date")
                plt.show()

                # Add results for stacked bar chart "Optimal energy supply" to process dictionaries
                el_price_scenario_dict[process][run][amp]['results']['Optimal result'] = pm.value(m.objective)
                el_price_scenario_dict[process][run][amp]['results']['CAPEX'] = \
                    bat_cap * c_bat * disc_rate / (1 - (1 + disc_rate) ** -bat_lifetime) + \
                    ElB_cap * c_ElB * disc_rate / (1 - (1 + disc_rate) ** -ElB_lifetime) + \
                    TES_cap * c_TES_A * disc_rate / (1 - (1 + disc_rate) ** -TES_lifetime) + \
                    H2E_cap * c_H2E * disc_rate / (1 - (1 + disc_rate) ** -H2E_lifetime) + \
                    H2B_cap * c_H2B * disc_rate / (1 - (1 + disc_rate) ** -H2B_lifetime) + \
                    H2S_cap * c_H2S * disc_rate / (1 - (1 + disc_rate) ** -H2S_lifetime)
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] = \
                    bat_cap * c_bat + \
                    ElB_cap * c_ElB + \
                    TES_cap * c_TES_A + \
                    H2E_cap * c_H2E + \
                    H2B_cap * c_H2B + \
                    H2S_cap * c_H2S
                el_price_scenario_dict[process][run][amp]['results']['OPEX'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] - \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX']
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions'] = total_scope_2_CO2
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] - \
                    el_price_scenario_dict[process][run][amp]['results']['Cost for EUA']
                el_price_scenario_dict[process][run][amp]['results']['required area'] = \
                    bat_cap * bat_areaftpr + ElB_cap * ElB_areaftpr + \
                    TES_cap * TES_areaftpr_A + H2E_cap * H2E_areaftpr + \
                    H2B_cap * H2B_areaftpr + H2S_cap * H2S_areaftpr
                el_price_scenario_dict[process][run][amp]['results']['CHP gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP heat gen to TES'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess heat gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to CP'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to battery'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP excess power gen'] = 0
                el_price_scenario_dict[process][run][amp]['results']['CHP power gen to grid'] = 0
                el_price_scenario_dict[process][run][amp]['results']['total grid consumption'] = Grid_gen
                el_price_scenario_dict[process][run][amp]['results']['total natural gas consumption'] = 0
                el_price_scenario_dict[process][run][amp]['results']['grid to CP'] = \
                    result['Electricity from grid to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to battery'] = \
                    result['Electricity from grid to battery'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electric boiler'] = \
                    result['Electricity from grid to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to electrolyser'] = \
                    result['Electricity from grid to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['grid to TES'] = \
                    result['Electricity from grid to TES'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB gen to CP'] = \
                    result['Heat from electric boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['ElB size'] = ElB_cap
                el_price_scenario_dict[process][run][amp]['results']['Battery size'] = bat_cap
                el_price_scenario_dict[process][run][amp]['results']['battery to ElB'] = \
                    result['Electricity from battery to electric boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to CP'] = \
                    result['Electricity from battery to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to electrolyser'] = \
                    result['Electricity from battery to electrolyser'].sum()
                el_price_scenario_dict[process][run][amp]['results']['battery to TES'] = \
                    result['Electricity from battery to TES'].sum()
                if bat_cap > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours Battery'] \
                        = len(battery_hours_with_simultaneous_charging_and_discharging[
                                  battery_hours_with_simultaneous_charging_and_discharging > 0])
                else:
                    el_price_scenario_dict[process][run][amp]['results']['Simultaeous charging and discharging hours'] \
                        = 0
                el_price_scenario_dict[process][run][amp]['results']['TES size'] = TES_cap
                el_price_scenario_dict[process][run][amp]['results']['TES to CP'] = \
                    result['Heat from TES to process'].sum()
                if TES_cap > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours TES'] \
                        = len(TES_hours_with_simultaneous_charging_and_discharging[
                                  TES_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['electrolyser size'] = H2E_cap
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler size'] = H2B_cap
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen storage size'] = H2S_cap
                el_price_scenario_dict[process][run][amp]['results']['Hydrogen boiler to CP'] = result[
                    'Heat from H2 boiler to process'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to boiler'] = \
                    result['Hydrogen from electrolyser to H2 boiler'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from electrolyser to storage'] = \
                    result['Hydrogen from electrolyser to storage'].sum()
                el_price_scenario_dict[process][run][amp]['results']['H2 from storage to boiler'] = \
                    result['Hydrogen from storage to H2 boiler'].sum()
                if H2S_cap > 0:
                    el_price_scenario_dict[process][run][amp]['results'][
                        'Simultaneous charging and discharging hours H2S'] \
                        = len(H2S_hours_with_simultaneous_charging_and_discharging[
                                  H2S_hours_with_simultaneous_charging_and_discharging > 0])
                el_price_scenario_dict[process][run][amp]['results']['grid connection cap'] = gr_connection
                el_price_scenario_dict[process][run][amp]['results']['discount rate'] = disc_rate
                el_price_scenario_dict[process][run][amp]['results']['available area [m^2]'] = available_area
                el_price_scenario_dict[process][run][amp]['results']['max. power flow from grid [MW]'] = grid_P_out_max

                # 'extra' entries (processed data)
                el_price_scenario_dict[process][run][amp]['results']['Optimal result [million eur]'] \
                    = pm.value(m.objective) / 1E6
                el_price_scenario_dict[process][run][amp]['results']['CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Non-annualized CAPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['Share of CAPEX in total cost [%]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['CAPEX'] / \
                    el_price_scenario_dict[process][run][amp]['results']['Optimal result'] * 100
                el_price_scenario_dict[process][run][amp]['results']['OPEX [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['OPEX'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['scope 1 emissions [kiloton]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['Cost for EUA [million eur]'] = 0
                el_price_scenario_dict[process][run][amp]['results']['scope 2 emissions [kiloton]'] = \
                    total_scope_2_CO2 / 1E3
                el_price_scenario_dict[process][run][amp]['results']['Fuel cost [million eur]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['Fuel cost'] / 1E6
                el_price_scenario_dict[process][run][amp]['results']['required area [km^2]'] = \
                    el_price_scenario_dict[process][run][amp]['results']['required area'] / 1E6

                # energy flows
                el_price_scenario_dict[process][run][amp]['energy flows'] = result

    return el_price_scenario_dict
