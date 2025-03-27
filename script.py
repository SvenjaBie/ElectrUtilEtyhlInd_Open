import pyomo.environ as pm
import pandas as pd
import os
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from pathlib import Path
from functions import optimisation_run_GB, optimisation_run_CHP_feed_in, benchmark_CHP_scheduling_optimisation, \
    optimisation_run_fully_electrified, optimisation_run_fully_electrified_ESCAPE, benchmark_GB_scheduling_optimisation

# IMPORT INPUT DATA
reporoot_dir = Path(__file__).resolve().parent
# make list with all data which follows the same order of "years" list
# for natural gas prices in eur/MWh(electricity)
ng_prices_all_years = {
    '2018': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2018.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
    '2019': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2019.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
    '2020': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2020.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
    '2021': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2021.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
    '2022': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2022.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
    '2023': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2023.csv'), delimiter=";",
                        index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False)}
# convert Index to DatetimeIndex
for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
    ng_prices_all_years[year].index = pd.to_datetime(ng_prices_all_years[year].index, format='mixed', dayfirst=False)

# for electricity prices in eur/MWh(electricity)
el_prices_all_years = {
    '2018': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2018.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True),
    '2019': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2019.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True),
    '2020': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2020.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True),
    '2021': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2021.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True),
    '2022': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2022.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True),
    '2023': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2023.csv'),
                        delimiter=";",
                        usecols=[0, 1], parse_dates=True, dayfirst=True)}

# turn first column into Datetime index
for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
    # first, cut the required characters from the string in the first column
    for i in range(0, len(el_prices_all_years[year])):
        el_prices_all_years[year].iloc[i, 0] = el_prices_all_years[year].iloc[i, 0][0:17]

    # then, convert the remaining characters into a datetime object and use them as new index of the dataframes
    el_prices_all_years[year].index = pd.to_datetime(el_prices_all_years[year].iloc[:, 0], dayfirst=True)

    # delete the first column (which is the new index)
    el_prices_all_years[year] = el_prices_all_years[year].iloc[:, 1:]

# for ETS prices in eur/ton
ETS_prices_all_years = {
    '2018': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Sandbag_2018data.csv'),
                        delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
    '2019': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Sandbag_2019data.csv'),
                        delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
    '2020': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2020data.csv'),
                        delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
    '2021': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2021data.csv'),
                        delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
    '2022': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/EMBER_Coal2Clean_EUETSPrices_2022.csv'),
                        delimiter=";", index_col=0, parse_dates=True),
    '2023': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2023data.csv'),
                        delimiter=";", index_col=0, parse_dates=True, dayfirst=True)}
# for CO2 footprint power grid in ton
# TODO: Look for real data from 2018-2020 and replace those
CO2_emiss_grid_all_years = {'2018': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2021_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True),
                            '2019': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2021_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True),
                            '2020': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2021_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True),
                            '2021': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2021_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True),
                            '2022': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2022_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True),
                            '2023': pd.read_csv(os.path.join(reporoot_dir, r'CO2_ftpr_data'
                                                                           r'/carbonintensity_NL_2023_hourly.csv'),
                                                delimiter=";", index_col=0,
                                                usecols=["Datetime (UTC)", "Carbon Intensity gCO2eq/kWh (direct)"],
                                                parse_dates=True, dayfirst=True)}

# # plot the price data
# # resample electricity price data to get average daily prices and save the daily prices in a new dataframe
# el_prices_all_years_daily = pd.DataFrame
# el_prices_all_years_daily = el_prices_all_years
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     el_prices_all_years_daily[year] = el_prices_all_years[year].resample('1d', axis='index').mean()
#     # mean_el_price = el_prices_all_years_daily[year]['Day-ahead Price [EUR/MWh]'].mean()
#     # print(mean_el_price)
#
# # resample natural gas price data to fill the weekend days and save the daily prices in a new dataframe
# ng_prices_all_years_daily = pd.DataFrame
# ng_prices_all_years_daily = ng_prices_all_years
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     ng_prices_all_years_daily[year] = ng_prices_all_years[year].resample('1d', axis='index').ffill()
#     #remove the last row to end at 31st of december
#     ng_prices_all_years_daily[year] = ng_prices_all_years_daily[year][:-1]
#     # mean_ng_price = ng_prices_all_years_daily[year]['Open'].mean()
#     # print(mean_ng_price)
#
# # resample ETS price data to fill the weekend days and save the daily prices in a new dataframe
# ETS_prices_all_years_daily = pd.DataFrame
# ETS_prices_all_years_daily = ETS_prices_all_years
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     ETS_prices_all_years_daily[year] = ETS_prices_all_years[year].resample('1d', axis='index').ffill()
#     # remove the last row to end at 31st of december
#     ETS_prices_all_years_daily[year] = ETS_prices_all_years_daily[year][:-1]
#     # mean_ETS_price = ETS_prices_all_years_daily[year]['Price'].mean()
#     # print(mean_ETS_price)

# # store the data in lists for the plot
# dates_list = []
# el_price_list = []
# ng_price_list = []
# ETS_price_list = []
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     dates_list.extend(el_prices_all_years_daily[year].index.date.tolist())
#     el_price_list.extend(el_prices_all_years_daily[year]['Day-ahead Price [EUR/MWh]'].tolist())
#     ng_price_list.extend(ng_prices_all_years_daily[year]['Open'].tolist())
#     ETS_price_list.extend(ETS_prices_all_years_daily[year]['Price'].tolist())

# # make the plot
# fig, axs = plt.subplots(3, sharex=True, sharey=True)
# axs[0].plot(dates_list, el_price_list)
# axs[0].set_title("Electricity price", fontsize=18, weight='bold')
# axs[0].set_ylabel('EUR/MWh', fontsize=18, weight='bold')
# axs[0].tick_params(labelsize=18, width=4)
# axs[1].plot(dates_list, ng_price_list)
# axs[1].set_title("Natural gas price", fontsize=16, weight='bold')
# axs[1].set_ylabel('EUR/MWh', fontsize=18, weight='bold')
# axs[1].tick_params(labelsize=18, width=4)
# axs[2].plot(dates_list, ETS_price_list)
# axs[2].set_title("EU ETS allowance price", fontsize=18, weight='bold')
# axs[2].set_ylabel('EUR/ton', fontsize=18, weight='bold')
# axs[2].tick_params(labelsize=18, width=4)
# plt.show()
# print('stop')


# # # __________________________Run optimisation for all years individually _____________________________________________
# # ________________Optional: Run simulation for different grid connection capacities____________________________________
# # define years which are included in the scenarios:
# years = ['2018', '2019', '2020', '2021', '2022', '2023']
#
# # define the number of hours for which the optimisation should run
# hours = 8000
#
# # define amp_values for which data variability should be amplified. Empty if no amplification should happen, otherwise
# # it should look like this: amp_values = [1.050, 1.100, 1.150, ...]
# amp_values = []  # [1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200]
#
# # define variability values
# # (If variability has not been amplified, this is only 'original'. Otherwise, variability_values should look like this:
# # variability_values = ['original', 'amp 1.050', 'amp 1.100', ....]. TODO: It has to match with amp_values!)
# variability_values = ['original']
#
# # define grid connection capacity (gr_cap is a factor which is multiplied with the utility demand of the process. If
# # gr_cap = 1, the entire utility demand can be supplied via the grid.)
# gr_cap_values = [1.0]
# for gr_cap in gr_cap_values:
#     print("Grid cap is: " + str(gr_cap))
#
#     # define discount rate
#     disc_rate = 0.1
#
#     # define minimal load of the legacy technology
#     min_load_CHP = 0.5
#     min_load_GB = min_load_CHP
#
#     # prepare dictionary where all individual dicts are stored
#     single_scenarios_dict = {}
#
#     # run the optimisation for all years
#     for year in years:
#         print('Starting ' + year)
#         # hand over the required data input
#         price_el_hourly = el_prices_all_years[year]
#         price_ng_orig = ng_prices_all_years[year]
#         price_EUA_orig = ETS_prices_all_years[year]
#         CO2_emiss_grid_hourly = CO2_emiss_grid_all_years[year]
#         capex_data = {'c_ElB': 70000, 'c_bat': 300e3, 'c_TES': 23000, 'c_H2E': 700e3, 'c_H2B': 35000, 'c_H2S': 10000}
#
#         # call the (single scenario) optimisation and collect the results for the individual years in single_scenarios_dict
#         single_scenarios_dict[year] = optimisation_run_GB(price_el_hourly, price_ng_orig, price_EUA_orig,
#                                                                    CO2_emiss_grid_hourly, amp_values,
#                                                                    variability_values, gr_cap,
#                                                                    hours, disc_rate, capex_data, min_load_GB)
#
#         # single_scenarios_dict[year] = optimisation_run_fully_electrified_ESCAPE(price_el_hourly, price_ng_orig,
#         #                                                                  CO2_emiss_grid_hourly, amp_values,
#         #                                                                  variability_values, hours, disc_rate)
#
#     # store the single_scenarios_dict dictionary as pickle file
#     # TODO:Change prefix according to the years and the function
#     prefix = 'single_scenarios_dict_GB' + '_gr-cap ' + str(gr_cap) + '_' + '_dis-rate ' + str(disc_rate) + \
#              '_min-load-GB_' + str(min_load_CHP) + '_' + years[0] + ' to ' + years[len(years) - 1] + '_woSpaceConstr'
#
#     timestamp_format = "{:%Y%m%dT%H%M}"
#     timestamp = timestamp_format.format(datetime.now())
#     output_filename = f"{prefix}__{timestamp}.pickle"
#
#     with open(output_filename, 'wb') as handle:
#         pickle.dump(single_scenarios_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("Finished saving all_single_scenarios_dict")

## ________________________________________ CAPEX sensitivity analysis___________________________________________________
# define years which are included in the scenarios:
years = ['2018', '2019', '2020', '2021', '2022', '2023']
# define which technologies should be included in the scenarios:
technologies = ['Bat', 'TES', 'H2E']
# define CAPEX scenarios
capex_scenarios = ['Low', 'High']

# define capex data according to 'technologies' and 'capex_scenarios' lists
capex_data_dict = {
    'Bat-High': {'c_ElB': 70000, 'c_bat': 300e3 * (1 + 0.25), 'c_TES': 23000, 'c_H2E': 700e3, 'c_H2B': 35000, 'c_H2S': 10000},
    'Bat-Low': {'c_ElB': 70000, 'c_bat': 300e3 * (1 - 0.25), 'c_TES': 23000, 'c_H2E': 700e3, 'c_H2B': 35000, 'c_H2S': 10000},
    'TES-High': {'c_ElB': 70000, 'c_bat': 300e3, 'c_TES': 23000 * (1 + 0.25), 'c_H2E': 700e3, 'c_H2B': 35000, 'c_H2S': 10000},
    'TES-Low': {'c_ElB': 70000, 'c_bat': 300e3, 'c_TES': 23000 * (1 - 0.25), 'c_H2E': 700e3, 'c_H2B': 35000, 'c_H2S': 10000},
    'H2E-High': {'c_ElB': 70000, 'c_bat': 300e3, 'c_TES': 23000, 'c_H2E': 700e3 * (1 + 0.25), 'c_H2B': 35000, 'c_H2S': 10000},
    'H2E-Low': {'c_ElB': 70000, 'c_bat': 300e3, 'c_TES': 23000, 'c_H2E': 700e3 * (1 - 0.25), 'c_H2B': 35000, 'c_H2S': 10000},

}

# define the number of hours for which the optimisation should run
hours = 8000

# define amp_values for which data variability should be amplified. Empty if no amplification should happen, otherwise
# it should look like this: amp_values = [1.050, 1.100, 1.150, ...]
amp_values = []  # [1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200]

# define variability values
# (If variability has not been amplified, this is only 'original'. Otherwise, variability_values should look like this:
# variability_values = ['original', 'amp 1.050', 'amp 1.100', ....]. TODO: It has to match with amp_values!)
variability_values = ['original']

# define grid connection capacity (gr_cap is a factor which is multiplied with the utility demand of the process. If
# gr_cap = 1, the entire utility demand can be supplied via the grid.)
gr_cap = 1.0

# define discount rate
disc_rate = 0.1

# define minimal load of the legacy technology
min_load_CHP = 0.5
min_load_GB = min_load_CHP

# prepare dictionary where all individual dicts are stored
single_scenarios_dict = {year: {capex_scenario: {} for capex_scenario in capex_data_dict} for year in years}

# run the optimisation for all years
for year in years:
    print('Starting ' + year)
    for technology in technologies:
        for capex_scenario in capex_scenarios:
            print("Starting capex scenario " + str(technology + '-' + capex_scenario))
            # hand over the required data input
            capex_data = capex_data_dict[str(technology + '-' + capex_scenario)]
            price_el_hourly = el_prices_all_years[year]
            price_ng_orig = ng_prices_all_years[year]
            price_EUA_orig = ETS_prices_all_years[year]
            CO2_emiss_grid_hourly = CO2_emiss_grid_all_years[year]

            # call the (single scenario) optimisation and collect the results for the individual years in single_scenarios_dict
            single_scenarios_dict[year][str(technology + '-' + capex_scenario)] = \
                optimisation_run_CHP_feed_in(price_el_hourly, price_ng_orig, price_EUA_orig, CO2_emiss_grid_hourly,
                                             amp_values, variability_values, gr_cap, hours, disc_rate, capex_data,
                                             min_load_CHP)

# store the single_scenarios_dict dictionary as pickle file
# TODO:Change prefix according to the runs
prefix = 'SA_TC_woSpatCons_with_CHP' + '_gr-cap ' + str(gr_cap) + '_' + '_dis-rate ' + str(disc_rate) + \
         '_min-load_' + str(min_load_CHP) + '_' + years[0] + ' to ' + years[len(years) - 1]
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(single_scenarios_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving all_single_scenarios_dict")

prefix = 'SA_TCdata_'
timestamp_format = "{:%Y%m%dT%H%M}"
timestamp = timestamp_format.format(datetime.now())
output_filename = f"{prefix}__{timestamp}.pickle"
with open(output_filename, 'wb') as handle:
    pickle.dump(capex_data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Finished saving capex data")

# ## _________Run optimisation for benchmark system for all years individually. Optional: Run simulation for different
# ## grid capacities__________
#
# # define years which are included in the scenarios:
# years = ['2018', '2019', '2020', '2021', '2022', '2023']
#
# # define the number of hours for which the optimisation should run
# hours = 8000
#
# # define amp_values for which data variability should be amplified. Empty if no amplification should happen, otherwise
# # it should look like this: amp_values = [1.050, 1.100, 1.150, ...]
# amp_values = []  # [1.025, 1.050, 1.075, 1.100, 1.125, 1.150, 1.175, 1.200]
#
# # define variability values
# # (If variability has not been amplified, this is only 'original'. Otherwise, variability_values should look like this:
# # variability_values = ['original', 'amp 1.050', 'amp 1.100', ....]. TODO: It has to match with amp_values!)
# variability_values = [
#     'original']  # ['original', 'amp 1.025', 'amp 1.050', 'amp 1.075', 'amp 1.100', 'amp 1.125', 'amp 1.150', 'amp 1.175', 'amp 1.200']
#
# # define grid connection capacity (gr_cap is a factor which is multiplied with the utility demand of the process. If
# # gr_cap = 1, the entire utility demand can be supplied via the grid.)
# gr_cap = 1
# CHP_min_load = 0.3
# GB_cap_min = CHP_min_load
#
# # prepare dictionary where all individual dicts are stored
# benchmark_single_scenarios_dict = {}
#
# # run the optimisation for all years
# for year in years:
#     # hand over the required data input
#     price_el_hourly = el_prices_all_years[year]
#     price_ng_orig = ng_prices_all_years[year]
#     price_EUA_orig = ETS_prices_all_years[year]
#     CO2_emiss_grid_hourly = CO2_emiss_grid_all_years[year]
#
#     # call the (single scenario) optimisation
#     print('Starting ' + year)
#     # benchmark_single_scenario_dict = benchmark_CHP_scheduling_optimisation(price_el_hourly, price_ng_orig,
#     #                                                                        price_EUA_orig, CO2_emiss_grid_hourly,
#     #                                                                        amp_values, variability_values, hours,
#     #                                                                        gr_cap, CHP_min_load)
#     benchmark_single_scenario_dict = benchmark_GB_scheduling_optimisation(price_el_hourly, price_ng_orig,
#                                                                            price_EUA_orig, CO2_emiss_grid_hourly,
#                                                                            amp_values, variability_values, hours,
#                                                                            GB_cap_min)
#
#     # collect the results for the individual years in single_scenarios_dict
#     benchmark_single_scenarios_dict[year] = benchmark_single_scenario_dict
#
# # store the single_scenarios_dict dictionary as pickle file
# # TODO:Change prefix according to the years
# prefix = 'benchmark_single_scenarios_dict' + '_gr-cap ' + str(gr_cap) + '_' + '_min-load-GB0.3_' + years[0] + ' to ' + \
#          years[len(years) - 1]
# # prefix = 'benchmark_single_scenarios_dict_CHP_' + years[0] + ' to ' + years[len(years) - 1]
# timestamp_format = "{:%Y%m%dT%H%M}"
# timestamp = timestamp_format.format(datetime.now())
# output_filename = f"{prefix}__{timestamp}.pickle"
# with open(output_filename, 'wb') as handle:
#     pickle.dump(benchmark_single_scenarios_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print("Finished saving all_single_scenarios_dict")

## ---------------------- Access the results and export them to csv files (post processing) ------------------------------
# with open('SA_TC_woSpatCons_with_CHP_gr-cap 1.0__dis-rate 0.1_min-load_0.5_2018 to 2023__20250201T0109'
#           '.pickle', 'rb') as handle:
#     single_scenarios_dict = pickle.load(handle)
#
# years = ['2023']

# # for multi-year dicts:
# results_df = pd.DataFrame.from_records(
#     [
#         (year, process, amp, parameter, value)
#         for year, process_dict in single_scenarios_dict.items()
#         for process, amp_dict in process_dict.items()
#         for amp, parameter_dict in amp_dict['non-optimized'].items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['Optimal result [million eur]', 'CAPEX [million eur]', 'Non-annualized CAPEX [million eur]',
#                          'Share of CAPEX in total cost [%]', 'OPEX [million eur]', 'discount rate',
#                          'scope 1 emissions [kilotonne]', 'scope 2 emissions [kiloton]', 'Cost for EUA',
#                          'available area [m^2]', 'required area', 'grid connection cap',
#                          'CHP power gen to grid', 'battery to grid', 'ElB size', 'Battery size', 'TES size',
#                          'electrolyser size', 'Hydrogen boiler size', 'Hydrogen storage size',
#                          'max. power flow from grid [MW]', 'Simultaneous charging and discharging hours Battery',
#                          'Simultaneous charging and discharging hours TES']
#     ],
#     columns=(['year', 'process', 'amp', 'parameter', 'value'])
# )
# print(results_df)

## for multi-year multi-scenario dicts:
# results_df = pd.DataFrame.from_records(
#     [
#         (year, TC_scenario, process, amp, parameter, value)
#         for year, scenario_dict in single_scenarios_dict.items()
#         for TC_scenario, process_dict in scenario_dict.items()
#         for process, amp_dict in process_dict.items()
#         for amp, parameter_dict in amp_dict['non-optimized'].items()
#         for parameter, value in parameter_dict['results'].items()
#         if parameter in ['Optimal result [million eur]', 'CAPEX [million eur]', 'Non-annualized CAPEX [million eur]',
#                          'Share of CAPEX in total cost [%]', 'OPEX [million eur]', 'discount rate',
#                          'scope 1 emissions [kilotonne]', 'scope 2 emissions [kiloton]', 'Cost for EUA',
#                          'available area [m^2]', 'required area', 'grid connection cap',
#                          'CHP power gen to grid', 'battery to grid', 'ElB size', 'Battery size', 'TES size',
#                          'electrolyser size', 'Hydrogen boiler size', 'Hydrogen storage size',
#                          'max. power flow from grid [MW]', 'Simultaneous charging and discharging hours Battery',
#                          'Simultaneous charging and discharging hours TES']
#     ],
#     columns=(['year', 'TC_scenario', 'process', 'amp', 'parameter', 'value'])
# )
#
# filename = 'results_' \
#            'SA_TC_woSpatCons_with_CHP_gr-cap 1.0__dis-rate 0.1_min-load_0.5_2018 to 2023__20250201T0109' \
#            '.csv'
# results_csv_data = results_df.to_csv(filename, index=False)
# print('Saved csv file containing the results.')

# _______________________________________________________________________________________________________________________
print("End of the script")
