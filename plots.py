import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# # ------------------------------------------ Sankey diagrams  ------------------------------------------------------#
# # For new systems with CHP
# # single_scenarios_dict_CHP_gr-cap 1.0__dis-rate 0.1_min-load-CHP_0.3_2018 to 2023__20240710T1332
# # single_scenarios_dict_CHP_gr-cap 1.0__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240701T1043
# with open('single_scenarios_dict_CHP_gr-cap 1.0__dis-rate 0.1_min-load-CHP_0.3_2018 to 2023__20240710T1332'
#           '.pickle', 'rb') as handle:
#     output_dict = pickle.load(handle)
#
# year = '2022'
# process = 'Olefins'
# amp = 'original'
# # calculate losses
# CHP_losses = output_dict[year][process]['non-optimized'][amp]['results']['total natural gas consumption'] \
#              - (output_dict[year][process]['non-optimized'][amp]['results']['CHP heat gen to CP']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to CP']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to battery']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP heat gen to TES']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP excess heat gen']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP excess power gen']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to grid']
#                 )
# print(CHP_losses)
# battery_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to battery'] \
#                  + output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to battery'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to CP'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser']
# print(battery_losses)
# boiler_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to electric boiler'] \
#                 + output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'] \
#                 - (output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to CP']
#                    + output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'])
# print(boiler_losses)
# electrolyser_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to electrolyser'] \
#                       + output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser'] \
#                       - (output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to storage']
#                          + output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to boiler'])
# H2storage_losses = output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to storage'] \
#                    - output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler']
# H2boiler_losses = output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to boiler'] \
#                   + output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler'] \
#                   - output_dict[year][process]['non-optimized'][amp]['results']['Hydrogen boiler to CP']
# TES_losses = output_dict[year][process]['non-optimized'][amp]['results']['CHP heat gen to TES'] \
#              + output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'] \
#              - output_dict[year][process]['non-optimized'][amp]['results']['TES to CP']
# print(TES_losses)
#
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=["Grid electricity", "Natural gas", "CHP", "Battery", "Electric boiler", "Electrolyser", "Hydrogen tank",
#                "Hydrogen boiler", "Thermal energy storage", "Power demand", "Heat demand", "Losses",
#                "Excess heat from CHP", "Excess power from CHP"],
#         color="blue"
#     ),
#     link=dict(
#         source=[0, 0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 8, 2, 3, 4, 5, 6, 7, 8, 2, 2, 2],
#         target=[9, 3, 4, 5, 2, 10, 9, 3, 8, 9, 4, 5, 10, 8, 6, 7, 7, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 0],
#         value=[output_dict[year][process]['non-optimized'][amp]['results']['grid to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to battery'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to electric boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to electrolyser'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['total natural gas consumption'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP heat gen to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to battery'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP heat gen to TES'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser'],
#                output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['H2 from electrolyser to storage'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['H2 from electrolyser to boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['Hydrogen boiler to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['TES to CP'],
#                CHP_losses,
#                battery_losses,
#                boiler_losses,
#                electrolyser_losses,
#                H2storage_losses,
#                H2boiler_losses,
#                TES_losses,
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP excess heat gen'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP excess power gen'],
#                output_dict[year][process]['non-optimized'][amp]['results']['CHP power gen to grid']
#                ]
#     ))])
#
# fig.show()

# # For new systems with GB
# # single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240624T1203
# # single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.3_2018 to 2023__20240710T1403
# # SA_capex_single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load_0.5_2018 to 2023__20240719T1434
# with open('single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.3_2018 to 2023__20240710T1403'
#           '.pickle', 'rb') as handle:
#     output_dict = pickle.load(handle)
#
# year = '2023'
# #scenario = 'Bat-Low'
# process = 'Ethylbenzene'
# amp = 'original'
# # calculate losses
# GB_losses = output_dict[year][process]['non-optimized'][amp]['results']['total natural gas consumption'] \
#              - (output_dict[year][process]['non-optimized'][amp]['results']['GB heat gen to CP']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['GB heat gen to TES']
#                 + output_dict[year][process]['non-optimized'][amp]['results']['GB excess heat gen']
#                 )
# battery_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to battery'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to CP'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'] \
#                  - output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser']
# boiler_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to electric boiler'] \
#                 + output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'] \
#                 - (output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to CP']
#                    + output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'])
# electrolyser_losses = output_dict[year][process]['non-optimized'][amp]['results']['grid to electrolyser'] \
#                       + output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser'] \
#                       - (output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to storage']
#                          + output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to boiler'])
# H2storage_losses = output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to storage'] \
#                    - output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler']
# H2boiler_losses = output_dict[year][process]['non-optimized'][amp]['results']['H2 from electrolyser to boiler'] \
#                   + output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler'] \
#                   - output_dict[year][process]['non-optimized'][amp]['results']['Hydrogen boiler to CP']
# TES_losses = output_dict[year][process]['non-optimized'][amp]['results']['GB heat gen to TES'] \
#              + output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'] \
#              - output_dict[year][process]['non-optimized'][amp]['results']['TES to CP']
#
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=["Grid electricity", "Natural gas", "Gas boiler", "Battery", "Electric boiler", "Electrolyser", "Hydrogen tank",
#                "Hydrogen boiler", "Thermal energy storage", "Power demand", "Heat demand", "Losses",
#                "Excess heat from gas boiler"],
#         color="blue"
#     ),
#     link=dict(
#         source=[0, 0, 0, 0, 1, 2,  2, 3, 3, 3, 4,  4, 5, 5, 6,  7,  8,  2,  3,  4,  5,  6,  7,  8,  2],
#         target=[9, 3, 4, 5, 2, 10, 8, 9, 4, 5, 10, 8, 6, 7, 7, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12],
#         value=[output_dict[year][process]['non-optimized'][amp]['results']['grid to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to battery'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to electric boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['grid to electrolyser'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['total natural gas consumption'],
#                output_dict[year][process]['non-optimized'][amp]['results']['GB heat gen to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['GB heat gen to TES'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to ElB'],
#                output_dict[year][process]['non-optimized'][amp]['results']['battery to electrolyser'],
#                output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['ElB gen to TES'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['H2 from electrolyser to storage'],
#                output_dict[year][process]['non-optimized'][amp]['results']
#                ['H2 from electrolyser to boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['H2 from storage to boiler'],
#                output_dict[year][process]['non-optimized'][amp]['results']['Hydrogen boiler to CP'],
#                output_dict[year][process]['non-optimized'][amp]['results']['TES to CP'],
#                GB_losses,
#                battery_losses,
#                boiler_losses,
#                electrolyser_losses,
#                H2storage_losses,
#                H2boiler_losses,
#                TES_losses,
#                output_dict[year][process]['non-optimized'][amp]['results']['GB excess heat gen']
#                ]
#     ))])
#
# fig.show()

## For ESCAPE paper (fully electrified system)
# with open('single_scenarios_dict_fully_electrified_ESCAPE_gr-cap 1.0__dis-rate 0.12022__20240528T1149.pickle',
#           'rb') as handle:
#     outputs = pickle.load(handle)
#
# output_dict = outputs
#
# amp = 'original'
# year = '2022'
# # calculate losses
# battery_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to battery'] \
#                  - output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to CP'] \
#                  - output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to ElB'] \
#                  - output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to electrolyser'] \
#                  - output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to TES']
# boiler_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to electric boiler'] \
#                 + output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to ElB'] \
#                 - output_dict[year]['Olefins']['non-optimized'][amp]['results']['ElB gen to CP']
# electrolyser_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to electrolyser'] \
#                       + output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to electrolyser'] \
#                       - (output_dict[year]['Olefins']['non-optimized'][amp]['results'][
#                              'H2 from electrolyser to storage']
#                          + output_dict[year]['Olefins']['non-optimized'][amp]['results'][
#                              'H2 from electrolyser to boiler'])
# H2storage_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['H2 from electrolyser to storage'] \
#                    - output_dict[year]['Olefins']['non-optimized'][amp]['results']['H2 from storage to boiler']
# H2boiler_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['H2 from electrolyser to boiler'] \
#                   + output_dict[year]['Olefins']['non-optimized'][amp]['results']['H2 from storage to boiler'] \
#                   - output_dict[year]['Olefins']['non-optimized'][amp]['results']['Hydrogen boiler to CP']
# TES_losses = output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to TES'] \
#              + output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to TES'] \
#              - output_dict[year]['Olefins']['non-optimized'][amp]['results']['TES to CP']
#
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=40,
#         line=dict(color="black", width=1),
#         label=["Grid electricity", "Natural gas", "CHP", "Battery", "Electric boiler", "Electrolyser", "Hydrogen tank",
#                "Hydrogen boiler", "Thermal energy storage", "Power demand", "Heat demand", "Losses"],
#         color="#636363"
#     ),
#     link=dict(
#         source=[0, 0, 0, 0, 0, 3, 3, 3, 3, 4, 5, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8],
#         target=[9, 3, 4, 5, 8, 9, 4, 5, 8, 10, 6, 7, 7, 10, 10, 11, 11, 11, 11, 11, 11],
#         value=[output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to CP'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to battery'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to electric boiler'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to electrolyser'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['grid to TES'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to CP'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to ElB'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to electrolyser'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['battery to TES'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['ElB gen to CP'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']
#                ['H2 from electrolyser to storage'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']
#                ['H2 from electrolyser to boiler'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['H2 from storage to boiler'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['Hydrogen boiler to CP'],
#                output_dict[year]['Olefins']['non-optimized'][amp]['results']['TES to CP'],
#                battery_losses,
#                boiler_losses,
#                electrolyser_losses,
#                H2storage_losses,
#                H2boiler_losses,
#                TES_losses
#                ],
#         color=['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue',
#                'lightblue', 'lightblue', '#e66101', 'green', 'green', 'green', '#e66101', '#e66101',
#                'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey']
#     ))])
# fig.update_layout(
#     font_color="black",
#     font_size=20,
#     title_font_family="Times New Roman",
# )
# fig.show()

# # For benchmark systems -------------------------------------------------#
# with open('benchmark_single_scenarios_dict_CHP_2018 to 2023__20240701T1611.pickle',
#           'rb') as handle:
#     outputs_benchmark = pickle.load(handle)
# benchmark_output_dict = outputs_benchmark
#
# amp = 'original'
# year = '2020'
#
# # calculate losses
# CHP_losses = benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                  'total natural gas consumption'] \
#              - (benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                     'CHP heat gen to CP']
#                 + benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                     'CHP power gen to CP']
#                 + benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                     'CHP excess power gen']
#                 + benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                     'CHP power gen to grid'])
#
# fig = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=5,
#         thickness=40,
#         line=dict(color="black", width=1),
#         label=["Grid electricity", "Natural gas", "CHP", "Battery", "Electric boiler", "Electrolyser", "Hydrogen tank",
#                "Hydrogen boiler", "Thermal energy storage", "Power demand", "Heat demand", "Losses", "Excess power",
#                "Sold power"],
#         color="#636363"
#     ),
#     link=dict(
#         source=[0, 1, 2, 2, 2, 2, 2],
#         target=[9, 2, 10, 9, 11, 12, 13],
#         value=[benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results']['grid to CP'],
#                benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results']
#                ['total natural gas consumption'],
#                benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                    'CHP heat gen to CP'],
#                benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                    'CHP power gen to CP'],
#                CHP_losses,
#                benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                    'CHP excess power gen'],
#                benchmark_output_dict[year]['Olefins benchmark']['non-optimized']['original']['results'][
#                    'CHP power gen to grid']
#                ],
#         color=['lightblue', 'goldenrod', '#e66101', 'purple', 'lightgrey', 'purple', 'purple']
#     ))])
# fig.update_layout(
#     font_color="black",
#     font_size=20,
#     title_font_family="Times New Roman",
# )
#
# fig.show()


# # --------------------------------------- Energy (and hydrogen) flows and Opex ----------------------------------------#
# # IMPORT INPUT DATA
# reporoot_dir = Path(__file__).resolve().parent
# # make list with all data which follows the same order of "years" list
# # for natural gas prices in eur/MWh(electricity)
# ng_prices_all_years = {
#     '2018': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2018.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
#     '2019': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2019.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
#     '2020': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2020.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
#     '2021': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2021.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
#     '2022': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2022.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False),
#     '2023': pd.read_csv(os.path.join(reporoot_dir, r'gas_price_data/Dutch TTF 2023.csv'), delimiter=";",
#                         index_col=0, usecols=[0, 1], parse_dates=True, dayfirst=False)}
# # convert Index to DatetimeIndex
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     ng_prices_all_years[year].index = pd.to_datetime(ng_prices_all_years[year].index, format='mixed', dayfirst=False)
#
# # for electricity prices in eur/MWh(electricity)
# el_prices_all_years = {
#     '2018': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2018.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True),
#     '2019': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2019.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True),
#     '2020': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2020.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True),
#     '2021': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2021.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True),
#     '2022': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2022.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True),
#     '2023': pd.read_csv(os.path.join(reporoot_dir, r'electricity_price_data/Day-ahead Prices_ENTSO-E_2023.csv'),
#                         delimiter=";",
#                         usecols=[0, 1], parse_dates=True, dayfirst=True)}
#
# # turn first column into Datetime index
# for year in ['2018', '2019', '2020', '2021', '2022', '2023']:
#     # first, cut the required characters from the string in the first column
#     for i in range(0, len(el_prices_all_years[year])):
#         el_prices_all_years[year].iloc[i, 0] = el_prices_all_years[year].iloc[i, 0][0:17]
#
#     # then, convert the remaining characters into a datetime object and use them as new index of the dataframes
#     el_prices_all_years[year].index = pd.to_datetime(el_prices_all_years[year].iloc[:, 0], dayfirst=True)
#
#     # delete the first column (which is the new index)
#     el_prices_all_years[year] = el_prices_all_years[year].iloc[:, 1:]
#
# # for ETS prices in eur/ton
# ETS_prices_all_years = {
#     '2018': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Sandbag_2018data.csv'),
#                         delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
#     '2019': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Sandbag_2019data.csv'),
#                         delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
#     '2020': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2020data.csv'),
#                         delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
#     '2021': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2021data.csv'),
#                         delimiter=";", index_col=0, parse_dates=True, dayfirst=True),
#     '2022': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/EMBER_Coal2Clean_EUETSPrices_2022.csv'),
#                         delimiter=";", index_col=0, parse_dates=True),
#     '2023': pd.read_csv(os.path.join(reporoot_dir, r'ETS_price_data/WebPlotDigitizer_Ember_2023data.csv'),
#                         delimiter=";", index_col=0, parse_dates=True, dayfirst=True)}

# # Benchmark with Gas Boiler
# with open('benchmark_single_scenarios_dict_GB_2018 to 2023__20240701T1000.pickle',
#           'rb') as handle:
#     outputs_benchmark = pickle.load(handle)
# benchmark_output_dict = outputs_benchmark
# year = '2020'
# process = 'PET'
# amp = 'original'
#
# fig, axs = plt.subplots(2, sharex=True)
# # # grid flows
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'], label='Electricity from grid to process',
#             color='lightcoral', marker='.')
# # all flows from the GB to the process
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from GB to process'], label='Heat from GB to process', color='red',
#             marker='.')
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat excess from GB'], label='Excess Heat from GB', color='coral', marker='s')
#
# axs[0].set_ylabel("MW")
# axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
#
# # # plot prices for clarification
# # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
# #             markersize=0.75)
# # axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
# #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
# #             markersize=0.75)
# # axs[1].set_ylabel("EUR/MWh")
# # axs[1].legend()
#
# plt.xlabel("Date")
# plt.show()

# # Benchmark with CHP
# with open('benchmark_single_scenarios_dict_CHP_2018 to 2023__20240701T1703.pickle',
#           'rb') as handle:
#     outputs_benchmark = pickle.load(handle)
# benchmark_output_dict = outputs_benchmark
# year = '2022'
# process = 'Olefins benchmark'
# amp = 'original'
#
# fig, axs = plt.subplots(2, sharex=True)
# # # grid flows
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'], label='Electricity from grid to process',
#             marker='*')
# # all flows from the CHP to the process
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from CHP to process'], label='Heat from GB to process',
#             marker='.')
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to process'], label='Electricity from CHP to process',
#             marker='.')
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to grid'], label='Electricity from CHP to process',
#             marker='s')
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity excess from CHP'], label='Electricity excess from CHP',
#             marker='.')
# axs[0].plot(benchmark_output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat excess from CHP'], label='Excess Heat from CHP', marker='s')
#
# axs[0].set_ylabel("MW")
# axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
#
# # # plot prices for clarification
# # axs[1].plot(price_el_hourly.iloc[:hours, count], label='Electricity price', color='b', marker='o',
# #             markersize=0.75)
# # axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
# #             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
# #             markersize=0.75)
# # axs[1].set_ylabel("EUR/MWh")
# # axs[1].legend()
#
# plt.xlabel("Date")
# plt.show()
#
# # New system with CHP
# with open('single_scenarios_dict_CHP_gr-cap 1.0__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240701T1043'
#           '.pickle', 'rb') as handle:
#     output_dict = pickle.load(handle)
#
# year = '2023'
# process = 'Olefins'
# amp = 'original'
# hours = 8000
# time_step = 1
#
# # Price data
# price_ng_orig = ng_prices_all_years[year]
# price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
#
# price_el_hourly = el_prices_all_years[year]
# price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
# price_el_hourly.index = price_ng_hourly.index
# price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
#
# price_EUA_orig = ETS_prices_all_years[year]
# price_EUA_hourly = price_EUA_orig.resample('{}h'.format(time_step)).ffill()
# EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
# price_EUA_hourly.index = price_ng_hourly.index
#
# # # calculate cost for using natural gas as price for the gas + cost for CO2 emissions
# price_EUA_hourly_MWh = price_EUA_hourly * 0.2  # eur/ton * 0.2 ton(CO2)/MWh(natural gas) = eur/MWh(natural gas)
# price_NG_use_hourly = pd.DataFrame({'Cost of using natural gas': None}, index=price_ng_hourly.index)
# price_NG_use_hourly['Cost of using natural gas'] = price_EUA_hourly_MWh['Price'] + price_ng_hourly['Open']
#
# # energy flows and prices in one figure for analysis
# fig, axs = plt.subplots(2, sharex=True)
# # # grid flows
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'],
#             label='Electricity from grid to process',
#             color='lightcoral', marker='d')
# total_grid_out = output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'electric boiler'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'battery'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'electrolyser']
# axs[0].plot(total_grid_out, label='Total flow from grid', marker='D')
#
# # all flows from the CHP to the process and from the CHP to the grid
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to process'],
#             label='Electricity from CHP to process',
#             color='brown', marker='.')
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from CHP to process'],
#             label='Heat from CHP to process', color='red',
#             marker='.')
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity excess from CHP'],
#             label='Excess electricity from CHP', color='black',
#             marker='s')
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat excess from CHP'],
#             label='Excess Heat from CHP', color='coral', marker='s')
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to grid'],
#             marker='*',
#             label='Electricity from CHP to grid', color='chocolate')
# # battery flows
# if output_dict[year][process]['non-optimized'][amp]['results']['Battery size'] > 0:
#     tot_inflow = output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to battery'] + \
#                  output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to battery']
#     axs[0].plot(tot_inflow, label='Total inflow to battery', marker='X')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to battery'],
#                 label='Electricity from grid to battery',
#                 color='gold', marker='d')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from CHP to battery'],
#                 label='Electricity from CHP to battery',
#                 color='tan', marker='.')
#     tot_outlow = output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from battery to process'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows'][
#                      'Electricity from battery to electric boiler'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows'][
#                      'Electricity from battery to electrolyser']
#     axs[0].plot(tot_outlow, label='Total outflow from battery', marker='X')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from battery to process'],
#                 label='Electricity from battery to process', color='darkkhaki', marker='s')
#     if output_dict[year][process]['non-optimized'][amp]['results']['ElB size'] > 0:
#         axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows'][
#                         'Electricity from battery to electric boiler'],
#                     label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
#     if output_dict[year][process]['non-optimized'][amp]['results']['electrolyser size'] > 0:
#         axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows'][
#                         'Electricity from battery to electrolyser'],
#                     label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
#     # axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Battery SOE'], label='Battery SOE', marker='2')
# # # electric boiler flows
# if output_dict[year][process]['non-optimized'][amp]['results']['ElB size'] > 0:
#     axs[0].plot(
#         output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to electric boiler'],
#         label='Electricity from grid to electric boiler', color='seagreen', marker='d')
#     axs[0].plot(
#         output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from electric boiler to process'],
#         label='Heat from electric boiler to process', color='turquoise', marker='.')
#     if output_dict[year][process]['non-optimized'][amp]['results']['TES size'] > 0:
#         axs[0].plot(
#             output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from electric boiler to TES'],
#             label='Heat from electric boiler to TES',
#             color='lime', marker='.')
# # TES flows
# if output_dict[year][process]['non-optimized'][amp]['results']['TES size'] > 0:
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from TES to process'],
#                 label='Heat from TES to process',
#                 color='deepskyblue', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from CHP to TES'],
#                 label='Heat from CHP to TES',
#                 marker='.')
# # # Hydrogen flows
# if output_dict[year][process]['non-optimized'][amp]['results']['electrolyser size'] > 0:
#     axs[0].plot(
#         output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to electrolyser'],
#         label='Electricity from grid to electrolyser', color='royalblue', marker='d')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from H2 boiler to process'],
#                 label='Heat from H2 boiler to process',
#                 color='blueviolet', marker='.')
#     axs[0].plot(
#         output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from electrolyser to H2 boiler'],
#         color='darkmagenta',
#         label='Hydrogen from electrolyser to H2 boiler', marker='.')
#     axs[0].plot(
#         output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from electrolyser to storage'],
#         color='fuchsia',
#         label='Hydrogen from electrolyser to storage', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from storage to H2 boiler'],
#                 color='deeppink',
#                 label='Hydrogen from storage to H2 boiler', marker='.')
# axs[0].axhline(y=output_dict[year][process]['non-optimized'][amp]['results']['grid connection cap'], color='grey',
#                linestyle='--', label='Grid connection capacity')
# axs[0].set_ylabel("MW")
# axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
#
# # plot prices for clarification
# axs[1].plot(price_el_hourly.iloc[:hours], label='Electricity price', color='b', marker='o',
#             markersize=0.75)
# axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
#             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
#             markersize=0.75)
# axs[1].set_ylabel("EUR/MWh")
# axs[1].legend()
#
# plt.xlabel("Date")
# plt.show()


# # New system with Gas boiler
# # single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.3_2018 to 2023__20240710T1403
# # single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240624T1203
# with open('single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240624T1203'
#           '.pickle', 'rb') as handle:
#     output_dict = pickle.load(handle)
#
# year = '2022'
# process = 'Ethylbenzene'
# amp = 'original'
# hours = 8000
# time_step = 1
#
# # Price data
# price_ng_orig = ng_prices_all_years[year]
# price_ng_hourly = price_ng_orig.resample('{}h'.format(time_step)).ffill()
#
# price_el_hourly = el_prices_all_years[year]
# price_el_hourly.fillna(method='ffill', inplace=True)  # replace NaN values with previous non-NaN value
# price_el_hourly.index = price_ng_hourly.index
# price_el_hourly.rename(columns={'Day-ahead Price [EUR/MWh]': 'Original data'}, inplace=True)
#
# price_EUA_orig = ETS_prices_all_years[year]
# price_EUA_hourly = price_EUA_orig.resample('{}h'.format(time_step)).ffill()
# EUA_row_NaN = price_EUA_hourly[price_EUA_hourly.isna().any(axis=1)]
# price_EUA_hourly.index = price_ng_hourly.index
#
# # # calculate cost for using natural gas as price for the gas + cost for CO2 emissions
# price_EUA_hourly_MWh = price_EUA_hourly * 0.2  # eur/ton * 0.2 ton(CO2)/MWh(natural gas) = eur/MWh(natural gas)
# price_NG_use_hourly = pd.DataFrame({'Cost of using natural gas': None}, index=price_ng_hourly.index)
# price_NG_use_hourly['Cost of using natural gas'] = price_EUA_hourly_MWh['Price'] + price_ng_hourly['Open']
#
# # energy flows and prices in one figure for analysis
# fig, axs = plt.subplots(2, sharex=True)
# # # grid flows
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'], label='Electricity from grid to process',
#             color='lightcoral',marker='d')
# total_grid_out = output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to process'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'electric boiler'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'battery'] \
#                  + output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to '
#                                                                                     'electrolyser']
# axs[0].plot(total_grid_out, label='Total flow from grid', marker='D')
#
# # all flows from the GB to the process
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from GB to process'], label='Heat from GB to process', color='red',
#             marker='.')
# axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat excess from GB'], label='Excess Heat from GB', color='coral', marker='s')
# # battery flows
# if output_dict[year][process]['non-optimized'][amp]['results']['Battery size'] > 0:
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to battery'], label='Electricity from grid to battery',
#                 color='gold', marker='d')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from battery to process'],
#                 label='Electricity from battery to process', color='darkkhaki', marker='s')
#     if output_dict[year][process]['non-optimized'][amp]['results']['ElB size'] > 0:
#         axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from battery to electric boiler'],
#                     label='Electricity from battery to electric boiler', color='olivedrab', marker='s')
#     if output_dict[year][process]['non-optimized'][amp]['results']['electrolyser size'] > 0:
#         axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from battery to electrolyser'],
#                     label='Electricity from battery to electrolyser', color='yellowgreen', marker='s')
#     #axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Battery SOE'], label='Battery SOE', marker='2')
# # # electric boiler flows
# if output_dict[year][process]['non-optimized'][amp]['results']['ElB size'] > 0:
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to electric boiler'],
#                 label='Electricity from grid to electric boiler', color='seagreen', marker='d')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from electric boiler to process'],
#                 label='Heat from electric boiler to process', color='turquoise', marker='.')
#     if output_dict[year][process]['non-optimized'][amp]['results']['TES size'] > 0:
#         axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from electric boiler to TES'],
#                     label='Heat from electric boiler to TES',
#                     color='lime', marker='.')
# # TES flows
# if output_dict[year][process]['non-optimized'][amp]['results']['TES size'] > 0:
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from TES to process'], label='Heat from TES to process',
#                 color='deepskyblue', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from GB to TES'], label='Heat from GB to TES',
#                 marker='.')
# # # Hydrogen flows
# if output_dict[year][process]['non-optimized'][amp]['results']['electrolyser size'] > 0:
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Electricity from grid to electrolyser'],
#                 label='Electricity from grid to electrolyser', color='royalblue', marker='d')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Heat from H2 boiler to process'], label='Heat from H2 boiler to process',
#                 color='blueviolet', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from electrolyser to H2 boiler'], color='darkmagenta',
#                 label='Hydrogen from electrolyser to H2 boiler', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from electrolyser to storage'], color='fuchsia',
#                 label='Hydrogen from electrolyser to storage', marker='.')
#     axs[0].plot(output_dict[year][process]['non-optimized'][amp]['energy flows']['Hydrogen from storage to H2 boiler'], color='deeppink',
#                 label='Hydrogen from storage to H2 boiler', marker='.')
# axs[0].axhline(y=output_dict[year][process]['non-optimized'][amp]['results']['grid connection cap'], color='grey',
#                linestyle='--', label='Grid connection capacity')
# axs[0].set_ylabel("MW")
# axs[0].legend(ncols=5, bbox_to_anchor=(0.5, 1.01), loc='lower center', fontsize='small')
#
# # plot prices for clarification
# axs[1].plot(price_el_hourly.iloc[:hours], label='Electricity price', color='b', marker='o',
#             markersize=0.75)
# axs[1].plot(price_NG_use_hourly.iloc[:hours, 0],
#             label='Cost of using natural gas (incl. CO2 emission allowance)', color='r', marker='o',
#             markersize=0.75)
# axs[1].set_ylabel("EUR/MWh")
# axs[1].legend()
# plt.xlabel("Date")
# plt.show()

##________________________________________ Grid connection capacity study figure _______________________________________
# # With gas boiler
# with open('single_scenarios_dict_GB_gr-cap 0.5__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240822T1630'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_50 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 0.6__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240822T1641'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_60 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 0.7__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240822T1654'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_70 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 0.8__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240717T1324'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_80 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 0.9__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240709T1635'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_90 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 1.0__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240624T1203'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_100 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 1.1__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240709T1722'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_110 = pickle.load(handle)
# with open('single_scenarios_dict_GB_gr-cap 1.2__dis-rate 0.1_min-load-GB_0.5_2018 to 2023__20240717T1342'
#           '.pickle', 'rb') as handle:
#     output_dict_GB_120 = pickle.load(handle)

# with CHP
# with open('single_scenarios_dict_CHP_gr-cap 0.5__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240822T1548'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_50 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 0.6__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240822T1558'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_60 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 0.7__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240822T1608'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_70 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 0.8__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240709T1804'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_80 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 0.9__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240709T1546'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_90 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 1.0__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240701T1043'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_100 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 1.1__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240709T1735'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_110 = pickle.load(handle)
# with open('single_scenarios_dict_CHP_gr-cap 1.2__dis-rate 0.1_min-load-CHP_0.5_2018 to 2023__20240709T1745'
#           '.pickle', 'rb') as handle:
#     output_dict_CHP_120 = pickle.load(handle)


# x = [50, 60, 70, 80, 90, 100, 110, 120]
# yearE = '2022'
# yearF = '2023'
# process = 'Ethylbenzene'
# y1_yearE = [output_dict_GB_50[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_60[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_70[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_80[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_90[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_100[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_110[yearE][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_120[yearE][process]['non-optimized']['original']['results']['ElB size']] #Electric boiler capacity
# y1_yearF = [output_dict_GB_50[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_60[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_70[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_80[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_90[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_100[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_110[yearF][process]['non-optimized']['original']['results']['ElB size'],
#              output_dict_GB_120[yearF][process]['non-optimized']['original']['results']['ElB size']] #Electric boiler capacity
# y21_yearE = [output_dict_GB_50[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_60[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_70[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_80[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_90[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_100[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_110[yearE][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_120[yearE][process]['non-optimized']['original']['results']['TES size']] #TES capacity
# y22_yearE = [output_dict_GB_50[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_60[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_70[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_80[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_90[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_100[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_110[yearE][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_120[yearE][process]['non-optimized']['original']['results']['Battery size']] #Battery capacity
# y21_yearF = [output_dict_GB_50[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_60[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_70[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_80[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_90[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_100[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_110[yearF][process]['non-optimized']['original']['results']['TES size'],
#              output_dict_GB_120[yearF][process]['non-optimized']['original']['results']['TES size']] #TES capacity
# y22_yearF = [output_dict_GB_50[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_60[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_70[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_80[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_90[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_100[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_110[yearF][process]['non-optimized']['original']['results']['Battery size'],
#              output_dict_GB_120[yearF][process]['non-optimized']['original']['results']['Battery size']] #Battery capacity
#
# fig, axs = plt.subplots(2, sharex=True)
#
# ax1 = axs[0]
# ax1.plot(x, y1_yearE, '--o', markersize=10, label='Electric boiler', color='indigo')  #color='crimson', 'orange', 'darkviolet'
# ax1.set_ylabel("Power [MW]", fontsize=20)
# ax1.tick_params(axis='both', labelsize=18)
# ax1.legend(bbox_to_anchor=(0.0, 1.01), loc='lower left', fontsize='x-large')
# ax2 = ax1.twinx()
# ax2.plot(x, y21_yearE, '--d', markersize=10, label='Thermal energy storage', color='mediumorchid')  #color='darkgrey', darkkhaki
# ax2.plot(x, y22_yearE, '--P', markersize=10, label='Battery', color='thistle')  #color='limegreen', plum
# ax2.set_ylabel("Capacity [MWh]", fontsize=20)
# ax2.tick_params(axis='both', labelsize=18)
# ax2.legend(ncols=2, bbox_to_anchor=(1.0, 1.01), loc='lower right', fontsize='x-large')
# ax1.set_title("2022", fontsize=20)
#
#
# ax3 = axs[1]
# ax3.plot(x, y1_yearF, '--o', markersize=10, label='Electric boiler', color='indigo')
# ax3.set_ylabel("Power [MW]", fontsize=20)
# ax3.tick_params(labelsize=18)
# ax4 = ax3.twinx()
# ax4.plot(x, y21_yearF, '--d', markersize=10, label='Thermal energy storage', color='mediumorchid')
# ax4.plot(x, y22_yearF, '--P', markersize=10, label='Battery', color='thistle')
# ax4.set_ylabel("Capacity [MWh]", fontsize=20)
# ax4.tick_params(axis='both', labelsize=18)
# ax3.set_title("2023", fontsize=20)
#
# fig.supxlabel('Percentage of grid connection capacity required for complete electrification', fontsize=18)
#
# plt.show()

