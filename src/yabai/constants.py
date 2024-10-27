import numpy as np

# Coefficients for Bühlmann ZH-L16
ZH_L16 = {
    'C': {
        'N2': {
            'ht': np.array([5.0, 8.0, 12.5, 18.5, 27.0, 38.3, 54.3, 77.0, 109.0, 146.0, 187.0, 239.0, 305.0, 390.0,
                            498.0, 635.0]),
            'a': np.array([1.1696, 1.0000, 0.8618, 0.7562, 0.6200, 0.5043, 0.4410, 0.4000, 0.3750, 0.3500, 0.3295,
                           0.3065, 0.2835, 0.2610, 0.2480, 0.2327]),
            'b': np.array([0.5578, 0.6514, 0.7222, 0.7825, 0.8126, 0.8434, 0.8693, 0.8910, 0.9092, 0.9222, 0.9319,
                           0.9403, 0.9477, 0.9544, 0.9602, 0.9653])},
        'He': {
            'ht': np.array([1.88, 3.02, 4.72, 6.99, 10.21, 14.48, 20.53, 29.11, 41.20, 55.19, 70.69, 90.34, 115.29,
                            147.42, 188.24, 240.03]),
            'a': np.array([1.6189, 1.3830, 1.1919, 1.0458, 0.9220, 0.8205, 0.7305, 0.6502, 0.5950, 0.5545, 0.5333,
                           0.5189, 0.5181, 0.5176, 0.5172, 0.5119]),
            'b': np.array([0.4770, 0.5747, 0.6527, 0.7223, 0.7582, 0.7957, 0.8279, 0.8553, 0.8757, 0.8903, 0.8997,
                           0.9073, 0.9122, 0.9171, 0.9217, 0.9267])}}}

# NOAA table for CNS calculation
noaa_cns_points = {0.5: 900., 0.6: 720., 0.7: 570., 0.8: 450., 0.9: 360., 1.0: 300., 1.1: 270, 1.2: 240, 1.3: 210.,
                   1.4: 180., 1.5: 180., 1.6: 150.}
noaa_cns_equations = {(0.5, 0.6): (-1800., 1800.), (0.6, 0.7): (-1500., 1620.), (0.7, 0.8): (-1200., 1410.),
                      (0.8, 0.9): (-900., 1170.), (0.9, 1.1): (-600., 900.), (1.1, 1.5): (-300., 570.),
                      (1.5, 1.6): (-750., 1245.)}

# Water vapour pressure
pw = 0.0567
