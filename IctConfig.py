import math

std_linewght = {"edge_lat": 30,
                "server_lat": 30,
                "subnetwork_lat": 25,
                "core_lat": 20,
                "edge_br": 25e3,  # [kByte/s]
                "core_br": 200e3,  # [kByte/s]
                "sub_br": 100e3,
                "server_br": 25e3}  # [kByte/s]
screenin_lineweight = {"edge_lat": 30,
                       "server_lat": 30,
                       "subnetwork_lat": 25,
                       "core_lat": 20,
                       "edge_br": 250e3,  # [kByte/s]
                       "core_br": 2000e3,  # [kByte/s]
                       "sub_br": 1000e3,
                       "server_br": 10000e3}  # [kByte/s]

__core_data_rate = 100e3  # [kByte/s]
__sub_data_rate = 100e3  # [kByte/s]
__edge_data_rate = 25e3  # [kByte/s]
__server_data_rate = 150e3
# # Those values are calculated such that the given link delays (20, 25, 30 ms) mirror the transmission delay
# # formula to calculate link rate: max_packet_size/delay = 255*8 bit / [20, 25, 30]ms
# core_data_rate = 12.75e3  # 102 kbit/s, 12.75 kByte/s
# sub_data_rate = 10.2e3  # 81.6 kbit/s
# edge_data_rate = 8.5e3  # 68 kbit/s

max_p_size = 255  # [Byte]
# Those values are calculated such that the given link delays (20, 25, 30 ms) mirror the transmission delay
# formula to calculate link rate: max_packet_size/delay = 255*8 bit / [20, 25, 30]ms
__core_lat = math.ceil((max_p_size / __core_data_rate) * 1000)  # [ms]
__subnetwork_lat = math.ceil((max_p_size / __sub_data_rate) * 1000)  # [ms]
__edge_lat = math.ceil((max_p_size / __edge_data_rate) * 1000)  # [ms]
__server_lat = math.ceil((max_p_size / __server_data_rate) * 1000)  # [ms]
nc_lineweight = {"edge_lat": __edge_lat,
                 "server_lat": __server_lat,
                 "subnetwork_lat": __subnetwork_lat,
                 "core_lat": __core_lat,
                 "edge_br": __edge_data_rate,
                 "core_br": __core_data_rate,
                 "sub_br": __sub_data_rate,
                 "server_br": 150e3}

CPU_RES = {0: 0,
           1: 0.500,  # num cores
           2: 1,  # num cores
           3: 4}  # num cores
MEM_RES = {0: 0,
           1: 0.2,  # GB
           2: 2,  # GB
           3: 8}  # GB
STO_RES = {0: 0,
           1: 0.500,  # GB
           2: 4,  # GB
           3: 16}  # GB

S1_RES = {"cpu": 6,  # num cores
          "mem": 16,  # GB
          "sto": 50  # GB
          }

S3_RES = {"cpu": 8,  # num cores
          "mem": 24,  # GB
          "sto": 50  # GB
          }

S2_RES = {"cpu": 4,  # num cores
          "mem": 8,  # GB
          "sto": 20}  # GB

S12_RES = {"cpu": 20,  # num cores
           "mem": 40,  # GB
           "sto": 100}  # GB
S_PI = {
    "cpu": 4,
    "mem": 1,
    "sto": 16
}

medium = {"cpu": 2,  # num cores
          "mem": 32,  # GB
          "sto": 1000}  # GB

large = {"cpu": 4,  # num cores
         "mem": 64,  # GB
         "sto": 1000}  # GB

SERVER_RES = {"S1": S1_RES,
              "S2": S2_RES,
              "S3": S3_RES,
              "Trafo 0-1": medium,
              "Trafo 0-12": large}
