import os
import sys
import time
import math
import json

from binance.client import Client
from binance.websockets import BinanceSocketManager

from tensorflow import keras
import numpy as np

import matplotlib.pyplot as plt


### GLOBALS

TOTAL_CASH = 0

MAX_TRADES_OPEN = 5
OPEN_TRADES = 0

OPEN_ORDER_IDS = []
OUTSTANDING_ORDER_IDS = []

MAX_BUY_VALUE = 1.00
STANDARD_BUY_AMT = float(50.0)

APIKEY = os.environ.get('BINANCE_API')
SECRETKEY = os.environ.get('BINANCE_SECRET')

BINANCE_CLIENT = None
BINANCE_SOCKET_MGR = None

DEPTH_SOCKET_KEY = None
SYMB_TICK_SOCKET_KEY = None


DOGE_PRICE = None
DOGE_DEPTH = None


### UTILITY FUNCTIONS

def average(l):
    n = len(l)
    s = 0
    for e in l:
        s = s + e
    return s/n


### BINANCE FUNCTIONS

def sell_doge(amt,price):
    try:
        order_info = client.order_limit_sell(symbol='DOGEUSDT',
                                                quantity=amt,
                                                price=price)
        return order_info
    except:
        print('order failed')
        return None

def buy_doge(amt,price):
    try:
        order_info = client.order_limit_buy(symbol='DOGEUSDT',
                                                quantity=amt,
                                                price=price)
        return order_info
    except:
        print('order failed')
        return None

def check_doge_order(order_id):
    return order_info

def process_depth_msg(msg):
    global BINANCE_SOCKET_MGR
    global DEPTH_SOCKET_KEY
    global DOGE_DEPTH
    '''
    if msg['e'] == 'error':
        BINANCE_SOCKET_MGR.stop_socket(DEPTH_SOCKET_KEY)
        BINANCE_SOCKET_MGR.start_depth_socket('DOGEUSDT', process_depth_msg)
    '''
    DOGE_DEPTH = msg

def process_symb_msg(msg):
    global BINANCE_SOCKET_MGR
    global SYMB_TICK_SOCKET_KEY
    global DOGE_PRICE
    if msg['e'] == 'error':
        BINANCE_SOCKET_MGR.stop_socket(SYMB_TICK_SOCKET_KEY)
        BINANCE_SOCKET_MGR.start_symbol_ticker_socket('DOGEUSDT', process_symb_msg)
    else:
        DOGE_PRICE = msg



### NEURAL NETWORK FUNCTIONS

def load_model(filename):
    return keras.models.load_model(filename)


### ANALYSIS FUNCTIONS

def get_doge_info():

    global DOGE_PRICE
    global DOGE_DEPTH

    doge_price = float(DOGE_PRICE['b'])
    doge_high =  float(DOGE_PRICE['h'])
    doge_volume = float(DOGE_PRICE['v'])

    asks = DOGE_DEPTH['asks'][1:10]
    bids = DOGE_DEPTH['bids'][1:10]
    doge_asks = 0
    doge_bids = 0

    for i in range(0,len(asks)):
        doge_asks += float(asks[i][1])/float(asks[i][0])

    for i in range(0,len(bids)):
        doge_bids += float(bids[i][1])/float(bids[i][0])

    return [doge_price, doge_volume, doge_high, doge_bids, doge_asks]

def pressure_index(doge_info):

    price = doge_info[0]
    volume = doge_info[1]
    all_high = doge_info[2]
    bid = doge_info[3]
    ask = doge_info[4]

    market_transfers = volume/price
    demand = bid - ask
    close_to_high = 1/((all_high-price)+1)
    pressure = close_to_high*(market_transfers/demand)
    pressure = pressure/math.log(abs(pressure))
    if pressure <= 0:
        return -math.sqrt(-pressure+0.0001)/50
    return math.sqrt(pressure)/50.0

def get_pressure_indices():
    pi_list = []
    for i in range(0,45):
        pi_list.append(pressure_index(get_doge_info()))
        time.sleep(2)
    return pi_list


if __name__ == '__main__':

    try:
        BINANCE_CLIENT = Client(APIKEY,SECRETKEY)
        BINANCE_CLIENT.API_URL='https://api.binance.us/api'
        BINANCE_SOCKET_MGR = BinanceSocketManager(BINANCE_CLIENT, user_timeout=60)

        DEPTH_SOCKET_KEY = BINANCE_SOCKET_MGR.start_depth_socket('DOGEUSDT', process_depth_msg, depth='10')
        SYMB_TICK_SOCKET_KEY = BINANCE_SOCKET_MGR.start_symbol_ticker_socket('DOGEUSDT', process_symb_msg)

        BINANCE_SOCKET_MGR.start()

        time.sleep(3)

        # nn = load_model('doge_trainer_2021-02-01 15:52:42.142926')
        # pis = get_pressure_indices()
        i_s = []
        pis = []
        prices = []
        bought = False
        sold = True
        buy_price = 0
        profit = 0
        for i in range(0,10000):
            i_s.append(i)
            doge_info = get_doge_info()
            price = doge_info[0]
            prices.append(price)
            pi = pressure_index(doge_info)
            pis.append(pi)
            '''
            del pis[0]
            decision = nn.predict(np.array(pis).reshape(1,45))
            sys.stdout.flush()
            sys.stdout.write("\rBUY: {}, SELL: {}, HOLD: {}".format(decision[0,0],decision[0,1],decision[0,2]))
            if decision[0,0] >= 0.8 and sold:
                # print("\nBUY AT {}\n".format(price))
                bought = True
                sold = False
                buy_price = price
            if decision[0,1] >= 0.85 and bought:
                # if price > buy_price:
                print("\nSOLD AT {} BOUGHT AT {}\n".format(price,buy_price))
                bought = False
                sold = True
            '''
            print(f"time {i}: selling at {price} with pi: {pi}      \r", end="")
            time.sleep(2)

        BINANCE_SOCKET_MGR.close()

        with open('pressure_index_train_data.txt','w') as pi:
            json.dump(pis,pi)

        with open('prices_test_data.txt','w') as pr:
            json.dump(prices,pr)

        fig, ax = plt.subplots(2)
        ax[0].scatter(i_s, pis, s=1, label='in-the-moment pi', color='b')
        ax[0].set_xlabel('time')
        ax[0].set_ylabel('pressure index')


        ax[1].scatter(i_s, prices, s=1, label='prices', color='g')
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('prices')
        plt.legend()
        plt.show()

    except KeyboardInterrupt:
        BINANCE_SOCKET_MGR.close()
        pass


    '''
    with open('pressure_index_test_data.txt','w') as pi:
        json.dump(pis,pi)

    with open('prices_test_data.txt','w') as pr:
        json.dump(prices,pr)
    '''
    '''
    fig, ax = plt.subplots(2)
    ax[0].scatter(i_s, pis, s=1, label='in-the-moment pi', color='b')
    ax[0].set_xlabel('time')
    ax[0].set_ylabel('pressure index')


    ax[1].scatter(i_s, prices, s=1, label='prices', color='g')
    ax[1].set_xlabel('time')
    ax[1].set_ylabel('prices')
    plt.legend()
    plt.show()
    '''
