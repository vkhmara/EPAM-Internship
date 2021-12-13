import numpy as np

def EMA(ser, beta):
    smoothed_ser = [ser[0]]
    for x in ser:
        smoothed_ser.append((1 - beta) * x + beta * smoothed_ser[-1])
    return smoothed_ser[1:]

def inv_EMA(smoothed_ser, beta):
    assert beta != 1, 'invalid value of beta for inversion'
    y = np.append([smoothed_ser[0]], smoothed_ser)
    return (y[1:] - beta * y[:-1]) / (1 - beta)

def MA(ser, window):
    y = np.array(ser)
    smoothed_ser = [sum(ser[:k + 1]) / (k + 1) for k in range(window - 1)]
    return np.append(smoothed_ser,
                     (sum(y[k:-(window - 1 - k)] for k in range(window - 1)) + y[window - 1:]) / window)

def inv_MA(smoothed_ser, window):
    y = np.array(smoothed_ser)
    ser = [y[0]] + [(k + 1) * y[k] - k * y[k - 1] for k in range(1, window)]
    for k in range(window, len(y)):
        ser.append(window * (y[k] - y[k - 1]) + ser[k - window])
    return np.array(ser)