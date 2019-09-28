import numpy as np
from scipy.stats import norm
import sys

def binomialTree(price0, strike, vol, interest, time, dividend, putcall, binomial_steps, binomial_model, exercise):
    """

    :param price0: Current underlying asset price
    :param strike: strike price of the option
    :param vol: annualized volatility.
    :param interest: interest rate continuously compounded
    :param time: time in years (6 months == 6/12 == .5)
    :param dividend: annualized
    :param putcall: option type, either put or call
            'c' = call
            'p' = put
    :param binomial_steps: Only when using the binomial tree, the number of increments
    :param binomial_model:
            'base' - each step up is exp^(vol * sqrt(time/binomial_steps)) step down is exp^(-vol * sqrt(time/binomial_steps))
            'cox-ross-rubenstein' - in progress
            'jarrow-rudd' - in progress, the lognormal version of base
    :param exercise: the type of exercise for the option
            'a' - american, early exercise permitted
            'e' - european, exercise at expiration only
    :return: list with results
    """

    underlying = np.zeros((binomial_steps + 1, binomial_steps +1))
    option = np.zeros((binomial_steps + 1, binomial_steps +1))
    dt = time / binomial_steps

    sgn = 0
    if putcall == 'c':
        sgn = 1
    elif putcall == 'p':
        sgn = -1
    else:
        print("putcall variable is not properly defined, must be 'c' or 'p'")
        sys.exit()

    if binomial_model == 'base':
        u = np.exp((interest - dividend) * dt + vol * np.sqrt(dt))
        d = np.exp((interest - dividend) * dt - vol * np.sqrt(dt))
    elif binomial_model == 'cox-ross-rubenstein':
        u = np.exp(vol * np.sqrt(dt))
        d = 1/u
    elif binomial_model == 'jarrow-rudd':
        u = np.exp((interest-dividend-(vol**2)/2)*dt+vol*np.sqrt(dt))
        d = np.exp((interest-dividend-(vol**2)/2)*dt-vol*np.sqrt(dt))
    else:
        print("Value unrecognized in binomial_tree variable")
        sys.exit()

    for i in range(0,underlying.shape[0]):
        underlying[0,i] = price0 * u**(i)
    for i in range(1,underlying.shape[0]):
        underlying[i,:] = np.insert(underlying[i-1,:]*d, 0,0)[:-1]
    underlying = np.where(underlying < 0, 0, underlying)

    a = np.exp((interest-dividend)*dt)
    pu = (a-d)/(u-d)
    pd0 = 1 - pu
    pv0 = np.exp( -interest * dt)


    if putcall == 'c':
        option[:,binomial_steps] = underlying[:,(binomial_steps)] - strike
        option = np.where(option < 0, 0, option)
        if exercise == 'e':
            for i in reversed(range(binomial_steps + 1)):
                option[0:i,(i-1)] = pv0 * (pu * option[0:i,i] + pd0 * option[1:(i+1),i])
        else:
            for i in reversed(range(binomial_steps + 1)):
                tval = pv0 * (pu * option[0:i,i] + pd0 * option[1:(i+1),i])
                ex = (underlying[0:i,i] - strike)
                option[0:i,(i-1)] = np.maximum(tval, ex)
    elif putcall == 'p':
        option[:,binomial_steps] = strike - underlying[:,(binomial_steps)]
        option = np.where(option < 0, 0, option)
        for i in reversed(range(binomial_steps + 1)):
            option[0:i,(i-1)] = pv0 * (pu * option[0:i,i] + pd0 * option[1:(i+1),i])
        else:
            for i in reversed(range(binomial_steps + 1)):
                tval = pv0 * (pu * option[0:i,i] + pd0 * option[1:(i+1),i])
                ex = strike - underlying[0:i,i]
                option[0:i,(i-1)] = np.maximum(tval, ex)

    return round(float(option[0, 0]), 3)

def blackScholes(price0, strike, vol, interest, time, dividend, putcall):
    d1 = (np.log(price0/ strike) + (interest - dividend + (vol**2)/2) * time)/ (vol * np.sqrt(time))
    d2 = d1 - vol * np.sqrt(time)
    pn1 = norm.cdf(d1)
    pn2 = norm.cdf(d2)
    pn3 = norm.cdf(-d2)
    pn4 = norm.cdf(-d1)
    def n1(x):
        return np.exp(-(x**2)/ 2)/ np.sqrt(2* np.pi)

    if putcall == 'c':
        optionPrice = price0 * np.exp(-dividend * time) * pn1 - strike * np.exp(-interest * time) * pn2
        optionDelta = pn1 * np.exp(-dividend * time)
        optionGamma = n1(d1) * np.exp(-dividend*time)/ (price0 * vol * np.sqrt(time))
        optionTheta = dividend* price0* pn1* np.exp(-dividend* time) - interest* strike* np.exp(-interest* time)* pn2 - price0* n1(d1)* vol* np.exp(-dividend* time)/(2* np.sqrt(time))
        optionVega = price0* n1(d1)* np.sqrt(time)* np.exp(-dividend * time)/ 100
    elif putcall == 'p':
        optionPrice = strike * np.exp(-interest * time) * pn3 - price0 * np.exp(-dividend * time) * pn4
        optionDelta = (pn1 - 1) * np.exp(-dividend * time)
        optionGamma = n1(d1) * np.exp(-dividend*time)/ (price0 * vol * np.sqrt(time))
        optionTheta = -dividend* price0* pn4* np.exp(-dividend* time) + interest* strike* np.exp(-interest* time)* pn3 - price0* n1(d1)* vol* np.exp(-dividend* time)/(2* np.sqrt(time))
        optionVega = price0* n1(d1)* np.sqrt(time)* np.exp(-dividend * time)/ 100
    else:
        print("Incorrect value in the putcall var, must be a 'p' or 'c'")

    return optionPrice, optionDelta, optionGamma, optionTheta, optionVega


#testing the output
if __name__ == '__main__':
    print(binomialTree(100, 90, .25, 0.01, 2/12, 0, 'p', 30, 'base', 'e'))
    print(blackScholes(100, 90, .25, 0.01, 2/12, 0, 'p'))
