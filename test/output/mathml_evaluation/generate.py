#!/usr/bin/env python3
import numpy as np

plus = lambda a: a + a
minus = lambda a: a - 10.01
uminus = lambda a: -a
times = lambda a: a * a
divide = lambda a: a / 10.01

root = np.sqrt
power_half = lambda a: a ** 0.5
power2 = lambda a: a ** 2
power3 = lambda a: a ** 3
power4 = lambda a: a ** 4

exp = np.exp
ln = np.log
log = np.log10

# Higher precision functions are just implemented as the lower precision versions...
# Note that the log1p function here is 1 + log(x), and NOT log(x + 1) as it is usually defined
#expm1 = np.expm1
#log1p = np.log1p
expm1 = lambda a: np.exp(a) - 1
log1p = lambda a: 1 + np.log(a)

floor = np.floor
ceiling = np.ceil

sec = lambda a: 1 / np.cos(a)
csc = lambda a: 1 / np.sin(a)
cot = lambda a : 1 / np.tan(a)

sech = lambda a: 2 / (np.exp(a) + np.exp(-a))
csch = lambda a: 2 / (np.exp(x) - np.exp(-a))
coth = lambda a: (np.exp(a) + np.exp(-a)) / (np.exp(a) - np.exp(-a))

arcsec = lambda a: np.arccos(1 / a)
arccsc = lambda a: np.arcsin(1 / a)
arccot = lambda a: np.arctan(1 / a)

arccsch = lambda a: np.log(1 / a + np.sqrt(1 / a**2 + 1))
arcsech = lambda a: np.log(1 / a + np.sqrt(1 / a**2 - 1))
arccoth = lambda a: 0.5 * np.log((a + 1) / (a - 1))

ops = [
    plus,       # 1
    minus,      # 2
    uminus,
    times,
    divide,     # 5

    root,       # 6
    power_half, # 7
    power2,
    power3,
    power4,     # 10

    exp,        # 11
    expm1,      # 12
    ln,         # 13
    log,        # 14
    log1p,      # 15

    abs,
    floor,
    ceiling,

    np.sin, np.cos, np.tan,
    sec, csc, cot,
    np.sinh, np.cosh, np.tanh,
    sech, csch, coth,
    np.arcsin, np.arccos, np.arctan,
    arcsec, arccsc, arccot,
    np.arcsinh, np.arccosh, np.arctanh,
    arcsech, arccsch, arccoth,
]

# Stop after abs
ops = ops[:16]

np.seterr(all='ignore')

eol = '\n'
with open('outputs_combined_pos.csv', 'w') as f:
    f.write('# All outputs for positive inputs' + eol)
    for i in range(-300, 325, 25):
        # The code below has some funny bits, to more closely emulate whatever
        # it was that created the original output file
        x = np.float64(float(f'1.23456789e{i}'))
        y = [op(x) for op in ops]
        y = [0 if a == 0 else a for a in y]             # 0.0 to 0
        y = [1 if a == 1 else a for a in y]             # 1.0 to 1
        y = [-1 if a == -1 else a for a in y]           # -1.0 to -1
        y = [np.nan if np.isnan(a) else a for a in y]   # -nan to nan
        f.write(','.join([str(a) for a in y]) + eol)

eol = '\n'
with open('outputs_combined_neg.csv', 'w') as f:
    f.write('# All outputs for negative inputs' + eol)
    for i in range(-300, 325, 25):
        # The code below has some funny bits, to more closely emulate whatever
        # it was that created the original output file
        #x = np.float64(-1.23456789 * 10**i)
        x = np.float64(float(f'-1.23456789e{i}'))
        y = [op(x) for op in ops]
        y = [0 if a == 0 else a for a in y]             # 0.0 to 0
        y = [1 if a == 1 else a for a in y]             # 1.0 to 1
        y = [-1 if a == -1 else a for a in y]           # -1.0 to -1
        y = [np.nan if np.isnan(a) else a for a in y]   # -nan to nan
        f.write(','.join([str(a) for a in y]) + eol)

