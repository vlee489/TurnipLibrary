# uncompyle6 version 3.6.5
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.6 (default, Jan 30 2020, 09:44:41) 
# [GCC 9.2.1 20190827 (Red Hat 9.2.1-1)]
# Embedded file name: /home/nago/src/scraps/turnip/turnip.py
# Size of source mod 2**32: 9476 bytes
from __future__ import annotations
import enum, json, logging, math, sys
from typing import Optional
import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP
logging.basicConfig(level=(logging.DEBUG))

def find_lower_bound(value: 'int', base: 'int', precision: 'str'='0.0001') -> 'float':
    """
    This awful function computes the smallest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    lower_int = value - 1
    lower_bound = Decimal(lower_int / base)
    lower_approx = float(lower_bound.quantize((Decimal('0.0001')),
      rounding=ROUND_DOWN))
    while math.ceil(lower_approx * base) < value:
        lower_approx += 0.0001

    return lower_approx


def find_upper_bound(value: 'int', base: 'int', precision: 'str'='0.0001') -> 'float':
    """
    This awful function computes the largest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    upper_bound = Decimal(value / base)
    upper_approx = float(upper_bound.quantize((Decimal('0.0001')),
      rounding=ROUND_UP))
    while math.ceil(upper_approx * base) > value:
        upper_approx -= 0.0001

    return upper_approx


class Price:

    def __init__(self, price: 'float', upper: 'Optional[float]'=None):
        if price == upper or upper is None:
            self._actual = price
            self._lower = None
            self._upper = None
        else:
            self._actual = None
            self._lower = price
            self._upper = upper

    @property
    def lower(self):
        if self._actual is not None:
            return self._actual
        return self._lower

    @property
    def upper(self):
        if self._actual is not None:
            return self._actual
        return self._upper

    @property
    def precise(self):
        if self._actual is not None:
            return self._actual
        raise RuntimeError

    def __repr__(self):
        try:
            return '{}({})'.format(self.__class__.__name__, self.precise)
        except RuntimeError:
            return '{}({}, {})'.format(self.__class__.__name__, self.lower, self.upper)

    def __str__(self):
        if self._actual is not None:
            return str(math.ceil(self.precise))
        return f"[{math.ceil(self.lower)}, {math.ceil(self.upper)}]"

    def _op(self, func):
        if self._actual is not None:
            return type(self)(func(self.precise))
        return type(self)(func(self.lower), func(self.upper))

    def __mul__(self, other):
        return self._op(lambda a: a * other)

    def __add__(self, other):
        return self._op(lambda a: a + other)

    def __sub__(self, other):
        return self._op(lambda a: a - other)

    def __lt__(self, other):
        return self.upper < other

    def __gt__(self, other):
        return self.lower > other

    def __le__(self, other):
        return self.upper <= other

    def __ge__(self, other):
        return self.lower >= other

    def __contains__(self, item):
        return self.lower <= item <= self.upper

    def ceil_contains(self, item: 'int'):
        return math.ceil(self.lower) <= item <= math.ceil(self.upper)


class PriceModifier:

    def __init__(self, base: 'Price', parent: 'Optional[PriceModifier]'=None):
        self._base = base
        self._parent = parent
        self._static_low = None
        self._static_high = None
        if parent is None:
            self._static_low = 0.85
            self._static_high = 0.9
        self._fixed_price = None

    def __str__(self) -> 'str':
        low = self.modifier_low
        high = self.modifier_high
        return f"x[{low:0.4f}, {high:0.4f}]"

    @property
    def modifier_low(self) -> 'float':
        if self._static_low is not None:
            return self._static_low
        return self._parent.modifier_low - 0.05

    @property
    def modifier_high(self) -> 'float':
        if self._static_high is not None:
            return self._static_high
        return self._parent.modifier_high - 0.03

    @property
    def price(self):
        if self._fixed_price:
            return Price(self._fixed_price)
        return Price(self.modifier_low * self._base.lower, self.modifier_high * self._base.upper)

    def fix_price(self, price: 'int') -> 'None':
        price_window = self.price
        if price < math.ceil(price_window.lower):
            msg = 'Cannot fix price at {:d}, below model minimum {:d}'.format(price, math.ceil(price_window.lower))
            raise ArithmeticError(msg)
        if price > math.ceil(price_window.upper):
            msg = 'Cannot fix price at {:d}, above model maximum {:d}'.format(price, math.ceil(price_window.upper))
            raise ArithmeticError(msg)
        current_mod_low = self.modifier_low
        current_mod_high = self.modifier_high
        logging.debug('current modifier range: [{:0.4f}, {:0.4f}]'.format(current_mod_low, current_mod_high))
        modifiers = []
        for bound in (self._base.lower, self._base.upper):
            modifiers.append(find_lower_bound(price, bound))
            modifiers.append(find_upper_bound(price, bound))

        new_mod_low = min(modifiers)
        new_mod_high = max(modifiers)
        logging.debug('fixed modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))
        if new_mod_low > current_mod_high:
            msg = 'New low modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_low, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        if new_mod_high < current_mod_low:
            msg = 'New high modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_high, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        assert new_mod_low <= new_mod_high, 'God has left us'
        new_mod_low = max(new_mod_low, current_mod_low)
        new_mod_high = min(new_mod_high, current_mod_high)
        logging.debug('clamped modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))
        self._static_low = new_mod_low
        self._static_high = new_mod_high
        self._fixed_price = price


class TimePeriod(enum.Enum):
    Sunday_AM = 0
    Sunday_PM = 1
    Monday_AM = 2
    Monday_PM = 3
    Tuesday_AM = 4
    Tuesday_PM = 5
    Wednesday_AM = 6
    Wednesday_PM = 7
    Thursday_AM = 8
    Thursday_PM = 9
    Friday_AM = 10
    Friday_PM = 11
    Saturday_AM = 12
    Saturday_PM = 13


class Model:

    def __init__(self, initial: 'Price', timeline: 'Dict[TimePeriod, PriceModifier]'):
        self.initial = initial
        self.timeline = timeline

    def __str__(self):
        lines = [
         f"Initial price: {str(self.initial)}"]
        for time, modifier in self.timeline.items():
            col = f"{time.name}:"
            price = f"{str(modifier.price)}"
            modifier = f"{str(modifier)}"
            lines.append(f"{col:13} {price:10} {modifier:10}")

        return '\n'.join(lines)


def decay_model():
    initial = Price(90.0, 110.0)
    parent = None
    modifiers = {}
    for i in range(2, 14):
        time = TimePeriod(i)
        mod = PriceModifier(initial, parent)
        modifiers[time] = mod
        parent = mod

    model = Model(initial, modifiers)
    return model


def turnip_sale_price():
    return Price(90.0, 110.0)


def decay(initial: 'Price'):
    high_rate = 0.9
    low_rate = 0.85
    prices = {}
    for i in range(2, 14):
        low_price = initial * low_rate
        high_price = initial * high_rate
        prices[TimePeriod(i)] = (
         low_price, high_price)
        low_rate -= 0.05
        high_rate -= 0.03

    return prices


def main():
    if len(sys.argv) > 1:
        initial = Price(int(sys.argv[1]))
    else:
        initial = turnip_sale_price()
    prices = decay(initial)
    for time, (low, high) in prices.items():
        col = f"{time.name}:"
        print(f"{col:13} [{low}, {high}]")

    data = {2:95, 
     3:90, 
     4:86, 
     5:83}
    res = identify(initial, data)
    if res is not None:
        print(f"Pattern is {res}?")


def identify(initial, data):
    decay_map = decay(initial)
    for i in range(2, 14):
        time = TimePeriod(i)
        price = data.get(i, None)
        model = decay_map[time]
        print(f"Time: {time.name}; price: {price}; model: {model}")
        price_range = Price(model[0].lower, model[1].upper)
        if price is None:
            continue
        if price_range.ceil_contains(price):
            print(f"{time.name} - price {price} fits in model's projection {str(price_range)}")
        else:
            print(f"{time.name} - price {price} does not fit in model's projection {str(price_range)}")
            return

    return 'decay'


if __name__ == '__main__':
    main()
