#!/usr/bin/env python3

from __future__ import annotations

from decimal import Decimal, ROUND_DOWN, ROUND_UP
import enum
import logging
import math
import sys
from typing import Callable, Dict, Generator, Optional, Type, Union


logging.basicConfig(level=(logging.DEBUG))


def find_lower_bound(value: int, base: int, precision: str = '0.0001') -> float:
    """
    This awful function computes the smallest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    lower_int = value - 1
    lower_bound = Decimal(lower_int / base)
    lower_approx = float(lower_bound.quantize((Decimal(precision)),
                                              rounding=ROUND_DOWN))
    while math.ceil(lower_approx * base) < value:
        lower_approx += 0.0001

    return lower_approx


def find_upper_bound(value: int, base: int, precision: str = '0.0001') -> float:
    """
    This awful function computes the largest float x such that:
    ceil(x * base) == value
    to within some arbitrary quantized precision, e.g. 0.0001.
    """
    upper_bound = Decimal(value / base)
    upper_approx = float(upper_bound.quantize((Decimal(precision)),
                                              rounding=ROUND_UP))
    while math.ceil(upper_approx * base) > value:
        upper_approx -= 0.0001

    return upper_approx


class Price:
    def __init__(self, price: float, upper: Optional[float] = None):
        self._lower = price
        self._upper = price if upper is None else upper

    @property
    def is_precise(self) -> bool:
        return self._lower == self._upper

    @property
    def lower(self) -> float:
        return self._lower

    @property
    def upper(self) -> float:
        return self._upper

    @property
    def precise(self) -> float:
        if not self.is_precise:
            raise ValueError("No precise value available for a range")
        return self._lower

    @precise.setter
    def precise(self, value: float) -> None:
        self._lower = value
        self._upper = value

    def __repr__(self) -> str:
        if self.is_precise:
            return '{}({})'.format(self.__class__.__name__, self.precise)
        return '{}({}, {})'.format(self.__class__.__name__, self.lower, self.upper)

    def __str__(self) -> str:
        if self.is_precise:
            return str(math.ceil(self.precise))
        return f"[{math.ceil(self.lower)}, {math.ceil(self.upper)}]"

    def _op(self, func: Callable[[float], float]) -> Price:
        if self.is_precise:
            return type(self)(func(self.precise))
        return type(self)(func(self.lower), func(self.upper))

    def __mul__(self, other: float) -> Price:
        return self._op(lambda a: a * other)

    def __add__(self, other: float) -> Price:
        return self._op(lambda a: a + other)

    def __sub__(self, other: float) -> Price:
        return self._op(lambda a: a - other)

    def __lt__(self, other: float) -> bool:
        return self.upper < other

    def __gt__(self, other: float) -> bool:
        return self.lower > other

    def __le__(self, other: float) -> bool:
        return self.upper <= other

    def __ge__(self, other: float) -> bool:
        return self.lower >= other

    def __contains__(self, item: float) -> bool:
        return self.lower <= item <= self.upper

    def ceil_contains(self, item: int) -> bool:
        return math.ceil(self.lower) <= item <= math.ceil(self.upper)



class Modifier:
    def __init__(self,
                 base: Price,
                 parent: Optional[Modifier] = None):
        self._base = base
        self._parent = parent
        self._exact_price = None

    def __str__(self) -> str:
        low = self.lower_bound
        high = self.upper_bound
        return f"x[{low:0.4f}, {high:0.4f}]"

    def _default_lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def lower_bound(self) -> float:
        return self._default_lower_bound()

    def _default_upper_bound(self) -> float:
        raise NotImplementedError

    @property
    def upper_bound(self) -> float:
        return self._default_upper_bound()

    @property
    def price(self) -> Price:
        if self._exact_price is not None:
            return Price(self._exact_price)

        return Price(
            self.lower_bound * self._base.lower,
            self.upper_bound * self._base.upper
        )


class RootModifier(Modifier):
    def _default_lower_bound(self) -> float:
        return 1.00

    def _default_upper_bound(self) -> float:
        return 1.00


class StaticModifier(Modifier):
    default_lower = 0.00
    default_upper = 1.00

    def _default_lower_bound(self) -> float:
        return self.default_lower

    def _default_upper_bound(self) -> float:
        return self.default_upper


class InitialDecay(StaticModifier):
    default_lower = 0.85
    default_upper = 0.90


class WideLoss(StaticModifier):
    default_lower = 0.40
    default_upper = 0.90


class MediumLoss(StaticModifier):
    default_lower = 0.6
    default_upper = 0.8


class SmallProfit(StaticModifier):
    default_lower = 0.90
    default_upper = 1.40


class MediumProfit(StaticModifier):
    default_lower = 1.4
    default_upper = 2.0


class LargeProfit(StaticModifier):
    default_lower = 2.0
    default_upper = 6.0


class DynamicModifier(Modifier):
    delta_lower = 0.00
    delta_upper = 0.00

    def _default_lower_bound(self) -> float:
        assert self._parent is not None, "DynamicModifier needs a Parent"
        return self._parent.lower_bound + self.delta_lower

    def _default_upper_bound(self) -> float:
        assert self._parent is not None, "DynamicModifier needs a Parent"
        return self._parent.upper_bound + self.delta_upper


class SlowDecay(DynamicModifier):
    delta_lower = -0.05
    delta_upper = -0.03


class RapidDecay(DynamicModifier):
    delta_lower = -0.10
    delta_upper = -0.04

    # FIXME: this nasty baby below

#    def fix_price(self, price: 'int') -> 'None':
#        price_window = self.price
#        if price < math.ceil(price_window.lower):
#            msg = 'Cannot fix price at {:d}, below model minimum {:d}'.format(price, math.ceil(price_window.lower))
#            raise ArithmeticError(msg)
#        if price > math.ceil(price_window.upper):
#            msg = 'Cannot fix price at {:d}, above model maximum {:d}'.format(price, math.ceil(price_window.upper))
#            raise ArithmeticError(msg)
#        current_mod_low = self.modifier_low
#        current_mod_high = self.modifier_high
#        logging.debug('current modifier range: [{:0.4f}, {:0.4f}]'.format(current_mod_low, current_mod_high))
#        modifiers = []
#        for bound in (self._base.lower, self._base.upper):
#            modifiers.append(find_lower_bound(price, bound))
#            modifiers.append(find_upper_bound(price, bound))
#
#        new_mod_low = min(modifiers)
#        new_mod_high = max(modifiers)
#        logging.debug('fixed modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))
#        if new_mod_low > current_mod_high:
#            msg = 'New low modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_low, current_mod_low, current_mod_high)
#            raise ArithmeticError(msg)
#        if new_mod_high < current_mod_low:
####            msg = 'New high modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_high, current_mod_low, current_mod_high)
#            raise ArithmeticError(msg)
#        assert new_mod_low <= new_mod_high, 'God has left us'
#        new_mod_low = max(new_mod_low, current_mod_low)
#        new_mod_high = min(new_mod_high, current_mod_high)
#        logging.debug('clamped modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))
#        self._static_low = new_mod_low
#        self._static_high = new_mod_high
#        self._fixed_price = price


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
    def __init__(self, initial: Price):
        self.initial = initial
        self.timeline: Dict[TimePeriod, Modifier] = {}

    @property
    def name(self) -> str:
        raise NotImplementedError("Generic Model has no name!")

    def __str__(self) -> str:
        lines = [
            f"# Model: {self.name}\n",
        ]
        for time, modifier in self.timeline.items():
            col = f"{time.name}:"
            price = f"{str(modifier.price)}"
            modrange = f"{str(modifier)}"
            modtype = f"{modifier.__class__.__name__}"
            lines.append(f"{col:13} {price:10} {modrange:20} ({modtype})")
        return '\n'.join(lines)

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[Model, None, None]:
        yield cls(Price(initial))

    @classmethod
    def permutations(cls, initial: Optional[int] = None) -> Generator[Model, None, None]:
        if initial is not None:
            assert 90 <= initial <= 110, "Initial should be within [90, 110]"
            lower = initial
            upper = initial
        else:
            lower = 90
            upper = 110

        for tprice in range(lower, upper + 1):
            yield from cls.inner_permutations(tprice)


class DecayModel(Model):
    def __init__(self, initial: Price):
        super().__init__(initial)
        mod: Modifier
        parent: Modifier

        # Slice 2: InitialDecay
        mod = InitialDecay(self.initial, None)
        self.timeline[TimePeriod(2)] = mod
        parent = mod

        # Slices 3-13: Decay
        for i in range(3, 14):
            time = TimePeriod(i)
            mod = SlowDecay(self.initial, parent)
            self.timeline[time] = mod
            parent = mod

    @property
    def name(self) -> str:
        return f"Decay@{self.initial.precise}"


class BumpModel(Model):
    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):
        if not 2 <= pattern_start <= 9:
            raise ValueError("pattern_start must be between 2 and 9, inclusive")
        super().__init__(initial)
        self._pattern_start = pattern_start
        self._pattern_peak = TimePeriod(pattern_start + 3)


        def _get_pattern(value: int) -> Type[Modifier]:
            # The default:
            cls: Type[Modifier] = SlowDecay

            # Pattern takes priority:
            if pattern_start <= value < pattern_start + 2:
                cls = SmallProfit
            elif value == pattern_start + 2:
                cls = MediumProfit  # FIXME
            elif value == pattern_start + 3:
                cls = MediumProfit  # FIXME
            elif value == pattern_start + 4:
                cls = MediumProfit  # FIXME
            elif value == pattern_start + 5:
                cls = WideLoss

            # Non-pattern values:
            elif value == 2:
                # Week starts with WideLoss (unless it was in a pattern already!)
                cls = WideLoss

            return cls

        parent = None
        # Slices 2-13: Magic!
        for i in range(2, 14):
            time = TimePeriod(i)
            mod = _get_pattern(i)(self.initial, parent)
            self.timeline[time] = mod
            parent = mod

    @property
    def name(self) -> str:
        return f"Bump@{self.initial.precise}; peak@{self._pattern_peak.name}"

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[BumpModel, None, None]:
        # Pattern can start on 2nd-9th slot
        # [Monday AM - Thursday PM]
        for patt in range(2, 10):
            yield cls(Price(initial), patt)


class SpikeModel(Model):
    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):
        if not 3 <= pattern_start <= 9:
            raise ValueError("pattern_start must be between 3 and 9, inclusive")
        super().__init__(initial)
        self._pattern_start = pattern_start
        self._pattern_peak = TimePeriod(pattern_start + 2)


        def _get_pattern(value: int) -> Type[Modifier]:
            # The default:
            cls: Type[Modifier] = SlowDecay

            # Pattern takes priority:
            if value == pattern_start:
                cls = SmallProfit
            elif value == pattern_start + 1:
                cls = MediumProfit
            elif value == pattern_start + 2:
                cls = LargeProfit
            elif value == pattern_start + 3:
                cls = MediumProfit
            elif value == pattern_start + 4:
                cls = SmallProfit
            elif value >= pattern_start + 5:
                # Week finishes out with independent low prices
                cls = WideLoss
            elif value == 2:
                # Normal start-of-week pattern
                cls = InitialDecay

            return cls

        parent = None
        # Slices 2-13: Magic!
        for i in range(2, 14):
            time = TimePeriod(i)
            mod = _get_pattern(i)(self.initial, parent)
            self.timeline[time] = mod
            parent = mod

    @property
    def name(self) -> str:
        return f"Spike@{self.initial.precise}; peak@{self._pattern_peak.name}"

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[BumpModel, None, None]:
        # Pattern can start on 3rd-9th slot
        # [Monday PM - Thursday PM]
        for patt in range(3, 10):
            yield cls(Price(initial), patt)


class MultiModel:
    def __init__(self, initial: Optional[int] = None):
        i = 0
        self._models = {}

        for model in DecayModel.permutations(initial):
            self._models[i] = model
            i += 1

        for model in BumpModel.permutations(initial):
            self._models[i] = model
            i += 1

        for model in SpikeModel.permutations(initial):
            self._models[i] = model
            i += 1

    def fix_price(self, time: Union[str, TimePeriod], price: int) -> None:
        if not self._models:
            assert RuntimeError("No viable models to fix prices on!")

        if isinstance(time, TimePeriod):
            timeslice = time
        else:
            timeslice = TimePeriod[time]

        remove_queue = []
        for index, model in self._models.items():
            try:
                raise ArithmeticError("WIP sorry")
                #model.timeline[timeslice].fix_price(price)
            except ArithmeticError as exc:
                print(f"Ruled out model: {model.name}")
                print(f"  Reason: {timeslice.name} price={price} not possible:")
                print(f"    {str(exc)}")
                remove_queue.append(index)

        for i in remove_queue:
            del self._models[i]

    def __bool__(self) -> bool:
        return bool(self._models)

    def __len__(self) -> int:
        return len(self._models)

    def __str__(self) -> str:
        if not self:
            return "--- No Viable Models ---"
        return "\n\n".join([str(model) for model in self._models.values()])


def main() -> None:
    if len(sys.argv) > 1:
        multi = MultiModel(int(sys.argv[1]))
    else:
        multi = MultiModel()
    print(multi)


if __name__ == '__main__':
    main()
