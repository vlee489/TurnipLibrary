#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, ROUND_DOWN, ROUND_UP
import enum
import json
import logging
import math
from typing import Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple, Type, Union
from typing import Counter as CounterType


def find_lower_bound(value: int, base: float, precision: str = '0.0001') -> float:
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


def find_upper_bound(value: int, base: float, precision: str = '0.0001') -> float:
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


class RangeSet:
    def __init__(self) -> None:
        self._set: Set[int] = set()

    def add(self, value: int) -> None:
        self._set.add(value)

    def __str__(self) -> str:
        sortlist = sorted(list(self._set))
        ranges = []

        seed = sortlist.pop(0)
        current = [seed, seed]

        for item in sortlist:
            if item == current[1] + 1:
                current[1] = item
            else:
                ranges.append((current[0], current[1]))
                current = [item, item]
        ranges.append((current[0], current[1]))

        chunks = []
        for prange in ranges:
            if prange[0] == prange[1]:
                chunk = f"{prange[0]}"
            else:
                chunk = f"[{prange[0]}, {prange[1]}]"
            chunks.append(chunk)

        if len(chunks) == 1:
            return str(chunks[0])
        return "{" + ", ".join(chunks) + "}"


class Price:
    def __init__(self,
                 price: float,
                 upper: Optional[float] = None,
                 ufilter: Optional[Callable[[float], float]] = None):
        self._lower = price
        self._upper = price if upper is None else upper
        self._user_filter = ufilter

    def _filter(self, value: float) -> float:
        if self._user_filter is not None:
            return self._user_filter(math.ceil(value))
        return math.ceil(value)

    @property
    def is_atomic(self) -> bool:
        return self._lower == self._upper

    @property
    def lower(self) -> float:
        return self._filter(self._lower)

    @property
    def upper(self) -> float:
        return self._filter(self._upper)

    @property
    def value(self) -> float:
        if not self.is_atomic:
            raise ValueError("No single value available; this is a range")
        return self.lower

    @property
    def raw(self) -> Tuple[float, float]:
        return (self._lower, self._upper)

    def __repr__(self) -> str:
        if self.is_atomic:
            return '{}({})'.format(self.__class__.__name__, self._lower)
        return '{}({}, {})'.format(self.__class__.__name__, self._lower, self._upper)

    def __str__(self) -> str:
        if self.is_atomic:
            return str(self.value)
        return f"[{self.lower}, {self.upper}]"


class Modifier:
    def __init__(self,
                 base: Price,
                 parent: Optional[Modifier] = None):
        self._base = base
        self._parent = parent
        self._exact_price: Optional[float] = None
        self._static_low: Optional[float] = None
        self._static_high: Optional[float] = None
        self.sub1 = False

    def __str__(self) -> str:
        low = self.lower_bound
        high = self.upper_bound
        return f"x[{low:0.4f}, {high:0.4f}]"

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def parent(self) -> Modifier:
        if self._parent is None:
            raise ValueError("Modifier node has no parent")
        return self._parent

    def _default_lower_bound(self) -> float:
        raise NotImplementedError

    @property
    def lower_bound(self) -> float:
        if self._static_low is not None:
            return self._static_low
        return self._default_lower_bound()

    def _default_upper_bound(self) -> float:
        raise NotImplementedError

    @property
    def upper_bound(self) -> float:
        if self._static_high is not None:
            return self._static_high
        return self._default_upper_bound()

    @property
    def price(self) -> Price:
        if self._exact_price is not None:
            return Price(self._exact_price)

        lower = self.lower_bound * self._base.lower
        upper = self.upper_bound * self._base.upper

        if self.sub1:
            return Price(lower, upper, ufilter=lambda f: f - 1)

        return Price(lower, upper)

    def tighten_bounds(self,
                       lower: Optional[float] = None,
                       upper: Optional[float] = None) -> None:
        if lower is not None and upper is not None:
            if lower > upper:
                raise ValueError("Invalid Argument (Range is reversed): [{lower}, {upper}]")

        if lower is not None:
            if self.lower_bound <= lower <= self.upper_bound:
                self._static_low = lower

        if upper is not None:
            if self.lower_bound <= upper <= self.upper_bound:
                self._static_high = upper

    def fix_price(self, price: int) -> None:
        # Protection against typos and fat fingers:
        if self._exact_price is not None:
            if self._exact_price != price:
                raise ValueError("Invalid Argument: Cannot re-fix price range with new price")

        price_window = self.price

        # Check that the new price isn't immediately outside of what we presently consider possible
        if price < price_window.lower:
            msg = f"Cannot fix price at {price:d}, below model minimum {price_window.lower:d}"
            raise ArithmeticError(msg)
        if price > price_window.upper:
            msg = f"Cannot fix price at {price:d}, above model maximum {price_window.upper:d}"
            raise ArithmeticError(msg)

        # Calculate our presently understood modifier bounds
        current_mod_low = self.lower_bound
        current_mod_high = self.upper_bound
        logging.debug('current modifier range: [{:0.4f}, {:0.4f}]'.format(current_mod_low, current_mod_high))

        # Sometimes a price given is (modifier * price) - 1;
        # to compute the correct coefficients, we need to add one!
        if self.sub1:
            unfiltered_price = price + 1
        else:
            unfiltered_price = price

        # For the price given, determine the modifiers that could have produced that value.
        # NB: Accommodate an unknown base price by computing against the lower and upper bounds of the base price.
        #
        # NB2: The boundaries here can change over time, depending on a few things:
        # (1) If the base price range is refined in the future, this calculation could change.
        #     (At present, MultiModel always builds models with fixed base prices, not ranges.)
        #
        # (2) If we are a dynamic modifier node and any of our parents refine THEIR bounds, this could change.
        #     This is generally only a problem when there are gaps in the data;
        #     Subsequent constraints will be necessarily less accurate.
        #
        # FIXME: Once this computation is performed, it's completely static. Oops!
        #        This code needs to be a little more dynamically adaptable.

        modifiers = []
        for bound in (self._base.lower, self._base.upper):
            modifiers.append(find_lower_bound(unfiltered_price, bound))
            modifiers.append(find_upper_bound(unfiltered_price, bound))
        new_mod_low = min(modifiers)
        new_mod_high = max(modifiers)
        logging.debug('fixed modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))

        # If our modifiers are out of scope, we can reject the price for this model.
        if new_mod_low > current_mod_high:
            msg = 'New low modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_low, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        if new_mod_high < current_mod_low:
            msg = 'New high modifier ({:0.4f}) out of range [{:0.4f}, {:0.4f}]'.format(new_mod_high, current_mod_low, current_mod_high)
            raise ArithmeticError(msg)
        assert new_mod_low <= new_mod_high, 'God has left us'

        # Our calculated modifier range might overlap with the base model range.
        # Base: [    LxxxxxxH      ]
        # Calc: [ lxxxxxh          ]
        # Calc: [        lxxxxh    ]
        #
        # i.e. our low end may be lower then the existing low end,
        # or the high end might be higher then the existing high end.
        #
        # The assertions in the code block above only assert that:
        # (1) l <= H
        # (2) h >= L
        #
        # Which means that these formulations are valid:
        # Base: [    LxxxxxxH      ]
        # Calc: [           lxxxxh ]
        # Calc: [ lxxh             ]
        #
        # This is fine, it just means that we can rule out some of the modifiers
        # in the range of possibilities.
        # Clamp the new computed ranges to respect the original boundaries.
        new_mod_low = max(new_mod_low, current_mod_low)
        new_mod_high = min(new_mod_high, current_mod_high)
        logging.debug('clamped modifier range: [{:0.4f}, {:0.4f}]'.format(new_mod_low, new_mod_high))

        self.tighten_bounds(new_mod_low, new_mod_high)
        self._exact_price = price


class RootModifier(Modifier):
    def _default_lower_bound(self) -> float:
        return 0.00

    def _default_upper_bound(self) -> float:
        return 6.00


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
    """
    DynamicModifier inherits the modifier coefficients of its parent.
    It can modify these coefficients with two deltas.
    """
    delta_lower = 0.00
    delta_upper = 0.00

    def _default_lower_bound(self) -> float:
        return self.parent.lower_bound + self.delta_lower

    def _default_upper_bound(self) -> float:
        return self.parent.upper_bound + self.delta_upper


class Passthrough(DynamicModifier):
    @property
    def name(self) -> str:
        return f"{self.parent.name}*"

    def fix_price(self, price: int) -> None:
        self.parent.fix_price(price)
        super().fix_price(price)


class CappedPassthrough(DynamicModifier):
    # Inherit the *static* lower bound of our parent
    def _default_lower_bound(self) -> float:
        assert isinstance(self.parent, StaticModifier)
        return self.parent.default_lower

    # Upper bound is our parent's dynamic upper bound, as normal for Dynamic nodes

    def fix_price(self, price: int) -> None:
        super().fix_price(price)
        self.parent.tighten_bounds(lower=self.lower_bound)

    @property
    def name(self) -> str:
        return f"{self.parent.name} (Capped)"


class SlowDecay(DynamicModifier):
    delta_lower = -0.05
    delta_upper = -0.03


class RapidDecay(DynamicModifier):
    delta_lower = -0.10
    delta_upper = -0.04


AnyTime = Union[str, int, 'TimePeriod']

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

    @classmethod
    def normalize(cls, value: AnyTime) -> TimePeriod:
        if isinstance(value, TimePeriod):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value]


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
            modtype = f"{modifier.name}"
            lines.append(f"{col:13} {price:10} {modrange:20} ({modtype})")
        return '\n'.join(lines)

    def histogram(self) -> Dict[str, CounterType[int]]:
        histogram: Dict[str, CounterType[int]] = {}
        for time, modifier in self.timeline.items():
            price_low = math.ceil(modifier.price.lower)
            price_high = math.ceil(modifier.price.upper)
            counter = histogram.setdefault(time.name, Counter())
            for price in range(price_low, price_high + 1):
                counter[price] += 1
        return histogram

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

    def fix_price(self, time: AnyTime, price: int) -> None:
        timeslice = TimePeriod.normalize(time)
        self.timeline[timeslice].fix_price(price)


class TripleModel(Model):
    def __init__(self,
                 initial: Price,
                 length_phase1: int,
                 length_decay1: int,
                 length_phase2: int):

        super().__init__(initial)

        if not 0 <= length_phase1 <= 6:
            raise ValueError("Phase1 length must be between [0, 6]")
        self._length_phase1 = length_phase1

        if not 2 <= length_decay1 <= 3:
            raise ValueError("Decay1 length must be 2 or 3")
        self._length_decay1 = length_decay1
        self._length_decay2 = 5 - length_decay1

        remainder = 7 - length_phase1
        if not 1 <= length_phase2 <= remainder:
            raise ValueError(f"Phase2 must be between [1, {remainder}]")
        self._length_phase2 = length_phase2
        self._length_phase3 = remainder - length_phase2

        assert (self._length_phase1
                + self._length_phase2
                + self._length_phase3
                + self._length_decay1
                + self._length_decay2) == 12

        chain: List[Modifier] = []
        decay_class: Type[Modifier]

        def _push_node(mod_cls: Type[Modifier]) -> None:
            mod = mod_cls(self.initial, chain[-1] if chain else None)
            chain.append(mod)

        # Phase 1 [0, 6]
        for _ in range(0, self._length_phase1):
            _push_node(SmallProfit)

        # Decay 1 [2, 3]
        decay_class = MediumLoss
        for _ in range(0, self._length_decay1):
            _push_node(decay_class)
            decay_class = RapidDecay

        # Phase 2 [1, 6]
        for _ in range(0, self._length_phase2):
            _push_node(SmallProfit)

        # Decay 2 [2, 3]
        decay_class = MediumLoss
        for _ in range(0, self._length_decay2):
            _push_node(decay_class)
            decay_class = RapidDecay

        # Phase 3 [0, 6]
        for _ in range(0, self._length_phase3):
            _push_node(SmallProfit)

        # Build timeline
        assert len(chain) == 12
        for i, mod in enumerate(chain):
            time = TimePeriod(i + 2)
            self.timeline[time] = mod

    @property
    def phases(self) -> List[int]:
        return [
            self._length_phase1,
            self._length_decay1,
            self._length_phase2,
            self._length_decay2,
            self._length_phase3,
        ]

    @property
    def name(self) -> str:
        return f"Triple@{str(self.initial)} {str(self.phases)}"

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[TripleModel, None, None]:
        for phase1 in range(0, 6 + 1):  # [0, 6] inclusive
            for decay1 in (2, 3):
                for phase2 in range(1, 7 - phase1 + 1):  # [1, 7 - phase1] inclusive
                    yield cls(Price(initial), phase1, decay1, phase2)


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
        return f"Decay@{str(self.initial)}"


class PeakModel(Model):
    _pattern_latest = 9
    _pattern_earliest: int
    _peak_time: int
    _name: str

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        low = self._pattern_earliest
        high = self._pattern_latest
        if not low <= pattern_start <= high:
            raise ValueError(f"pattern_start must be between {low} and {high}, inclusive")

        super().__init__(initial)
        self._pattern_start = pattern_start
        self._pattern_peak = TimePeriod(pattern_start + self._peak_time)
        self._tail = 9 - pattern_start

    @property
    def peak(self) -> TimePeriod:
        return self._pattern_peak

    @classmethod
    def inner_permutations(cls, initial: int) -> Generator[PeakModel, None, None]:
        for patt in range(cls._pattern_earliest, cls._pattern_latest + 1):
            yield cls(Price(initial), patt)

    @property
    def name(self) -> str:
        return f"{self._name}@{str(self.initial)}; peak@{self.peak.name}"


class BumpModel(PeakModel):
    _pattern_earliest = 2  # Monday AM
    _peak_time = 3         # Fourth price of pattern
    _name = "Bump"

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        super().__init__(initial, pattern_start)

        chain: List[Modifier] = []
        decay_class: Type[Modifier]

        def _push_node(mod_cls: Type[Modifier],
                       parent_override: Optional[Modifier] = None) -> Modifier:
            if parent_override is None:
                my_parent = chain[-1] if chain else None
            else:
                my_parent = parent_override
            mod = mod_cls(self.initial, my_parent)
            chain.append(mod)
            return mod

        decay_class = WideLoss
        for _ in range(2, pattern_start):
            _push_node(decay_class)
            decay_class = SlowDecay

        # Pattern starts:
        _push_node(SmallProfit)
        _push_node(SmallProfit)

        # And then gets weird!
        # Create an unlisted parent that represents the peak shared across the next three prices.
        cap = MediumProfit(self.initial, chain[-1])
        _push_node(CappedPassthrough, cap).sub1 = True
        _push_node(Passthrough, cap)
        _push_node(CappedPassthrough, cap).sub1 = True

        # Alright, phew.
        decay_class = WideLoss
        for _ in range(0, self._tail):
            _push_node(decay_class)
            decay_class = SlowDecay

        # Build timeline
        assert len(chain) == 12
        for i, mod in enumerate(chain):
            time = TimePeriod(i + 2)
            self.timeline[time] = mod


class SpikeModel(PeakModel):
    _pattern_earliest = 3  # Monday PM
    _peak_time = 2         # Third price of pattern
    _name = "Spike"

    def __init__(self,
                 initial: 'Price',
                 pattern_start: int):

        super().__init__(initial, pattern_start)

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


class MultiModel:
    """
    A collection of several models that we can aggregate data about.
    """
    def __init__(self, initial: Optional[int] = None):
        self._initial = initial
        self._models: Dict[int, Model] = {}

    @property
    def models(self) -> Generator[Model, None, None]:
        for model in self._models.values():
            yield model

    def fix_price(self, time: AnyTime, price: int) -> None:
        if not self:
            assert RuntimeError("No viable models to fix prices on!")

        timeslice = TimePeriod.normalize(time)
        remove_queue = []
        for index, model in self._models.items():
            try:
                model.fix_price(timeslice, price)
            except ArithmeticError as exc:
                logging.info(f"Ruled out model: {model.name}")
                logging.debug(f"  Reason: {timeslice.name} price={price} not possible:")
                logging.debug(f"    {str(exc)}")
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
        return "\n\n".join([str(model) for model in self.models])

    def histogram(self) -> Dict[str, CounterType[int]]:
        histogram: Dict[str, CounterType[int]] = {}
        for model in self.models:
            mhist = model.histogram()
            for timename in mhist:
                counter = histogram.setdefault(timename, Counter())
                counter.update(mhist[timename])
        return histogram

    def summary(self) -> None:
        print('')
        print("  Summary: ")
        print("    {:13} {:23} {:23} {:6}".format('Time', 'Price', 'Likely', 'Odds'))
        hist = self.histogram()
        for time, pricecounts in hist.items():
            # Gather possible prices
            pset = RangeSet()
            for price in pricecounts.keys():
                pset.add(price)

            # Determine likeliest price(s)
            n_possibilities = sum(pricecounts.values())
            likeliest = max(pricecounts.items(), key=lambda x: x[1])
            likelies = list(filter(lambda x: x[1] >= likeliest[1], pricecounts.items()))

            sample_size = len(likelies) * likeliest[1]
            pct = 100 * (sample_size / n_possibilities)

            rset = RangeSet()
            for likely in likelies:
                rset.add(likely[0])

            time_col = f"{time}:"
            price_col = f"{str(pset)};"
            likely_col = f"{str(rset)};"
            chance_col = f"({pct:0.2f}%)"
            print(f"    {time_col:13} {price_col:23} {likely_col:23} {chance_col:6}")

    def detailed_report(self) -> None:
        raise NotImplementedError

    def report(self, show_summary: bool = True) -> None:
        if len(self) == 1:
            for model in self.models:
                print(model)
        else:
            self.detailed_report()
            if show_summary:
                self.summary()

    def chatty_fix_price(self, time: AnyTime, price: int) -> None:
        timeslice = TimePeriod.normalize(time)
        self.fix_price(time, price)
        print(f"Added {timeslice.name} @ {price};")
        self.report()


class PeakModels(MultiModel):
    """
    PeakModels gathers aggregate data about Bump and Spike models.
    """
    _interior_class: Type[Model]

    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(self._interior_class.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        buckets: Dict[TimePeriod, List[Model]] = {}
        prices: Dict[TimePeriod, RangeSet] = {}
        global_prices = RangeSet()
        for model in self._models.values():
            assert isinstance(model, (BumpModel, SpikeModel))  # Shut up mypy, I know it's bad OO
            peak = model.peak
            buckets.setdefault(peak, []).append(model)
            prices.setdefault(peak, RangeSet()).add(int(model.initial.value))
            global_prices.add(int(model.initial.value))

        print(f"{self._interior_class.__name__} Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")
        print(f"  base price: {str(global_prices)}")

        if len(buckets) != 1:
            print('')
            print(f"  {len(buckets)} possible peak times:")

        for key, _models in buckets.items():
            groupprices = prices[key]
            bullet = '' if len(buckets) == 1 else '- '
            print(f"  {bullet}peak time: {key}")
            if len(buckets) == 1:
                # Don't re-print the prices for just one group.
                continue
            print(f"      base price: {str(groupprices)}")


class BumpModels(PeakModels):
    _interior_class = BumpModel


class SpikeModels(PeakModels):
    _interior_class = SpikeModel


class DecayModels(MultiModel):
    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(DecayModel.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        global_prices = RangeSet()
        for model in self._models.values():
            global_prices.add(int(model.initial.value))

        print(f"Decay Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")
        print(f"  base price: {str(global_prices)}")


class TripleModels(MultiModel):
    def __init__(self, initial: Optional[int] = None):
        super().__init__(initial)
        for i, model in enumerate(TripleModel.permutations(initial)):
            self._models[i] = model

    def detailed_report(self) -> None:
        # Alright, long story short:
        # This method (and _analyze_models) loop over the known parameters (params)
        # and their values, and look for parameters which are now "fixed" in the remaining data.
        #
        # If any are found to have only one possible value, we print that value and
        # delete it from the dict.
        #
        # If there are parameters with multiple values, loop over each value
        # and recursively call _analyze_models on just those models.

        params = {
            'phase1': lambda m: m.phases[0],
            'decay1': lambda m: m.phases[1],
            'phase2': lambda m: m.phases[2],
            'price': lambda m: int(m.initial.value),
        }

        print(f"Triple Model Analyses:")
        print(f"  {len(self._models)} model(s) remaining")

        mlist: List[TripleModel] = []
        for model in self._models.values():
            assert isinstance(model, TripleModel)
            mlist.append(model)

        self._analyze_models(mlist, params)

    def _analyze_models(self,
                        models: List[TripleModel],
                        params: Dict[str, Callable[[TripleModel], int]],
                        indent: int = 2) -> None:
        indent_str = ' ' * indent

        buckets: Dict[str, Dict[int, List[TripleModel]]] = {}
        for model in models:
            for param, param_fn in params.items():
                pdict = buckets.setdefault(param, {})
                pvalue = param_fn(model)
                pdict.setdefault(pvalue, []).append(model)

        remove_queue = []
        for param, pvalues in buckets.items():
            if len(pvalues) == 1:
                pvalue = list(pvalues.keys())[0]
                print(f"{indent_str}{param}: {pvalue}")
                remove_queue.append(param)

        for remove in remove_queue:
            buckets.pop(remove)
            params.pop(remove)

        if len(buckets) == 1:
            # Only one parameter left, so just print it now, as a list.
            pset = RangeSet()
            for param, pvalues in buckets.items():
                for pvalue in pvalues.keys():
                    pset.add(pvalue)
            print(f"{indent_str}{param}: {str(pset)}")
            return

        for param, pvalues in buckets.items():
            params.pop(param)
            for pvalue in pvalues.keys():
                print(f"{indent_str}- {param}: {pvalue}")
                pcopy = params.copy()
                self._analyze_models(pvalues[pvalue], pcopy, indent + 2)
            return


class MetaModel(MultiModel):
    # This is either a horrid abuse of OO, or brilliant.
    # We override only methods that refer to self._models directly;
    # otherwise, self.models takes care of it.

    def __init__(self,
                 initial: Optional[int],
                 groups: Iterable[MultiModel]):
        super().__init__(initial)
        self._children = list(groups)

    @classmethod
    def blank(cls, initial: Optional[int] = None) -> MetaModel:
        groups = [
            TripleModels(initial),
            SpikeModels(initial),
            DecayModels(initial),
            BumpModels(initial),
        ]
        return cls(initial, groups)

    @property
    def models(self) -> Generator[Model, None, None]:
        for child in self._children:
            yield from child.models

    def fix_price(self, time: AnyTime, price: int) -> None:
        for child in self._children:
            child.fix_price(time, price)

    def __bool__(self) -> bool:
        return len(self) != 0

    def __len__(self) -> int:
        return sum(len(child) for child in self._children)

    def detailed_report(self) -> None:
        self._children = list(filter(None, self._children))

        if not self._children:
            print("--- No Viable Model Groups ---")
            return

        if len(self._children) == 1:
            self._children[0].report(show_summary=False)
            return

        print("Meta-Analysis: ")
        print("total possible models: {}".format(len(self)))
        print('----------')
        for child in self._children:
            if child:
                child.report()
                print('----------')


def main() -> None:
    parser = argparse.ArgumentParser(description='Gaze into the runes')
    parser.add_argument('archipelago', metavar='archipelago-json',
                        help='JSON file with turnip price data for the week')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Turn on some informational messages')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Turn on debugging messages')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        logging.basicConfig(level=logging.INFO)

    with open(args.archipelago) as infile:
        pdata = json.load(infile)

    island_models = {}
    for name, data in pdata.items():
        base = data.get('Sunday_AM', None)
        data['Sunday_AM'] = None

        logging.info(f" == {name} island == ")
        model = MetaModel.blank(base)
        logging.info(f"  (%d models)  ", len(model))

        for time, price in data.items():
            if price is None:
                continue
            logging.info(f"[{time}]: fixing price @ {price}")
            model.fix_price(time, price)

        island_models[name] = model

    for island, model in island_models.items():
        print(f"{island}")
        print('-' * len(island))
        print('')
        model.report()
        print('')

    # The initial doesn't matter here, it's ignored.
    print('-' * 80)
    archipelago = MetaModel(100, island_models.values())
    archipelago.summary()
    print('-' * 80)


if __name__ == '__main__':
    main()
