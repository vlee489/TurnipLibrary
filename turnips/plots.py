#!/usr/bin/env python3

from collections import Counter
from typing import Iterable, Sequence

import matplotlib.pyplot as plt

from turnips.meta import MetaModel
from turnips.model import TRIPLE, SPIKE, DECAY, BUMP, Model, ModelEnum
from turnips.multi import MultiModel
from turnips.ttime import TimePeriod
from turnips.markov import MARKOV

def plot_models_range(name: str,
                      models: Sequence[Model],
                      previous: ModelEnum,
                      add_points: bool = False) -> None:
    '''
    Plot a fill_between for all models' low and high values using an
    alpha (transparency) equal to 1/num_models. Plot a regular line
    for all fixed prices.

    Shows ~probability of various prices based on your possible models.
    '''

    colors = {
        TRIPLE: 'orange',
        SPIKE: 'green',
        DECAY: 'red',
        BUMP: 'blue',
    }

    _fig, ax = plt.subplots()

    # when is last data point (assumes contiguous), plot a line for those
    # TODO: figure out how do do this when it isn't contiguous
    a_model = models[0]

    for day in range(2, 14):
        if a_model.timeline[TimePeriod(day)].price.is_atomic:
            continue
        else:
            # no precise data!
            if day == 2:
                final_precise = 2
            else:
                final_precise = day - 1

                # plot known data
                vals = [a_model.timeline[TimePeriod(day)].price.value
                        for day in range(2, final_precise + 1)]
                days = range(2, final_precise + 1)
                plt.plot(days, vals, c='black')
            break

    model_counts = Counter(x.model_type for x in models)
    remaining_model_types = model_counts.keys()
    remaining_probability = sum(MARKOV[previous][rem_mod]
                                for rem_mod in remaining_model_types)
    adjusted_priors = {model: MARKOV[previous][model] / remaining_probability
                       for model in model_counts.keys()}

    days = range(final_precise, 14)
    for model in models:
        low_vals = [model.timeline[TimePeriod(day)].price.lower
                    for day in range(final_precise, 14)]
        high_vals = [model.timeline[TimePeriod(day)].price.upper
                     for day in range(final_precise, 14)]

        if previous != ModelEnum.unknown:
            alpha = adjusted_priors[model.model_type] / model_counts[model.model_type]
        else:
            alpha = 1 / len(models)

        plt.fill_between(days, low_vals, high_vals, alpha=alpha, color=colors[model.model_type])

        if add_points:
            plt.scatter(days, low_vals, c='black', s=2)
            plt.scatter(days, high_vals, c='black', s=2)

    # cosmetics
    msummary = '+'.join(['{}_{{{}}}^{{{:.2f}}}'.format(t, l, adjusted_priors[l])
                         for l, t in model_counts.items()])
    ax.set_title(f'Island {name}: ${len(models)}_{{total}}={msummary}$, Last: {previous}')
    ax.set_ylabel('Turnip Price')
    ax.set_xticklabels(['Mon AM', 'Mon PM', 'Tue AM', 'Tue PM', 'Wed AM', 'Wed PM',
                        'Thu AM', 'Thu PM', 'Fri AM', 'Fri PM', 'Sat AM', 'Sat PM'])
    ax.xaxis.set_ticks(range(2, 14))
    plt.xticks(rotation=45)
    plt.grid(axis='both', which='both', ls='--')
    plt.tight_layout()

    plt.show()


def global_plot(island_models: Iterable[MultiModel]) -> None:
    '''Plot a histogram, per day, for all possible prices in the archipelago.

    This isn't exactly a probability curve since the histogram isn't scaled
    by the real probabilities.'''

    arch = MetaModel(100, island_models)

    hist = arch.histogram()

    num_ticks = len(range(2, 14))
    _fig, ax = plt.subplots(num_ticks, sharex=True)

    for i, (time, pricecounts) in enumerate(hist.items()):
        ax[i].hist(list(pricecounts.elements()), 50)
        ax[i].text(.8, .5, time,
                   horizontalalignment='center',
                   transform=ax[i].transAxes)

    ax[i].set_xlabel('Turnip Price')

    plt.show()
