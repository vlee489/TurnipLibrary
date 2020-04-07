#!/usr/bin/env python3

from typing import Any, Dict

from turnips.meta import MetaModel
from turnips.multi import MultiModel, BumpModels
from turnips.plots import plot_models_range, global_plot
# pylint: disable=unused-import
from turnips.model import TRIPLE, SPIKE, DECAY, BUMP, UNKNOWN

def main() -> None:
    pdata: Dict[str, Dict[str, Any]] = {
        'Trimex': {
            'Previous':    BUMP,
            'Sunday_AM':   110,
            'Monday_AM':   58,
            'Monday_PM':   None,
            'Tuesday_AM':  None,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Aduien': {
            'Previous':    BUMP,
            'Sunday_AM':   91,
            'Monday_AM':   113,
            'Monday_PM':   106,
            'Tuesday_AM':  98,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Torishima': {
            'Previous':    SPIKE,
            'Sunday_AM':   106,
            'Monday_AM':   105,
            'Monday_PM':   134,
            'Tuesday_AM':  121,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'File Isle': {
            'Previous':    BUMP,
            'Sunday_AM':   96,
            'Monday_AM':   83,
            'Monday_PM':   79,
            'Tuesday_AM':  None,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Kibo': {
            'Previous':    BUMP,
            'Sunday_AM':   93,
            'Monday_AM':   73,
            'Monday_PM':   69,
            'Tuesday_AM':  65,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Calidris': {
            'Previous':    BUMP,
            'Sunday_AM':   99,
            'Monday_AM':   98,
            'Monday_PM':   93,
            'Tuesday_AM':  None,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Harvest': {
            'Previous':    BUMP,
            'Sunday_AM':   110,
            'Monday_AM':   81,
            'Monday_PM':   77,
            'Tuesday_AM':  73,
            'Tuesday_PM':  68,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'Wolfshire': {
            'Previous':    None,
            'Sunday_AM':   107,
            'Monday_AM':   95,
            'Monday_PM':   92,
            'Tuesday_AM':  88,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
        'covid19': {
            'Previous':    None,
            'Sunday_AM':   97,
            'Monday_AM':   None,
            'Monday_PM':   None,
            'Tuesday_AM':  70,
            'Tuesday_PM':  None,
            'Wednesday_AM':None,
            'Wednesday_PM':None,
            'Thursday_AM': None,
            'Thursday_PM': None,
            'Friday_AM':   None,
            'Friday_PM':   None,
            'Saturday_AM': None,
            'Saturday_PM': None,
        },
    }

    island_models = {}
    for name, data in pdata.items():
        base = data.get('Sunday_AM', None)
        data['Sunday_AM'] = None

        model: MultiModel
        if data['Previous'] is not None:
            model = MetaModel.blank(base)
        else:
            model = BumpModels()

        for time, price in data.items():
            if price is None:
                continue
            if time == 'Previous':
                continue
            model.fix_price(time, price)

        island_models[name] = model

        # for island, model in island_models.items():
        #     print(f"{island}")
        #     print('-' * len(island))
        #     print('')
        #     model.report()
        #     print('')
        # archipelago = MetaModel(100, island_models.values())
        # archipelago.summary()
        # raise SystemExit

    for name, idata in island_models.items():
        previous = pdata[name]['Previous']
        plot_models_range(name, list(idata.models), previous)

    global_plot(island_models.values())


if __name__ == '__main__':
    main()
