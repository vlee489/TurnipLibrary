#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging

from turnips.meta import MetaModel


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
