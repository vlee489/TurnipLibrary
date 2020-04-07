#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging

from turnips import archipelago

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

    island_models = archipelago.load(args.archipelago)
    archipelago.summary(island_models)


if __name__ == '__main__':
    main()
