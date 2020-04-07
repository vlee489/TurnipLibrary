import logging
from typing import Any, Dict, Optional

import pydantic
from pydantic import validator
import yaml

from turnips.model import ModelEnum
from turnips.meta import MetaModel
from turnips.multi import MultiModel, BumpModels
from turnips.ttime import TimePeriod


# pylint: disable=too-few-public-methods, no-self-use, no-self-argument

class Base(pydantic.BaseModel):
    class Config:
        extra = 'forbid'


class IslandModel(Base):
    previous_week: ModelEnum = ModelEnum.unknown
    initial_week: Optional[bool] = False
    timeline: Dict[TimePeriod, Optional[int]]

    @validator('previous_week', pre=True)
    def coerce_model(cls, value: Any) -> Any:
        if isinstance(value, str):
            return ModelEnum[value.lower()]
        return value

    @validator('timeline', pre=True)
    def normalize(cls, value: Any) -> Any:
        if isinstance(value, Dict):
            return {
                TimePeriod.normalize(key): price
                for key, price in value.items()
            }
        return value

    @validator('initial_week')
    def cajole(cls, initial_week: bool, values: Dict[str, Any]) -> Any:
        if values['previous_week'] != ModelEnum.unknown and initial_week:
            raise ValueError("Cannot set initial_week = True when previous_week is set")
        return initial_week


class Archipelago(Base):
    islands: Dict[str, IslandModel]


def load_json(filename: str) -> Archipelago:
    return Archipelago.parse_file(filename)


def load_yaml(filename: str) -> Archipelago:
    with open(filename, "r") as infile:
        ydoc = yaml.safe_load(infile)
    return Archipelago.parse_obj(ydoc)


def process_data(data: Archipelago) -> Dict[str, MultiModel]:
    island_models = {}

    for name, idata in data.islands.items():
        logging.info(f" == {name} island == ")

        base = idata.timeline.get(TimePeriod.Sunday_AM, None)
        models: MultiModel
        if idata.initial_week:
            models = BumpModels(base)
        else:
            models = MetaModel.blank(base)

        logging.info(f"  (%d models)  ", len(models))

        for time, price in idata.timeline.items():
            if price is None:
                continue
            if time.value < TimePeriod.Monday_AM.value:
                continue
            logging.info(f"[{time.name}]: fixing price @ {price}")
            models.fix_price(time, price)
        island_models[name] = models

    return island_models


def load(filename: str) -> Dict[str, MultiModel]:
    if '.yml' in filename or '.yaml' in filename:
        data = load_yaml(filename)
    else:
        data = load_json(filename)

    return process_data(data)


def summary(island_models: Dict[str, MultiModel]) -> None:
    for island, model in island_models.items():
        print(f"{island}")
        print('-' * len(island))
        print('')
        model.report()
        print('')

    # The initial doesn't matter here, it's ignored.
    print('Archipelago Summary')
    print('-' * 80)
    archipelago = MetaModel(100, island_models.values())
    archipelago.summary()
    print('-' * 80)
