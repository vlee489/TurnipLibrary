from typing import Dict, Optional

import pydantic
from pydantic import validator

from turnips.model import ModelEnum
from turnips.ttime import TimePeriod


class Base(pydantic.BaseModel):
    class Config:
        extra = 'forbid'


class IslandModel(Base):
    previous_week: ModelEnum = ModelEnum.unknown
    initial_week: Optional[bool] = False
    timeline: Dict[TimePeriod, Optional[int]]

    @validator('previous_week', pre=True)
    def coerce_model(cls, value):
        if isinstance(value, str):
            return ModelEnum[value.lower()]
        return value

    @validator('timeline', pre=True)
    def normalize(cls, value):
        if isinstance(value, Dict):
            return {
                TimePeriod.normalize(key): price
                for key, price in value.items()
            }
        return value

    @validator('initial_week')
    def cajole(cls, initial_week: bool, values):
        if values['previous_week'] != ModelEnum.unknown and initial_week:
            raise ValueError("Cannot set initial_week = True when previous_week is set")


class Archipelago(Base):
    islands: Dict[str, IslandModel]
