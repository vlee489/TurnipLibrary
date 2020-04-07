Turnips
=======

They are high in vitamin K.


What is this?
-------------

It's a Stalk Market price analysis library. If you give it the prices
observed on your island, it will help you predict your prices for the
rest of the week.


How?
----

Stalk Market prices in AC:NH are determined by four possible patterns
you can get for the week. Each pattern has a distinctive shape; by
entering in prices already seen, we can usually rule out many
possibilities.

In some cases, some patterns look too close to each other, though;
this happens often between a "Decay" week and a "Spike" week, which
can look identical until the spike happens.


How do I use this?
------------------

This is really meant as a library and not as an end-user tool.

The library can be used by creating a MetaModel() object, like this:

`>>> prices = turnips.MetaModel.blank()`

then you can get a report:

`>>> prices.report()`

Or just the probability for the week:

`>>> prices.summary()`

If you know what price was being offered on your island *and you did
not buy turnips from Daisy Mae for the first time*, you can narrow
down the possibilities:

`>>> prices = turnips.MetaModel.blank(98)`

To start entering prices, start at Monday_AM whenever possible, and then:

- `>>> prices.fix_price('Monday_AM', 53)`
- `>>> prices.fix_price('Monday_PM', 49)`

or, you can use numerical digits to represent the timeslots; this is equivalent:

- `>>> prices.fix_price(2, 53)`
- `>>> prices.fix_price(3, 49)`


How do I use the multi-island summary report?
---------------------------------------------

If you create a JSON file called `islands.json` and fill it like this::

  { "my_island": {
      "Sunday_AM": 110,
      "Monday_AM": 53,
      "Monday_PM": 49
    },
    "tom's island": {
      "Sunday_AM": 93,
      "Monday_AM": 85,
      "Monday_PM": 81
    }
  }

you can run `turnips islands.json` from the command line to see a
multi-group probability report and forecast.
