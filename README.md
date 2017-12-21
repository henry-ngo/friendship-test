# friendship-test
Python package to determine whether a detected companion ("friend") is bound, by testing for common proper motion of the friend. This package will compute the expected shift in a friend's separation and position angle over time if the friend was actually an (infinitely) distant background star instead of a gravitionally bound companion. A true, bound friend would remain at the same separation and position angle* while a background source would lie on this computed track. A Monte Carlo process accounts for uncertainties on the friend's astrometry and primary star's position and proper motions.

*Note: This code does not yet consider small shifts in separation and position angle due to orbital motion of the friend around the primary star.
