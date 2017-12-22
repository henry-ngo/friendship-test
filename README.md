# friendship-test
Python package to determine whether a detected companion ("friend") is bound, by testing for common proper motion of the friend. This package will compute the expected shift in a friend's separation and position angle over time if the friend was actually an (infinitely) distant background star instead of a gravitionally bound companion. A true, bound friend would remain at the same* separation and position angle while a background source would lie on this computed track. A Monte Carlo process accounts for uncertainties on the friend's astrometry and primary star's position and proper motions.

This code was used to test for common proper motion of the detected stellar compainons reported in the following papers:
- [Ngo et al. (2015): "Friends of Hot Jupiters. II. No correspondence Between hot-Jupiter spin-orbit misalignment and the incidence of directly imaged stellar companions"] (http://iopscience.iop.org/article/10.1088/0004-637X/800/2/138/meta)
- [Ngo et al. (2016): "Friends of Hot Jupiters. IV. Stellar companions beyond 50 AU might facilitate giant planet formation, but most are unlikely to cause Kozai-Lidov migration"] (http://iopscience.iop.org/article/10.3847/0004-637X/827/1/8/meta)
- [Ngo et al. (2017): "No difference in orbital parameters of RV-detected giant planets between 0.1-5 au in single vs. multi-stellar systems"] (dx.doi.org/10.3847/1538-3881/aa6cac)

*Note: This code does not yet consider small shifts in separation and position angle due to orbital motion of the friend around the primary star.

The current version of this repository contains unedited code from Friends of Hot Jupiters and related work and was written for NGS-AO images taken by Keck/NIRC2 and Palomar/PHARO. Generalization to other images coming in future releases.
