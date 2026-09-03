# Rolling-Contact (Ball and Roller) Bearings
Source: Machine Design Coursework Notes — Bearings Module

## Section BR1 (p. 1): Load-Life Relationship
Bearing life is statistical, not deterministic. The basic relationship between load and life
(L10 life, in millions of revolutions, i.e., the life at which 90% of a bearing population
survives) is:
L10 = (C / P)^a
where C is the basic dynamic load rating (catalog value), P is the equivalent dynamic load, and
a = 3 for ball bearings, a = 10/3 for roller bearings. This exponent difference reflects different
observed fatigue-life sensitivity to load between point contact (ball) and line contact (roller).

## Section BR2 (p. 2): Equivalent Dynamic Load
When a bearing sees combined radial and axial load, the equivalent dynamic load P is:
P = X*Fr + Y*Fa
where Fr and Fa are the applied radial and axial loads, and X, Y are radial/axial load factors
from the manufacturer's catalog, which depend on the ratio Fa/(Co) or Fa/Fr relative to threshold
values e (also catalog-specific).

## Section BR3 (p. 3): Reliability Adjustment
Catalog L10 values assume 90% reliability. For applications needing higher reliability (e.g., 95%,
99%), a reliability factor a1 is applied: L_na = a1 * L10, where a1 < 1 and decreases sharply as
required reliability increases (e.g., a1 ≈ 0.62 at 95%, a1 ≈ 0.21 at 99%), because bearing fatigue
life follows a Weibull distribution with a long lower tail, so pushing reliability higher costs
disproportionately more life margin.

## Section BR4 (p. 4): Selecting a Bearing from Desired Life and Speed
Design life is normally specified in hours at a known speed, so it must be converted to millions
of revolutions before applying the L10 formula: L10 (millions of rev) = 60*n*Lh / 1e6, where n is
speed in rpm and Lh is desired life in hours. Solving the L10 equation for required C gives the
minimum catalog dynamic load rating to select a bearing from a manufacturer's table.

## Section BR5 (p. 5): Preload and Stiffness in Angular Contact Bearings
Angular-contact and tapered-roller bearings are often used in preloaded pairs (duplex mounting)
to remove internal clearance, which increases stiffness and positioning accuracy but also raises
operating temperature and reduces fatigue life if preload is set too high — there is a design
trade-off between stiffness/precision and bearing life that does not exist for standard
deep-groove ball bearings run with normal internal clearance.
