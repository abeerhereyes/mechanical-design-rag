# Welded Joints
Source: Machine Design Coursework Notes — Welded Joints Module

## Section W1 (p. 1): Fillet Weld Throat and Treating the Weld as a Line
For a fillet weld of leg size h, the effective throat t = 0.707*h. A key simplification used for
weld group analysis under general loading is to treat the weld bead as a line (ignoring throat
thickness in the geometry, applying it only as a stress multiplier at the end). This lets weld
groups be analyzed with the same primary + secondary shear stress method used for riveted/bolted
patterns, using a unit throat area of 1 (per unit length) and then dividing final stress by
0.707*h at the end.

## Section W2 (p. 2): Primary (Direct) Shear Stress
For a weld group carrying a direct shear force F through the centroid, the primary shear stress
is tau' = F / A, where A = 0.707*h*L is the total throat area summed over all weld segments of
total length L.

## Section W3 (p. 3): Secondary (Torsional) Shear Stress
When the load is offset from the weld group centroid by distance e, a torsional moment
M = F*e is induced. The secondary shear stress at any point is tau'' = M*r / J, where r is the
distance from the centroid to that point and J is the polar second moment of area of the weld
group treated as a line (unit throat). For common weld patterns (single line, rectangular
pattern, circular pattern), J can be looked up from standard tables rather than integrated
directly. The two shear components tau' and tau'' are combined as vectors (not scalars) because
they act in different directions at each point on the weld, and the critical point is generally
the one farthest from the centroid where the two vectors are most nearly aligned.

## Section W4 (p. 4): Allowable Stress and Electrode Matching
Weld metal strength is governed by electrode designation (e.g., E60xx has Sy ≈ 50 ksi, Sut ≈
62 ksi; E70xx has Sy ≈ 60 ksi, Sut ≈ 70 ksi). Design practice is to match or slightly overmatch
electrode strength to base metal strength, and allowable shear stress on the throat is commonly
taken as tau_allow = 0.30 * Sut of the electrode classification for static loads (AISC-style
convention used in machine design coursework problems).

## Section W5 (p. 5): Fatigue Considerations
Weld toes and roots are severe stress concentrators; even a weld with adequate static strength can
fail well below its static rating under cyclic loading due to Kf (fatigue stress concentration
factor) values often in the 1.5-2.7 range depending on weld geometry and quality. AWS fatigue
design curves (categorized by weld detail class, e.g., Category C, E) are used in preference to
computing Kf from first principles for welds, because weld-toe geometry is too irregular to model
reliably from a simple stress-concentration chart.
