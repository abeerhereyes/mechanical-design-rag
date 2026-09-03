# Helical Compression and Extension Springs
Source: Machine Design Coursework Notes — Springs Module

## Section S1 (p. 1): Spring Rate
For a helical coil spring made of round wire, the spring rate k is:
k = (G * d^4) / (8 * D^3 * Na)
where G is the shear modulus of the wire material, d is wire diameter, D is mean coil diameter,
and Na is the number of active coils. Spring rate is independent of the applied load (linear
spring) as long as the material stays elastic and coils don't bind.

## Section S2 (p. 2): Shear Stress and the Wahl Correction Factor
Direct (uncorrected) shear stress in the wire is tau = 8*F*D / (pi*d^3). This underestimates the
true peak stress because it ignores curvature effects (stress concentration on the inside of the
coil) and direct shear superposition. The Wahl factor Kw corrects for both:
Kw = (4C - 1)/(4C - 4) + 0.615/C
where C = D/d is the spring index (typically designed in the range 4-12; C < 4 is hard to form,
C > 12 tends to tangle and buckle). Corrected stress: tau_max = Kw * 8*F*D / (pi*d^3).

## Section S3 (p. 3): Buckling and Slenderness
Compression springs loaded beyond a critical slenderness ratio (free length L0 / mean diameter D)
can buckle sideways like a column. As a rule of thumb, springs with L0/D > 4 should be checked for
buckling, and springs with L0/D > 5 are usually guided by a rod or tube, or a lower slenderness
design is preferred to avoid the buckling check entirely.

## Section S4 (p. 4): Fatigue Life and Set Removal
Springs subject to cyclic loading are evaluated similarly to other fatigue problems, but with two
practical design levers rarely available elsewhere: (1) shot peening the wire surface can raise
the fatigue strength substantially by inducing beneficial residual compressive stress, and (2) the
spring can be intentionally overstressed once during manufacture ("set removed") to induce
favorable residual stresses that oppose the working stress direction, effectively pre-hardening the
critical region before it ever sees service load.

## Section S5 (p. 5): Extension Spring End Stresses
Unlike compression springs, extension springs have highly stressed hook or loop ends, and these
end features — not the coil body — are frequently the fatigue-critical location, sometimes even
more critical than the body stress computed from the Wahl-corrected formula in Section S2.
