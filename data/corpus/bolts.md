# Bolted and Threaded Joints
Source: Machine Design Coursework Notes — Bolted Joints Module

## Section B1 (p. 1): Proof Strength and Preload
The proof strength Sp of a bolt is the stress at which the bolt begins to take a permanent set,
and it is slightly lower than the yield strength Sy. For SAE Grade 5 bolts, Sp is approximately
85,000 psi (586 MPa); for Grade 8, Sp is approximately 120,000 psi (827 MPa). Recommended
preload Fi for reused, non-permanent connections is Fi = 0.75 * Fp, where Fp = At * Sp is the
proof load and At is the tensile stress area of the threaded section. For permanent connections,
Fi = 0.90 * Fp is commonly recommended.

## Section B2 (p. 2): Tensile Stress Area
The tensile stress area At for a metric thread is computed as:
At = (pi/4) * (dp - 0.9382*p)^2
where dp is the basic major diameter and p is the thread pitch. For unified (UN/UNC) threads,
At = (pi/4) * (d - 0.9743/n)^2, where n is threads per inch.

## Section B3 (p. 3): Joint Stiffness and Load Sharing
When an external tensile load P is applied to a preloaded bolted joint, the load is shared between
the bolt and the clamped members according to their relative stiffness. The fraction of external
load carried by the bolt, C, is given by C = kb / (kb + km), where kb is bolt stiffness and km is
member (clamped material) stiffness. The resultant bolt load is Fb = Fi + C*P, and the resultant
load on the members is Fm = Fi - (1-C)*P, valid as long as Fm > 0 (joint separation has not
occurred). Because km is typically 3 to 5 times larger than kb in practice, C is usually small
(0.2-0.3), meaning most of the external load is absorbed by relaxing the clamped members rather
than by stretching the bolt — this is why preload is so effective against fatigue.

## Section B4 (p. 4): Fatigue of Bolted Joints
Because preload keeps the mean bolt stress high but the *alternating* stress amplitude low
(the bolt only sees the C*P/At portion cyclically), correctly preloaded bolts have much better
fatigue life than the peak-load figure would suggest. Endurance limits for standard bolt
materials, fully corrected, are typically quoted directly from test data (e.g., Se ≈ 18.6 ksi for
SAE Grade 5, axial loading, per Shigley's Table 8-17) rather than derived from Sut via the usual
Marin factor approach, because rolled-thread fatigue behavior is dominated by thread-root
geometry.

## Section B5 (p. 5): Torque-Preload Relationship
Wrench torque T required to achieve a target preload Fi is approximated by T = K * Fi * d, where
d is the nominal bolt diameter and K is the nut factor (typically K ≈ 0.2 for standard, non-plated,
steel-on-steel threads under average conditions). Because K itself can vary +/-30% between
"identical" lubricated bolts, torque-based preload control has an inherent large uncertainty band;
this is why torque-angle or direct-tension methods are preferred for critical joints.
