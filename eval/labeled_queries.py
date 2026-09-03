"""
Phase 5: Labeled evaluation set.

25 hand-written queries against the 20-chunk corpus, each with the ground-truth
relevant chunk id(s). This is what lets us compute retrieval quality metrics
(precision@k, recall@k, MRR) WITHOUT needing an LLM judge or API key -- these
are objective, deterministic metrics computed purely from retrieved-id vs
gold-id overlap. (Faithfulness/answer-relevance, by contrast, need an LLM-as-
judge over generated answers -- see eval/README_EVAL_NOTES.md for how RAGAS
plugs in once a real generation LLM is wired up.)
"""

LABELED_QUERIES = [
    {"query": "How much preload should I use for a bolt in a joint I'll disassemble and reuse?",
     "relevant": ["bolts.md::B1"]},
    {"query": "What formula gives the tensile stress area of a metric thread?",
     "relevant": ["bolts.md::B2"]},
    {"query": "How is external load split between the bolt and the clamped members?",
     "relevant": ["bolts.md::B3"]},
    {"query": "Why do correctly preloaded bolts have good fatigue life despite high mean stress?",
     "relevant": ["bolts.md::B3", "bolts.md::B4"]},
    {"query": "What torque should I apply to reach a target bolt preload, and how reliable is that method?",
     "relevant": ["bolts.md::B5"]},
    {"query": "Why do we treat a weld bead as a line when analyzing a weld group?",
     "relevant": ["welds.md::W1"]},
    {"query": "How do I compute direct shear stress in a fillet weld group?",
     "relevant": ["welds.md::W2"]},
    {"query": "How is torsional shear stress computed for an eccentrically loaded weld group?",
     "relevant": ["welds.md::W3"]},
    {"query": "What electrode strength should I pick relative to the base metal for a weld?",
     "relevant": ["welds.md::W4"]},
    {"query": "Why are AWS fatigue category curves used instead of computing Kf for welds?",
     "relevant": ["welds.md::W5"]},
    {"query": "What is the formula for the spring rate of a helical compression spring?",
     "relevant": ["springs.md::S1"]},
    {"query": "What is the Wahl factor and why is it needed to correct spring shear stress?",
     "relevant": ["springs.md::S2"]},
    {"query": "When should I check a compression spring for buckling?",
     "relevant": ["springs.md::S3"]},
    {"query": "How does shot peening improve spring fatigue life?",
     "relevant": ["springs.md::S4"]},
    {"query": "Why are hook ends often the fatigue-critical location in extension springs?",
     "relevant": ["springs.md::S5"]},
    {"query": "What is the load-life relationship for a ball bearing?",
     "relevant": ["bearings.md::BR1"]},
    {"query": "How do I combine radial and axial load into an equivalent dynamic bearing load?",
     "relevant": ["bearings.md::BR2"]},
    {"query": "How does required reliability affect the adjusted bearing life?",
     "relevant": ["bearings.md::BR3"]},
    {"query": "How do I convert a desired bearing life in hours at a given speed into L10 millions of revolutions?",
     "relevant": ["bearings.md::BR4"]},
    {"query": "What's the tradeoff with preloading angular contact bearings in a duplex pair?",
     "relevant": ["bearings.md::BR5"]},
    {"query": "Why does roller bearing life use a different load exponent than ball bearing life?",
     "relevant": ["bearings.md::BR1"]},
    {"query": "What's the recommended spring index range and why?",
     "relevant": ["springs.md::S2"]},
    {"query": "Why is torque-based bolt preload control considered unreliable for critical joints?",
     "relevant": ["bolts.md::B5"]},
    {"query": "What determines whether a bolted joint separates under external load?",
     "relevant": ["bolts.md::B3"]},
    {"query": "Why does weld fatigue strength depend so much on weld-toe geometry?",
     "relevant": ["welds.md::W5"]},
]

if __name__ == "__main__":
    print(f"{len(LABELED_QUERIES)} labeled queries")
