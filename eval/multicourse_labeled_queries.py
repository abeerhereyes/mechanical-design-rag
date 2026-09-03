"""Small source/page gold set for multi-course retrieval regression tests."""

LABELED_QUERIES = [
    {
        "course_id": "aerodynamics",
        "query": "Who teaches the Aerodynamics course?",
        "source": "Aerodynamic_M-1.pdf",
        "pages": {1},
    },
    {
        "course_id": "aerodynamics",
        "query": "What topics are listed in the aerodynamics syllabus?",
        "source": "Aerodynamic_M-1.pdf",
        "pages": {2},
    },
    {
        "course_id": "aerodynamics",
        "query": "What Mach number was used in the double cone heat flux experiment?",
        "source": "Aerodynamic_M-1.pdf",
        "pages": {18},
    },
    {
        "course_id": "aerodynamics",
        "query": "What is the stream function for uniform flow plus a source?",
        "source": "Aerodynamic_M-1.pdf",
        "pages": {38, 39, 40},
    },
    {
        "course_id": "qrm",
        "query": "Explain producer risk and consumer risk in acceptance sampling",
        "source": "Quality_Reliability_Maintenance_Acceptance_Sampling_Notes.pdf",
        "pages": {7, 8, 9},
    },
    {
        "course_id": "qrm",
        "query": "What is an operating characteristic curve?",
        "source": "Acceptance_Sampling_OC_Curve_ME327E.pdf",
        "pages": {2, 3},
    },
    {
        "course_id": "qrm",
        "query": "How is average outgoing quality AOQ calculated?",
        "source": "Sampling_Plan_L15.pdf",
        "pages": {2, 3, 4, 5},
    },
    {
        "course_id": "qrm",
        "query": "What are prevention appraisal internal and external failure costs?",
        "source": "Quality_Cost_L5_Note.pdf",
        "pages": {2},
    },
    {
        "course_id": "qrm",
        "query": "When should a p control chart be used?",
        "source": "Quality_Reliability_Maintenance_Control_Chart_3_Note.pdf",
        "pages": set(range(16, 25)),
    },
    {
        "course_id": "qrm",
        "query": "Explain the Weibull distribution and its parameters",
        "source": "Quality_Reliability_Maintenance_L8.pdf",
        "pages": {2, 3, 4, 5, 6},
    },
]
