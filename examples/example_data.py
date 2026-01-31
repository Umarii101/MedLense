"""
Example clinical notes for testing the system.
These are synthetic examples for demonstration purposes.
"""

EXAMPLE_CASES = {
    "case_1_respiratory": {
        "clinical_note": """
Patient: 72-year-old female

Chief Complaint: Progressive shortness of breath and fatigue over past 2 months

History of Present Illness:
Patient reports gradual onset of dyspnea on exertion that has worsened over the past 
8 weeks. Initially noticed difficulty with stairs, now experiences SOB with minimal 
activity. Denies chest pain, palpitations, or syncope. Reports bilateral ankle swelling 
that worsens throughout the day. Occasional dry cough, worse at night. Denies fever, 
recent illness, or weight changes.

Past Medical History:
- Congestive Heart Failure (EF 40% on last echo 6 months ago)
- Hypertension
- Hyperlipidemia
- Chronic Kidney Disease Stage 3

Current Medications:
- Furosemide 40mg daily
- Lisinopril 10mg daily
- Metoprolol 50mg twice daily
- Atorvastatin 40mg nightly

Physical Exam:
Vitals: BP 156/92, HR 88, RR 22, SpO2 92% on room air, Temp 98.1°F
General: Appears fatigued, mildly dyspneic
Heart: Regular rhythm, S3 gallop present, 2/6 systolic murmur
Lungs: Bilateral crackles at bases
Extremities: 2+ pitting edema bilateral lower extremities

Labs:
- BNP: 850 pg/mL (elevated)
- Creatinine: 1.8 mg/dL (baseline 1.5)
- Potassium: 4.2 mEq/L
""",
        "patient_age": 72,
        "image_type": "Chest X-Ray",
        "expected_findings": ["heart failure exacerbation", "volume overload", "renal function decline"]
    },
    
    "case_2_trauma": {
        "clinical_note": """
Patient: 28-year-old male

Chief Complaint: Right ankle pain and swelling following sports injury

History of Present Illness:
Patient was playing basketball approximately 2 hours ago when he landed awkwardly 
on his right foot after jumping. Heard a "pop" and experienced immediate severe pain 
in right ankle. Unable to bear weight. Swelling began immediately. Denies loss of 
consciousness, head injury, or other trauma. Ice applied at scene.

Past Medical History: None significant

Current Medications: None

Social History: Physically active, no tobacco/alcohol

Physical Exam:
Vitals: BP 128/76, HR 78, RR 16, Temp 98.6°F, Pain 8/10
General: Alert, in moderate distress from pain
Right Lower Extremity:
- Significant swelling over lateral ankle
- Ecchymosis present
- Tenderness over lateral malleolus and anterior talofibular ligament
- Limited range of motion due to pain
- Neurovascularly intact (pulses 2+, normal sensation, capillary refill <2 sec)
- Unable to bear weight

Ottawa Ankle Rules: Positive (tenderness at malleolus)
""",
        "patient_age": 28,
        "image_type": "Ankle X-Ray",
        "expected_findings": ["ankle injury", "possible fracture", "ligament injury"]
    },
    
    "case_3_pediatric": {
        "clinical_note": """
Patient: 8-year-old female

Chief Complaint: Fever and ear pain for 2 days

History of Present Illness:
Per mother, patient developed fever to 101.5°F two days ago. Yesterday began 
complaining of right ear pain, describing it as "really hurting inside." Pain 
worse when lying down. No pulling or tugging on ear. Reports some hearing 
difficulty in right ear. Denies sore throat, cough, or nasal congestion. 
Decreased appetite but maintaining hydration. No vomiting or diarrhea.

Past Medical History:
- Recurrent otitis media (4 episodes in past year)
- Mild intermittent asthma (well-controlled)

Current Medications:
- Albuterol inhaler PRN (rarely used)

Physical Exam:
Vitals: Temp 100.8°F, HR 108, RR 20, BP 98/62, SpO2 99%
General: Appears mildly uncomfortable but playful
HEENT:
- Right TM: Erythematous, bulging, decreased mobility on pneumatic otoscopy
- Left TM: Normal appearance, good mobility
- Oropharynx: Clear, no erythema
Neck: Supple, no lymphadenopathy
Lungs: Clear bilaterally
""",
        "patient_age": 8,
        "image_type": None,
        "expected_findings": ["acute otitis media", "recurrent infections"]
    },
    
    "case_4_chronic_disease": {
        "clinical_note": """
Patient: 55-year-old male

Chief Complaint: Routine diabetes follow-up

History of Present Illness:
Patient here for 3-month diabetes follow-up. Reports generally feeling well. 
Blood glucose readings at home typically 120-160 mg/dL fasting, 180-220 mg/dL 
postprandial. Adherent to medications but admits dietary indiscretions on 
weekends. Walking 30 minutes 3-4 times per week. Denies polyuria, polydipsia, 
polyphagia. No visual changes. Annual eye exam scheduled next month.

Past Medical History:
- Type 2 Diabetes Mellitus (diagnosed 5 years ago)
- Hypertension
- Hyperlipidemia
- Obesity (BMI 34)

Current Medications:
- Metformin 1000mg twice daily
- Glipizide 10mg daily
- Lisinopril 20mg daily
- Atorvastatin 20mg nightly

Physical Exam:
Vitals: BP 138/84, HR 76, Weight 235 lbs (down 3 lbs from last visit)
General: Well-appearing, no acute distress
Cardiovascular: Regular rate and rhythm
Feet: Intact sensation to monofilament, pedal pulses 2+ bilaterally, no ulcers

Labs (drawn this morning):
- HbA1c: 7.8% (goal <7%, previous 8.2%)
- Fasting glucose: 145 mg/dL
- Creatinine: 1.1 mg/dL, eGFR: 72 mL/min
- LDL: 98 mg/dL
- Microalbumin/Creatinine ratio: 35 mg/g (mildly elevated)
""",
        "patient_age": 55,
        "image_type": None,
        "expected_findings": ["diabetes management", "improving control", "early nephropathy"]
    }
}


def get_example_case(case_name: str):
    """Get an example case by name"""
    return EXAMPLE_CASES.get(case_name, EXAMPLE_CASES["case_1_respiratory"])


def list_example_cases():
    """List all available example cases"""
    print("Available Example Cases:")
    print("-" * 80)
    for name, case in EXAMPLE_CASES.items():
        preview = case["clinical_note"][:100].replace("\n", " ")
        print(f"\n{name}:")
        print(f"  Age: {case['patient_age']}")
        print(f"  Preview: {preview}...")
        if case.get("image_type"):
            print(f"  Image Type: {case['image_type']}")
    print("-" * 80)


if __name__ == "__main__":
    list_example_cases()
