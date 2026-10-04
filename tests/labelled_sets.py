"""
Labelled sets for Step 5 / Step 6 metric tests (blueprint §7.1).

Expanded for Phase 2/3 reporting:
- Step 5: 21 no-retrieval turns (PRESENTATION_ONLY + SUMMARIZE_VARIANTS + INCOMPLETE)
- Step 6: 28 compound utterances (COMPOUND)
"""

# Long-enough utterances whose intent becomes stable well before the end.
# "Eligible" for the G2 early-retrieval metric.
EARLY_ELIGIBLE = [
    "I need to plan a customer workshop in Pune for 30 people and I need the cancellation policy and the catering options",
    "We are organising a team offsite in Mumbai for 45 employees and want to know the venue capacity and parking availability",
    "I am planning a leadership retreat in Goa for 20 managers and I want the catering options and the refund policy",
    "Our team travelled to Singapore last week on an approved trip and I want to claim reimbursement for the hotel and meals",
    "We are hosting a client summit in Delhi for 60 guests and need to know which venue fits and what the cancellation terms are",
    "I booked a training workshop in Chennai for 25 attendees and now want to understand the late cancellation fee and the catering menu",
    "I am travelling to Bangalore for a product demo and need to know how much hotel accommodation I can claim per night",
    "We want to arrange a hackathon in Hyderabad for 100 participants and need to check the venue seating and the catering capacity",
    "For the annual meeting in Kolkata with 80 people I need the parking details and the accessibility features of the venue",
    "Our Pune workshop has 35 registered participants and I need the refund policy and the catering options for vegetarians",
]

# Presentation-only requests: must SUPPRESS, zero retrievals.
PRESENTATION_ONLY = [
    "Can you repeat that in two bullets please",
    "Make that shorter",
    "Translate that into Hindi",
    "Say that again more simply",
    "Reword that for a manager",
    "Rephrase the last answer",
    "Give me that in bullet points",
    "Simplify the explanation",
    "Put that into a bulleted list",
    "Can you rephrase the response in simpler terms",
    "Shorten the last paragraph please",
]

# Presentation-only phrased with "summarize" (US spelling).
SUMMARIZE_VARIANTS = [
    "Summarize that in one line",
    "Please summarize the answer for my boss",
]

# Utterances that trail off. Replayed with is_final=False on every chunk.
INCOMPLETE = [
    "Um I think we",
    "Can you tell me about the",
    "So the thing is that",
    "I was wondering if",
    "Okay and then the",
    "I need to plan a customer workshop in Pune for 30 people and",
    "Well we were considering if the",
    "Please check whether Venue B has",
]

# Real questions that must NOT be suppressed and MUST retrieve.
REAL_QUESTIONS = [
    "What is the seated capacity of Venue B?",
    "Is there a weekend booking surcharge",
    "How do I claim reimbursement for a domestic trip",
    "What is the cancellation policy for workshops in Pune",
]

# (utterance, gold number of distinct intents) — Step 6 / G3.
# First 12 are original starter items; 16 appended to reach 28 items.
COMPOUND = [
    ("What is the seated capacity of Venue B?", 1),
    ("What is the cancellation policy for workshops?", 1),
    ("I need to plan a customer workshop in Pune for 30 people and I need the cancellation policy and the catering options", 3),
    ("Can you tell me the venue capacity in Pune and the catering options", 2),
    ("I want to know the cancellation policy and the refund timeline and whether catering is included", 3),
    ("How do I claim travel reimbursement", 1),
    ("What is the domestic reimbursement rule and what extra approvals does an international trip need", 2),
    ("Which venue fits 50 people and does it have parking", 2),
    ("Does the catering menu include vegetarian options", 1),
    ("I need the capacity of Venue A, the cancellation deadline, and the dietary options", 3),
    ("What is the late cancellation fee", 1),
    ("Tell me about the reimbursement deadline and also how international trips differ", 2),
    # Appended items for G3 evaluation scaling
    ("Can you give me the exact address of Venue C?", 1),
    ("Is parking free for attendees at Venue A?", 1),
    ("Who approves international travel exceptions?", 1),
    ("What is the cutoff notice for dietary restrictions in catering?", 1),
    ("What is the capacity of Venue C and does it have parking facilities?", 2),
    ("Tell me the cancellation timeline for Venue B and how the catering deposit works", 2),
    ("How much can I claim for domestic hotel stay and what meals are covered", 2),
    ("Can we bring outside caterers for Venue A and what is the cleanup policy", 2),
    ("What is the seated capacity for Venue B as well as the audio visual facilities", 2),
    ("I want the refund policy for late cancellations and the travel reimbursement limits", 2),
    ("Can I get the parking space count at Venue A plus the public transport options", 2),
    ("What are the international flight booking rules and who must sign the memo", 2),
    ("Does Venue B offer valet parking and what is the daily garage rate", 2),
    ("I need the seated capacity for Venue B, the parking fees, and the cancellation cutoff", 3),
    ("Can you tell me the domestic travel allowance, the hotel limit per night, and the meal limits", 3),
    ("Tell me the cancellation fees for Venue A, the catering notice period, and the parking spaces", 3),
]


def heldout(items, fraction=0.3):
    """Deterministic hold-out: the last `fraction` of the list."""
    n = max(1, int(len(items) * fraction))
    return items[-n:]
