import sys
sys.path.insert(0, ".")
from tests.labelled_sets import EARLY_ELIGIBLE, heldout
from simulator.transcript_simulator import simulate_utterance
from controller.controller import RetrievalController
from controller.stage2 import GroqStage2Confirm
from telemetry.bus import TelemetryBus
from llm.groq_client import GroqLLMClient

groq = GroqLLMClient()
print("--- Heldout items test ---")
for i, u in enumerate(heldout(EARLY_ELIGIBLE)):
    ctrl = RetrievalController(stage2=GroqStage2Confirm(client=groq), telemetry=TelemetryBus(), session_id='s', trace_id='t')
    chunks = simulate_utterance(u, session_id='s', seed=42)
    decs = [(c.seq, c.is_final, ctrl.decide(c).decision.value) for c in chunks]
    provs = [d for d in decs if d[2] == 'RETRIEVE_PROVISIONAL']
    print(f"Item {i+1}: early_fired={len(provs) > 0} (prov chunks: {provs})")
    print(f"       utt: {u}")
