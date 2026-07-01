from fastapi import FastAPI
from app.models import TravelClaim, ApprovalDecision
from app.agent.graph import travel_reimbursement_graph
from app.agent.state import AgentState


app = FastAPI(
    title="Travel Reimbursement Approval Agent",
    version="1.0.0",
)

SERVICE_NAME = "Travel Reimbursement Approval Agent"


@app.get("/")
def root() -> dict[str, str]:
    return {"service": SERVICE_NAME, "version": app.version}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": SERVICE_NAME}

@app.post("/approve", response_model=ApprovalDecision)
def approve(claim: TravelClaim):
    state = AgentState(
        claim=claim,
        duplicate_result=None,
        validation_result=None,
        receipt_result=None,
        policy_context="",
        policy_section_titles=[],
        ai_decision=None,
        final_decision=None,
        current_step="start",
        errors=[],
    )
    result = travel_reimbursement_graph.invoke(state)
    return result["final_decision"]