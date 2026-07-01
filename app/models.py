from datetime import date
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class ExpenseCategory(str, Enum):
    FLIGHT = "flight"
    HOTEL = "hotel"
    MEALS = "meals"
    TRANSPORT = "transport"
    OTHER = "other"


class ApprovalStatus(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PARTIALLY_APPROVED = "partially_approved"
    MANUAL_REVIEW = "manual_review"


class ExpenseItem(BaseModel):
    category: ExpenseCategory = Field(description="Type of travel expense")
    amount: Decimal = Field(gt=0, description="Expense amount in the stated currency")
    currency: str = Field(
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code (e.g. USD, EUR)",
    )
    description: str = Field(
        min_length=1,
        max_length=500,
        description="Brief description of the expense",
    )
    receipt_attached: bool = Field(
        description="Whether a receipt or supporting document is attached",
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.strip().upper()


class TravelClaim(BaseModel):
    employee_id: str = Field(
        min_length=1,
        max_length=50,
        description="Unique employee identifier",
    )
    employee_name: str = Field(
        min_length=1,
        max_length=100,
        description="Full name of the employee submitting the claim",
    )
    department: str = Field(
        min_length=1,
        max_length=100,
        description="Employee department or cost center",
    )
    trip_start_date: date = Field(description="First day of the business trip")
    trip_end_date: date = Field(description="Last day of the business trip")
    destination: str = Field(
        min_length=1,
        max_length=200,
        description="Primary trip destination (city, country, or region)",
    )
    purpose: str = Field(
        min_length=1,
        max_length=500,
        description="Business purpose of the trip",
    )
    expenses: list[ExpenseItem] = Field(
        min_length=1,
        description="List of itemized expenses for the trip",
    )
    claim_id: str = Field(
        min_length=1,
        max_length=50,
        description="Unique reimbursement claim identifier",
    )

    @model_validator(mode="after")
    def validate_trip_dates(self) -> "TravelClaim":
        if self.trip_end_date < self.trip_start_date:
            raise ValueError("trip_end_date must be on or after trip_start_date")
        return self


class ApprovalDecision(BaseModel):
    decision: ApprovalStatus = Field(description="Final approval outcome for the claim")
    approved_amount: Decimal = Field(
        ge=0,
        description="Total amount approved for reimbursement",
    )
    rejected_amount: Decimal = Field(
        ge=0,
        description="Total amount rejected from the claim",
    )
    missing_documents: list[str] = Field(
        default_factory=list,
        description="Required documents that were not provided",
    )
    violated_policies: list[str] = Field(
        default_factory=list,
        description="Company travel policies that were violated",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Agent confidence score for this decision (0.0 to 1.0)",
    )
    explanation: str = Field(
        min_length=1,
        max_length=2000,
        description="Human-readable rationale for the approval decision",
    )
