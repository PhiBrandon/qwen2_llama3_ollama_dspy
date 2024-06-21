from langfuse import Langfuse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import dspy

load_dotenv()


review_text = open("code_review.txt", "r").read()


class ReviewSummary(BaseModel):
    """Summary of code changes"""

    summary: str = Field(..., description="Summary of the changes in the code file.")


class ReviewSeverity(BaseModel):
    """Implied severity of the code changes"""

    severity: Literal["Critical", "Serious", "Minor"]
    explanations: list[str] = Field(
        ..., description="Explanation for severity selection"
    )


class Category(BaseModel):
    category: str
    explanation: str = Field(..., description="Explanation for category selection.")


class ReviewCategory(BaseModel):
    """Categories that the code changes could be. Examples... readability, maintainability, security, etc..."""

    categories: list[Category]


class Review(BaseModel):
    """Review of New code in relation to Old code."""

    summary: ReviewSummary
    severity: ReviewSeverity
    category: ReviewCategory


class RawSummary(dspy.Signature):
    """Summary of the code changes"""

    code_changes: str = dspy.InputField()
    summary: ReviewSummary = dspy.OutputField()


class RawSeverity(dspy.Signature):
    """Severity of the code changes. Make sure to provide detailed explanation of the severity classification. Possible sevirities: Critical, Serious, Minor"""

    code_changes: str = dspy.InputField()
    severity: ReviewSeverity = dspy.OutputField()


class RawCategory(dspy.Signature):
    """Categories that changes of could be. Examples: readability, maintainability, security. Give explanations for each category that is chosen."""

    code_changes: str = dspy.InputField()
    categories: ReviewCategory = dspy.OutputField()


class SummaryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.structured_summary = dspy.TypedPredictor(RawSummary)

    def forward(self, code_changes):
        structured = self.structured_summary(code_changes=code_changes)

        return structured


class SeverityModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.structured_severity = dspy.TypedPredictor(RawSeverity)

    def forward(self, code_changes):
        structured = self.structured_severity(code_changes=code_changes)
        return structured


class CategoryModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.structured_category = dspy.TypedPredictor(RawCategory)

    def forward(self, code_changes):
        structured = self.structured_category(code_changes=code_changes)
        return structured


class ReviewModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.summary = SummaryModule()
        self.severity = SeverityModule()
        self.category = CategoryModule()

    def forward(self, code_changes):
        summary = self.summary(code_changes=code_changes).summary
        severity = self.severity(code_changes=code_changes).severity
        category = self.category(code_changes=code_changes).categories
        return Review(summary=summary, severity=severity, category=category)

#ol_model = "llama3" # crashes
#ol_model = "phi3"
ol_model = "phi3:instruct"
#ol_model = "phi3:medium" #crashes
#ol_model = "phi3:14b-medium-128k-instruct-q4_0" # crashes !!
#ol_model = "deepseek-coder-v2" #128k needs timeout_s=300


#client = dspy.OllamaLocal(model="qwen2-7b:latest", max_tokens=10000)
client = dspy.OllamaLocal(model=ol_model, max_tokens=4000,temperature=0.002, timeout_s=300 )

dspy.configure(lm=client)

review = ReviewModule()
review_output: Review = review(code_changes=review_text)

print("Model: " + ol_model)
print("Review - Results")
print(review_output.summary)

print()
print("SeverityModule - Results")
print(review_output.severity)

print()
print("CategoryModule - Results")
print(review_output.category)