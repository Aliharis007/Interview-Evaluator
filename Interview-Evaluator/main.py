# backend/main.py
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, conlist
from typing import Dict, List, Any

from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel

# === Environment Setup ===
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = "us-central1" # Or your desired GCP region for Vertex AI
MODEL_NAME = "gemini-2.5-flash" # Your chosen Gemini model

# Initialize Vertex AI client and Gemini model
aiplatform.init(project=PROJECT_ID, location=LOCATION)
gemini_model = GenerativeModel(MODEL_NAME)

app = FastAPI(
    title="AI Interview Evaluation API",
    description="API for evaluating candidate performance using Google Gemini based on provided transcriptions.",
    version="1.0.0"
)

# === Evaluation Traits ===
EVALUATION_TRAITS = [
    "Resilience",
    "Self-Confidence",
    "Teamwork",
    "Influential",
    "Communication",
    "Ownership Mind-set",
    "Drive",
    "Discipline",
    "Creative Execution",
    "Customer Centricity"
]

# === API Request Models using Pydantic ===
class QuestionAnswerPair(BaseModel):
    """Represents a single question and its transcribed text answer."""
    question: str = Field(..., description="The interview question text.")
    transcribedText: str = Field(..., description="The transcribed text of the candidate's answer in Urdu.")

class InterviewEvaluationRequest(BaseModel):
    """The request body for evaluating an entire interview."""
    job_role_name: str = Field(..., description="The name of the job role being interviewed for.")
    responses: List[QuestionAnswerPair] = Field(..., min_length=1, description="A list of questions and their corresponding transcribed answers. This list cannot be empty.")

# === Function to Evaluate Response with Gemini ===
async def evaluate_response_with_gemini(
    job_role_name: str,
    interview_data: List[Dict[str, Any]] # List of {"question": ..., "transcribedText": ...}
) -> Dict[str, Any]:
    """
    Evaluates the candidate's interview responses using the Gemini model
    based on predefined traits and generates a concise report with scoring.
    """
    full_transcript = ""
    for i, qa in enumerate(interview_data):
        # Construct the prompt using both question and transcribed answer
        full_transcript += f"Question {i+1}: {qa['question']}\n"
        full_transcript += f"Candidate's Answer {i+1}: {qa['transcribedText']}\n\n"

    prompt = f"""
    You are an AI interview evaluator for the role of "{job_role_name}".
    Below is the transcript of an interview in the Urdu language. Your task is to evaluate the candidate's responses
    based on the following traits. Provide a score for each trait on a scale of 1 to 5,
    where 1 is Poor, 2 is Below Average, 3 is Average, 4 is Good, and 5 is Excellent. The 
    scores should reflect the candidate's performance in the interview and evaluation should be based on the provided traits.
    The evaluation should be consistent with the candidate's responses in the transcript. You must expect comprehensive answers from the 
    candidate. The overall summary should be professional, comprehensive, definite, and focused on actionable feedback
    for the candidate, highlighting strengths and areas for improvement with an improved tone and criteria.

    Interview Transcript:
    {full_transcript}

    Evaluation Traits: {', '.join(EVALUATION_TRAITS)}

    Provide the output as a JSON object with the following structure:
    {{
        "scores": {{
            "Resilience": <score 1-5>,
            "Self-Confidence": <score 1-5>,
            "Teamwork": <score 1-5>,
            "Influential": <score 1-5>,
            "Communication": <score 1-5>,
            "Ownership Mind-set": <score 1-5>,
            "Drive": <score 1-5>,
            "Discipline": <score 1-5>,
            "Creative Execution": <score 1-5>,
            "Customer Centricity": <score 1-5>
        }},
        "overall_summary": "<concise summary of performance>"
    }}
    """

    # Define the expected JSON schema for Gemini's response to ensure structured output
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "scores": {
                "type": "OBJECT",
                "properties": {trait: {"type": "INTEGER"} for trait in EVALUATION_TRAITS},
                "required": EVALUATION_TRAITS
            },
            "overall_summary": {"type": "STRING"}
        },
        "required": ["scores", "overall_summary"]
    }

    try:
        # Call Gemini model to generate content based on the prompt and schema
        response = gemini_model.generate_content(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": response_schema, # Ensure Gemini returns valid JSON
                "temperature": 0.0 # Set temperature to 0.0 for deterministic behavior
            }
        )
        # Parse the JSON string received from Gemini
        return json.loads(response.text)
    except Exception as e:
        print(f"Error generating evaluation report with Gemini: {e}")
        # Return a structured error response for consistency
        return {"error": f"Could not generate report: {e}"}


# === API Endpoint: Evaluate Interview ===
@app.post("/evaluate-interview", response_model=Dict[str, Any])
async def evaluate_interview_endpoint(request: InterviewEvaluationRequest):
    """
    Receives a list of interview questions and transcribed text for candidate responses,
    and generates an evaluation report using Gemini.
    """
    try:
        print("Generating evaluation report with Gemini based on provided transcriptions...")
        # The request already contains the transcribed text, so we can pass it directly
        evaluation_report = await evaluate_response_with_gemini(
            request.job_role_name,
            [response.dict() for response in request.responses] # Convert Pydantic models to dicts
        )
        print("Evaluation report generated.")

        # Calculate overall weighted score from Gemini's scores
        overall_weighted_score = 0
        if "scores" in evaluation_report and isinstance(evaluation_report["scores"], dict):
            overall_weighted_score = sum(evaluation_report["scores"].values())
            print(f"Overall Weighted Score: {overall_weighted_score}/50")
        else:
            print("Warning: 'scores' not found or not a dictionary in evaluation report. Overall score cannot be calculated.")

        # Return the comprehensive JSON response
        return JSONResponse(content={
            "status": "success",
            "job_role_name": request.job_role_name,
            "transcribed_answers": [response.dict() for response in request.responses],
            "evaluation_report": evaluation_report,
            "overall_weighted_score_out_of_50": overall_weighted_score
        })

    except Exception as e:
        print(f"An unhandled error occurred during interview evaluation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")