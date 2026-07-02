import asyncio
import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.heuristic_scorer import HeuristicScorer
from tools.prompt_optimizer import PromptOptimizer
from tools.external_llm import ExternalLLMService
from tools.prompt_diagnostics import find_prompt_issues
from tools.prompt_validator import validate_rewrite
from tools.recommendation_engine import build_recommendations
from dataset_builder.prompt_templates import detect_archetype, modularity_for
from tools.db import init_db, create_user, verify_user, save_optimization_history, get_user_history, set_feedback

# Configure structured logging for all promptee modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "error_log.txt"),
            mode="a",
        ),
    ],
)
logger = logging.getLogger("promptee.api")

# ── Diagnostic-Guided Refinement (Pass 2) ─────────────────────────────────
# When a Pass 1 rewrite scores below this quality threshold, a second
# rewrite pass is attempted with a targeted meta-prompt that tells the
# model exactly which quality dimensions are weak. This addresses the
# "re-optimization stall" where feeding an already-optimized prompt back
# into the generic rewriter produces no improvement.
REFINEMENT_QUALITY_THRESHOLD = 0.85

# Maximum quality threshold for attempting refinement — if Pass 1 already
# scores at or above this, no second pass is needed.
REFINEMENT_MAX_ATTEMPTS = 1  # Additional attempts beyond Pass 1


def _build_refinement_prompt(score_result: dict) -> str | None:
    """
    Analyse scorer diagnostics and build a targeted system prompt for Pass 2.

    Examines which specificity categories scored zero and which clarity
    sub-components are weak, then constructs explicit instructions for the
    model to add the missing elements.

    Returns None if no actionable gaps are found (the prompt is already
    well-optimised and a second pass would not help).
    """
    gaps: list[str] = []

    # ── Specificity gaps: which categories scored zero? ────────────────
    if score_result.get("specificity_persona", 0) == 0:
        gaps.append(
            "Add a clear persona or role declaration at the start "
            "(e.g. 'You are a senior [relevant expert]')."
        )
    if score_result.get("specificity_negation", 0) == 0:
        gaps.append(
            "Add at least one explicit constraint or boundary "
            "(e.g. 'Do not include...', 'Avoid...', 'Never...')."
        )
    if score_result.get("specificity_ranges", 0) == 0:
        gaps.append(
            "Add specific numeric bounds where appropriate "
            "(e.g. 'between 3 and 5 examples', 'at least 200 words', "
            "'no more than 10 items')."
        )
    if score_result.get("specificity_formats", 0) == 0:
        gaps.append(
            "Specify the desired output format explicitly "
            "(e.g. 'Return the response as a numbered list', "
            "'Output in JSON', 'Use markdown with headings')."
        )
    if score_result.get("specificity_entities", 0) == 0:
        gaps.append(
            "Include specific named entities, technologies, or concrete "
            "references relevant to the task rather than generic terms."
        )

    # ── Clarity gaps: which sub-components are low? ────────────────────
    clarity_actionability = score_result.get("clarity_actionability", 1.0)
    if clarity_actionability < 0.5:
        gaps.append(
            "Use more direct action verbs (e.g. 'Explain', 'Create', "
            "'List', 'Compare', 'Analyze') instead of vague phrasing."
        )

    clarity_structure = score_result.get("clarity_structure", 1.0)
    if clarity_structure < 0.5:
        gaps.append(
            "Add structural formatting where it improves clarity "
            "(e.g. numbered steps, bullet points, or labeled sections)."
        )

    clarity_completeness = score_result.get("clarity_completeness", 1.0)
    if clarity_completeness < 0.6:
        gaps.append(
            "Ensure the prompt covers key instruction components: "
            "objective/goal, specific requirements, and expected output."
        )

    if not gaps:
        return None

    # Build the refinement system prompt
    gap_instructions = "\n".join(f"- {g}" for g in gaps)
    return (
        "You are an expert Prompt Rewriter and Prompt Quality Optimizer.\n\n"
        "The prompt below has already been partially optimized but still has "
        "specific quality gaps. Your task is to REFINE it by addressing the "
        "gaps listed below while preserving everything that is already good.\n\n"
        "Quality gaps to address:\n"
        f"{gap_instructions}\n\n"
        "Rules:\n"
        "- Preserve the original intent and all existing good structure.\n"
        "- Only ADD or IMPROVE the specific elements listed above.\n"
        "- Do NOT remove existing constraints, sections, or details.\n"
        "- Do NOT answer the prompt or generate the requested output.\n"
        "- Do NOT add irrelevant requirements.\n"
        "- Return only the refined prompt."
    )


# Initialize singleton tools (lightweight — no GPU models yet)
scorer = HeuristicScorer()
optimizer = PromptOptimizer()
ext_llm = ExternalLLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy models at startup, clean up on shutdown."""
    init_db()
    try:
        optimizer.load_model()
        logger.info("Startup complete — optimizer model loaded.")
    except RuntimeError as e:
        logger.warning(
            f"Optimizer model could not be loaded at startup: {e}. "
            f"Rewrite requests will fail until the model is available."
        )
    yield
    logger.info("Shutdown complete.")


app = FastAPI(title="Prompt Optimization Pipeline", lifespan=lifespan)

# Local dev origins are always allowed. Production origins (e.g. the Vercel URL)
# are appended from the comma-separated FRONTEND_ORIGINS env var so the image
# does not need a rebuild when the deployed frontend domain changes.
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
]
_extra_origins = [
    o.strip() for o in os.environ.get("FRONTEND_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional legacy static-frontend mount. The current production frontend ships
# to Vercel from the Next.js project under Frontend/ (capital F), so this path
# is only present in older local checkouts. Skip the mount when missing — the
# API runs headless behind a CORS-allowed Vercel client.
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
else:
    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return {"status": "ok", "service": "promptee-api", "docs": "/docs"}


class PromptRequest(BaseModel):
    prompt: str
    user_id: int = None
    # When False (default), skip the external Gemini benchmarking round-trips
    # to keep end-to-end latency low. The UI can opt in for side-by-side comparison.
    benchmark: bool = False


class OptimizationResponse(BaseModel):
    raw_prompt: str
    optimized_prompt: str
    raw_score: dict
    optimized_score: dict
    external_llm_response_raw: str
    external_llm_response_optimized: str
    external_llm_status_raw: str = "off"          # off | ok | error
    external_llm_status_optimized: str = "off"
    external_llm_error_raw: str | None = None
    external_llm_error_optimized: str | None = None
    improvement_score: float
    rewrite_metadata: dict
    issues: list
    recommendations: list[str]
    institutional_guideline: str
    validation: dict
    run_id: int | None = None


class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/signup")
async def signup(request: AuthRequest):
    success, message = create_user(request.username, request.password)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"message": message}


@app.post("/signin")
async def signin(request: AuthRequest):
    success, user_id = verify_user(request.username, request.password)
    if not success:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Signin successful", "user_id": user_id, "username": request.username}


@app.get("/history/{user_id}")
async def get_history(user_id: int):
    history = get_user_history(user_id)
    return {"history": history}


class SaveHistoryRequest(BaseModel):
    user_id: int
    raw_prompt: str
    optimized_prompt: str
    raw_score: dict
    optimized_score: dict
    improvement_score: float

@app.post("/save_history")
async def save_history(request: SaveHistoryRequest):
    run_id = save_optimization_history(
        request.user_id,
        request.raw_prompt,
        request.optimized_prompt,
        request.raw_score,
        request.optimized_score,
        request.improvement_score
    )
    return {"message": "History saved successfully", "run_id": run_id}


class FeedbackRequest(BaseModel):
    run_id: int
    user_id: int
    feedback: str | None  # "like" | "dislike" | None (to clear)


@app.post("/feedback")
async def post_feedback(request: FeedbackRequest):
    if request.feedback not in (None, "like", "dislike"):
        raise HTTPException(status_code=400, detail="feedback must be 'like', 'dislike', or null")
    ok = set_feedback(request.run_id, request.user_id, request.feedback)
    if not ok:
        raise HTTPException(status_code=404, detail="Run not found for this user")
    return {"message": "Feedback saved"}


@app.post("/optimize_prompt", response_model=OptimizationResponse)
async def optimize_prompt(request: PromptRequest):
    raw = request.prompt

    if not raw.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # 1. Deterministic diagnostics on the raw prompt (cheap, <50ms — keep on main thread)
    issues = find_prompt_issues(raw)
    archetype = detect_archetype(raw)
    modularity = modularity_for(archetype)

    loop = asyncio.get_event_loop()

    # 2 & 3. Raw scoring (CPU) and rewrite (GPU) are both heavy and independent of
    # each other — run them concurrently. spaCy / sentence-transformers / model.generate
    # all release the GIL during their hot loops, so a ThreadPoolExecutor gives real overlap.
    try:
        raw_score, optimized = await asyncio.gather(
            loop.run_in_executor(None, scorer.evaluate, raw),
            loop.run_in_executor(None, optimizer.rewrite, raw),
        )
    except RuntimeError as e:
        logger.error(f"Optimizer not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Prompt optimizer model is not loaded. Please try again later.",
        )

    # 4. Compute Optimized Score (with semantic preservation gate)
    opt_score = scorer.evaluate(raw, optimized)

    # 5. Safety / Boundary checks (Self-Annealing)
    improvement = opt_score["quality_improvement"]

    # Reject if: semantic preservation gate failed OR no improvement
    if opt_score["rejected"] or improvement < 0:
        logger.info(
            f"Rewrite rejected (rejected={opt_score['rejected']}, improvement={improvement:.4f}). "
            f"Falling back to raw prompt."
        )
        optimized = raw
        opt_score = raw_score
        improvement = 0.0

    # 5b. Diagnostic-Guided Refinement (Pass 2)
    # If Pass 1 produced a valid rewrite but quality is below the threshold,
    # identify specific scorer gaps and run a targeted second pass.
    pass2_attempted = False
    if (
        optimized != raw  # Pass 1 was accepted (not rejected/fallen-back)
        and opt_score["candidate_quality"] < REFINEMENT_QUALITY_THRESHOLD
    ):
        refinement_sys_prompt = _build_refinement_prompt(opt_score)
        if refinement_sys_prompt is not None:
            logger.info(
                f"Pass 1 quality={opt_score['candidate_quality']:.4f} < "
                f"threshold={REFINEMENT_QUALITY_THRESHOLD}. Attempting "
                f"diagnostic-guided refinement (Pass 2)."
            )
            pass2_attempted = True
            try:
                # Pass 2: refine the Pass 1 output with targeted instructions
                pass2_candidate = await loop.run_in_executor(
                    None,
                    lambda: optimizer.rewrite(
                        optimized,
                        sys_prompt_override=refinement_sys_prompt,
                        user_prompt_template=(
                            "Current prompt to refine:\n{0}\n\n"
                            "Refined prompt only:"
                        ),
                    ),
                )

                # Score Pass 2 against the original raw prompt
                pass2_score = scorer.evaluate(raw, pass2_candidate)
                pass2_quality = pass2_score["candidate_quality"]
                pass1_quality = opt_score["candidate_quality"]

                if (
                    not pass2_score["rejected"]
                    and pass2_quality > pass1_quality
                ):
                    logger.info(
                        f"Pass 2 improved quality: {pass1_quality:.4f} → "
                        f"{pass2_quality:.4f} (+{pass2_quality - pass1_quality:.4f}). "
                        f"Accepting Pass 2 result."
                    )
                    optimized = pass2_candidate
                    opt_score = pass2_score
                    improvement = opt_score["quality_improvement"]
                else:
                    logger.info(
                        f"Pass 2 did not improve (Pass 1={pass1_quality:.4f}, "
                        f"Pass 2={pass2_quality:.4f}, rejected={pass2_score['rejected']}). "
                        f"Keeping Pass 1 result."
                    )

            except Exception as e:
                logger.warning(
                    f"Pass 2 refinement failed: {e}. Keeping Pass 1 result.",
                    exc_info=True,
                )

    # 6. Post-rewrite validation (deterministic guard)
    validation = validate_rewrite(raw, optimized)

    # 7. Build educational recommendations from issues + scores + archetype
    recommendation_result = build_recommendations(
        raw_prompt=raw,
        issues=issues,
        score_result=opt_score,
        archetype=archetype.value,
    )

    # 8. External LLM Benchmarking — opt-in only. Skipping this saves 1–3s of network
    # round-trip per request; the UI can toggle benchmark=True for the comparison view.
    if request.benchmark:
        raw_result, opt_result = await asyncio.gather(
            loop.run_in_executor(None, ext_llm.generate_response, raw),
            loop.run_in_executor(None, ext_llm.generate_response, optimized),
        )
        resp_raw = raw_result["text"]
        resp_opt = opt_result["text"]
        status_raw = "ok" if raw_result["ok"] else "error"
        status_opt = "ok" if opt_result["ok"] else "error"
        err_raw = raw_result["error"]
        err_opt = opt_result["error"]
        if not raw_result["ok"]:
            logger.warning(f"External LLM (raw) failed: {err_raw}")
        if not opt_result["ok"]:
            logger.warning(f"External LLM (optimized) failed: {err_opt}")
    else:
        resp_raw, resp_opt = "", ""
        status_raw = status_opt = "off"
        err_raw = err_opt = None

    resp = OptimizationResponse(
        raw_prompt=raw,
        optimized_prompt=optimized,
        raw_score=raw_score,
        optimized_score=opt_score,
        external_llm_response_raw=resp_raw,
        external_llm_response_optimized=resp_opt,
        external_llm_status_raw=status_raw,
        external_llm_status_optimized=status_opt,
        external_llm_error_raw=err_raw,
        external_llm_error_optimized=err_opt,
        improvement_score=round(improvement, 4),
        rewrite_metadata={
            "archetype": archetype.value,
            "modularity": modularity.value,
            "adapter_safe_mode": True,
            "runtime_generation_policy": "structured_adapter_aligned",
            "refinement_pass2_attempted": pass2_attempted,
        },
        issues=issues,
        recommendations=recommendation_result["recommendations"],
        institutional_guideline=recommendation_result["institutional_guideline"],
        validation=validation,
    )

    if request.user_id is not None:
        resp.run_id = save_optimization_history(
            request.user_id,
            raw,
            optimized,
            raw_score,
            opt_score,
            round(improvement, 4)
        )

    return resp


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
