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
from tools.db import init_db, create_user, verify_user, save_optimization_history, get_user_history

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


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
    improvement_score: float
    rewrite_metadata: dict
    issues: list
    recommendations: list[str]
    institutional_guideline: str
    validation: dict


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
    save_optimization_history(
        request.user_id,
        request.raw_prompt,
        request.optimized_prompt,
        request.raw_score,
        request.optimized_score,
        request.improvement_score
    )
    return {"message": "History saved successfully"}


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
        resp_raw, resp_opt = await asyncio.gather(
            loop.run_in_executor(None, ext_llm.generate_response, raw),
            loop.run_in_executor(None, ext_llm.generate_response, optimized),
        )
    else:
        resp_raw, resp_opt = "", ""

    resp = OptimizationResponse(
        raw_prompt=raw,
        optimized_prompt=optimized,
        raw_score=raw_score,
        optimized_score=opt_score,
        external_llm_response_raw=resp_raw,
        external_llm_response_optimized=resp_opt,
        improvement_score=round(improvement, 4),
        rewrite_metadata={
            "archetype": archetype.value,
            "modularity": modularity.value,
            "adapter_safe_mode": True,
            "runtime_generation_policy": "structured_adapter_aligned",
        },
        issues=issues,
        recommendations=recommendation_result["recommendations"],
        institutional_guideline=recommendation_result["institutional_guideline"],
        validation=validation,
    )

    if request.user_id is not None:
        save_optimization_history(
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
