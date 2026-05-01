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
from tools.prompt_optimizer import PromptOptimizer, detect_answer_shape
from tools.external_llm import ExternalLLMService

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


class OptimizationResponse(BaseModel):
    raw_prompt: str
    optimized_prompt: str
    raw_score: dict
    optimized_score: dict
    external_llm_response_raw: str
    external_llm_response_optimized: str
    improvement_score: float


@app.post("/optimize_prompt", response_model=OptimizationResponse)
async def optimize_prompt(request: PromptRequest):
    raw = request.prompt

    if not raw.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    # 1. Compute Raw Score
    raw_score = scorer.evaluate(raw)

    # 2. Generate Rewrite
    try:
        optimized = optimizer.rewrite(raw)
    except RuntimeError as e:
        logger.error(f"Optimizer not available: {e}")
        raise HTTPException(
            status_code=503,
            detail="Prompt optimizer model is not loaded. Please try again later.",
        )

    # 2b. Answer-shape guardrail — catches the case where the model produced
    # an actual answer / code instead of a rewritten prompt. Cheap heuristic
    # (no model call); runs before scoring so we skip a wasted evaluate().
    is_answer_shaped, reasons = detect_answer_shape(raw, optimized)
    if is_answer_shaped:
        logger.info(
            f"Answer-shape detected, falling back to raw prompt. "
            f"signals={reasons}"
        )
        optimized = raw

    # 3. Compute Optimized Score (with semantic + specificity-recall gates)
    opt_score = scorer.evaluate(raw, optimized)

    # 4. Safety / Boundary checks (Self-Annealing)
    improvement = opt_score["quality_improvement"]

    # Reject if: any scorer gate failed OR no improvement
    if opt_score["rejected"] or improvement < 0:
        logger.info(
            f"Rewrite rejected (rejected={opt_score['rejected']}, "
            f"improvement={improvement:.4f}, "
            f"semantic={opt_score['semantic_preservation']:.4f}, "
            f"specificity_recall={opt_score['specificity_recall']:.4f}). "
            f"Falling back to raw prompt."
        )
        optimized = raw
        opt_score = raw_score
        improvement = 0.0

    # 5. External LLM Benchmarking (run in thread pool to avoid blocking event loop)
    import asyncio

    loop = asyncio.get_event_loop()
    resp_raw, resp_opt = await asyncio.gather(
        loop.run_in_executor(None, ext_llm.generate_response, raw),
        loop.run_in_executor(None, ext_llm.generate_response, optimized),
    )

    return OptimizationResponse(
        raw_prompt=raw,
        optimized_prompt=optimized,
        raw_score=raw_score,
        optimized_score=opt_score,
        external_llm_response_raw=resp_raw,
        external_llm_response_optimized=resp_opt,
        improvement_score=round(improvement, 4),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
