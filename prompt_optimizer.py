import os
import json
import math
import time
import random
from typing import TypedDict, Any, Literal

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph

# Configuration
SAMPLE_SIZE = 20  # Number of samples to evaluate on

class OptimizerState(TypedDict, total=False):
    prompt: str
    best_prompt: str
    best_score: float
    attempts: list[dict[str, Any]]
    iter: int
    max_iters: int
    samples: list[dict[str, Any]]
    _last_predictions_runs: list[list[str]]


def load_dataset(csv_path: str, sample_size: int = SAMPLE_SIZE, seed: int | None = None) -> list[dict[str, Any]]:
    print(f"📊 CSV 파일 로드 중: {csv_path}")
    # Handle complex CSV with multiline content
    df = pd.read_csv(csv_path, on_bad_lines='skip')
    print(f"✅ CSV 로드 완료: {len(df)}개 행 발견")
    
    if len(df) < sample_size:
        raise ValueError(f"Dataset has only {len(df)} rows, but sample_size={sample_size}")
    
    if seed is not None:
        print(f"🎯 {sample_size}개 샘플 추출 중 (seed={seed})")
    else:
        print(f"🎯 {sample_size}개 랜덤 샘플 추출 중")
    if seed is not None:
        sampled = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    else:
        sampled = df.sample(n=sample_size).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for _, r in sampled.iterrows():
        rows.append({
            "title": str(r.get("title", "")),
            "content": str(r.get("content", "")),
            "label": str(r.get("label", "")).strip(),
        })
    print(f"✅ 샘플 준비 완료: {len(rows)}개")
    return rows


DEFAULT_BASE_PROMPT = (
    """
당신은 매우 엄격하고 정확한 텍스트 분류기입니다. 당신의 유일한 임무는 주어진 기사의 핵심 주제가 '자동차' 자체(예: 신차 출시, 자동차 기술, 자동차 산업 동향)인지 아닌지를 판단하는 것입니다. 사소한 언급이나 간접적인 연관성은 전부 무시해야 합니다. 당신의 판단은 오직 '1' 또는 '0'으로만 출력되어야 합니다. 다른 어떤 단어나 설명도 추가해서는 안 됩니다.

아래 규칙을 반드시 따르십시오.
[규칙]
1. 핵심 주제가 자동차인가? 기사의 제목과 본문 전체를 고려했을 때, 핵심 주제가 명백하게 자동차, 자동차 산업, 자동차 기술, 신차 리뷰 등이어야만 '1'입니다.
2. 자동차의 단순 언급은 '0'입니다. 교통사고, 교통 체증, 특정 인물이 차를 탔다는 내용, 배경에 자동차가 등장하는 사건 등은 자동차가 핵심 주제가 아니므로 '0'입니다.
3. 주제가 모호하면 '0'입니다. 자동차 외에 다른 주제(예: 경제, 기술, 사회)와 동등한 비중으로 다뤄지거나, 자동차가 단지 예시로 사용된 경우 무조건 '0'입니다.
4. 출력은 오직 '1' 또는 '0'입니다. 어떠한 경우에도 다른 설명, 문장, 주석을 포함해서는 안 됩니다.
    """
    .strip()
)


def format_news_for_user(row: dict[str, Any]) -> str:
    return f"[기사]\n제목: {row['title']}\n내용: {row['content']}"


def parse_label(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`")
    # Normalize to just '0' or '1'
    if "1" in t and "0" not in t:
        return "1"
    if "0" in t and "1" not in t:
        return "0"
    # Fallback: take first char if valid
    if t[:1] in {"0", "1"}:
        return t[:1]
    return "0"


def classify_once(eval_llm: ChatOpenAI, prompt: str, row: dict[str, Any]) -> str:
    msgs = [
        SystemMessage(content=prompt),
        HumanMessage(content=format_news_for_user(row)),
    ]
    res = eval_llm.invoke(msgs)
    if isinstance(res, AIMessage):
        return parse_label(res.content)
    return "0"


def evaluate_prompt(
    eval_llm: ChatOpenAI,
    prompt: str,
    samples: list[dict[str, Any]],
    runs: int = 3,
) -> dict[str, Any]:
    print(f"🔍 프롬프트 평가 시작 ({runs}회 수행)")
    accuracy_runs: list[float] = []
    predictions_runs: list[list[str]] = []
    for run_idx in range(runs):
        print(f"  📝 {run_idx+1}/{runs}번째 평가 중...")
        preds: list[str] = []
        correct = 0
        for i, row in enumerate(samples):
            yhat = classify_once(eval_llm, prompt, row)
            preds.append(yhat)
            if yhat == str(row["label"]).strip():
                correct += 1
            print(f"    샘플 {i+1}: 예측={yhat}, 정답={row['label']}, {'✅' if yhat == str(row['label']).strip() else '❌'}")
        acc = correct / len(samples)
        accuracy_runs.append(acc)
        predictions_runs.append(preds)
        print(f"  🎯 {run_idx+1}회차 정확도: {acc:.3f}")

    accuracy = sum(accuracy_runs) / len(accuracy_runs)
    L = len(prompt)
    # score favors accuracy, lightly penalizes very long prompts
    length_term = math.sqrt(max(0.0, 1.0 - (L / 3000.0) ** 2))
    score = 0.9 * accuracy + 0.1 * length_term
    print(f"📊 평균 정확도: {accuracy:.3f}, 최종 점수: {score:.3f}")

    return {
        "accuracy": accuracy,
        "score": score,
        "accuracy_runs": accuracy_runs,
        "predictions_runs": predictions_runs,
    }


def aggregate_errors(samples: list[dict[str, Any]], predictions_runs: list[list[str]]) -> list[dict[str, Any]]:
    # Collect misclassified cases across runs (use majority mistake indicator)
    errors: list[dict[str, Any]] = []
    n = len(samples)
    runs = len(predictions_runs)
    for i in range(n):
        votes = [predictions_runs[r][i] for r in range(runs)]
        # If any run misclassified, include for analysis
        gold = str(samples[i]["label"]).strip()
        if any(v != gold for v in votes):
            errors.append({
                "title": samples[i]["title"],
                "content": samples[i]["content"],
                "label": gold,
                "pred_votes": votes,
            })
    # Limit to keep the improve prompt short
    return errors[:8]


def improve_prompt_with_llm(
    llm: ChatOpenAI,
    current_prompt: str,
    best_prompt: str,
    best_score: float,
    errors: list[dict[str, Any]],
) -> str:
    sys = SystemMessage(
        content=(
            "당신은 프롬프트 엔지니어입니다. 아래 분류 태스크의 프롬프트를 더 간결하고 정확하게 개선해 주세요.\n"
            "반드시 '출력은 오직 0 또는 1' 규율을 유지하고, 오분류를 줄일 수 있는 간결한 규칙을 보강하세요.\n"
            "너무 장황한 문장은 피하고, 금지/허용 기준을 명확히 정제하세요.\n"
            "응답은 JSON 형식으로만, key는 improved_prompt 하나만 포함하세요."
            "점수의 계산 식은 다음과 같습니다: score = 0.9 * accuracy + 0.1 * sqrt(max(0, 1 - (L/3000)^2)), 여기서 L은 프롬프트 길이입니다.\n" \
            "당신의 목표는 accuracy를 높이되, 프롬프트가 너무 길어지지 않도록 하는 것입니다.\n" \
            "아래는 현재 프롬프트, 최고 프롬프트, 최고 점수, 그리고 최근 평가에서의 오분류 사례입니다.\n" \
            "오분류 사례는 반드시 분석하여 개선에 반영해야 합니다." \
            "\n\n"
        )
    )
    user = HumanMessage(
        content=json.dumps({
            "current_prompt": current_prompt,
            "best_prompt": best_prompt,
            "best_score": round(best_score, 5),
            "common_errors": errors,
        }, ensure_ascii=False)
    )
    res = llm.invoke([sys, user])
    text = res.content if isinstance(res, AIMessage) else "{}"
    try:
        obj = json.loads(text)
        improved = str(obj.get("improved_prompt", "")).strip()
        if improved:
            return improved
    except Exception:
        pass
    # Fallback heuristic: append a brief refinement
    return (
        current_prompt
        + "\n\n[추가 규칙]\n5. 자동차 '관세/정책/주가/투자' 등 메타 주제는 자동차 자체가 아니므로 0으로 답하십시오."
    )


def evaluate_node(state: OptimizerState, eval_llm: ChatOpenAI) -> OptimizerState:
    print(f"\n🚀 === 반복 {state.get('iter', 0) + 1} - 평가 단계 ===")
    
    # Generate new samples for this iteration
    csv_path = os.path.join(os.path.dirname(__file__), "final.csv")
    new_samples = load_dataset(csv_path)
    state["samples"] = new_samples
    print(f"🎲 새로운 랜덤 샘플 생성")
    
    eval_result = evaluate_prompt(eval_llm, state["prompt"], state["samples"], runs=1)
    attempt = {
        "iteration": state.get("iter", 0),
        "prompt": state["prompt"],
        "accuracy": eval_result["accuracy"],
        "score": eval_result["score"],
        "accuracy_runs": eval_result["accuracy_runs"],
    }
    state.setdefault("attempts", []).append(attempt)
    
    current_best = state.get("best_score", -1)
    if eval_result["score"] > current_best:
        print(f"🎉 새로운 최고 점수! {current_best:.3f} → {eval_result['score']:.3f}")
        state["best_score"] = eval_result["score"]
        state["best_prompt"] = state["prompt"]
    else:
        print(f"📈 현재 점수: {eval_result['score']:.3f} (최고: {current_best:.3f})")
    
    # Save progress after each evaluation
    print("💾 중간 저장 중...")
    save_current_progress(state)
    
    # store last predictions to guide improvement
    print(f"🔍 평가 결과 predictions_runs 길이: {len(eval_result['predictions_runs'])}")
    state["_last_predictions_runs"] = eval_result["predictions_runs"]
    return state


def improve_node(state: OptimizerState, improve_llm: ChatOpenAI) -> OptimizerState:
    print(f"\n🔧 === 개선 단계 ===")
    preds = state.get("_last_predictions_runs", [])  # type: ignore[assignment]
    print(f"🔍 디버깅: predictions_runs 길이={len(preds)}")
    if preds:
        print(f"🔍 첫 번째 run 예측: {preds[0]}")
        samples = state["samples"]
        print(f"🔍 실제 라벨: {[str(s['label']).strip() for s in samples]}")
    
    errors = aggregate_errors(state["samples"], preds) if preds else []
    print(f"❌ 분석된 오류 케이스: {len(errors)}개")
    
    if errors:
        print("📋 오류 케이스 상세:")
        for i, err in enumerate(errors[:3]):  # 처음 3개만 출력
            print(f"  {i+1}. 제목: {err['title'][:50]}...")
            print(f"     예측: {err['pred_votes']}, 정답: {err['label']}")
    
    print("🤖 프롬프트 개선 중...")
    improved = improve_prompt_with_llm(
        llm=improve_llm,
        current_prompt=state["prompt"],
        best_prompt=state.get("best_prompt", state["prompt"]),
        best_score=float(state.get("best_score", 0.0)),
        errors=errors,
    )
    
    print(f"✏️ 프롬프트 개선 완료 (길이: {len(state['prompt'])} → {len(improved)})")
    print(f"\n📝 === 개선된 프롬프트 ===")
    print(improved)
    print("=" * 50)
    
    state["prompt"] = improved
    state["iter"] = int(state.get("iter", 0)) + 1
    return state


def should_continue(state: OptimizerState) -> Literal["improve", "evaluate", "__end__"]:
    # Check if last evaluation was perfect (accuracy = 1.0)
    attempts = state.get("attempts", [])
    if attempts:
        last_attempt = attempts[-1]
        last_accuracy = last_attempt.get("accuracy", 0.0)
        if last_accuracy >= 1.0:
            print("🎯 완벽한 정확도 달성! 개선 건너뛰고 다음 평가로...")
            # Increment iter and go to next evaluation
            state["iter"] = int(state.get("iter", 0)) + 1
            # Check if we've reached max iterations after incrementing
            if int(state.get("iter", 0)) >= int(state.get("max_iters", 5)):
                print("✅ 최대 반복 횟수 도달, 종료")
                return END
            return "evaluate"
    
    # Normal case: check iterations before improvement
    if int(state.get("iter", 0)) >= int(state.get("max_iters", 5)):
        return END
    return "improve"


def save_history(history_path: str, run_record: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    existing: list[dict[str, Any]] = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(run_record)
    # Sort by best_score descending for convenience
    existing.sort(key=lambda x: x.get("best_score", 0.0), reverse=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def save_current_progress(state: OptimizerState) -> None:
    """Save current progress during execution"""
    run_id = int(time.time())
    out_dir = os.path.join(os.path.dirname(__file__), "experiments")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save intermediate record
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_id)),
        "status": "in_progress",
        "current_iteration": state.get("iter", 0),
        "best_score": state.get("best_score"),
        "best_prompt": state.get("best_prompt"),
        "attempts": state.get("attempts", []),
        "sample_size": 10,
        "eval_runs": 1,
        "eval_model": "gpt-4o-mini",
        "improve_model": "gpt-5",
    }
    
    # Save to intermediate file
    progress_path = os.path.join(out_dir, "current_progress.json")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_best_prompt_from_history(history_path: str) -> str | None:
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expecting a list of run records
        if isinstance(data, list) and data:
            # Get run with max best_score
            best_run = max(
                (r for r in data if isinstance(r, dict)),
                key=lambda r: r.get("best_score", float("-inf")),
            )
            bp = best_run.get("best_prompt")
            if isinstance(bp, str) and bp.strip():
                return bp
            # Fallback: scan attempts in best_run
            attempts = best_run.get("attempts", []) if isinstance(best_run, dict) else []
            best_attempt = None
            best_score = float("-inf")
            for a in attempts:
                if not isinstance(a, dict):
                    continue
                s = a.get("score", None)
                p = a.get("prompt", None)
                if isinstance(s, (int, float)) and isinstance(p, str) and s > best_score:
                    best_score = float(s)
                    best_attempt = p
            if best_attempt:
                return best_attempt
        return None
    except Exception:
        return None


def main() -> None:
    print("🎯 === 프롬프트 최적화 에이전트 시작 ===\n")
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    print("🤖 모델 초기화 중...")
    # Evaluation model: gpt-4o-mini for classification
    eval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, openai_api_key=api_key)
    # Improvement model: gpt-5 for prompt optimization
    improve_llm = ChatOpenAI(model="gpt-5", temperature=0.2, openai_api_key=api_key)
    print("✅ 평가용: gpt-4o-mini, 개선용: gpt-5")

    # Samples will be generated fresh for each iteration

    # Build LangGraph
    def _evaluate(state: OptimizerState) -> OptimizerState:
        return evaluate_node(state, eval_llm)

    def _improve(state: OptimizerState) -> OptimizerState:
        return improve_node(state, improve_llm)

    builder = StateGraph(OptimizerState)
    builder.add_node("evaluate", _evaluate)
    builder.add_node("improve", _improve)
    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", should_continue)
    builder.add_edge("improve", "evaluate")
    graph = builder.compile()

    # Load highest-scoring prompt from history if available
    out_dir = os.path.join(os.path.dirname(__file__), "experiments")
    history_path = os.path.join(out_dir, "prompts_history.json")
    print(f"📚 이전 실험 기록 확인 중: {history_path}")
    best_from_history = load_best_prompt_from_history(history_path)
    if best_from_history:
        print("✅ 이전 최고 점수 프롬프트 발견, 이를 시작점으로 사용")
    else:
        print("📝 기록 없음, 기본 프롬프트로 시작")
    initial_prompt = best_from_history or DEFAULT_BASE_PROMPT
    init_state: OptimizerState = {
        "prompt": initial_prompt,
        "best_prompt": initial_prompt,
        "best_score": -1.0,
        "attempts": [],
        "iter": 0,
        "max_iters": 5,  # 5 improvements per run
        "samples": [],  # Will be populated in each evaluation
    }

    # Run graph
    print(f"\n🏁 LangGraph 최적화 시작 (최대 {init_state['max_iters']}회 반복)")
    final_state = graph.invoke(init_state)

    # Prepare final run record
    run_id = int(time.time())
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_id)),
        "status": "completed",
        "best_score": final_state.get("best_score"),
        "best_prompt": final_state.get("best_prompt"),
        "attempts": final_state.get("attempts", []),
        "sample_indices_seed": 42,
        "sample_size": 10,
        "eval_runs": 1,
        "eval_model": "gpt-4o-mini",
        "improve_model": "gpt-5",
    }

    # Save final artifacts
    print(f"\n💾 최종 결과 저장 중...")
    os.makedirs(out_dir, exist_ok=True)
    save_history(history_path, record)
    
    # Clean up intermediate progress file
    progress_path = os.path.join(out_dir, "current_progress.json")
    if os.path.exists(progress_path):
        os.remove(progress_path)
        print("🧹 중간 진행 파일 정리 완료")

    best_prompt_path = os.path.join(out_dir, f"best_prompt_{run_id}.txt")
    with open(best_prompt_path, "w", encoding="utf-8") as f:
        f.write(str(final_state.get("best_prompt", "")))

    # Console summary
    print("\n🎉 === 프롬프트 최적화 완료 ===")
    print(f"🏆 최고 점수: {final_state.get('best_score'):.4f}")
    print(f"🔄 총 시도: {len(final_state.get('attempts', []))}회")
    print(f"📂 히스토리: {history_path}")
    print(f"📄 최고 프롬프트: {best_prompt_path}")
    print("\n완료! 🎯")


if __name__ == "__main__":
    main()


