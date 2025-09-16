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


def load_dataset(
    csv_path: str, sample_size: int = SAMPLE_SIZE, seed: int | None = None
) -> list[dict[str, Any]]:
    print(f"📊 CSV 파일 로드 중: {csv_path}")
    # Handle complex CSV with multiline content
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    print(f"✅ CSV 로드 완료: {len(df)}개 행 발견")

    if len(df) < sample_size:
        raise ValueError(
            f"Dataset has only {len(df)} rows, but sample_size={sample_size}"
        )

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
        rows.append(
            {
                "title": str(r.get("title", "")),
                "content": str(r.get("content", "")),
                "label": str(r.get("label", "")).strip(),
            }
        )
    print(f"✅ 샘플 준비 완료: {len(rows)}개")
    return rows


# DEFAULT_BASE_PROMPT = """
# 당신은 매우 엄격하고 정확한 텍스트 분류기입니다. 당신의 유일한 임무는 주어진 기사의 핵심 주제가 '자동차' 자체(예: 신차 출시, 자동차 기술, 자동차 산업 동향)인지 아닌지를 판단하는 것입니다. 사소한 언급이나 간접적인 연관성은 전부 무시해야 합니다. 당신의 판단은 오직 '1' 또는 '0'으로만 출력되어야 합니다. 다른 어떤 단어나 설명도 추가해서는 안 됩니다.

# 아래 규칙을 반드시 따르십시오.
# [규칙]
# 1. 핵심 주제가 자동차인가? 기사의 제목과 본문 전체를 고려했을 때, 핵심 주제가 명백하게 자동차, 자동차 산업, 자동차 기술, 신차 리뷰 등이어야만 '1'입니다.
# 2. 자동차의 단순 언급은 '0'입니다. 교통사고, 교통 체증, 특정 인물이 차를 탔다는 내용, 배경에 자동차가 등장하는 사건 등은 자동차가 핵심 주제가 아니므로 '0'입니다.
# 3. 주제가 모호하면 '0'입니다. 자동차 외에 다른 주제(예: 경제, 기술, 사회)와 동등한 비중으로 다뤄지거나, 자동차가 단지 예시로 사용된 경우 무조건 '0'입니다.
# 4. 출력은 오직 '1' 또는 '0'입니다. 어떠한 경우에도 다른 설명, 문장, 주석을 포함해서는 안 됩니다.
#     """.strip()
DEFAULT_BASE_PROMPT = """
System: 당신은 매우 엄격하고 정확한 텍스트 분류기입니다. 당신의 유일한 임무는 주어진 기사의 핵심 주제가 '자동차' 자체(예: 신차 출시, 자동차 기술, 자동차 산업 동향)인지 아닌지를 판단하는 것입니다. 사소한 언급이나 간접적인 연관성은 전부 무시해야 합니다. 당신의 판단은 오직 '1' 또는 '0'으로만 출력되어야 합니다. 다른 어떤 단어나 설명도 추가해서는 안 됩니다.

User: 아래 규칙을 반드시 따르십시오.

[규칙]

핵심 주제가 자동차인가?: 기사의 제목과 본문 전체를 고려했을 때, 핵심 주제가 명백하게 자동차, 자동차 시장, 자동차 산업, 자동차와 직접적 관련이 있는 기술(예: 자율주행, 전기차 배터리 등), 신차 리뷰, 자동차 생산 등이어야만 '1'입니다.

자동차의 단순 언급은 '0'입니다: 교통사고, 교통 체증, 특정 인물이 차를 탔다는 내용, 배경에 자동차가 등장하는 사건, 자동차를 포함한 관세에 대한 내용 등은 자동차가 핵심 주제가 아니므로 '0'입니다.

주제가 모호하면 '0'입니다: 자동차 외에 다른 주제(예: 경제, 기술, 사회)와 동등한 비중으로 다뤄지거나, 자동차가 단지 예시로 사용된 경우 무조건 '0'입니다. 예시: 미래 모빌리티 시장에 대한 기사

출력은 오직 '1' 또는 '0'입니다: 어떠한 경우에도 다른 설명, 문장, 주석을 포함해서는 안 됩니다.

[분류 예시]
[긍정 예시 (자동차 기사 = 1)]

입력: "현대자동차가 전기차 라인업을 강화하기 위해 새로운 SUV 모델 '아이오닉 7'의 티저 이미지를 공개했다. 이 모델은 한 번 충전으로 600km 주행을 목표로 한다."

출력: 1

입력: "기아가 4세대 카니발의 페이스리프트 모델을 출시했다. 새로운 디자인과 하이브리드 파워트레인을 추가하여 상품성을 대폭 개선한 것이 특징이다."

출력: 1

입력: "최근 자동차 업계에서는 자율주행 기술의 레벨 3 상용화를 두고 치열한 경쟁이 벌어지고 있다. 특히 라이다 센서와 카메라 기술의 융합이 핵심 과제로 떠올랐다."

출력: 1

[부정 예시 (자동차 기사가 아님 = 0)]

입력: "오늘 새벽 강변북로에서 3중 추돌 사고가 발생하여 출근길 극심한 정체가 빚어졌다. 경찰은 운전자들을 상대로 정확한 사고 원인을 조사 중이다."

출력: 0

입력: "유명 배우 A씨가 자신의 SNS에 최근 구매한 슈퍼카 사진을 올려 화제가 되고 있다."

출력: 0

입력: "정부가 다가오는 휴가철을 맞아 고속도로 통행료를 한시적으로 면제하는 방안을 검토하고 있다고 밝혔다."

출력: 0

입력: "LG전자가 미래 성장 동력으로 전장 사업을 낙점하고, 차량용 인포테인먼트 시스템 공급을 확대하고 있다."

출력: 0

[이제 당신이 분류할 차례입니다. 규칙을 엄격하게 적용하여 '1' 또는 '0'으로만 답하십시오. THINK STEP BY STEP]
    """.strip()


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
            print(
                f"    샘플 {i+1}: 예측={yhat}, 정답={row['label']}, {'✅' if yhat == str(row['label']).strip() else '❌'}"
            )
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


def aggregate_errors(
    samples: list[dict[str, Any]], predictions_runs: list[list[str]]
) -> list[dict[str, Any]]:
    # Collect misclassified cases across runs (use majority mistake indicator)
    errors: list[dict[str, Any]] = []
    n = len(samples)
    runs = len(predictions_runs)
    for i in range(n):
        votes = [predictions_runs[r][i] for r in range(runs)]
        # If any run misclassified, include for analysis
        gold = str(samples[i]["label"]).strip()
        if any(v != gold for v in votes):
            errors.append(
                {
                    "title": samples[i]["title"],
                    "content": samples[i]["content"],
                    "label": gold,
                    "pred_votes": votes,
                }
            )
    # Limit to keep the improve prompt short
    return errors[:8]


def analyze_prompt_history(history_path: str) -> dict[str, Any]:
    """분석용으로 상위/하위 점수 프롬프트들과 패턴을 추출한다."""
    if not os.path.exists(history_path):
        return {"top_prompts": [], "bottom_prompts": []}

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not isinstance(history, list) or len(history) == 0:
            return {"top_prompts": [], "bottom_prompts": []}

        # 모든 시도(attempts)를 수집하여 점수별 정렬
        all_attempts: list[dict[str, Any]] = []
        for run in history:
            if isinstance(run, dict) and "attempts" in run:
                attempts = run.get("attempts", [])
                for attempt in attempts:
                    if (
                        isinstance(attempt, dict)
                        and "score" in attempt
                        and "prompt" in attempt
                    ):
                        all_attempts.append(
                            {
                                "score": float(attempt["score"]),
                                "accuracy": float(attempt.get("accuracy", 0)),
                                "prompt": str(attempt["prompt"]),
                                "iteration": attempt.get("iteration", 0),
                                "run_id": run.get("run_id", "unknown"),
                            }
                        )

        if len(all_attempts) == 0:
            return {"top_prompts": [], "bottom_prompts": []}

        # 점수별 정렬
        all_attempts.sort(key=lambda x: x["score"], reverse=True)

        # 상위 3개, 하위 3개 추출
        top_3 = all_attempts[:3]
        bottom_3 = all_attempts[-3:] if len(all_attempts) >= 3 else []

        return {
            "top_prompts": top_3,
            "bottom_prompts": bottom_3,
        }

    except Exception as e:
        print(f"⚠️ 히스토리 분석 중 오류: {e}")
        return {"top_prompts": [], "bottom_prompts": []}


def improve_prompt_with_llm(
    llm: ChatOpenAI,
    current_prompt: str,
    best_prompt: str,
    best_score: float,
    errors: list[dict[str, Any]],
    history_path: str | None = None,
) -> str:
    # 히스토리 분석 수행
    history_analysis = {"top_prompts": [], "bottom_prompts": []}
    if history_path and os.path.exists(history_path):
        history_analysis = analyze_prompt_history(history_path)
        print(
            f"📊 히스토리 분석 완료: 상위 {len(history_analysis['top_prompts'])}개, 하위 {len(history_analysis['bottom_prompts'])}개 프롬프트 분석"
        )

    sys = SystemMessage(
        content=(
            "당신은 텍스트 분류 태스크 전문 프롬프트 엔지니어입니다. 이전 실험 히스토리를 분석하여 프롬프트를 전략적으로 개선해주세요.\n\n"
            "**최적화 목표:**\n"
            "- 점수 공식: score = 0.9 * accuracy + 0.1 * sqrt(max(0, 1 - (L/3000)^2))\n"
            "- 정확도(accuracy) 향상이 최우선, 프롬프트 길이는 2차적 고려사항\n"
            "- 반드시 '출력은 오직 0 또는 1' 규율을 유지해야 함\n\n"
            "**히스토리 기반 학습 전략:**\n"
            "1. **성공 패턴 학습**: 상위 점수 프롬프트들의 공통 특징을 식별하고 활용\n"
            "   - 어떤 키워드, 구조, 규칙이 높은 정확도를 이끌어냈는지 분석\n"
            "   - 성공한 분류 기준의 명확성과 구체성 수준 파악\n"
            "2. **실패 패턴 회피**: 하위 점수 프롬프트들의 문제점을 식별하고 개선\n"
            "   - 애매한 표현, 과도한 복잡성, 불분명한 경계 등 문제 요소 제거\n"
            "   - 오분류를 유발한 규칙의 모호성이나 누락 파악\n"
            "3. **점진적 개선**: 현재 프롬프트의 약점을 타겟팅하여 정밀 개선\n\n"
            "**개선 우선순위:**\n"
            "1. 오분류 사례 분석을 통한 즉시 수정 가능한 규칙 보완\n"
            "2. 이전 고득점 프롬프트의 성공 요소 적용\n"
            "3. 경계 케이스에 대한 명확한 판정 기준 제시\n"
            "4. 불필요한 중복이나 장황함 제거\n"
            "5. 일관성 있는 용어와 구조 사용\n\n"
            "**분석 관점:**\n"
            "- **정확도 차이**: 상위와 하위 프롬프트 간 정확도 차이의 핵심 원인\n"
            "- **길이 효율성**: 비슷한 길이 대비 더 높은 정확도를 달성한 패턴\n"
            "- **규칙 구체성**: 추상적 vs 구체적 규칙의 효과 차이\n"
            "- **키워드 효과**: 특정 키워드나 표현의 분류 성능 영향\n\n"
            "**응답 형식:**\n"
            "JSON 형식으로만 응답하며, 다음 키들을 포함하세요:\n"
            "```json\n"
            "{\n"
            '  "improved_prompt": "개선된 프롬프트 전문",\n'
            '  "improvement_reasoning": "히스토리 분석을 바탕으로 한 개선 근거 (구체적 변경사항과 이유)",\n'
            '  "expected_score_change": "예상 점수 변화 (예: +0.05)",\n'
            '  "key_changes": ["주요 변경사항 1", "주요 변경사항 2", "주요 변경사항 3"]\n'
            "}\n"
            "```\n\n"
            "**중요 주의사항:**\n"
            "- 이전 실험 데이터가 있다면 반드시 활용하여 개선 방향 결정\n"
            "- 과적합이 안 되도록 주의\n"
            "- 성공한 프롬프트의 구조나 표현을 참고하되, 맹목적 복사는 지양\n"
            "- 자동차 분류 태스크의 특성상 '직접성'과 '구체성'이 핵심 성공 요인"
        )
    )

    # 분석 데이터 구성
    analysis_data = {
        "current_prompt": current_prompt,
        "best_prompt": best_prompt,
        "best_score": round(best_score, 5),
        "recent_errors": errors,
    }

    # 히스토리 분석 결과 추가
    if history_analysis["top_prompts"]:
        analysis_data["top_scoring_prompts"] = [
            {
                "score": p["score"],
                "accuracy": p["accuracy"],
                "prompt_preview": p["prompt"][:200] + "...",
            }
            for p in history_analysis["top_prompts"]
        ]

    if history_analysis["bottom_prompts"]:
        analysis_data["low_scoring_prompts"] = [
            {
                "score": p["score"],
                "accuracy": p["accuracy"],
                "prompt_preview": p["prompt"][:200] + "...",
            }
            for p in history_analysis["bottom_prompts"]
        ]

    user = HumanMessage(content=json.dumps(analysis_data, ensure_ascii=False, indent=2))

    try:
        res = llm.invoke([sys, user])
        text = res.content if isinstance(res, AIMessage) else "{}"

        # JSON 파싱 시도
        try:
            obj = json.loads(text)
            improved = str(obj.get("improved_prompt", "")).strip()
            reasoning = str(obj.get("improvement_reasoning", "")).strip()
            expected_change = obj.get("expected_score_change", "")
            key_changes = obj.get("key_changes", [])

            if improved:
                print(f"🎯 개선 근거: {reasoning}")
                if expected_change:
                    print(f"📈 예상 점수 변화: {expected_change}")
                if key_changes:
                    print(f"🔑 주요 변경사항: {', '.join(key_changes)}")
                return improved
        except json.JSONDecodeError:
            print("⚠️ LLM 응답 JSON 파싱 실패, 휴리스틱 개선 적용")

    except Exception as e:
        print(f"⚠️ LLM 호출 오류: {e}, 휴리스틱 개선 적용")

    # Fallback heuristic: append a brief refinement based on errors
    fallback_improvements = []

    # 오류 패턴 기반 휴리스틱 개선
    if errors:
        error_titles = [err.get("title", "") for err in errors]
        error_patterns = []

        # 관세/무역 관련 오류가 많으면
        if any(
            "관세" in title or "무역" in title or "협상" in title
            for title in error_titles
        ):
            error_patterns.append("관세·무역·협상 중심 기사는 자동차가 언급되어도 0")

        # 모빌리티 플랫폼 관련 오류가 많으면
        if any(
            "모빌리티" in title or "플랫폼" in title or "서비스" in title
            for title in error_titles
        ):
            error_patterns.append(
                "모빌리티 플랫폼·서비스 논의는 차량 자체가 핵심이 아니므로 0"
            )

        # 기술 관련 오류가 많으면
        if any(
            "기술" in title or "개발" in title or "AI" in title
            for title in error_titles
        ):
            error_patterns.append(
                "범용 기술 개발은 자동차 적용이 명확히 언급되지 않으면 0"
            )

        if error_patterns:
            fallback_improvements.extend(error_patterns)

    # 기본 개선사항
    if not fallback_improvements:
        fallback_improvements.append("모호한 경계 케이스는 무조건 0으로 분류")

    return (
        current_prompt
        + "\n\n[추가 정제 규칙]\n"
        + "\n".join(f"- {improvement}" for improvement in fallback_improvements)
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

    # 히스토리 경로 설정
    out_dir = os.path.join(os.path.dirname(__file__), "experiments")
    history_path = os.path.join(out_dir, "prompts_history.json")

    improved = improve_prompt_with_llm(
        llm=improve_llm,
        current_prompt=state["prompt"],
        best_prompt=state.get("best_prompt", state["prompt"]),
        best_score=float(state.get("best_score", 0.0)),
        errors=errors,
        history_path=history_path,
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
        "sample_size": SAMPLE_SIZE,
        "eval_runs": 1,
        "eval_model": "gpt-4o-mini",
        "improve_model": "gpt-5",
    }

    # Save to intermediate file
    progress_path = os.path.join(out_dir, "current_progress.json")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def load_best_prompt_from_history(history_path: str) -> tuple[str | None, float]:
    """히스토리에서 최고 점수 프롬프트와 점수를 반환한다."""
    if not os.path.exists(history_path):
        return None, -1.0
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
            bs = best_run.get("best_score", -1.0)
            if isinstance(bp, str) and bp.strip():
                return bp, float(bs)
            # Fallback: scan attempts in best_run
            attempts = (
                best_run.get("attempts", []) if isinstance(best_run, dict) else []
            )
            best_attempt = None
            best_score = float("-inf")
            for a in attempts:
                if not isinstance(a, dict):
                    continue
                s = a.get("score", None)
                p = a.get("prompt", None)
                if (
                    isinstance(s, (int, float))
                    and isinstance(p, str)
                    and s > best_score
                ):
                    best_score = float(s)
                    best_attempt = p
            if best_attempt:
                return best_attempt, best_score
        return None, -1.0
    except Exception:
        return None, -1.0


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
    best_from_history, best_score_from_history = load_best_prompt_from_history(
        history_path
    )
    if best_from_history:
        print(
            f"✅ 이전 최고 점수 프롬프트 발견 (점수: {best_score_from_history:.4f}), 이를 시작점으로 사용"
        )
    else:
        print("📝 기록 없음, 기본 프롬프트로 시작")
    initial_prompt = best_from_history or DEFAULT_BASE_PROMPT
    initial_best_score = best_score_from_history if best_from_history else -1.0
    init_state: OptimizerState = {
        "prompt": initial_prompt,
        "best_prompt": initial_prompt,
        "best_score": initial_best_score,
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
        "sample_size": SAMPLE_SIZE,
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
