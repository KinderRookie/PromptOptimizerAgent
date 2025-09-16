# README
프롬프트 자동 개선을 위한 프로젝트

## 저장소 복사 방법
```bash
git clone https://github.com/KinderRookie/PromptOptimizerAgent.git
cd PromptOptimizerAgent
```

- 코드 변경 사항이 있는 경우에는 항상 commit message를 작성하도록 합니다. 
- README.md 하단 영역에 제거한 기사 id와 제목을 간단히 기록합니다. 

samples.csv 파일에 원본 기사들이 들어있습니다.
final.csv 파일에는 증강된 기사들이 들어있습니다.
./experiments 폴더에 실험 결과가 쌓이게 됩니다. 새로 시작하고 싶다면 지우고 시작해주세요. 

## 구동 방법
```bash
# 구동에 필요한 OPENAI_API_KEY를 .env 파일에 넣어준다
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 값을 넣어준다
# 띄어쓰기 없이 넣어야 한다

# uv로 환경 설정을 해준다
uv sync

# 가상 환경 venv를 활성화 한다
source .venv/bin/activate

# 스크립트를 실행한다.
python prompt_optimizer.py
```

./experiments에 결과가 쌓이게 된다.

현재: 샘플링 20개
개선 모델: gpt-5

## 제거한 기사 목록
2025-09-17:00:19:23
- 없음
