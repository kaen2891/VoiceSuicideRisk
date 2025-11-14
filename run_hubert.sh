#!/bin/bash
# ===========================================
#  Run all 5 folds × 4 task variants
#  (Run from project root directory)
# ===========================================

echo "======================================="
echo "     Starting 5-Fold × 4-Task Runs     "
echo "======================================="

# Fold loop
for fold in {3..5}
do
    echo ""
    echo "==============================="
    echo "▶ Running Fold ${fold}"
    echo "==============================="

    # Task variants
    for task in "" "_color" "_incongruent" "_word"
    do
        # 쉘 파일 경로 설정
        sh_file="./script/hubert_5sec_30sec_fold${fold}${task}.sh"

        if [ -f "$sh_file" ]; then
            echo "▶ Executing: $sh_file"
            bash "$sh_file"

            # 로그를 남기고 싶다면 아래 줄로 대체하세요:
            # bash "$sh_file" > "./logs/fold${fold}${task}.log" 2>&1

            echo "✅ Finished: $sh_file"
            echo "---------------------------------------"
        else
            echo "❌ File not found: $sh_file"
        fi
    done
done

echo ""
echo "🎯 All folds and tasks completed!"
echo "======================================="