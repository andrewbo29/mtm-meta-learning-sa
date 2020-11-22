python ../src/train.py ../datasets/ \
    --run-name 06-cifarfs-2way-5shot \
    --dataset cifarfs \
    --num-ways 2 \
    --num-shots 5 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 8 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ../results

python ../src/test.py ../results/06-cifarfs-2way-5shot/config.json --num-steps 10 --use-cuda