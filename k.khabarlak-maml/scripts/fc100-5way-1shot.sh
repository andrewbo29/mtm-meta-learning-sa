python ../src/train.py ../datasets/ \
    --run-name 13-fc100-5way-1shot \
    --dataset fc100 \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.01 \
    --hidden-size 32 \
    --batch-size 4 \
    --num-workers 8 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ../results

python ../src/test.py ../results/13-fc100-5way-1shot/config.json --num-steps 10 --use-cuda