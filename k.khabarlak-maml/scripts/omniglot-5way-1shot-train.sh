python ../src/train.py ../datasets/omniglot/ \
    --run-name 01-omniglot-5way-1shot \
    --dataset omniglot \
    --num-ways 5 \
    --num-shots 1 \
    --num-steps 1 \
    --step-size 0.4 \
    --no-max-pool \
    --batch-size 32 \
    --num-workers 8 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ../results