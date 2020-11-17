python ../src/train.py ../datasets/omniglot/ \
    --dataset omniglot \
    --num-ways 20 \
    --num-shots 1 \
    --num-steps 5 \
    --step-size 0.1 \
    --no-max-pool \
    --batch-size 6 \
    --num-workers 4 \
    --num-epochs 600 \
    --use-cuda \
    --output-folder ../results