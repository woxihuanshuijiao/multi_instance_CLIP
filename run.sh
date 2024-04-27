
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node 6 /home/wangyuhan/clip/open_clip/src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="/home/wangyuhan/clip/open_clip/1m.csv"  \
    --csv-img-key img \
    --csv-caption-key caption \
    --warmup 500 \
    --batch-size=2 \
    --pretrained=openai \
    --lr=5e-5 \
    --wd=0.1 \
    --epochs=6 \
    --workers=8 \
    --accum-freq 8 \
    --model ViT-L-14-336 \
   --resume /home/wangyuhan/open_clip-main_lab/weights/epoch_1_transfered.pt


# CUDA_VISIBLE_DEVICES=0 /home/yanshuang/anaconda3/envs/localization/bin/python /home/wangyuhan/clip/open_clip/src/training/main.py \
#     --save-frequency 1 \
#     --zeroshot-frequency 1 \
#     --train-data "/home/wangyuhan/clip/open_clip/1m.csv"  \
#     --csv-img-key img \
#     --csv-caption-key caption \
#     --warmup 5000 \
#     --batch-size=2 \
#     --pretrained=openai \
#     --lr=1e-4 \
#     --wd=0.1 \
#     --epochs=30 \
#     --workers=8 \
#     --model ViT-L-14-336 \
#     --resume /home/wangyuhan/open_clip-main_lab/weights/epoch_1_transfered.pt