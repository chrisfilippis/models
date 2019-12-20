set -e

cd 'C:\Projects\tf_models\models\research'
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"
echo "${WORK_DIR}"

python "${WORK_DIR}"/test.py

# python "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --train_split="trainval" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --train_crop_size="1024,768" \
#   --train_batch_size=4 \
#   --training_number_of_steps="10" \
#   --fine_tune_batch_norm=true \
#   --tf_initial_checkpoint="C:\Projects\tf_models\models\research\deeplab\datasets\tooth\init_models\deeplabv3_pascal_train_aug\model.ckpt" \
#   --train_logdir="C:\Projects\tf_models\models\research\deeplab\datasets\tooth\exp\train_on_trainval_set\train" \
#   --dataset_dir="C:\Projects\tf_models\models\research\deeplab\datasets\tooth\output\tfrecord"