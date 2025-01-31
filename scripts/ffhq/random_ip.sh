gpu=0
model_dir="configs/model_config.yaml"
diffusion_dir="configs/diffusion_config.yaml"
xy=0
guidance12=0
data_size=100
p=0.285
task_dir="configs/inpainting_config_random.yaml"
data_name="ffhq"
scale_list=(3.1)
sub_scale_list=(19.0)
noise=0.05
for scale in ${scale_list[@]};do
    for sub_scale in ${sub_scale_list[@]};do
        python3 sample_condition.py --model_config=${model_dir} --diffusion_config=${diffusion_dir} --task_config=${task_dir} --scale ${scale} --sub_scale ${sub_scale} --xy ${xy} --guidance12 ${guidance12} --gpu ${gpu} --data_size ${data_size} --noise ${noise} --data_name ${data_name} --p ${p}
    done
done