from functools import partial
import os
import argparse
import yaml
from datasets import load_from_disk
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms as T
from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
import numpy as np
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_size', type=int, default=30)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--scale', type=float, default=0.8)
    parser.add_argument('--sub_scale', type=float, default=4.0)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--xy', type=int, default=0)
    parser.add_argument('--guidance12', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='ffhq')
    parser.add_argument('--data_dir', type=str, default='data/ffhq')
    parser.add_argument('--p', type=float, default='0.5')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    task_config['conditioning']['params']['scale']=args.scale
    task_config['conditioning']['params']['sub_scale']=args.sub_scale
    print('model config:',model_config,'\n')
    print('diffusion config:',diffusion_config,'\n')
    print('task config:',task_config,'\n')

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    if measure_config['noise']['name']!='poisson':
        measure_config['noise']['sigma']=args.noise
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()
    # model for y
    model1=model
    # Prepare conditioning method
    cond_config = task_config['conditioning']
    if measure_config['operator'] ['name'] in {'super_resolution'}:
        condx='ps_x_sr'
        condy='ps_y_sr'
    elif measure_config['operator']['name'] == 'inpainting':
        condx='ps_x_ip'
        condy='ps_y_ip'
    else:
        condx='ps_x'
        condy='ps_y'
    cond_method_x = get_conditioning_method(condx, operator, noiser,p=args.p, **cond_config['params'])
    cond_method_y = get_conditioning_method(condy, operator, noiser, **cond_config['params'])

    measurement_cond_fn_x = cond_method_x.conditioning
    if (args.guidance12==0):
        measurement_cond_fn_y = cond_method_y.conditioning_ind
        print('2 guidance related : no')
    elif (args.guidance12==1):
        measurement_cond_fn_y = cond_method_y.conditioning_d
        print('2 guidance related : yes')
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    
    
    if measure_config['noise']['name']=='poisson':
        if(task_config['measurement']['operator']['name']=='super_resolution'):
            args.save_dir='./res_{t}_poisson/{s}_{no}_Ytransform_measurementXT/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_{p}_m1'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],no=args.noise,s2=task_config['conditioning']['params']['sub_scale'],p=args.p)
        else:
            args.save_dir='./res_{t}_poisson/{s}_{no}/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_{p}_m1'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],no=args.noise,s2=task_config['conditioning']['params']['sub_scale'],p=args.p)
    else:
        if (task_config['measurement']['operator']['name']=='inpainting'):
            if(task_config['measurement']['mask_opt']['mask_type']=='box'): 
                args.save_dir='./res_{t}/{s}_{i}_{r}_{no}/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_55_m1_measurementXT'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],i=task_config['measurement']['mask_opt']['mask_type'],r=task_config['measurement']['mask_opt']['mask_len_range'][0],no=args.noise,n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],s2=task_config['conditioning']['params']['sub_scale'])
            elif(task_config['measurement']['mask_opt']['mask_type']=='random'): 
                args.save_dir='./res_{t}/{s}_{i}_{r}_{no}/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_08scale_m1_measurementXT_noAonY'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],i=task_config['measurement']['mask_opt']['mask_type'],r=task_config['measurement']['mask_opt']['mask_prob_range'][0],no=args.noise,n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],s2=task_config['conditioning']['params']['sub_scale'])
        elif(task_config['measurement']['operator']['name']=='super_resolution'):
            args.save_dir='./res_{t}/{s}_{no}_Ytransform_measurementXT/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_{p}_m1'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],no=args.noise,s2=task_config['conditioning']['params']['sub_scale'],p=args.p)
        else:
            args.save_dir='./res_{t}/{s}_{no}/{t}_{n}/xy_{xy}_guide12_{g12}/s_{s1}_subs_{s2}_n1m1_{p}_m1'.format(t=args.data_name,s=task_config['measurement']['operator']['name'],n=args.data_size,xy=args.xy,g12=args.guidance12,s1=task_config['conditioning']['params']['scale'],no=args.noise,s2=task_config['conditioning']['params']['sub_scale'],p=args.p)
    

    print('saving directory:', args.save_dir)
    out_path=args.save_dir

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if (args.data_name=='ffhq'):
        dataset = load_from_disk("data/ffhq_{s}".format(s=args.data_size))['test']
        dataset = get_dataset(name='ffhq',data=dataset['image'], transforms=transform)
        loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    elif(args.data_name=='imagenet'):
        data = np.load('data/VIRTUAL_imagenet256_labeled.npz')
        images = data['arr_0'][0:args.data_size]
        from PIL import Image
        transformed_images = []
        for img in images:
            pil_img = Image.fromarray(img)
            tensor_img = transform(pil_img)
            transformed_images.append(tensor_img)
        loader = get_dataloader(transformed_images, batch_size=1, num_workers=0, train=False)

    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn_x = partial(cond_method_x.conditioning, mask=mask)
            if (args.guidance12==0):
                    measurement_cond_fn_y = partial(cond_method_y.conditioning_ind, mask=mask)
                    print('2 guidance related : no')
            elif (args.guidance12==1):
                    measurement_cond_fn_y = partial(cond_method_y.conditioning_d, mask=mask)
                    print('2 guidance related : yes')
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)
        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)

        #y not uses updated x
        if (args.xy==0):
            sample_fn = partial(sampler.p_sample_loop_xyind, model=model, measurement_cond_fn_x=measurement_cond_fn_x,measurement_cond_fn_y=measurement_cond_fn_y,task=measure_config['operator'] ['name'],model1=model1)
            print('uses updated x for y: no')
        #y uses updated x
        elif (args.xy==1):
            sample_fn = partial(sampler.p_sample_loop_xyd, model=model, measurement_cond_fn_x=measurement_cond_fn_x,measurement_cond_fn_y=measurement_cond_fn_y)
            print('uses updated x for y: yes')
        
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        y_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        sample = sample_fn(x_start=x_start,y_start=y_start,measurement=y_n.float(), record=True, save_root=out_path)
        sample = (sample/2+0.5)
        os.makedirs(out_path, exist_ok=True)
        for img_dir in ['input', 'recon', 'progress', 'label']:
            os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

if __name__ == '__main__':
    main()
