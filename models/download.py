import gdown
# find the links of pretrained diffusion ckpts in DPS code base and replace the below url with the download link
url = ''
output_path = 'ffhg_10m.pt'
gdown.download(url, output_path, quiet=False,fuzzy=True)