import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

model_name = 'RealESRGAN_x4plus'
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4

model_path = './Real-ESRGAN/weights/RealESRGAN_x4plus.pth'
dni_weight = None
tile = 0
tile_pad = 10
pre_pad = 0
fp32 = True
gpu_id = None
outscale = 4.0

upsampler = RealESRGANer(
    scale=netscale,
    model_path=model_path,
    dni_weight=dni_weight,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=not fp32,
    gpu_id=gpu_id)

def enhance_image(path, save_folder='./'):
    imgname, extension = os.path.splitext(os.path.basename(path))
    print('Testing', imgname)

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img_mode = 'RGBA'
    else:
        img_mode = None
    
    print ('Running upsampler, image shape:', img.shape)    
    output, _ = upsampler.enhance(img, outscale=outscale)
    
    extension = extension[1:]
    if img_mode == 'RGBA':  # RGBA images should be saved in png format
        extension = 'png'
    
    save_path = os.path.join(save_folder, f'upscaled_{imgname}.{extension}')
    print ('Saving', save_path)
    cv2.imwrite(save_path, output)
    return save_path

if __name__ == '__main__':
    #path = './watch.jpg'
    #output = './'
    #enhance_image(path, output)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, help="path to product image for enhancement")
    parser.add_argument("--save_dir", type=str, default="./", help="folder to save enhanced image that has been scaled up")
    args = parser.parse_args()

    enhance_image(args.image_path, args.save_dir)
