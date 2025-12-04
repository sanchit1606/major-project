import os
# Fix OpenMP runtime conflict on Windows - MUST be set before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os.path as osp
import imageio.v2 as iio
import numpy as np
import argparse

import tigre.algorithms as algs
from src.config.configloading import load_config
from src.evaluator import Evaluator
from src.utils import get_psnr_3d, get_ssim_3d, cast_to_image
from tqdm import tqdm
from pdb import set_trace as stx
import time



def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_github/FDK/chest_50.yaml",
                        help="configs file path")
    parser.add_argument("--algorithm", default="fdk",
                        help="the algorithm to use for reconstruction")
    parser.add_argument("--category", default="chest",
                        help="the category of the tested scene")
    parser.add_argument("--output_path", default=f"output", 
                        help="path to the output folder")
    parser.add_argument("--nview", type=int, default=50,
                        help="the number of views in iterative algs")
    parser.add_argument("--gpu_id", default="1", help="gpu to use")
    return parser

parser = config_parser()
args = parser.parse_args()
cfg = load_config(args.config)

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


class Eval(Evaluator):
    def __init__(self):
        """
        Basic network evaluator.
        """
        super().__init__(cfg, device="cpu")
        print(f"[Start] exp: {cfg['exp']['expname']}")

    def eval_step(self):
        """
        Evaluation step
        """
        # Convert custom ConeGeometry to TIGRE's Geometry format
        from tigre.utilities.geometry import Geometry
        geo = Geometry()
        
        # Copy geometry parameters (convert from meters back to millimeters for TIGRE)
        # Ensure proper array shapes (TIGRE expects specific shapes)
        geo.DSD = float(self.train_dset.geo.DSD * 1000)  # m to mm, ensure scalar
        geo.DSO = float(self.train_dset.geo.DSO * 1000)  # m to mm, ensure scalar
        geo.nDetector = np.array(self.train_dset.geo.nDetector, dtype=np.float64).reshape(2)  # Ensure (2,) shape
        geo.dDetector = np.array(self.train_dset.geo.dDetector * 1000, dtype=np.float64).reshape(2)  # m to mm
        geo.sDetector = np.array(self.train_dset.geo.sDetector * 1000, dtype=np.float64).reshape(2)  # m to mm
        geo.nVoxel = np.array(np.flip(self.train_dset.geo.nVoxel), dtype=np.float64).reshape(3)  # Flip to align with data
        geo.dVoxel = np.array(np.flip(self.train_dset.geo.dVoxel) * 1000, dtype=np.float64).reshape(3)  # m to mm
        geo.sVoxel = np.array(np.flip(self.train_dset.geo.sVoxel) * 1000, dtype=np.float64).reshape(3)  # m to mm
        geo.offOrigin = np.array(self.train_dset.geo.offOrigin * 1000, dtype=np.float64).reshape(3)  # m to mm
        geo.offDetector = np.array(self.train_dset.geo.offDetector * 1000, dtype=np.float64).reshape(2)  # m to mm
        geo.accuracy = self.train_dset.geo.accuracy
        geo.mode = self.train_dset.geo.mode
        geo.filter = self.train_dset.geo.filter
        
        del self.eval_dset.geo, self.train_dset.image
        
        # 3D density
        print("Reconstructing 3D density...")
        # stx()
        view_num = args.nview
        print("The number of projections is %d" % view_num)
        lmbda = 1
        lambdared = 0.999
        initmode = None
        verbose = True

        start_time = time.time()

        if args.algorithm == "fdk":
            image_pred = algs.fdk(self.train_dset.projs.cpu().numpy()[:view_num], 
                                geo, 
                                self.train_dset.angles[:view_num], 
                                filter="ram_lak")
        
        elif args.algorithm == "sart":
            image_pred = algs.sart(self.train_dset.projs.cpu().numpy()[:view_num],
                                geo, 
                                self.train_dset.angles[:view_num],
                                niter=6,
                                lmbda=lmbda,
                                lmbda_red=lambdared,
                                init=initmode,
                                verbose=verbose)
        
        elif args.algorithm == "asd_pocs":
            image_pred = algs.asd_pocs(self.train_dset.projs.cpu().numpy()[:view_num], 
                                geo, 
                                self.train_dset.angles[:view_num],
                                niter=6,
                                lmbda=lmbda,
                                lmbda_red=lambdared,
                                init=initmode,
                                verbose=verbose)
        
        else:
            raise NotImplementedError
        
        end_time = time.time()
        CT_reconstruct_time = end_time - start_time
        
        image = self.eval_dset.image.cpu().numpy()
        image_pred = np.flip(image_pred.transpose(2,1,0), axis=2)

        loss = {
            "psnr_3d": get_psnr_3d(image_pred, image),
            "ssim_3d": get_ssim_3d(image_pred, image),
            "CT_reconstruct_time": CT_reconstruct_time
        }
        print(loss)

            
        # Save
        eval_save_dir = osp.join(args.output_path, args.algorithm, args.category)
        os.makedirs(eval_save_dir, exist_ok=True)
        print("Output path: %s" % eval_save_dir)
        np.save(osp.join(eval_save_dir, "image_pred.npy"), image_pred)
        np.save(osp.join(eval_save_dir, "image_gt.npy"), image)

        with open(osp.join(eval_save_dir, "stats.txt"), "w") as f: 
            for key, value in loss.items():
                if isinstance(value, float):
                    f.write("%s: %f\n" % (key, value))
                else:
                    f.write("%s: %f\n" % (key, value.item()))

        # add different CT cut
        ct_pred_dir_H = osp.join(eval_save_dir, "CT", "H", "ct_pred")
        ct_gt_dir_H = osp.join(eval_save_dir, "CT", "H", "ct_gt")
        ct_pred_dir_W = osp.join(eval_save_dir, "CT", "W", "ct_pred")
        ct_gt_dir_W = osp.join(eval_save_dir, "CT", "W", "ct_gt")
        ct_pred_dir_L = osp.join(eval_save_dir, "CT", "L", "ct_pred")
        ct_gt_dir_L = osp.join(eval_save_dir, "CT", "L", "ct_gt")

        H, W, L = image_pred.shape
        print(image_pred.shape)

        os.makedirs(ct_pred_dir_H, exist_ok=True)
        os.makedirs(ct_gt_dir_H, exist_ok=True)
        os.makedirs(ct_pred_dir_W, exist_ok=True)
        os.makedirs(ct_gt_dir_W, exist_ok=True)
        os.makedirs(ct_pred_dir_L, exist_ok=True)
        os.makedirs(ct_gt_dir_L, exist_ok=True)


        for i in tqdm(range(H)):
            iio.imwrite(osp.join(ct_pred_dir_H, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[i,...])*255).astype(np.uint8))
            iio.imwrite(osp.join(ct_gt_dir_H, f"ct_gt_{str(i)}.png"), (cast_to_image(image[i,...])*255).astype(np.uint8))

        for i in tqdm(range(W)):
            iio.imwrite(osp.join(ct_pred_dir_W, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[:,i,:])*255).astype(np.uint8))
            iio.imwrite(osp.join(ct_gt_dir_W, f"ct_gt_{str(i)}.png"), (cast_to_image(image[:,i,:])*255).astype(np.uint8))

        for i in tqdm(range(L)):
            iio.imwrite(osp.join(ct_pred_dir_L, f"ct_pred_{str(i)}.png"), (cast_to_image(image_pred[...,i])*255).astype(np.uint8))
            iio.imwrite(osp.join(ct_gt_dir_L, f"ct_gt_{str(i)}.png"), (cast_to_image(image[...,i])*255).astype(np.uint8))

        return loss

start = time.time()
evaluator = Eval()
evaluator.eval_step()
end = time.time()
print("Time cost: %f" % (end-start))