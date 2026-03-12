#
# Standalone NeRF-style viewer for a trained Deformable 3D Gaussians checkpoint.
# Requires: pip install viser nerfview
#

import os
import sys
import time

from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel, DeformModel
from scene.gaussian_viewer import GaussianViewer
from gaussian_renderer import view
from utils.general_utils import safe_state

try:
    import viser
    VISER_FOUND = True
except ImportError:
    VISER_FOUND = False


def view_model(dataset: ModelParams, pipe: PipelineParams, iteration: int, port: int, share_url: bool, is_6dof: bool):
    if not VISER_FOUND:
        print("viser/nerfview not installed. Install with: pip install viser nerfview")
        sys.exit(1)

    print("Loading model at iteration {} from {}".format(iteration, dataset.model_path))

    ply_path = os.path.join(dataset.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    if not os.path.isfile(ply_path):
        print("Error: checkpoint not found at {}".format(ply_path))
        sys.exit(1)

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(ply_path)
    deform = DeformModel(is_blender=dataset.is_blender, is_6dof=is_6dof)
    deform.load_weights(dataset.model_path, iteration=iteration)

    server = viser.ViserServer(port=port, verbose=False)
    viewer = GaussianViewer(
        server=server,
        render_fn=view(gaussians, deform, pipe, is_6dof),
        is_dynamic=True,
        mode="rendering",
        share_url=share_url,
    )

    print("Viewer running at http://localhost:{}".format(port))
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive viewer for a trained Deformable 3D Gaussians model")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=30000,
                        help="Checkpoint iteration to load (default: 30000)")
    parser.add_argument("--port", "-p", type=int, default=8080,
                        help="Viewer port (default: 8080)")
    parser.add_argument("--share_url", action="store_true",
                        help="Request shareable viewer URL")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    if not args.model_path:
        print("Error: model path is required. Use -m or --model_path.")
        parser.print_help()
        sys.exit(1)

    safe_state(args.quiet)
    dataset = lp.extract(args)
    pipe = pp.extract(args)
    is_6dof = getattr(args, "is_6dof", False)
    view_model(dataset, pipe, args.iteration, args.port, args.share_url, is_6dof)
