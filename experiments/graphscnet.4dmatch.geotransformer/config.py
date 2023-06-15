import argparse
import os
import os.path as osp

from easydict import EasyDict as edict

from vision3d.utils.io import ensure_dir

_C = edict()

# exp
_C.exp = edict()
_C.exp.name = osp.basename(osp.dirname(osp.realpath(__file__)))
_C.exp.working_dir = osp.dirname(osp.realpath(__file__))
_C.exp.output_dir = osp.join("..", "..", "outputs", _C.exp.name)
_C.exp.checkpoint_dir = osp.join(_C.exp.output_dir, "checkpoints")
_C.exp.log_dir = osp.join(_C.exp.output_dir, "logs")
_C.exp.event_dir = osp.join(_C.exp.output_dir, "events")
_C.exp.cache_dir = osp.join(_C.exp.output_dir, "cache")
_C.exp.result_dir = osp.join(_C.exp.output_dir, "results")
_C.exp.seed = 7351

ensure_dir(_C.exp.output_dir)
ensure_dir(_C.exp.checkpoint_dir)
ensure_dir(_C.exp.log_dir)
ensure_dir(_C.exp.event_dir)
ensure_dir(_C.exp.cache_dir)
ensure_dir(_C.exp.result_dir)

# data
_C.data = edict()
_C.data.dataset_dir = "../../data/4DMatch"

# train data
_C.train = edict()
_C.train.batch_size = 1
_C.train.num_workers = 8
_C.train.use_augmentation = True
_C.train.return_corr_indices = True

# test data
_C.test = edict()
_C.test.batch_size = 1
_C.test.num_workers = 8
_C.test.return_corr_indices = True
_C.test.shape_names = None

# evaluation
_C.eval = edict()
_C.eval.acceptance_score = 0.4
_C.eval.acceptance_radius = 0.04
_C.eval.distance_limit = 0.1

# trainer
_C.trainer = edict()
_C.trainer.max_epoch = 40
_C.trainer.grad_acc_steps = 1

# optimizer
_C.optimizer = edict()
_C.optimizer.type = "Adam"
_C.optimizer.lr = 1e-4
_C.optimizer.weight_decay = 1e-6

# scheduler
_C.scheduler = edict()
_C.scheduler.type = "Step"
_C.scheduler.gamma = 0.95
_C.scheduler.step_size = 1

# model - Global
_C.model = edict()
_C.model.min_local_correspondences = 3
_C.model.max_local_correspondences = 128

_C.model.deformation_graph = edict()
_C.model.deformation_graph.num_anchors = 6
_C.model.deformation_graph.node_coverage = 0.08

# model - transformer
_C.model.transformer = edict()
_C.model.transformer.input_dim = 6
_C.model.transformer.hidden_dim = 256
_C.model.transformer.output_dim = 256
_C.model.transformer.num_heads = 4
_C.model.transformer.num_blocks = 3
_C.model.transformer.num_layers_per_block = 2
_C.model.transformer.sigma_d = 0.08
_C.model.transformer.dropout = None
_C.model.transformer.activation_fn = "ReLU"
_C.model.transformer.embedding_k = -1
_C.model.transformer.embedding_dim = 1

# model - classifier
_C.model.classifier = edict()
_C.model.classifier.input_dim = 256
_C.model.classifier.dropout = None

# Non-rigid ICP
_C.model.nicp = edict()
_C.model.nicp.corr_lambda = 5.0
_C.model.nicp.arap_lambda = 1.0
_C.model.nicp.lm_lambda = 0.01
_C.model.nicp.num_iterations = 5

# loss
_C.loss = edict()
_C.loss.focal_loss = edict()
_C.loss.focal_loss.weight = 1.0
_C.loss.consistency_loss = edict()
_C.loss.consistency_loss.weight = 1.0


def make_cfg():
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--link_output", dest="link_output", action="store_true", help="link output dir"
    )
    args = parser.parse_args()
    return args


def main():
    cfg = make_cfg()
    args = parse_args()
    if args.link_output:
        os.symlink(cfg.output_dir, "output")


if __name__ == "__main__":
    main()
