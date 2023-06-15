from vision3d.engine import SingleTester
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args, get_default_parser
from vision3d.utils.profiling import profile_cpu_runtime

# isort: split
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from model import create_model


def add_custom_args():
    parser = get_default_parser()
    parser.add_argument(
        "--benchmark",
        choices=["4DMatch-F", "4DLoMatch-F"],
        required=True,
        help="test benchmark",
    )


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        with profile_cpu_runtime("Data loader created"):
            data_loader = test_data_loader(cfg, self.args.benchmark)
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

    def test_step(self, iteration, data_dict):
        data_dict["registration"] = True
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        return result_dict

    def get_log_string(self, iteration, data_dict, output_dict, result_dict):
        shape_name = data_dict["shape_name"]
        src_frame = data_dict["src_frame"]
        tgt_frame = data_dict["tgt_frame"]
        message = f"{shape_name}, id0: {tgt_frame}, id1: {src_frame}"
        message += ", " + get_log_string(result_dict=result_dict)
        return message


def main():
    add_tester_args()
    add_custom_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
