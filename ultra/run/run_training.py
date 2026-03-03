# -*- coding: utf-8 -*-
import os
from typing import Union, Optional
from rich import print
import ultra
import torch
import torch.cuda
import torch.multiprocessing as mp
from torch.backends import cudnn
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.run.run_training import find_free_network_port
from nnunetv2.run.run_training import run_ddp, maybe_load_checkpoint

torch.set_float32_matmul_precision('high')


def get_trainer_from_args(dataset_name_or_id: Union[int, str],
                          configuration: str,
                          fold: int,
                          trainer_name: str = 'iSeekTrainer',
                          plans_identifier: str = 'nnUNetPlans',
                          use_compressed: bool = False,
                          device: torch.device = torch.device('cuda')):
    base_trainer = recursive_find_python_class(
        join(ultra.__path__[0], 'trainer'),
        trainer_name,
        'ultra.trainer'
    )
    if base_trainer is None:
        raise RuntimeError(f'Trainer {trainer_name} not found in '
                           f'ultra.trainer')
    assert issubclass(base_trainer, nnUNetTrainer), (f'Trainer {trainer_name} is not a subclass of nnUNetTrainer'
                                                     f'Please define a trainer that integrates nnUNetTrainer.')

    if dataset_name_or_id.startswith("Dataset"):
        pass
    else:
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError(f'Dataset name or id {dataset_name_or_id} '
                             f'must either be an integer or a valid dataset name with the pattern '
                             f'DatasetXXX_YYY where XXX are the three(!) task ID digits. Your '
                             f'input: {dataset_name_or_id}')

    preprocessed_dataset_folder_base = join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    plans_file = join(preprocessed_dataset_folder_base, plans_identifier + ".json")
    plans = load_json(plans_file)
    dataset_json = load_json(join(preprocessed_dataset_folder_base, "dataset.json"))
    base_trainer = base_trainer(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device)
    return base_trainer


def run_training(dataset_name_or_id: Union[str, int],
                 configuration: str, fold: Union[int, str],
                 trainer_class_name: str = 'nnUNetTrainer',
                 plans_identifier: str = 'nnUNetPlans',
                 pretrained_weights: Optional[str] = None,
                 num_gpus: int = 1,
                 use_compressed_data: bool = False,
                 export_validation_probabilities: bool = False,
                 continue_training: bool = False,
                 only_run_validation: bool = False,
                 disable_checkpointing: bool = False,
                 val_with_best: bool = False,
                 device: torch.device = torch.device('cuda')):
    if plans_identifier == "nnUNetPlans":
        print(f"Using nnUNetPlans ({trainer_class_name})")
    if isinstance(fold, str):
        if fold != 'all':
            try:
                fold = int(fold)
            except ValueError as e:
                print(
                    f'Unable to convert given value for fold to int: {fold}. '
                    f'fold must bei either "all" or an integer!')
                raise e
    if val_with_best:
        assert not disable_checkpointing, '--val_best is not compatible with --disable_checkpointing'

    if num_gpus > 1:
        assert device.type == 'cuda', (f"DDP training (triggered by num_gpus > 1) is only implemented for cuda devices."
                                       f" Your device: {device}")
        os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ.keys():
            port = str(find_free_network_port())
            print(f"using port {port}")
            os.environ['MASTER_PORT'] = port  # str(port)

        mp.spawn(run_ddp,
                 args=(
                     dataset_name_or_id,
                     configuration,
                     fold,
                     trainer_class_name,
                     plans_identifier,
                     use_compressed_data,
                     disable_checkpointing,
                     continue_training,
                     only_run_validation,
                     pretrained_weights,
                     export_validation_probabilities,
                     val_with_best,
                     num_gpus),
                 nprocs=num_gpus,
                 join=True)
    else:
        trainer = get_trainer_from_args(dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier,
                                        use_compressed_data, device=device)
        if disable_checkpointing:
            trainer.disable_checkpointing = disable_checkpointing

        assert not (
                continue_training and only_run_validation), f'Cannot set --c and --val flag at the same time. Dummy.'
        maybe_load_checkpoint(trainer, continue_training, only_run_validation, pretrained_weights)

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            trainer.run_training()

        if val_with_best:
            trainer.load_checkpoint(join(trainer.output_folder, 'checkpoint_best.pth'))
        trainer.perform_actual_validation(export_validation_probabilities)


def run_training_entry():
    import argparse
    parser = argparse.ArgumentParser("Run the segmentation models training [based on nnUNetv2]")
    parser.add_argument('dataset_name_or_id', type=str,
                        help="Dataset name or ID to train with")
    parser.add_argument('configuration', type=str,
                        help="Configuration that should be trained")
    parser.add_argument('fold', type=str,
                        help='Fold of the 5-fold cross-validation. Should be an int between 0 and 4.')
    parser.add_argument('-tr', type=str, required=False, default='UltraTrainerS3',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: UltraTrainerS3')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--val_best', action='store_true', required=False,
                        help='[OPTIONAL] If set, the validation will be performed with the checkpoint_best instead '
                             'of checkpoint_final. NOT COMPATIBLE with --disable_checkpointing! '
                             'WARNING: This will use the same \'validation\' folder as the regular validation '
                             'with no way of distinguishing the two!')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the training should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")
    args = parser.parse_args()

    assert args.device in ['cpu', 'cuda',
                           'mps'], (f'-device must be either cpu, mps or cuda. '
                                    f'Other devices are not tested/supported. Got: {args.device}.')

    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    run_training(args.dataset_name_or_id,
                 args.configuration,
                 args.fold,
                 args.tr,
                 args.p,
                 args.pretrained_weights,
                 args.num_gpus,
                 args.use_compressed,
                 args.npz,
                 args.c,
                 args.val,
                 args.disable_checkpointing,
                 args.val_best,
                 device=device)
