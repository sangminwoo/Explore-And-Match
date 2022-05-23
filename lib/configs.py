import argparse
from lib.utils.misc import dict_to_markdown


# meta config
parser = argparse.ArgumentParser(description='Video Grounding Transformer')
parser.add_argument('--root', type=str, default='/ROOT_DIR',
                    help='root directory of dataset')
parser.add_argument('--results_dir', type=str, default='results',
                    help='directory for saving results')
parser.add_argument('--device', type=str, default='0',
                    help='GPU ID')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1). if seed=0, seed is not fixed.')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging training status')
parser.add_argument('--val_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before validation')
parser.add_argument('--save_interval', type=int, default=50, metavar='N',
                    help='how many epochs to wait before saving a model')
parser.add_argument('--no_gpu', dest='use_gpu', action='store_false',
                    help='disable use of gpu')
parser.add_argument('--debug', action='store_true',
                    help='debug (fast) mode, break all loops, do not load all data into memory.')
parser.add_argument("--eval_untrained", action="store_true",
                    help="Evaluate on untrained model")
parser.add_argument('--log_dir', type=str, default='logs',
                    help='directory for saving logs')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint path to resume or evaluate, without --resume_all this only load model weights')
parser.add_argument('--resume_all', action='store_true',
                    help='if --resume_all, load optimizer/scheduler/epoch as well')
parser.add_argument('--att_visualize', action='store_true',
                    help='use for visualizing attention')
parser.add_argument('--corr_visualize', action='store_true',
                    help='use for visualizing correspondence between predictions and queries')
parser.add_argument('--dist_visualize', action='store_true',
                    help='use for visualizing predicted span distribution')


# training config
parser.add_argument('--start_epoch', type=int, default=None,
                    help='if None, will be set automatically when using --resume_all')
parser.add_argument('--end_epoch', type=int, default=200,
                    help='number of epochs to run')
parser.add_argument('--early_stop_patience', type=int, default=-1,
                    help='number of epochs to early stop, use -1 to disable early stop')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument("--lr_drop_step", type=int, default=20,
                    help="drop learning rate to 1/10 every lr_drop_step epochs")
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay (default=0.0001)')
parser.add_argument('--optimizer', type=str, default='adamw',
                    help='which optimizer to use (e.g., sgd, adam, adamw)')
parser.add_argument('--scheduler', type=str, default='steplr',
                    help='which scheduler to use')


# data config
parser.add_argument('--dataset', type=str, default='activitynet',
                    choices=['activitynet', 'charades'],
                    help='the name of dataset')
parser.add_argument('--data_type', type=str, default='features',
                    choices=['features', 'raw'],
                    help='type of data to load (pre-trained features vs. raw image texts)')
parser.add_argument('--num_input_frames', type=int, default=64,
                    help='number of frames to input.')
parser.add_argument('--num_input_sentences', type=int, default=4,
                    help='number of sentences to predict.')
parser.add_argument('--bs', type=int, default=16, # FIXME
                    help='batch size')
parser.add_argument('--eval_bs', type=int, default=1, # FIXME
                    help='batch size at inference, for query')
parser.add_argument('--num_workers', type=int, default=16,
                    help='num subprocesses used to load the data, 0: use main process')
parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                    help='No use of pin_memory for data loading.'
                         'If pin_memory=True, the data loader will copy Tensors into CUDA pinned memory before returning them.')
parser.add_argument('--checkpoint', type=str, default='./save',
                    help='dir to save checkpoint')
parser.add_argument("--no_norm_vfeat", dest='norm_vfeat', action="store_false",
                    help="No normalization performed on input video features")
parser.add_argument("--no_norm_tfeat", dest='norm_tfeat', action="store_false",
                    help="No normalization performed on input text features")
parser.add_argument("--txt_drop_ratio", default=0, type=float,
                    help="drop txt_drop_ratio tokens from text input. 0.1=10%")


# model config
parser.add_argument('--backbone', type=str, default='clip',
                    choices=['clip', 'c3d_lstm'],
                    help='choice of backbone network')
parser.add_argument('--method', type=str, default='joint',
                    choices=['stepwise', 'joint'],
                    help='choice of transformer design')
parser.add_argument('--hidden_dim', type=int, default=256,
                    help='hidden dimension of Transformer')
parser.add_argument('--nheads', type=int, default=8,
                    help='number of Transformer attention heads')
parser.add_argument('--enc_layers', type=int, default=4,
                    help='Number of encoding layers in the transformer')
parser.add_argument('--dec_layers', type=int, default=4,
                    help='Number of decoding layers in the transformer')
parser.add_argument('--vid_feat_dim', type=int, default=512,
                    help='video feature dim')
parser.add_argument('--txt_feat_dim', type=int, default=512,
                    help='text/query feature dim')
parser.add_argument('--num_proposals', default=10, type=int,
                    help='Number of learnable proposals')
parser.add_argument('--input_dropout', default=0.5, type=float,
                    help='Dropout applied in input')
# parser.add_argument('--shuffle_txt', action='store_true',
#                     help='enable shuffling the input sentences.')
parser.add_argument('--no_vid_pos', dest='use_vid_pos', action='store_false',
                    help='disable position_embedding for video.')
parser.add_argument('--no_txt_pos', dest='use_txt_pos', action='store_false',
                    help='disable position_embedding for text.')
parser.add_argument('--n_input_proj', type=int, default=2,
                    help='#layers to encoder input')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout applied in the transformer')
parser.add_argument('--dim_feedforward', type=int, default=1024,
                    help='Intermediate size of the feedforward layers in the transformer blocks')
parser.add_argument('--pre_norm', action='store_true',
                    help='apply normalize before attention')
parser.add_argument('--vid_position_embedding', default='sine', type=str,
                    choices=['trainable', 'sine', 'learned'],
                    help="Type of positional embedding to use on top of the image features")
parser.add_argument('--txt_position_embedding', default='sine', type=str,
                    choices=['trainable', 'sine', 'learned'],
                    help="Type of positional embedding to use on top of the text features")


# loss config
parser.add_argument('--set_cost_span', default=5, type=int,
                    help="L1 span coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=int,
                    help="giou span coefficient in the matching cost")
parser.add_argument('--set_cost_query', default=1, type=int,
                    help="Set guidance coefficient in the matching cost")
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help='Disable auxiliary decoding losses (loss at each layer)')
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")
parser.add_argument('--pred_label', default='sim', type=str,
                    choices=['att', 'sim', 'cos', 'pred'],
                    help="Criteria of label assignment for each query"
                         "att: use attention weight to predict label"
                         "sim: use similarity to predict label"
                         "cos: use cosine similarity to predict label"
                         "pred: predict label with class head")

# evaluation config
# parser.add_argument('--matcher', type=str, default='no_match',
#                     choices=['hungarian', 'bipartite', 'no_match'],
#                     help='type of matcher to use for evaluation'
#                          'hungarian: select min(src, tgt) that minimize overall cost'
#                          'bipartite: select max(src, tgt) that minimize overall cost'
#                          'no_match: do not assign src to tgt')
parser.add_argument('--span_type', default='cw', type=str,
                    choices=['cw', 'xx'],
                    help="Type of span (cw: center-width / xx: start-end)")
parser.add_argument("--no_sort_results", action="store_true",
                    help="do not sort results, use this for span visualization")
parser.add_argument("--max_before_nms", type=int, default=10)
parser.add_argument("--max_after_nms", type=int, default=10)
parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
parser.add_argument("--nms_thd", type=float, default=-1,
                       help="additionally use non-maximum suppression "
                            "(or non-minimum suppression for distance)"
                            "to post-processing the predictions. "
                            "-1: do not use nms. [0, 1]")

args = parser.parse_args()

# Display settings
print(dict_to_markdown(vars(args), max_str_len=120))