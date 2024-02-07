import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main")
    parser.add("--train_data_path", action="append", required=True)
    parser.add("--val_data_path", action="append", required=True)
    parser.add("--test_data_path", action="append", required=False)
    parser.add("--model_save_path", required=True)
    parser.add("--random_seed", type=int, default=-1)
	

    # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=200)
    parser.add("--sentence_level", type=str, required=True)
    parser.add("--sentence_frame_length", type=int, default=120)


    # model
    parser.add("--model", type=str, required=True)
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)

    # Autoencoder
    parser.add("--autoencoder_denoising", type=str, required=True)
    parser.add("--autoencoder_att", type=str, required=True)
    parser.add("--autoencoder_fixed_weight", type=str, required=True)
    parser.add("--autoencoder_conditioned", type=str, required=True)
    parser.add("--use_derivative", type=str, required=True)
    parser.add("--autoencoder_checkpoint", type=str, required=True)
    parser.add("--autoencoder_vae", type=str, required=True)
    # parser.add("--autoenoder_train_decoder", type=str, required=True)  # seems duplication
    parser.add("--autoencoder_freeze_encoder", type=str, required=True)
    parser.add("--autoencoder_vq", type=str, required=True)
    parser.add("--autoencoder_vq_components", type=str, required=True)
    parser.add("--autoencoder_vq_commitment_cost", type=str, required=True)

    # text2embedding
    parser.add("--text2_embedding_discrete", type=str, required=True)


    # similarity
    parser.add("--use_similarity", type=str, required=True)
    parser.add("--similarity_labels", type=str, required=True)
    parser.add("--data_for_sim", type=str, required=True)
    parser.add("--loss_label_weight", type=float)
    # dataset
    parser.add("--data_mean", action="append", type=float, nargs='*')
    parser.add("--data_std", action="append", type=float, nargs='*')
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--subdivision_stride_sentence", type=int, default=30)
    parser.add("--loader_workers", type=int, default=4)
    parser.add("--input_motion_dim", type=int, default=135)

    # Modalities
    parser.add("--Modality_Audio", type=str, required=True)
    parser.add("--Modality_Text", type=str, required=True)
    parser.add("--Modality_Gesture", type=str, required=True)

    # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--loss_l1_weight", type=float, default=50)
    parser.add("--loss_cont_weight", type=float, default=0.1)
    parser.add("--loss_var_weight", type=float, default=0.01)

    # Representation learning:
    parser.add("--rep_learning_checkpoint", type=str, default='')
    parser.add("--rep_learning_dim", type=int, default=-1)


    # GAN
    parser.add("--noise_dim", type=int, default=200)

    args = parser.parse_args()
    return args