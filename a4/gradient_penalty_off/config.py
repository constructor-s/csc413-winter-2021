SEED = 11

# Set the random seed manually for reproducibility.
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


args = AttrDict()
args_dict = {
              'image_size':32, 
              'g_conv_dim':32, 
              'd_conv_dim':64,
              'noise_size':100,
              'num_workers': 0,
              'train_iters':20000,
              'X':'Windows',  # options: 'Windows' / 'Apple'
              'Y': None,
              'lr':0.00003,
              'beta1':0.5,
              'beta2':0.999,
              'batch_size':32, 
              'checkpoint_dir': 'results/checkpoints_gan_gp1_lr3e-5_nogp',
              'sample_dir': 'results/samples_gan_gp1_lr3e-5_nogp',
              'load': None,
              'log_step':200,
              'sample_every':200,
              'checkpoint_every':1000,
              'spectral_norm': False,
              'gradient_penalty': False,
              'd_train_iters': 1
}
args.update(args_dict)

print_opts(args)
G, D = train(args)

generate_gif("results/samples_gan_gp1_lr3e-5_nogp")
