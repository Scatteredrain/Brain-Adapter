from .config import CfgNode as CN

_C = CN()
_C.VERSION = 1
# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.name = 'explainable_3d_clip'
_C.isTrain = True
_C.checkpoints_dir = './checkpoints'
_C.continue_train = False
_C.load_iter = 0
# which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter];
# otherwise, the code will load models by [epoch]
_C.epoch = None  # which epoch to load? set to latest to use latest cached model
_C.verbose = False

_C.num_gpus = 1

_C.manual_seed = 42
_C.save_ckpt = False
_C.debug = False

_C.loaders = CN()
_C.loaders.batch_size = 8
_C.loaders.cls_thresholds = []
_C.loaders.fold = 0
_C.loaders.fold_split = False
_C.loaders.fold_num = 5
_C.loaders.serial_batches = False
_C.loaders.num_threads = 8
_C.loaders.max_value = 40
_C.loaders.min_value = 0
_C.loaders.scaler = 1
_C.loaders.offset = 0

_C.loaders.train = CN()
_C.loaders.train.img_file_path = '../../data/train.lst'
_C.loaders.train.text_file_path = ''
_C.loaders.train.WL_WW = [40, 90]
# _C.loaders.train.max_value = 50
# _C.loaders.train.scaler = 10
_C.loaders.train.label_file_path = ''

_C.loaders.train.transformer = CN()

_C.loaders.train.transformer.raw = CN()
_C.loaders.train.transformer.raw.CropToFixed = CN()
_C.loaders.train.transformer.raw.CropToFixed.enabled = False
_C.loaders.train.transformer.raw.CropToFixed.size = [256, 256]
_C.loaders.train.transformer.raw.CropToFixed.centered = False
_C.loaders.train.transformer.raw.CropToFixed.mode = 'reflect'
_C.loaders.train.transformer.raw.PercentileNormalizer = CN()
_C.loaders.train.transformer.raw.PercentileNormalizer.enabled = False
_C.loaders.train.transformer.raw.PercentileNormalizer.pmin = 1.0
_C.loaders.train.transformer.raw.PercentileNormalizer.pmax = 99.6
_C.loaders.train.transformer.raw.Normalize = CN()
_C.loaders.train.transformer.raw.Normalize.enabled = False
_C.loaders.train.transformer.raw.Normalize.min_value = 0.0
_C.loaders.train.transformer.raw.Normalize.max_value = 255.0
_C.loaders.train.transformer.raw.RandomFlip = CN()
_C.loaders.train.transformer.raw.RandomFlip.enabled = False
_C.loaders.train.transformer.raw.RandomFlip.axes = [0, ]
_C.loaders.train.transformer.raw.ResizeCrop = CN()
_C.loaders.train.transformer.raw.ResizeCrop.enabled = False
_C.loaders.train.transformer.raw.ResizeCrop.inter_size = [286, 286]
_C.loaders.train.transformer.raw.ResizeCrop.target_size = [256, 256]
_C.loaders.train.transformer.raw.ResizeCrop.is_label = False
_C.loaders.train.transformer.raw.RandomRotate90 = CN()
_C.loaders.train.transformer.raw.RandomRotate90.enabled = False
_C.loaders.train.transformer.raw.RandomRotate = CN()
_C.loaders.train.transformer.raw.RandomRotate.enabled = False
_C.loaders.train.transformer.raw.RandomRotate.axes = [[2, 1]]
_C.loaders.train.transformer.raw.RandomRotate.angle_spectrum = 45
_C.loaders.train.transformer.raw.RandomRotate.mode = 'reflect'

_C.loaders.train.transformer.raw.ElasticDeformation = CN()
_C.loaders.train.transformer.raw.ElasticDeformation.enabled = False
_C.loaders.train.transformer.raw.ElasticDeformation.spline_order = 3
_C.loaders.train.transformer.raw.ElasticDeformation.alpha = 2000
_C.loaders.train.transformer.raw.ElasticDeformation.sigma = 50
_C.loaders.train.transformer.raw.ElasticDeformation.execution_probability = 0.2


_C.loaders.train.transformer.raw.GaussianBlur3D = CN()
_C.loaders.train.transformer.raw.GaussianBlur3D.enabled = False
_C.loaders.train.transformer.raw.GaussianBlur3D.execution_probability = 0.5
_C.loaders.train.transformer.raw.GaussianBlur3D.sigma = [0.1,2.0]

_C.loaders.train.transformer.raw.AdditiveGaussianNoise = CN()
_C.loaders.train.transformer.raw.AdditiveGaussianNoise.enabled = False
_C.loaders.train.transformer.raw.AdditiveGaussianNoise.execution_probability = 0.2
_C.loaders.train.transformer.raw.AdditiveGaussianNoise.scale = (.0,1.)

_C.loaders.train.transformer.raw.AdditivePoissonNoise = CN()
_C.loaders.train.transformer.raw.AdditivePoissonNoise.enabled = False
_C.loaders.train.transformer.raw.AdditivePoissonNoise.execution_probability = 0.2
_C.loaders.train.transformer.raw.ToTensor = CN()
_C.loaders.train.transformer.raw.ToTensor.enabled = True
_C.loaders.train.transformer.raw.ToTensor.expand_dims = False
_C.loaders.train.transformer.raw.RepeatDim = CN()
_C.loaders.train.transformer.raw.RepeatDim.enabled = False
_C.loaders.train.transformer.raw.RepeatDim.repeat_dim = 1
_C.loaders.train.transformer.raw.Standardize = CN()
_C.loaders.train.transformer.raw.Standardize.enabled = False

_C.loaders.test = CN()
_C.loaders.test.img_file_path = '../../data/test.lst'
_C.loaders.test.text_file_path = '../../data/test_description.lst'
_C.loaders.test.label_file_path = ''
_C.loaders.test.WL_WW = [40, 90]
# _C.loaders.test.max_value = 50
# _C.loaders.test.scaler = 10

_C.loaders.test.transformer = CN()
_C.loaders.test.transformer.raw = CN()
_C.loaders.test.transformer.raw.CropToFixed = CN()
_C.loaders.test.transformer.raw.CropToFixed.enabled = False
_C.loaders.test.transformer.raw.CropToFixed.size = [256, 256]
_C.loaders.test.transformer.raw.CropToFixed.centered = False
_C.loaders.test.transformer.raw.CropToFixed.mode = 'reflect'
_C.loaders.test.transformer.raw.PercentileNormalizer = CN()
_C.loaders.test.transformer.raw.PercentileNormalizer.enabled = False
_C.loaders.test.transformer.raw.PercentileNormalizer.pmin = 1.0
_C.loaders.test.transformer.raw.PercentileNormalizer.pmax = 99.6
_C.loaders.test.transformer.raw.Normalize = CN()
_C.loaders.test.transformer.raw.Normalize.enabled = False
_C.loaders.test.transformer.raw.Normalize.min_value = 0.0
_C.loaders.test.transformer.raw.Normalize.max_value = 255.0
_C.loaders.test.transformer.raw.ResizeCrop = CN()
_C.loaders.test.transformer.raw.ResizeCrop.enabled = False
_C.loaders.test.transformer.raw.ResizeCrop.inter_size = [286, 286]
_C.loaders.test.transformer.raw.ResizeCrop.target_size = [256, 256]
_C.loaders.test.transformer.raw.ResizeCrop.is_label = False
_C.loaders.test.transformer.raw.ToTensor = CN()
_C.loaders.test.transformer.raw.ToTensor.enabled = True
_C.loaders.test.transformer.raw.ToTensor.expand_dims = False
_C.loaders.test.transformer.raw.RepeatDim = CN()
_C.loaders.test.transformer.raw.RepeatDim.enabled = False
_C.loaders.test.transformer.raw.RepeatDim.repeat_dim = 1
_C.loaders.test.transformer.raw.Standardize = CN()
_C.loaders.test.transformer.raw.Standardize.enabled = False

_C.model = CN()
_C.model.backbone_name = 'biomedclip'
_C.model.backbone_ckpt = ''
_C.model.freeze_backbone = True
_C.model.freeze_text_backbone = True
_C.model.embed_dim = 512
_C.model.num_mha_heads = 1
_C.model.transformer_dropout = 0.3
_C.model.prompt_length = 0
_C.model.vision_ctx = 0
_C.model.use_vpt = False
_C.model.use_lora = False
_C.model.use_clip_adapter = False
_C.model.lora_rank = 4
_C.model.lora_alpha = 8
_C.model.clip_adapter_reduction = 4
_C.model.clip_adapter_ratio = 0.2
_C.model.add_cls_head = False
_C.model.add_reg_head = False
_C.model.add_retrieval_reg_head = False
_C.model.using_xpooling = False
_C.model.using_attnmil = False
_C.model.ct_mil = False
_C.model.add_multi_cls = False
_C.model.paper_core_mode = False
_C.model.max_fine_grained_sentences = 8
_C.model.use_uar = False
_C.model.uar_alpha = 2.0
_C.model.uar_lambda = 0.5
_C.model.uar_temperature = 0.07
_C.model.uar_prompt_template = 'This CT study shows: {}.'
_C.model.uar_class_names = []

_C.optimizer = CN()
_C.optimizer.lr = 0.0002
_C.optimizer.clip_lr = 0.00002
_C.optimizer.other_lr = 0.00001
_C.optimizer.beta1 = 0.5
_C.optimizer.weight_decay = 0.0

_C.loss = CN()
_C.loss.lambda_cls = 0.0
_C.loss.lambda_clip = 0.0
_C.loss.lambda_RPS = 0.0
_C.loss.lambda_consis = 0.0  # weight for contrastive loss
_C.loss.lambda_regress = 0.0
_C.loss.consis_temparature = 0.0
_C.loss.lambda_bmc = 0.0
_C.loss.lambda_categorical_ordinal_focal_loss = 0.0
_C.loss.lambda_ranking = 0.0
_C.loss.lambda_multi_cls = 0.0
_C.loss.lambda_align = 0.0
_C.loss.lambda_paper_consistency = 0.0
_C.loss.alignment_temperature = 0.07
_C.loss.using_LDS = False

_C.scheduler = CN()
_C.scheduler.n_epochs = 100
_C.scheduler.epoch_count = 1
_C.scheduler.accumulate_gradient_count = 1
_C.scheduler.cls_weight = 1
# the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...
_C.scheduler.n_epochs_decay = 100  # number of epochs to linearly decay learning rate to zero
_C.scheduler.lr_decay_iters = 50  # number of epochs to linearly decay learning rate to zero
_C.scheduler.lr_policy = 'linear'  # learning rate policy. [linear | step | plateau | cosine]

_C.display = CN()
_C.display.display_id = 0
_C.display.use_html = False
_C.display.display_server = "http://localhost"
_C.display.display_port = 8097
_C.display.display_env = 'main'
#  visdom display environment name (default is "main")
_C.display.display_winsize = 256
_C.display.display_ncols = 4
_C.display.use_wandb = True
_C.display.wandb_project_name = 'explainable_3d_clip'

_C.trainer = CN()
_C.trainer.print_freq = 10  # frequency of showing training results on console
_C.trainer.display_freq = 10  # frequency of showing training results on console
_C.trainer.update_html_freq = 1000  # frequency of showing training results on console
_C.trainer.save_latest_freq = 10  # frequency of saving the latest results
_C.trainer.save_epoch_freq = 10  # frequency of saving checkpoints at the end of epochs
_C.trainer.save_by_iter = False  # frequency of saving checkpoints at the end of epochs
_C.trainer.save_model_freq = 5
