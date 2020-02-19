# import matplotlib.pyplot as plt
# import matplotlib
# import numpy as np
# plt.rcParams['image.cmap'] = 'gist_earth'
# np.random.seed(98765)
# from tf_unet import image_gen
# from tf_unet import unet
# from tf_unet import util
# nx = 572
# ny = 572
# generator = image_gen.GrayScaleDataProvider(nx, ny, cnt=20)
# x_test, y_test = generator(1)
#
#
# net = unet.Unet(channels=generator.channels, n_class=generator.n_class, layers=3, features_root=16)
# trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
# path = trainer.train(generator, "./unet_trained", training_iters=32, epochs=10, display_step=2)
#



# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
import os
plt.rcParams['image.cmap'] = 'gist_earth'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from scripts.rfi_launcher import DataProvider
from tf_unet import unet

files = glob.glob('bgs_example_data/seek_cache/*')

data_provider = DataProvider(600, files)

net = unet.Unet(channels=data_provider.channels,
                n_class=data_provider.n_class,
                layers=3,
                features_root=64,
                cost_kwargs=dict(regularizer=0.001),
                )

trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))
path = trainer.train(data_provider, "./unet_trained_bgs_example_data",
                     training_iters=32,
                     epochs=30,
                     dropout=0.5,
                     display_step=2)

