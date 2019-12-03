import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import optimizers
import os
import gc
import parameter as para
import data_generators as dg
from SqueezeDet import SqueezeDet
from model_loss import total_loss, bbox_loss, class_loss, conf_loss


def train():
    img_file = "../kitti_data/img_train.txt"
    ytrue_file = "../kitti_data/ytrue_train.txt"
    val_file = "../kitti_data/img_val.txt"
    val_ytrue_file = "../kitti_data/ytrue_val.txt"
    checkpoint_dir = "../kitti_data/checkpoints"
    CUDA_VISIBLE_DEVICES = "0"
    
    # open files with images and ground truths files with full path names
    with open(img_file) as imgs:
        img_names = imgs.read().splitlines()
    imgs.close()
    with open(ytrue_file) as ytrues:
        ytrue_names = ytrues.read().splitlines()
    ytrues.close()
    with open(val_file) as vals:
        val_names = vals.read().splitlines()
    vals.close()
    with open(val_ytrue_file) as val_ytrues:
        val_ytrue_names = val_ytrues.read().splitlines()
    val_ytrues.close()
    val_names = val_names[0:32]
    val_ytrue_names = val_ytrue_names[0:32]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
    eps = 50
    
    # compute number of steps per epoch
    train_steps, mod = divmod(len(img_names), para.BATCH_SIZE)
    val_steps, mod = divmod(len(val_names), para.BATCH_SIZE)
    
    # print some run info
    print("Number of images: {}".format(len(img_names)))
    print("Number of validation images: {}".format(len(val_names)))
    print("Batch size: {}".format(para.BATCH_SIZE))
    print("Number of epochs: {}".format(eps))
    print("Number of steps: {}".format(train_steps))
    print("Number of validation steps: {}".format(val_steps))
    
    # tf config and session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    K.set_session(sess)
    
    # Initiate a model
    squeeze = SqueezeDet()
    
    # Set optimizer
    opt = optimizers.Adam(lr=0.001,  clipnorm=1.0)
    
    # Create train and validation generator
    train_generator = dg.DataGen(img_names, ytrue_names, batch_size=para.BATCH_SIZE, img_height=384, img_width=1248)
    valid_generator = dg.DataGen(val_names, val_ytrue_names, batch_size=para.BATCH_SIZE, img_height=384, img_width=1248)
    
    # Set callbacks
    cb = []
    # Set reducelronplateu callback
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,verbose=1, patience=5, min_lr=0.0)
    cb.append(reduce_lr)
    # Create checkpoint callback
    ckp_saver = ModelCheckpoint(checkpoint_dir + "/squeeze_ep50.h5", monitor='loss', verbose=1,
                                save_best_only=True,
                                save_weights_only=True, mode='auto', period=1)
    cb.append(ckp_saver)
    
    # Compile model
    squeeze.compile(optimizer=opt, loss=[total_loss], metrics=[bbox_loss, class_loss, conf_loss])
    
    # Train the model
    squeeze.fit_generator(train_generator, validation_data=valid_generator, steps_per_epoch=train_steps, validation_steps=val_steps, epochs=eps, callbacks=cb)
    gc.collect()
    

if __name__ == "__main__":
    train()