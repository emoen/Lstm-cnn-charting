from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Activation, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from keras import optimizers, layers
from keras import backend as K

#from efficientnet import EfficientNetB4
import efficientnet.keras as efn
from deepaugment.deepaugment import DeepAugment

def train_cnn():
    max_dataset_size = 1029#6330
    new_shape = (349, 385, 3)
    IMG_SHAPE = (349, 385)
    rb_imgs = np.empty(shape=(max_dataset_size,)+new_shape)
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    tensorboard_path= './tensorboard_softmax'
    checkpoint_path = './checkpoints_softmax/efficientnetB4.{epoch:03d}-{val_loss:.2f}.hdf5'

if __name__ == '__main__':
    test_charting()