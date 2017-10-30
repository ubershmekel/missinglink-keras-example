import missinglink
from keras.applications import vgg16

model = vgg16.VGG16()

missinglink_callback = missinglink.KerasCallback(
    owner_id='replace with owner id',
    project_token='replace with project token')

path = 'http://cmeimg-a.akamaihd.net/640/photos.demandstudios.com' + \
    '/getty/article/103/49/516464087.jpg'

missinglink_callback.generate_grad_cam(path, model)
