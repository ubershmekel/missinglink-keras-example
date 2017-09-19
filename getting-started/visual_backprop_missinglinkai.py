import missinglink
from keras.applications import vgg16

model = vgg16.VGG16()

missinglink_callback = missinglink.KerasCallback(
    owner_id='replace with owner id',
    project_token='replace with project token')

path = 'http://l7.alamy.com/zooms' + \
    '/b76d255dd51e493e8c0fd5d5aa85f96f/lumbermill-cp93p7.jpg'

missinglink_callback.visual_back_prop(path, model)
