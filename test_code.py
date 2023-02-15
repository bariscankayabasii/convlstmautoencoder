import glob
import tensorflow as tf
import matplotlib.pyplot as plt

def evaluate(net, test_img_path):
    
    test_img = glob.glob(test_img_path + '/*.jpg')
    random.shuffle(test_img)
    
    for img in test_img:
        
        img = tf.io.read_file(img)
        img = tf.io.decode_jpeg(img, channels = 3)
        
        if img.shape[1] > img.shape[0]:
            img = tf.image.resize(img, size = (1080, 1920), antialias = True)
        if img.shape[1] < img.shape[0]:
            img = tf.image.resize(img, size = (1920, 1080), antialias = True)
        
        img = img / 255
        img = tf.expand_dims(img, axis = 0)
        
        dehaze = net(img, training = False)

        plt.figure(figsize = (80, 80))
        
        display_list = [img[0], dehaze[0]]       
        title = ['Hazy Image', 'Dehazed Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i], fontsize = 65, y = 1.045)
            plt.imshow(display_list[i])
            plt.axis('off')
        
        plt.show()
        
if __name__='__main__':

    test_net = tf.keras.models.load_model('trained_model', compile = False)
    dehaze=evaluate(test_net, '../input/hazy-test-images')
