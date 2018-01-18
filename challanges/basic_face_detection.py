import tensorflow as tf
import os
import align.detect_face

from challange import challange
import imageio
import requests

#   setup facenet parameters
gpu_memory_fraction = 1.0
minsize = 50 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor



def get_faces(faces):
    for i in range(0, 800, 100):
        for j in range(0, 800, 100):
            yield i//100, j//100, faces[i:i+100, j:j+100]

def solver(data):
    faces_file = 'faces.jpg'
    open(faces_file, 'wb').write(requests.get(data['image_url']).content)
    faces = imageio.imread(faces_file)
    res = []

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

            for possible_face in get_faces(faces):
                x, y, img = possible_face
                bounding_boxes, _ = align.detect_face.detect_face( img, minsize, pnet, rnet, onet, threshold, factor)

                if len(bounding_boxes):
                    res.append([x,y])


    print(res)
    return {
        'face_tiles': res
    }



def main():
    challange('basic_face_detection', solver, post=True)


if __name__ == '__main__':
    main()
