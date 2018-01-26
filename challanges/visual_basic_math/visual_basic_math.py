from challange import challange
import imageio
import numpy as np
import scipy.misc
import tensorflow as tf
import uuid
import requests
import time
import operator


def is_all_white(row):
    for p in row:
        if p != 255.0:
            return False

    return True



def conver_to_grayscale(image):
    grey = np.zeros((image.shape[0], image.shape[1]), dtype=np.float64) 

    for rownum in range(len(image)):
       for colnum in range(len(image[rownum])):
          grey[rownum][colnum] = np.average(image[rownum][colnum])

    return grey


def split_by_whitespace(img):
    looking_for_chars=True
    y0=None

    for y in range(img.shape[0]):
        line = img[y]

        if looking_for_chars and not is_all_white(line):
            y0=y
            looking_for_chars=False
        elif not looking_for_chars and is_all_white(line):
            yield img[y0:y]
            looking_for_chars=True

def get_max_wh(chars):
    max_h = 0
    max_w = 0

    for char in chars:
        h, w = char.shape

        if h > max_h:
            max_h = h

        if w > max_w:
            max_w = w

    return max_h, max_w

def centered_char(size, char):
    c = np.zeros((size, size))
    c.fill(255.0)

    char_h, char_w = char.shape

    lower_h = (size // 2) - (char_h // 2) 
    upper_h = lower_h + char_h
    
    lower_w = (size // 2) - (char_w // 2)
    upper_w = lower_w + char_w

    c[lower_h:upper_h, lower_w:upper_w] = char
     
    return c


def ocr(img):
    img = conver_to_grayscale(img)

    for i, row in enumerate(split_by_whitespace(img)):
        for j, char in enumerate(split_by_whitespace(np.rot90(row, 3))):
            char = np.rot90(char)
            char = centered_char(43, char)
            char = scipy.misc.imresize(char, (28, 28))
            yield char

def fit_char(char):
    char = char.flatten().astype(np.float64)

    for p in range(0, char.shape[0]):
        char[p] = (255 - char[p]) / 255

    return char


def hottop_to_symbol(i):
    if i < 10:
        return i

    if i == 10:
        return operator.floordiv

    if i == 11:
        return operator.mul

    if i == 12:
        return operator.add

    if i == 13:
        return operator.sub



def to_symbols(img):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.import_meta_graph('./models/deep.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./models'))
        graph = tf.get_default_graph()

        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("fc2/y:0")
        keep_prob = graph.get_tensor_by_name("dropout/keep_prob:0")

        feed_dict = {
            x: [fit_char(char) for char in ocr(img)],
            keep_prob: 1.0,
        }

        argmax = tf.argmax(y, 1)
        classifications = sess.run(argmax, feed_dict)
        return [hottop_to_symbol(c) for c in classifications]

def to_ops(symbols):
    r = []
    
    op = None
    number = ""

    for s in symbols:
        if callable(s):

            if number:
                r.append((op, int(number)))

            op = s
            number = ""
        else:
            number = number + str(s)

    r.append((op, int(number)))
    op = s
    
    return r


def solver(data):
    output = 'numbers.png'
    open(output, 'wb').write(requests.get(data['image_url']).content)

    img = imageio.imread(output)
    symbols = to_symbols(img)
    ops = to_ops(symbols)


    total = 0

    for o in ops:
        op, n = o
        total = op(total, n)

    print(total)

    return {
        'result': total
    }

def collect_data(data):
    output = 'numbers.png'
    open(output, 'wb').write(requests.get(data['image_url']).content)

    img = imageio.imread(output)

    for char in ocr(img):
        imageio.imwrite('./chars/{}.png'.format(uuid.uuid4()), char)

    

def main():
    challange('visual_basic_math', solver, post=True)



if __name__ == '__main__':
    main()
