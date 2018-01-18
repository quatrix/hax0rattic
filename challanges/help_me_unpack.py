from challange import challange
from base64 import b64decode
from struct import unpack


def solver(data):
    """
    int: the signed integer value
    uint: the unsigned integer value
    short: the decoded short value
    float: surprisingly, the float value
    double: the double value - shockigly
    big_endian_double: you get the idea by now!
    """

    binary = b64decode(data['bytes'])

    res = unpack('iIhfd', binary[0:24])

    solution = {
        'int': res[0],
        'uint': res[1],
        'short': res[2],
        'float': res[3],
        'double': res[4],
        'big_endian_double': unpack('>d', binary[24:])[0],
    }

    return solution

def main():
    challange('help_me_unpack', solver, post=True)


if __name__ == '__main__':
    main()
