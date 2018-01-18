from challange import challange
from base64 import b64decode
from struct import unpack
import binascii


"""
useful links:

https://github.com/leonchen83/redis-replicator/wiki/RDB-dump-data-format
https://github.com/sripathikrishnan/redis-rdb-tools/wiki/Redis-RDB-Dump-File-Format

"""

class RDB:
    _translate = {
        0: 'string',
        1: 'list',
        2: 'set',
        3: 'sortedset',
        4: 'hash',
        9: 'zipmap',
        10: 'ziplist',
        11: 'set',
        12: 'ziplist_sortedset',
        13: 'hash',
    }

    def __init__(self, data):
        self.data = data
        self.pos = 0

        self.dbs = []

        self.parse()

    def validate_magic(self):
        self.magic = self.read(5)
        assert self.magic == b'REDIS'

    def get_version(self):
        self.version = self.read(2)
        assert self.version == b'\x00\x04'

    def read(self, n):
        d = self.data[self.pos:self.pos+n]
        self.pos += n

        if n == 1:
            return d[0]

        return d

    def read_until(self, niddle):
        assert len(niddle) == 1

        niddle = niddle[0]
        pos = self.pos

        while True:
            if pos >= len(self.data):
                raise Exception('did not find {}'.format(niddle))

            if self.data[pos] == niddle:
                d = self.data[self.pos:pos]
                self.pos = pos
                return d
            else:
                pos += 1

    def read_special_length(self, first_byte):
        fmt = 0b00111111 & first_byte

        if fmt == 0:
            return self.read(1)
        if fmt == 1:
            s_int = self.read(2)
            return unpack('h', s_int)[0]
        else:
            raise Exception('unimplmementeed')

    def read_length(self):
        first_byte = self.read(1)

        if first_byte & 0b11000000 == 0b00000000:
            return 0b00111111 & first_byte

        elif first_byte & 0b11000000 == 0b01000000:
            two_first = bytes([first_byte & 0b00111111, self.read(1)])
            return unpack('>h', two_first)[0]

        elif first_byte & 0b11000000 == 0b11000000:
            sl = self.read_special_length(first_byte)
            print('in read length', sl)
            return sl

        else:
            print('first_byte ::: {:b} {:x}'.format(first_byte, first_byte))
            raise Exception('should not get here')

    def parse_dbresize(self):
        """
        The first $length represents db key counts in selected db number.
        The second $length represents expired keys in selected db number.
        """

        # after reading selected db, we should be in dbresize
        assert self.data[self.pos] == b'\xfb'[0]
        self.pos += 1

        n_keys = self.read_length()
        n_expired_keys = self.read_length()

        return n_keys, n_expired_keys

    def read_string(self):
        key_length = self.read_length()
        return self.read(key_length)

    def read_record(self, c):
        expiry = None
        value_type = None
        value = None

        if c == b'\xfd'[0]:
            raise Exception('not implmemented')
        elif c == b'\xfc'[0]:
            expiry = unpack('Q', self.read(8))[0]
            print('key_expiry: {}'.format(expiry))

        if expiry is None:
            value_type = c
        else:
            value_type = self.read(1)

        print('value_type: {}'.format(value_type))

        key = self.read_string()
        print('key: {}'.format(key.decode('utf-8')))

        
        if value_type == 0:
            s_len = self.read(1)

            if s_len & 0b11000000 == 0b11000000:
                value = self.read_special_length(s_len)
            else:
                self.pos -= 1
                value = self.read_string()

        elif value_type == 11:
            envelop = self.read_string()
        elif value_type == 13:
            envelop = self.read_string()
        else:
            raise Exception('not implemented')

        print('value: {}'.format(value))
        print('---------')

        value = { 
            'value': value, 
            'expiry': expiry, 
            'value_type': self._translate[value_type]
        }

        return key.decode('utf-8'), value


    def read_records(self):
        records = {}

        print('------- records -------')
        while True:
            c = self.read(1)

            if c == b'\xfe'[0] or c == b'\xff'[0]:
                # end of database
                print('---------- end of records -------')
                self.pos -= 1
                return records

            k, v = self.read_record(c)

            records[k] = v

    def parse_dbselect(self):
        selected_db = self.read_length()
        print('selected db: {}'.format(selected_db))

        n_keys, n_expired_keys = self.parse_dbresize()
        print('keys: {} expired: {}'.format(n_keys, n_expired_keys))

        records = self.read_records()

        return {
            'selected_db': selected_db,
            'n_keys': n_keys,
            'n_expired_keys': n_expired_keys,
            'records': records
        }

    def parse(self):
        self.validate_magic()
        self.get_version()

        # find first dbselect
        d = self.read_until(b'\xfe')

        while True:
            if self.pos >= len(self.data):
                print('done')
                return

            c = self.read(1)
            
            if c == b'\xfe'[0]:
                self.dbs.append(self.parse_dbselect())




def solver(data):
    """
    need to find:
    -------------

    db_count: how many databases were initialized
    emoji_key_value: the value of the emoji key
    expiry_millis: the timestamp in milliseconds of the only key with an expiry time
    $check_type_of: the type of the key named $check_type_of (replace with the actual name)
    """


    looking_for = data['requirements']['check_type_of']
    data = b64decode(data['rdb'])
    data = b'REDIS\x00\x04' + data[9:]
    open('dump.rdb', 'wb').write(data)

    rdb = RDB(data)

    res = {
        'db_count': len(rdb.dbs),
    }

    dbs_len = len(rdb.dbs)

    for db in rdb.dbs:
        for k, v in db['records'].items():
            print('{} -- {}'.format(k, looking_for))
            if k == looking_for:
               res[looking_for] = v['value_type'] 

            if v['expiry'] is not None:
                res['expiry_millis'] = v['expiry']

            if k.encode('utf-8').startswith(b'\xf0'):
                res['emoji_key_value'] = v['value'].decode('utf-8')
    
    print(res)
    return res

def main():
    challange('the_redis_one', solver, post=True)



if __name__ == '__main__':
    main()
