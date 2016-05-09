# coding=utf-8


class Reader(object):
    def __init__(self):
        self.count = 0

    def read_segmented_content_file(self):
        with open('ordered_segmented_content_file.txt') as f:
            for line in f:
                print '正在读取第{0}行数据'.format(self.count)
                result = line.split(' ')
                for sector in result:
                    print sector
                self.count += 1

if __name__ == '__main__':
    reader = Reader()
    reader.read_segmented_content_file()
