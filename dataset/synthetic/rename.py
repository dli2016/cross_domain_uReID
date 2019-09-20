import os
import sys

# 1 - duke, 2 - market
def rename(dir_name, flag='1'):
    filelist = os.listdir(dir_name)
    total_num = len(filelist)

    print('%d files will be renamed ...' % total_num)
    cnt = 0
    for item in filelist:
        cnt += 1
        if item.endswith('.jpg'):
            src = os.path.join(dir_name, item)
            item_splited = item.split('_')
            new_pid = flag + item_splited[1]
            item_splited[1] = new_pid
            new_name = ''
            i = 0
            for part in item_splited:
                if i == 0:
                    append_str = part
                else:
                    append_str = '_' + part
                new_name += append_str
                i += 1
            dst = os.path.join(dir_name, new_name)
            print(dst)
            os.rename(src, dst)
    print('%d converted finally!' % cnt)

if __name__ == '__main__':
    dir_name = sys.argv[1]
    flag = sys.argv[2]
    rename(dir_name, flag)
