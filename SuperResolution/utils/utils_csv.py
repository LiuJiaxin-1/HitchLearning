import csv
def write_lines(file, info):
    with open(file, 'a') as f:
        f.writelines(info)