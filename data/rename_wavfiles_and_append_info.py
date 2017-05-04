from __future__ import print_function
import os
import sys
import csv
import pandas as pd

id_to_pianists = dict([
    ('p01', 'Ashkenazy'),
    ('p02', 'Gould'),
    ('p03', 'Argerich'),
    ('p04', 'Arrau'),
    ('p05', 'Rubinstein'),
    ('p06', 'Pollini')
])


def rename_wavfiles_and_append_info(dir_, filepath):
    """rename all .wav files in a particular pianist's folder 'pxx' to 'pxx_yyy.wav'
    and append the information to pianist_info.txt"""
    # f = open_info_file(_filepath)
    # df = check_info_file(_filepath)
    df = pd.DataFrame()

    # save the old filename of each .wav file,
    # then rename each file and append a new record to the info file
    for root, dirnames, filenames in os.walk(dir_):
        pianist_id = os.path.basename(root)  # folder name would be "pxx", the pianist id
        pianist_name = id_to_pianists[pianist_id]
        new_records = list()
        track_count = 1
        for fn in filenames:
            if fn.endswith('wav'):
                trackid = pianist_id + '_' + '{0:03}'.format(track_count)
                trackname = fn[:-4]
                # rename the file
                os.rename(os.path.join(root, fn), os.path.join(root, trackid + '.wav'))
                new_records.append({
                    'id': pianist_id,
                    'name': pianist_name,
                    'trackid': trackid,
                    'trackname': '"' + trackname + '"'
                })
                track_count += 1
        df = df.append(new_records)

    if os.stat(filepath).st_size == 0:
        df.to_csv(filepath, sep='|', index=False, mode='w', quoting=csv.QUOTE_NONE)
    else:
        df.to_csv(filepath, sep='|', header=False, index=False, mode='a', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    dir_ = sys.argv[1]
    if not os.path.isdir(dir_):
        print('Please specify the correct directory.')
        sys.exit(0)
    filepath = sys.argv[2]
    if not os.path.exists(filepath):
        try:
            f = open(filepath, 'w')
            f.close()
        except IOError:
            print('Error occurred when tring to create {}'.format(filepath))
            sys.exit(0)

    rename_wavfiles_and_append_info(dir_, filepath)
