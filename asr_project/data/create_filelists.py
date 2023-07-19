import os
from glob import glob
def dir_to_filelist(dir_path):
    txt_paths = glob(os.path.join(dir_path, 'txt/*'))
    wav_paths = glob(os.path.join(dir_path, 'wav/*'))

    assert len(txt_paths) == len(wav_paths)
    txt_paths = sorted(txt_paths)
    wav_paths = sorted(wav_paths)
    assert all([os.path.basename(p1).split('.')[0] == os.path.basename(p2).split('.')[0] for p1,p2 in zip(txt_paths,wav_paths)])

    lines = []
    for txt_path, wav_path in zip(txt_paths, wav_paths):
        with open(txt_path, 'r+') as f:
            text = f.readline()
        lines.append(os.path.relpath(wav_path) + "|" +  text + "\n")

    with open(os.path.join(dir_path, 'filelist.txt'), 'w+') as g:
        g.writelines(lines)

if __name__ == '__main__':
    pass
    # dir_to_filelist(os.path.abspath('./an4/an4/val/an4'))
    # dir_to_filelist(os.path.abspath('./an4/an4/train/an4'))
    # dir_to_filelist(os.path.abspath('./an4/an4/test/an4'))