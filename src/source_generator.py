import os
import os.path as osp
import sys
def source_generator(rgb_file, normal_file):
    rgb_file_abs = osp.abspath(rgb_file)
    normal_file_abs = osp.abspath(normal_file)
    rgb_files = os.listdir(rgb_file_abs)
    results = []
    for file in rgb_files:
        names = file.split('.')
        n_file = names[0] + '.txt'
        normal_file = osp.join(normal_file_abs, n_file)
        if osp.isfile(normal_file):
            results.append(osp.join(rgb_file_abs, file) + ' ' + normal_file)
    with open('source.txt', 'w') as txt_file:
        for item in results:
            txt_file.write('{}\n'.format(item))

if __name__ == '__main__':
	if len(sys.argv) != 3:
		raise ValueError('please parse: ---/path/to/rgb_dir /path/to/normal_dir')
		rgb = sys.argv[1]
		normal = sys.argv[2]
		source_generator(rgb, normal)





    
