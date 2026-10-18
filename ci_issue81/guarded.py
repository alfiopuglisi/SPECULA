import sys
import specula

if __name__ == '__main__':
    specula.main_simul(yml_files=sys.argv[1:], cpu=True)
    print('SIMULATION DONE')
