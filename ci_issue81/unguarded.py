import sys
import specula
specula.main_simul(yml_files=sys.argv[1:], cpu=True)
print('SIMULATION DONE')
