
class BaseList(list):
    '''
    Extend the standard Python list with a generation_time property
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @property
    def generation_time(self):
        '''Return the most recent generation time in the list'''
        if len(self) > 0:
            return max([x.generation_time for x in self])
        else:
            # Empty list, return the same as an uninitialized generation_time
            return -1
