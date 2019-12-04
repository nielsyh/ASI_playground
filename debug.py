
class Debug:

    def __init__(self, debug_level):
        self.debug_level = debug_level

    def dprint(self, string_to_print, d_level = self.debug_level):
        if(d_level):
            print(string_to_print)




