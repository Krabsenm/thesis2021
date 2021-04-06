
def str_to_list_fn(type=float, sep=','):
    def str_to_list(s):
        try:
            return [type(x) for x in s.split(sep)]
        except ValueError:
            raise ValueError('One or more of {} are not of type {}'.format(s.split(sep), type.__name__))
    return str_to_list